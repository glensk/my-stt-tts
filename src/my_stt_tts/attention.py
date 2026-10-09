# pylint: disable=too-many-lines  # one cohesive component: inbox store, monitor, briefing
"""Attention inbox of the Claude/Mac bridge: problems only, never successes (D6).

* :class:`AttentionStore` — a small sqlite inbox (``~/.local/state/mac-voice/attention.db``,
  directory 0700, db / -wal / -shm 0600, checked and fixed on every open). One row per
  problem, keyed by a dedupe key (the id is derived from it); a repeat bumps ``last_seen``.
  Items stay until a voice-verified per-item acknowledgement, or expire 7 days after they
  were last seen. It is the bridge's :class:`~my_stt_tts.bridge.ProblemSink`,
  :class:`~my_stt_tts.bridge.DeliveryTracker` and
  :class:`~my_stt_tts.bridge.BriefingProvider`. A corrupt db (never a locked one) is
  renamed ``attention.db.corrupt-<ts>``, rebuilt, and that is logged as one problem.
* :class:`Monitor` — a thread for the daemon's lifetime: every 15 s it reads
  ``ccc events --after CURSOR -j`` (cursor persisted, catch-up at start, idempotent replay:
  an already-seen event never reopens an acknowledged item) and asks
  ``ccc delivery -i ID -j`` about the ``unknown`` voice deliveries (accepted ones only
  every 5 min, to learn they completed). Alerts: a failed delivery, a
  StopFailure or a needs-input that follows a voice delivery, a delivery still ``unknown``
  after a 5-min grace (resolved automatically by later evidence). Long-running work is
  never an alert. New problems are spoken in a call through ``notify_in_call`` (Phase 7
  maps it to ``send_user_message("[system notice] …")``), otherwise they wait in the inbox
  and raise an optional content-free macOS banner.
* :func:`first_message_override` — the briefing for the next "voice on" as
  ``conversation_config_override``; the caller calls :meth:`AttentionStore.mark_briefed`
  once the conversation really started, :func:`report_startup_failure` when it did not.
* :func:`register` — wires store, monitor and the ``acknowledge_problem`` tool into a
  :class:`~my_stt_tts.bridge.BridgeController` when a bridge flag is on.

Nothing here stores message text: a row holds a kind, a subject (session name or action),
a short reason and timestamps.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import sqlite3
import stat
import subprocess
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bridge import (
    BRIEFING_CAP,
    BriefingProvider,
    Problem,
    Refusal,
    bridge_enabled,
    log_refused,
    log_tool,
)
from .bridge_text import normalise, redact
from .claude_sessions import resolve_ccc, run_group

if TYPE_CHECKING:
    from .bridge import BridgeController

log = logging.getLogger("my_stt_tts.attention")

DB_NAME = "attention.db"
DIR_MODE = 0o700
FILE_MODE = 0o600
EXPIRY_S = 7 * 24 * 3600.0
GRACE_S = 5 * 60.0
POLL_INTERVAL_S = 15.0
EVENTS_TIMEOUT_S = 10.0  # ccc events (it advances every open delivery first)
DELIVERY_TIMEOUT_S = 5.0  # ccc delivery -i
BANNER_TIMEOUT_S = 5.0  # osascript
#: stop() waits this long for a tick: one ccc call + one banner are the most in flight.
JOIN_TIMEOUT_S = EVENTS_TIMEOUT_S + BANNER_TIMEOUT_S + 1.0
ACCEPTED_POLL_S = 300.0  # accepted deliveries: only to learn they completed
DELIVERY_WINDOW_S = 6 * 3600.0  # a delivery is timed_out after 6 h without a Stop
CLOCK_SKEW_S = 60.0
FAILURES_BEFORE_PROBLEM = 3
EVENT_PAGES_PER_TICK = 20
SUBJECT_CAP = 60
REASON_CAP = 120
SCHEMA_VERSION = 1
SEVERITY_RANK = {"critical": 3, "error": 2, "warning": 1, "info": 0}

CURSOR_KEY = "events_cursor"
TERMINAL_STATES = frozenset({"completed", "needs_input", "failed", "timed_out"})

Runner = Callable[[Sequence[str], float], tuple[int, str]]
Listener = Callable[["Item", bool], None]


def default_path() -> Path:
    """``~/.local/state/mac-voice/attention.db`` (resolved at call time)."""
    return Path.home() / ".local" / "state" / "mac-voice" / DB_NAME


def run_command(argv: Sequence[str], timeout: float) -> tuple[int, str]:
    """``(exit code, stdout)`` of ``argv`` (no shell, empty stdin, stderr dropped).

    Runs in its own process group, killed as a whole on timeout (see
    :func:`~my_stt_tts.claude_sessions.run_group`).
    """
    try:
        res = run_group(argv, None, timeout)
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError:
        return 126, ""
    return res.returncode, res.stdout


# -- sanitising -------------------------------------------------------------------------------
def clean(text: object, cap: int) -> str:
    """Control characters → spaces, whitespace collapsed, secrets redacted, capped."""
    raw = str(text or "")
    plain = "".join(" " if unicodedata.category(ch) in {"Cc", "Cf"} else ch for ch in raw)
    return redact(" ".join(plain.split()), cap)


def problem_id(dedupe_key: str) -> str:
    """The deterministic id of the problem with ``dedupe_key``."""
    return hashlib.sha256(dedupe_key.encode()).hexdigest()[:16]


def default_dedupe_key(problem: Problem) -> str:
    """``<kind>:<normalised subject>`` — one item per thing that went wrong."""
    return f"{normalise(problem.kind) or 'problem'}:{normalise(problem.subject)}"


def _event_number(cursor: str | None) -> int | None:
    """The numeric ccc events cursor of a ``event:<n>`` source cursor, else None."""
    if not cursor or not cursor.startswith("event:"):
        return None
    tail = cursor.removeprefix("event:")
    return int(tail) if tail.isdigit() else None


def _as_int(value: object) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if text.isdigit() else None


def _parse_at(value: object, fallback: float) -> float:
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.timestamp()


# -- records ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Item:  # pylint: disable=too-many-instance-attributes  # one inbox row
    """One inbox row (times are epoch seconds; None = not yet)."""

    id: str
    dedupe_key: str
    kind: str
    subject: str
    reason: str
    severity: str
    opened_at: float
    last_seen: float
    resolved_at: float | None
    briefed_at: float | None
    acknowledged_at: float | None
    source_cursor: str | None
    expires_at: float
    count: int
    event_cursor: int | None = None  # highest ccc events cursor seen for this item

    def is_open(self, now: float) -> bool:
        return self.resolved_at is None and self.acknowledged_at is None and self.expires_at > now

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, SEVERITY_RANK["error"])


@dataclass(frozen=True)
class Delivery:
    """A message the voice sent to a session, watched until a terminal state."""

    delivery_id: str
    session_id: str
    subject: str
    state: str
    created_at: float
    unknown_since: float | None
    closed_at: float | None


_ITEM_COLUMNS = (
    "id, dedupe_key, kind, subject, reason, severity, opened_at, last_seen, resolved_at, "
    "briefed_at, acknowledged_at, source_cursor, expires_at, count, event_cursor"
)
_DELIVERY_COLUMNS = "delivery_id, session_id, subject, state, created_at, unknown_since, closed_at"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS problems (
    id TEXT PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    reason TEXT NOT NULL,
    severity TEXT NOT NULL,
    opened_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    resolved_at REAL,
    briefed_at REAL,
    acknowledged_at REAL,
    source_cursor TEXT,
    expires_at REAL NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    event_cursor INTEGER
);
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    unknown_since REAL,
    closed_at REAL
);
CREATE INDEX IF NOT EXISTS deliveries_session ON deliveries(session_id, closed_at);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


# -- the inbox --------------------------------------------------------------------------------
class AttentionStore:
    """The attention inbox (see module docstring). Thread-safe; one connection."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.clock: Callable[[], float] = clock
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._last_briefing: tuple[str, ...] = ()
        self._conn = self._open()

    # -- files ------------------------------------------------------------------------
    def _open(self) -> sqlite3.Connection:
        self._prepare_dir()
        try:
            conn = self._connect()
        except sqlite3.DatabaseError as exc:
            if not _is_corrupt(exc):  # locked / cannot open: not ours to throw away
                log.warning("⚠️  attention inbox unavailable (%s)", type(exc).__name__)
                raise
            log.warning("⚠️  attention inbox corrupt — quarantined and rebuilt")
            quarantined = self._quarantine()
            conn = self._connect()
            self._conn = conn
            self.record(
                Problem(
                    "attention_db_corrupt",
                    "attention inbox",
                    "the inbox was corrupt and has been rebuilt",
                    "warning",
                ),
                source_cursor=f"corrupt:{quarantined.name}",
            )
        return conn

    def _prepare_dir(self) -> None:
        directory = self.path.parent
        directory.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
        _ensure_mode(directory, DIR_MODE)
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, FILE_MODE)
            os.close(fd)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            row = conn.execute("PRAGMA quick_check").fetchone()
            if row is None or row[0] != "ok":
                raise sqlite3.DatabaseError("quick_check failed")
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(problems)")}
            if "event_cursor" not in columns:  # additive migration of an older inbox
                conn.execute("ALTER TABLE problems ADD COLUMN event_cursor INTEGER")
            conn.commit()
        except sqlite3.DatabaseError:
            conn.close()
            raise
        self.fix_modes()
        return conn

    def _quarantine(self) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}")
        n = 1
        while target.exists():
            target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{n}")
            n += 1
        self.path.rename(target)
        for suffix in ("-wal", "-shm"):
            side = self.path.with_name(self.path.name + suffix)
            if side.exists():
                side.rename(target.with_name(target.name + suffix))
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, FILE_MODE)
        os.close(fd)
        return target

    def fix_modes(self) -> list[Path]:
        """Directory 0700, db / -wal / -shm 0600; returns the paths that had to be fixed."""
        fixed = []
        if _ensure_mode(self.path.parent, DIR_MODE):
            fixed.append(self.path.parent)
        for suffix in ("", "-wal", "-shm"):
            target = self.path.with_name(self.path.name + suffix)
            if target.exists() and _ensure_mode(target, FILE_MODE):
                fixed.append(target)
        return fixed

    def close(self) -> None:
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._conn.close()

    # -- listeners --------------------------------------------------------------------
    def add_listener(self, fn: Listener) -> None:
        """``fn(item, announce)`` for every newly opened or reopened problem."""
        self._listeners.append(fn)

    def _emit(self, item: Item, announce: bool) -> None:
        for fn in list(self._listeners):
            try:
                fn(item, announce)
            except Exception:  # pylint: disable=broad-exception-caught  # never break a report
                log.warning("⚠️  attention listener failed", exc_info=True)

    # -- problems ---------------------------------------------------------------------
    def report(  # pylint: disable=too-many-arguments
        self,
        problem: Problem,
        *,
        dedupe_key: str | None = None,
        source_cursor: str | None = None,
        at: float | None = None,
        announce: bool = False,
    ) -> None:
        """:class:`~my_stt_tts.bridge.ProblemSink` entry point (see :meth:`record`)."""
        self.record(
            problem, dedupe_key=dedupe_key, source_cursor=source_cursor, at=at, announce=announce
        )

    def record(  # pylint: disable=too-many-arguments
        self,
        problem: Problem,
        *,
        dedupe_key: str | None = None,
        source_cursor: str | None = None,
        at: float | None = None,
        announce: bool = False,
    ) -> str:
        """Record ``problem``: ``opened`` / ``reopened`` / ``bumped`` / ``noop`` (a replay).

        ``announce`` asks for the problem to be spoken in a running call; direct reports
        from a tool default to False because the failing tool already tells the agent.
        A replayed source event never reopens an item: the same ``source_cursor``, a ccc
        events cursor at or below the item's highest one, or an event time ``at`` at or
        before its acknowledgement / resolution.
        """
        key = dedupe_key or default_dedupe_key(problem)
        now = self.clock()
        with self._lock:
            status, item = self._upsert(problem, key, _Source(source_cursor, at), now)
        if status in {"opened", "reopened"}:
            log.info("⚠️  problem: %s · %s", item.kind, item.subject)
            self._emit(item, announce)
        return status

    def _upsert(self, problem: Problem, key: str, src: _Source, now: float) -> tuple[str, Item]:
        subject = clean(problem.subject, SUBJECT_CAP) or "unknown"
        reason = clean(problem.reason, REASON_CAP) or problem.kind
        kind = clean(problem.kind, 40) or "problem"
        severity = problem.severity if problem.severity in SEVERITY_RANK else "error"
        cursor, seen = src.cursor, now if src.at is None else src.at
        current = self._get(key)
        if current is None:
            self._conn.execute(
                f"INSERT INTO problems ({_ITEM_COLUMNS}) "
                "VALUES (?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,?,1,?)",
                (
                    problem_id(key),
                    key,
                    kind,
                    subject,
                    reason,
                    severity,
                    seen,
                    seen,
                    cursor,
                    seen + EXPIRY_S,
                    src.event,
                ),
            )
            status = "opened"
        elif _replayed(current, src, now):
            return "noop", current  # an already-seen source event: idempotent replay
        else:
            events = [n for n in (current.event_cursor, src.event) if n is not None]
            event_cursor = max(events) if events else None
            if current.is_open(now):
                rank = max(current.rank, SEVERITY_RANK[severity])
                worst = next(name for name, r in SEVERITY_RANK.items() if r == rank)
                last = max(current.last_seen, seen)
                self._conn.execute(
                    "UPDATE problems SET last_seen=?, expires_at=?, reason=?, severity=?, "
                    "source_cursor=COALESCE(?, source_cursor), count=count+1, event_cursor=? "
                    "WHERE dedupe_key=?",
                    (last, last + EXPIRY_S, reason, worst, cursor, event_cursor, key),
                )
                status = "bumped"
            else:
                self._conn.execute(
                    "UPDATE problems SET kind=?, subject=?, reason=?, severity=?, opened_at=?, "
                    "last_seen=?, resolved_at=NULL, briefed_at=NULL, acknowledged_at=NULL, "
                    "source_cursor=?, expires_at=?, count=1, event_cursor=? WHERE dedupe_key=?",
                    (
                        kind,
                        subject,
                        reason,
                        severity,
                        seen,
                        seen,
                        cursor,
                        seen + EXPIRY_S,
                        event_cursor,
                        key,
                    ),
                )
                status = "reopened"
        self._conn.commit()
        item = self._get(key)
        assert item is not None
        return status, item

    def _get(self, key: str) -> Item | None:
        row = self._conn.execute(
            f"SELECT {_ITEM_COLUMNS} FROM problems WHERE dedupe_key=?", (key,)
        ).fetchone()
        return Item(*row) if row else None

    def get(self, dedupe_key: str) -> Item | None:
        with self._lock:
            return self._get(dedupe_key)

    def items(self) -> list[Item]:
        """Every row, open or not."""
        with self._lock:
            rows = self._conn.execute(f"SELECT {_ITEM_COLUMNS} FROM problems").fetchall()
        return [Item(*row) for row in rows]

    def open_items(self) -> list[Item]:
        """Unresolved, unacknowledged, unexpired — most severe first, unbriefed first."""
        now = self.clock()
        live = [item for item in self.items() if item.is_open(now)]
        return sorted(live, key=lambda i: (-i.rank, i.briefed_at is not None, -i.last_seen))

    def resolve(self, dedupe_key: str) -> bool:
        """Close an open problem because later evidence settled it (auto-resolve)."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE problems SET resolved_at=? WHERE dedupe_key=? AND resolved_at IS NULL "
                "AND acknowledged_at IS NULL",
                (self.clock(), dedupe_key),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def acknowledge(self, subject: str) -> int:
        """Acknowledge every open item about ``subject`` (normalised match); count."""
        wanted = normalise(subject)
        if not wanted:
            return 0
        now = self.clock()
        ids = [i.id for i in self.open_items() if normalise(i.subject) == wanted]
        with self._lock:
            for item_id in ids:
                self._conn.execute(
                    "UPDATE problems SET acknowledged_at=? WHERE id=?", (now, item_id)
                )
            self._conn.commit()
        return len(ids)

    def expire(self) -> int:
        """Delete items not seen for 7 days (acknowledged or not); count."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM problems WHERE expires_at<=?", (self.clock(),))
            self._conn.execute(
                "DELETE FROM deliveries WHERE created_at<=?", (self.clock() - EXPIRY_S,)
            )
            self._conn.commit()
            return cur.rowcount

    # -- briefing ---------------------------------------------------------------------
    def briefing(self) -> str | None:
        """One spoken paragraph about the open items (≤ 600 chars), or None."""
        items = self.open_items()
        text, used = compose_briefing(items)
        self._last_briefing = tuple(i.id for i in used)
        return text

    def mark_briefed(self, ids: Iterable[str] | None = None) -> int:
        """Stamp ``briefed_at`` (default: the items of the last :meth:`briefing`)."""
        chosen = tuple(self._last_briefing if ids is None else ids)
        now = self.clock()
        with self._lock:
            for item_id in chosen:
                self._conn.execute("UPDATE problems SET briefed_at=? WHERE id=?", (now, item_id))
            self._conn.commit()
        if ids is None:
            self._last_briefing = ()
        return len(chosen)

    # -- meta (event cursor) ----------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def forget_event_cursors(self) -> None:
        """ccc renumbered its events: the stored item cursors no longer compare."""
        with self._lock:
            self._conn.execute("UPDATE problems SET event_cursor=NULL")
            self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    # -- deliveries -------------------------------------------------------------------
    def track_delivery(self, delivery_id: str, session_id: str, subject: str, outcome: str) -> None:
        """:class:`~my_stt_tts.bridge.DeliveryTracker`: watch a voice delivery.

        ``outcome`` is the ``ccc send`` outcome, ``accepted`` or ``unknown``.
        """
        now = self.clock()
        unknown_since = now if outcome == "unknown" else None
        with self._lock:
            self._conn.execute(
                f"INSERT INTO deliveries ({_DELIVERY_COLUMNS}) VALUES (?,?,?,?,?,?,NULL) "
                "ON CONFLICT(delivery_id) DO NOTHING",
                (
                    delivery_id,
                    session_id,
                    clean(subject, SUBJECT_CAP) or "unknown",
                    outcome,
                    now,
                    unknown_since,
                ),
            )
            self._conn.commit()

    def open_deliveries(self) -> list[Delivery]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE closed_at IS NULL "
                "ORDER BY created_at"
            ).fetchall()
        return [Delivery(*row) for row in rows]

    def update_delivery(self, delivery_id: str, state: str, *, close: bool = False) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE deliveries SET state=?, "
                "unknown_since=CASE WHEN ?='unknown' THEN unknown_since ELSE NULL END, "
                "closed_at=CASE WHEN ? THEN ? ELSE closed_at END WHERE delivery_id=?",
                (state, state, int(close), self.clock(), delivery_id),
            )
            self._conn.commit()

    def correlated(
        self, session_id: str, at: float, delivery_id: str | None = None
    ) -> Delivery | None:
        """The voice delivery an event about ``session_id`` at ``at`` belongs to, if any."""
        with self._lock:
            if delivery_id:
                row = self._conn.execute(
                    f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE delivery_id=?",
                    (delivery_id,),
                ).fetchone()
                if row:
                    return Delivery(*row)
            row = self._conn.execute(
                f"SELECT {_DELIVERY_COLUMNS} FROM deliveries WHERE session_id=? "
                "AND created_at<=? AND ?<=created_at+? AND (closed_at IS NULL OR closed_at>=?) "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id, at + CLOCK_SKEW_S, at, DELIVERY_WINDOW_S, at - CLOCK_SKEW_S),
            ).fetchone()
        return Delivery(*row) if row else None


@dataclass(frozen=True)
class _Source:
    """Where a report comes from: its source cursor and, for events, the event time."""

    cursor: str | None
    at: float | None

    @property
    def event(self) -> int | None:
        return _event_number(self.cursor)


def _replayed(current: Item, src: _Source, now: float) -> bool:
    """True when ``src`` was already seen for ``current`` (a replay must not reopen it)."""
    if src.cursor is not None and src.cursor == current.source_cursor:
        return True
    known = current.event_cursor
    if src.event is not None and known is not None and src.event <= known:
        return True
    if src.at is None or current.is_open(now):
        return False
    closed = max(current.acknowledged_at or 0.0, current.resolved_at or 0.0)
    return bool(closed) and src.at <= closed


_CORRUPT_MARKERS = ("file is not a database", "malformed", "quick_check failed")


def _is_corrupt(exc: sqlite3.DatabaseError) -> bool:
    """Real corruption only: a lock or an unopenable file is an OperationalError."""
    text = str(exc).casefold()
    if any(marker in text for marker in _CORRUPT_MARKERS):
        return True
    return not isinstance(exc, sqlite3.OperationalError)


def _ensure_mode(path: Path, mode: int) -> bool:
    """chmod ``path`` to ``mode`` if it differs; True when it had to be fixed."""
    current = stat.S_IMODE(path.stat().st_mode)
    if current == mode:
        return False
    log.warning("⚠️  fixed mode of %s: %o → %o", path.name, current, mode)
    path.chmod(mode)
    return True


def notice_text(item: Item) -> str:
    """The spoken line for one problem: ``<subject>: <reason>``."""
    return f"{clean(item.subject, SUBJECT_CAP)}: {clean(item.reason, REASON_CAP)}"


def compose_briefing(items: Sequence[Item], cap: int = BRIEFING_CAP) -> tuple[str | None, list]:
    """``(paragraph, items it names)``: in order, as many as fit, then "and N more"."""
    if not items:
        return None, []
    total = len(items)
    head = (
        "One problem needs your attention."
        if total == 1
        else f"{total} problems need your attention."
    )
    tail = " Say got it and the name to clear one."

    def render(chosen: list[str], with_tail: bool) -> str:
        rest = total - len(chosen)
        body = (" " + " ".join(chosen)) if chosen else ""
        more = f" And {rest} more." if rest else ""
        return head + body + more + (tail if with_tail else "")

    chosen: list[str] = []
    for item in items:
        if len(render([*chosen, f"{notice_text(item)}."], True)) > cap:
            break
        chosen.append(f"{notice_text(item)}.")
    text = render(chosen, True)
    if len(text) > cap:
        text = render(chosen, False)
    return clean(text, cap), list(items[: len(chosen)])


# -- ccc envelopes ----------------------------------------------------------------------------
class EnvelopeError(Exception):
    """A ccc ``-j`` answer that is not a usable ``ok`` envelope."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


def parse_envelope(returncode: int, stdout: str) -> Mapping[str, Any]:
    """The ``data`` of a ccc JSON envelope (section 6), else :class:`EnvelopeError`."""
    import json  # pylint: disable=import-outside-toplevel  # keep module import light

    try:
        doc = json.loads(stdout)
    except (TypeError, ValueError) as exc:
        raise EnvelopeError("bad_json", f"exit {returncode}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION:
        raise EnvelopeError("bad_envelope", "schema_version")
    if not doc.get("ok"):
        error = doc.get("error")
        if not isinstance(error, dict):
            error = {}
        raise EnvelopeError(str(error.get("code") or "failed"), str(error.get("message") or ""))
    if returncode != 0:
        raise EnvelopeError("exit_mismatch", f"exit {returncode}")
    data = doc.get("data")
    if not isinstance(data, dict):
        raise EnvelopeError("bad_envelope", "data")
    return data


# -- monitor ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Alert:
    kind: str
    reason: str
    severity: str
    key_prefix: str


_EVENT_ALERTS = {
    "stop_failure": _Alert("stop_failure", "stopped with an error", "error", "failed"),
    "delivery_failed": _Alert("delivery_failed", "message failed", "error", "failed"),
    "needs_input": _Alert("needs_input", "needs your input", "warning", "needs_input"),
}


def _sanitise_kind(kind: str) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", str(kind).casefold()).strip()
    return words[:40].strip() or "problem"


def banner_argv(kind: str) -> list[str]:
    """``osascript`` argv for a content-free banner: only the problem kind, no subject."""
    script = f'display notification "{_sanitise_kind(kind)} needs attention" with title "mac-voice"'
    return ["osascript", "-e", script]


class Monitor:  # pylint: disable=too-many-instance-attributes  # the inbox's poller
    """Polls ccc events + voice deliveries for problems (see module docstring)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        store: AttentionStore,
        runner: Runner | None = None,
        *,
        interval: float = POLL_INTERVAL_S,
        notify_in_call: Callable[[str], Any] | None = None,
        call_active: Callable[[], bool] | None = None,
        banner: bool = True,
        ccc: Sequence[str] | None = None,
        grace: float = GRACE_S,
    ) -> None:
        self.store = store
        self.runner: Runner = runner or run_command
        self.interval = interval
        self.notify_in_call = notify_in_call
        self.call_active = call_active or (lambda: False)
        self.banner = banner
        self.ccc = tuple(ccc) if ccc is not None else None  # None: resolve_ccc() per call
        self.grace = grace
        self.failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_banners: set[str] | None = None
        self._tick_owner: int | None = None
        self._tick_lock = threading.Lock()
        self._polled: dict[str, float] = {}  # delivery_id → last `ccc delivery` poll
        store.add_listener(self._on_new)

    # -- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        """Start the poller thread (first tick at once: startup catch-up)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="attention-monitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> bool:
        """Stop and join the thread; False when it is still running after ``timeout``.

        The default waits :data:`JOIN_TIMEOUT_S` — the longest a tick takes to notice the
        stop (one ccc call and one banner). The store must stay open while it runs.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_TIMEOUT_S if timeout is None else timeout)
            if thread.is_alive():
                log.warning("⚠️  attention monitor still running — store left open")
                return False
        self._thread = None
        return True

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def set_notifier(self, fn: Callable[[str], Any] | None) -> None:
        """Phase 7: ``fn(text)`` speaks a problem in the call (``[system notice] <text>``)."""
        self.notify_in_call = fn

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # pylint: disable=broad-exception-caught  # keep polling
                log.warning("⚠️  attention monitor tick failed", exc_info=True)
            self._stop.wait(self.interval)

    # -- one tick ---------------------------------------------------------------------
    def poll_once(self) -> None:
        """Expire old items, read new events, check open deliveries, flush banners."""
        with self._tick_lock:
            self._tick_banners, self._tick_owner = set(), threading.get_ident()
            try:
                self.store.expire()
                self._poll_events()
                self._poll_deliveries()
            finally:
                kinds, self._tick_banners, self._tick_owner = self._tick_banners, None, None
            for kind in sorted(kinds):
                if self.stopping:
                    break  # the items stay in the inbox
                self._banner(kind)

    def _ccc(self, args: Sequence[str], timeout: float) -> Mapping[str, Any]:
        prefix = self.ccc
        if prefix is None:
            exe = resolve_ccc()
            if exe is None:
                raise EnvelopeError("ccc_missing", "ccc not found")
            prefix = (exe,)
        code, out = self.runner([*prefix, *args], timeout)
        return parse_envelope(code, out)

    def _poll_events(self) -> None:
        for _ in range(EVENT_PAGES_PER_TICK):
            if self.stopping:
                return
            cursor = self.store.get_meta(CURSOR_KEY)
            args = ["events", *(["--after", cursor] if cursor else []), "-j"]
            try:
                data = self._ccc(args, EVENTS_TIMEOUT_S)
                events = data.get("events")
                if not isinstance(events, list):
                    raise EnvelopeError("bad_envelope", "events")
            except EnvelopeError as exc:
                self._events_failed(exc)
                return
            self._events_ok()
            next_cursor = data.get("next_cursor")
            if next_cursor is None and events and isinstance(events[-1], dict):
                next_cursor = events[-1].get("cursor")
            self._check_rebuilt(cursor, next_cursor)
            for event in events:
                self._handle_event(event)
            if next_cursor is None or str(next_cursor) == (cursor or ""):
                return
            self.store.set_meta(CURSOR_KEY, str(next_cursor))
            if not events:
                return

    def _check_rebuilt(self, cursor: str | None, next_cursor: object) -> None:
        """A ``next_cursor`` below ours: ccc's events table was rebuilt — start over."""
        old, new = _as_int(cursor), _as_int(next_cursor)
        if old is None or new is None or new >= old:
            return
        log.warning("⚠️  ccc events cursor went back (%d → %d) — reset", old, new)
        self.store.forget_event_cursors()

    def _events_failed(self, exc: EnvelopeError) -> None:
        self.failures += 1
        if self.failures == 1:
            log.warning("⚠️  ccc events unreadable (%s)", exc.code)
        if self.failures >= FAILURES_BEFORE_PROBLEM:
            self.store.record(
                Problem("monitor_failed", "ccc events", "cannot read session events", "warning"),
                dedupe_key="monitor:events",
            )

    def _events_ok(self) -> None:
        if self.failures:
            self.failures = 0
            self.store.resolve("monitor:events")

    def _handle_event(self, event: object) -> None:
        if not isinstance(event, dict):
            return
        alert = _EVENT_ALERTS.get(str(event.get("kind", "")))
        session_id = event.get("session_id")
        if alert is None or not isinstance(session_id, str) or not session_id:
            return  # unknown kinds are ignored (forward compatible)
        now = self.store.clock()
        at = _parse_at(event.get("at"), now)
        if at < now - EXPIRY_S:
            return  # older than the inbox keeps anything
        raw = event.get("detail")
        detail: Mapping[str, Any] = raw if isinstance(raw, dict) else {}
        delivery_id = str(detail.get("delivery_id") or "") or None
        delivery = self.store.correlated(session_id, at, delivery_id)
        if delivery is None:
            return  # not after a voice delivery: not the voice's problem
        state = str(detail.get("state") or "")
        if delivery_id and delivery.delivery_id == delivery_id and state in TERMINAL_STATES:
            self.store.resolve(f"unconfirmed:{delivery_id}")
            self.store.update_delivery(delivery_id, state, close=True)  # ccc advanced it
        if state == "timed_out" and detail.get("reason") == "no_turn_end":
            return  # long-running work, not a problem
        cursor = event.get("cursor")
        self.store.report(
            Problem(alert.kind, delivery.subject, alert.reason, alert.severity),
            dedupe_key=f"{alert.key_prefix}:{session_id}",
            source_cursor=f"event:{cursor}" if cursor is not None else None,
            at=at,
            announce=True,
        )

    def _poll_deliveries(self) -> None:
        """``ccc delivery -i`` for ``unknown`` rows every tick; ``events`` advances the rest.

        Accepted rows are asked only every :data:`ACCEPTED_POLL_S` to learn that they
        completed (completion emits no event), which closes their correlation window.
        """
        now = self.store.clock()
        open_ids = set()
        for delivery in self.store.open_deliveries():
            open_ids.add(delivery.delivery_id)
            if self.stopping:
                return
            last = self._polled.get(delivery.delivery_id)
            if delivery.state != "unknown" and last is not None and now - last < ACCEPTED_POLL_S:
                continue
            self._polled[delivery.delivery_id] = now
            try:
                data = self._ccc(["delivery", "-i", delivery.delivery_id, "-j"], DELIVERY_TIMEOUT_S)
            except EnvelopeError:
                continue  # no evidence either way: keep the last known state
            self._delivery_state(delivery, str(data.get("state") or data.get("outcome") or ""))
        for gone in set(self._polled) - open_ids:
            del self._polled[gone]

    def _delivery_state(self, delivery: Delivery, state: str) -> None:
        unconfirmed = f"unconfirmed:{delivery.delivery_id}"
        if state in {"", "unknown", "sending"}:
            since = delivery.unknown_since
            if since is not None and self.store.clock() - since >= self.grace:
                self.store.report(
                    Problem(
                        "delivery_unknown", delivery.subject, "message not confirmed", "warning"
                    ),
                    dedupe_key=unconfirmed,
                    source_cursor=unconfirmed,
                    announce=True,
                )
            return
        self.store.resolve(unconfirmed)  # any later evidence settles the grace
        if state == "needs_input":
            self._alert(
                delivery, _EVENT_ALERTS["needs_input"], f"needs_input:{delivery.session_id}"
            )
        elif state == "failed":
            self._alert(delivery, _EVENT_ALERTS["delivery_failed"], f"failed:{delivery.session_id}")
        # completed / timed_out (long-running work) / accepted: silent
        self.store.update_delivery(delivery.delivery_id, state, close=state in TERMINAL_STATES)

    def _alert(self, delivery: Delivery, alert: _Alert, key: str) -> None:
        self.store.report(
            Problem(alert.kind, delivery.subject, alert.reason, alert.severity),
            dedupe_key=key,
            source_cursor=f"delivery:{delivery.delivery_id}:{alert.kind}",
            announce=True,
        )

    # -- delivery of new problems -----------------------------------------------------
    def _on_new(self, item: Item, announce: bool) -> None:
        if self.call_active():
            if announce and self._notify(item):
                self.store.mark_briefed([item.id])
            return
        banners = self._tick_banners
        if banners is not None and self._tick_owner == threading.get_ident():
            banners.add(item.kind)  # coalesced: one banner per kind and tick
        else:
            self._banner(item.kind)

    def _notify(self, item: Item) -> bool:
        notify = self.notify_in_call
        if notify is None:
            return False
        try:
            result = notify(notice_text(item))
        except Exception:  # pylint: disable=broad-exception-caught  # stays in the inbox
            log.warning("⚠️  in-call notice failed", exc_info=True)
            return False
        return result is not False

    def _banner(self, kind: str) -> None:
        if not self.banner:
            return
        post_banner(kind, self.runner)


def post_banner(kind: str, runner: Runner | None = None) -> bool:
    """A content-free macOS banner ("<kind> needs attention"); True when osascript ran."""
    code, _ = (runner or run_command)(banner_argv(kind), BANNER_TIMEOUT_S)
    return code == 0


# -- briefing at "voice on" -------------------------------------------------------------------
def first_message_override(store: BriefingProvider) -> dict[str, Any] | None:
    """``{"agent": {"first_message": <briefing>}}`` for ``conversation_config_override``.

    Needs the agent's platform setting that allows the ``first_message`` override (Phase 7).
    Call ``store.mark_briefed()`` only once the conversation has really started.
    """
    text = store.briefing()
    if not text:
        return None
    return {"agent": {"first_message": clean(text, BRIEFING_CAP)}}


def startup_failure_reason(exc: BaseException | str) -> str:
    """A short speakable reason for a failed conversation start (quota, auth, network)."""
    text = str(exc).casefold()
    if "quota_exceeded" in text or "quota exceeded" in text:
        return "ElevenLabs quota exceeded"
    if "1008" in text or "not allowed" in text:
        return "agent refused the start settings"
    if "401" in text or "unauthor" in text:
        return "ElevenLabs key rejected"
    return "voice agent did not start"


def report_startup_failure(
    store: AttentionStore, reason: str, *, banner: bool = False, runner: Runner | None = None
) -> str:
    """Record a silent SDK startup failure (e.g. ``3000 [quota_exceeded]``) as a problem.

    Unbriefed items stay unbriefed, so the next call briefs them again. A running
    :class:`Monitor` already raises the banner for the new item; pass ``banner=True``
    only when there is none.
    """
    status = store.record(
        Problem("voice_startup_failed", "voice agent", clean(reason, REASON_CAP), "error"),
        dedupe_key="startup:voice",
    )
    if banner:
        post_banner("voice startup failed", runner)
    return status


# -- the acknowledge tool + wiring -------------------------------------------------------------
def make_acknowledge_tool(
    controller: BridgeController, store: AttentionStore
) -> Callable[[str], str]:
    """``acknowledge_problem(subject)``: voice-verified "got it, <subject>" clears it."""

    def acknowledge_problem(subject: str) -> str:
        log_tool("acknowledge_problem", subject)
        cap = controller.mint()
        if isinstance(cap, Refusal):
            return f"refused: {cap.reason}"
        refusal = controller.caps.use(cap, "acknowledge_problem", {"subject": subject})
        if refusal is not None:
            return f"refused: {refusal.reason}"
        count = store.acknowledge(subject)
        if not count:
            log_refused("no open problem with that name")
            return "nothing open with that name"
        return f"ok: cleared {count}"

    return acknowledge_problem


def register(
    controller: BridgeController,
    store: AttentionStore | None = None,
    runner: Runner | None = None,
    *,
    env: Mapping[str, str] | None = None,
    start: bool = True,
) -> Monitor | None:
    """Wire the inbox into ``controller`` (only when a bridge flag is on).

    The store becomes the controller's problem sink, delivery tracker and briefing
    provider, the monitor starts and stops with the daemon (the store is closed only once
    the monitor thread has ended), and ``acknowledge_problem`` is registered. Phase 7 sets
    the in-call notifier with :meth:`Monitor.set_notifier`.
    """
    if not bridge_enabled(env):
        return None
    store = store or AttentionStore()
    controller.problems = store
    controller.deliveries = store
    controller.briefings = store
    monitor = Monitor(store, runner, call_active=lambda: controller.turns is not None)
    controller.register_tool("acknowledge_problem", make_acknowledge_tool(controller, store))

    def shutdown() -> None:
        if monitor.stop():
            store.close()

    controller.on_shutdown(shutdown)
    if start:
        monitor.start()
    return monitor
