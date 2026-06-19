"""Per-speaker persistent memory + provider-agnostic context assembly (G7).

This is a multi-user voice assistant: different members of a household talk to the
same Mac. To feel personal, the brain must (a) remember each *enrolled speaker*
across sessions, and (b) assemble that history into a prompt for *any* LLM backend
(Anthropic, OpenAI, Ollama, …) without coupling memory to a provider's message
schema. Three pieces, all provider-agnostic and unit-tested with fakes:

* :class:`MemoryStore` — a persistent, per-speaker conversation log on disk
  (**SQLite** by default; **JSON** for a ``.json`` path). Keyed by the speaker name
  from :mod:`my_stt_tts.speaker_id` (``unknown`` / ``ambiguous`` are stored under a
  shared bucket so a guest still gets in-session continuity without polluting an
  enrolled person's memory). Isolated per speaker: one person's turns never leak
  into another's recall.
* :class:`ContextAggregator` — assembles a bounded message list from (1) the
  persistent per-speaker history and (2) the live in-session turns, **independent
  of the provider**. The brain converts the neutral ``[{role, content}]`` list into
  whatever the backend wants (it already does for Anthropic/OpenAI).
* :class:`DialogueFlow` — a tiny structured-dialogue hook: named states with
  optional per-state system-prompt augmentation and a transition function, so a
  multi-turn flow (e.g. a reminder wizard) can steer the assistant without
  hard-coding it in the loop. Off by default (a single ``"default"`` state).

The store is opened lazily and degrades gracefully (a write/IO failure is logged,
never raised into the turn). ``memory_store`` config (a path) turns persistence on;
without it the brain keeps today's in-memory-only behaviour.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger("my_stt_tts.memory")

# Speakers whose identity is uncertain share one bucket so a guest gets in-session
# continuity without writing into an enrolled person's persistent memory.
GUEST_KEY = "_guest"


def speaker_key(name: str | None) -> str:
    """Normalize a speaker-id result to a stable storage key (G7).

    Enrolled names are used verbatim; ``unknown`` / ``ambiguous`` / ``None`` all map
    to the shared :data:`GUEST_KEY` bucket so unrecognized voices don't pollute a
    known person's recall but still get within-session continuity.
    """
    if not name or name in ("unknown", "ambiguous"):
        return GUEST_KEY
    return name


@runtime_checkable
class MemoryBackend(Protocol):
    """Persistence surface for per-speaker turns (SQLite or JSON)."""

    def append(self, speaker: str, role: str, content: str) -> None:
        """Persist one turn for ``speaker``."""
        ...

    def recent(self, speaker: str, limit: int) -> list[dict[str, str]]:
        """Return up to ``limit`` most recent ``{role, content}`` turns (oldest first)."""
        ...

    def clear(self, speaker: str) -> None:
        """Forget all stored turns for ``speaker``."""
        ...

    def amend_last_assistant(self, speaker: str, spoken: str) -> None:
        """Rewrite (or drop, if ``spoken`` is blank) the speaker's last assistant turn."""
        ...


class JsonMemoryBackend:
    """JSON-file persistence: ``{speaker: [{role, content, ts}, ...]}`` (G7).

    Simple + human-readable; good for small households. Thread-safe via a lock; the
    whole file is rewritten on append (fine at this scale). Per-speaker lists keep
    one person's turns fully isolated from another's.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, list[dict[str, str]]]:
        try:
            return dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return {}

    def append(self, speaker: str, role: str, content: str) -> None:
        with self._lock:
            data = self._load()
            data.setdefault(speaker, []).append(
                {"role": role, "content": content, "ts": str(int(time.time()))}
            )
            try:
                self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except OSError:
                log.warning("memory JSON write failed (%s)", self.path, exc_info=True)

    def recent(self, speaker: str, limit: int) -> list[dict[str, str]]:
        with self._lock:
            turns = self._load().get(speaker, [])
        return [{"role": t["role"], "content": t["content"]} for t in turns[-limit:]]

    def clear(self, speaker: str) -> None:
        with self._lock:
            data = self._load()
            data.pop(speaker, None)
            with _suppress_os():
                self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def amend_last_assistant(self, speaker: str, spoken: str) -> None:
        with self._lock:
            data = self._load()
            turns = data.get(speaker, [])
            if not turns or turns[-1].get("role") != "assistant":
                return
            if spoken:
                turns[-1]["content"] = spoken
            else:
                turns.pop()
            with _suppress_os():
                self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class SqliteMemoryBackend:
    """SQLite persistence: one row per turn, indexed by speaker (G7).

    The default backend — robust to concurrent writes (the local + network loops
    can both store turns), append-only, and queryable. The schema is a single
    ``turns(speaker, role, content, ts)`` table; ``recent`` pulls the newest N for a
    speaker and returns them oldest-first for prompt assembly.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + our lock: usable from the loop's worker threads.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS turns ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "speaker TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_turns_speaker ON turns(speaker)")
        self._db.commit()

    def append(self, speaker: str, role: str, content: str) -> None:
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO turns(speaker, role, content, ts) VALUES (?, ?, ?, ?)",
                    (speaker, role, content, time.time()),
                )
                self._db.commit()
            except sqlite3.Error:
                log.warning("memory SQLite write failed (%s)", self.path, exc_info=True)

    def recent(self, speaker: str, limit: int) -> list[dict[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM turns WHERE speaker = ? ORDER BY id DESC LIMIT ?",
                (speaker, limit),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def clear(self, speaker: str) -> None:
        with self._lock, _suppress_sqlite():
            self._db.execute("DELETE FROM turns WHERE speaker = ?", (speaker,))
            self._db.commit()

    def amend_last_assistant(self, speaker: str, spoken: str) -> None:
        with self._lock, _suppress_sqlite():
            row = self._db.execute(
                "SELECT id, role FROM turns WHERE speaker = ? ORDER BY id DESC LIMIT 1",
                (speaker,),
            ).fetchone()
            if row is None or row[1] != "assistant":
                return
            if spoken:
                self._db.execute("UPDATE turns SET content = ? WHERE id = ?", (spoken, row[0]))
            else:
                self._db.execute("DELETE FROM turns WHERE id = ?", (row[0],))
            self._db.commit()


class MemoryStore:
    """Per-speaker persistent memory keyed by the enrolled speaker name (G7).

    Wraps a :class:`MemoryBackend` (SQLite or JSON, chosen by the path suffix) and
    exposes a small speaker-keyed API the brain uses for cross-session recall. The
    speaker key is normalized via :func:`speaker_key`, so unrecognized voices share
    a guest bucket and never read/write an enrolled person's history.
    """

    def __init__(self, backend: MemoryBackend) -> None:
        self.backend = backend

    @classmethod
    def open(cls, path: str | Path) -> MemoryStore:
        """Open a store, picking JSON for a ``.json`` path and SQLite otherwise."""
        backend: MemoryBackend
        if str(path).endswith(".json"):
            backend = JsonMemoryBackend(path)
        else:
            backend = SqliteMemoryBackend(path)
        return cls(backend)

    def add_turn(self, speaker: str | None, role: str, content: str) -> None:
        """Persist one ``{role, content}`` turn for ``speaker`` (no-op on blank content)."""
        if not content:
            return
        self.backend.append(speaker_key(speaker), role, content)

    def history(self, speaker: str | None, limit: int) -> list[dict[str, str]]:
        """Return the most recent ``limit`` turns for ``speaker`` (oldest first)."""
        return self.backend.recent(speaker_key(speaker), limit)

    def forget(self, speaker: str | None) -> None:
        """Erase a speaker's stored history (e.g. a 'forget me' command)."""
        self.backend.clear(speaker_key(speaker))

    def amend_last_assistant(self, speaker: str | None, spoken: str) -> None:
        """Rewrite (or drop, if blank) the speaker's most recent assistant turn."""
        self.backend.amend_last_assistant(speaker_key(speaker), spoken)


@dataclass
class ContextAggregator:
    """Provider-agnostic context assembly for the LLM (G7).

    Produces a neutral ``[{role, content}]`` message list — the same shape the
    brain already feeds both the Anthropic and OpenAI paths — from three sources,
    in order: (1) any persistent per-speaker history loaded from the
    :class:`MemoryStore`, (2) the live in-session turns, capped to a budget. It is
    decoupled from the provider: the brain converts this neutral list to whatever
    the backend expects. Without a store it behaves exactly like the prior
    in-memory history.

    ``max_turns`` bounds the *combined* message count (×2 messages per turn) so the
    prompt stays small; the live session always wins over older persistent turns.
    """

    store: MemoryStore | None = None
    max_turns: int = 20
    live: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        """Record a user turn into the live session."""
        self.live.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        """Record an assistant turn into the live session."""
        self.live.append({"role": "assistant", "content": text})

    def assemble(self, speaker: str | None = None) -> list[dict[str, str]]:
        """Assemble the bounded message list: persistent recall + live, provider-agnostic.

        The persistent per-speaker turns are prepended (cross-session recall) and
        the live session appended; the whole thing is trimmed to the most recent
        ``max_turns`` turns so older recall yields to the current conversation.
        Deduplicates the overlap when live turns were also persisted.
        """
        budget = max(1, self.max_turns) * 2
        persisted: list[dict[str, str]] = []
        if self.store is not None:
            persisted = self.store.history(speaker, budget)
        # Concatenate persistent recall + live, dropping an exact overlap at the
        # seam (the live turns may have already been written to the store this
        # session), then trim to the budget so older recall yields to the present.
        combined = _dedupe_seam(persisted, self.live)
        return combined[-budget:]

    def persist(self, speaker: str | None, user_text: str, assistant_text: str) -> None:
        """Write a completed exchange to the per-speaker store (if persistence is on)."""
        if self.store is None:
            return
        self.store.add_turn(speaker, "user", user_text)
        if assistant_text:
            self.store.add_turn(speaker, "assistant", assistant_text)

    def amend_last_assistant(self, speaker: str | None, spoken: str) -> None:
        """Repair the last persisted assistant turn after a barge-in (G7 + G5).

        ``stream()`` persists the FULL generated reply optimistically; if a barge-in
        truncated what was actually voiced, this rewrites that last assistant row to
        the spoken prefix (or drops it when nothing was voiced) so persistent recall
        matches reality. No-op without a store.
        """
        if self.store is not None:
            self.store.amend_last_assistant(speaker, spoken)

    def reset_live(self) -> None:
        """Clear only the live session IN PLACE (persistent memory is untouched).

        Clears in place rather than rebinding ``live`` so any alias held elsewhere
        (e.g. ``Brain.history is context.live``) stays valid after a reset.
        """
        self.live.clear()


def _dedupe_seam(
    persisted: list[dict[str, str]], live: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Concatenate ``persisted`` + ``live``, removing a persisted tail that == live head.

    If the live session's opening turns were already persisted (common when memory
    is on), drop that duplicated tail from the persisted side so the assembled
    context doesn't show the same exchange twice.
    """
    if not persisted or not live:
        return [*persisted, *live]
    max_overlap = min(len(persisted), len(live))
    for k in range(max_overlap, 0, -1):
        if persisted[-k:] == live[:k]:
            return [*persisted[:-k], *live]
    return [*persisted, *live]


@dataclass
class DialogueFlow:
    """A tiny structured-dialogue / flow hook (G7).

    Named states, each optionally augmenting the system prompt and deciding the
    next state from the user's utterance. This lets a multi-turn flow (a reminder
    wizard, a confirmation step) steer the assistant without hard-coding it in the
    loop. Default: a single ``"default"`` state that augments nothing and never
    transitions — so the flow is a no-op until states are registered.

    ``states`` maps a name to a :class:`FlowState`. :meth:`augment` returns the
    extra system-prompt text for the current state; :meth:`advance` runs the
    current state's transition on an utterance and updates :attr:`current`.
    """

    states: dict[str, FlowState] = field(default_factory=dict)
    current: str = "default"

    def __post_init__(self) -> None:
        self.states.setdefault("default", FlowState("default"))

    def augment(self) -> str:
        """System-prompt augmentation for the current state (empty by default)."""
        return self.states.get(self.current, self.states["default"]).prompt

    def advance(self, user_text: str) -> str:
        """Run the current state's transition on ``user_text``; return the new state."""
        state = self.states.get(self.current)
        if state is not None and state.transition is not None:
            nxt = state.transition(user_text)
            if nxt in self.states:
                self.current = nxt
        return self.current

    def reset(self) -> None:
        """Return to the default state."""
        self.current = "default"


@dataclass
class FlowState:
    """One node in a :class:`DialogueFlow`: a prompt augmentation + a transition."""

    name: str
    prompt: str = ""
    # user_text -> next state name. None means "stay here".
    transition: Any = None


def make_memory_store(cfg: Any) -> MemoryStore | None:
    """Build a :class:`MemoryStore` from config, or None when persistence is off (G7).

    ``cfg.memory_store`` is a path: a ``.json`` suffix selects the JSON backend,
    anything else the SQLite backend. None / unset keeps the brain's in-memory-only
    behaviour (no cross-session recall). Never raises — a store that can't be opened
    logs and returns None so the loop still runs.
    """
    path = getattr(cfg, "memory_store", None)
    if not path:
        return None
    try:
        return MemoryStore.open(path)
    except Exception:  # bad path / permissions -> in-memory only
        log.warning(
            "could not open memory store at %s; continuing without it.", path, exc_info=True
        )
        return None


class _suppress_os:  # noqa: N801 — tiny OSError swallow
    def __enter__(self) -> _suppress_os:
        return self

    def __exit__(self, exc_type: type | None, *_exc: object) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)


class _suppress_sqlite:  # noqa: N801 — tiny sqlite3.Error swallow
    def __enter__(self) -> _suppress_sqlite:
        return self

    def __exit__(self, exc_type: type | None, *_exc: object) -> bool:
        return exc_type is not None and issubclass(exc_type, sqlite3.Error)
