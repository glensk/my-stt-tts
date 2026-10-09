# pylint: disable=too-many-lines  # one cohesive component: ccc client, targets, decisions, tools
"""Claude Code sessions by voice: list, read, brief, message and answer (via ``ccc … -j``).

Registered on the :class:`~my_stt_tts.bridge.BridgeController` by :func:`register` when
``MAC_VOICE_CLAUDE_BRIDGE=1``. Every ccc call goes through :class:`CccClient`: argv only,
message text only on stdin, the versioned envelope ``{"schema_version": 1, "ok", "data",
"error"}`` parsed strictly, every ``error.code`` mapped to a short speakable reason.

Tools (the voice agent's client tools):

* ``list_sessions()`` — numbered list (name, status, background marked, short AIM); the
  numbering is remembered so "number 2" works as a target afterwards.
* ``read_session(target, what)`` — the last reply or the last prompt, redacted + capped.
* ``brief_decision(target)`` — the spoken decision briefing (decision D5).
* ``send_message(target, text)`` / ``answer_decision(target, choice)`` — only a proposal,
  and only with the capability of what Albert just said (the target — name or list
  number — must be in it; a read or briefing since then voids it). The confirm prompt
  reads the message (redacted, ≤ 200 chars) or the chosen options back. The action runs
  after ``confirm_action(code)`` through the ``send`` / ``answer`` executors (one
  mutation at a time, final cancellation + deadline check).

Outcomes: an ``accepted`` / ``unknown`` delivery goes to ``controller.deliveries`` (the
attention monitor follows it); only failures after something was typed reach the inbox
(one item per session); a timeout or unreadable answer during send / answer is "not yet
confirmed", never "failed", and never retried. Every read tool call shares one 10 s
budget across its ccc calls.

Background sessions (D4) are listed and read, never messaged. A to-do-line decision is
answered with a message (``send``), a picker decision with ``ccc answer``. Every session
text passes :func:`~my_stt_tts.bridge_text.redact` before it reaches the agent, every
tool result voids unused capabilities (``note_context``), and logs never carry content.
"""

from __future__ import annotations

import contextlib
import difflib
import inspect
import json
import logging
import os
import re
import shutil
import signal
import string
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bridge import (
    BridgeController,
    Deadline,
    Mutation,
    PendingAction,
    Problem,
    Refusal,
    before_mutation,
    log_tool,
)
from .bridge_text import normalise, numbers_in, redact

log = logging.getLogger("my_stt_tts.claude_sessions")

FLAG = "MAC_VOICE_CLAUDE_BRIDGE"
CCC_ENV = "MAC_VOICE_CCC_BIN"  # optional override of the ccc executable
SCHEMA_VERSION = 1

READ_TIMEOUT_S = 3.0  # ccc sessions / inspect -N
BRIEF_TIMEOUT_S = 6.0  # inspect with the voice-brief LLM summary (ccc budget 10 s)
BRIEF_FALLBACK_TIMEOUT_S = 3.0  # then inspect -N (summary = first sentences)
TOOL_BUDGET_S = 10.0  # one tool call in total (list + re-list + inspect share it)
MUTATION_BUDGET_S = 20.0  # ccc send / answer
LOCK_TIMEOUT_S = 2.0  # wait this long for a running mutation, then refuse
LIST_TTL_S = 30.0  # names resolve against a list this fresh before re-fetching
CACHE_TTL_S = 600.0  # the numbered list + briefed decisions are forgotten after this

MESSAGE_MAX = 4000
OTHER_TEXT_MAX = 500  # ccc answer rejects longer free text
PROMPT_TEXT_CAP = 200  # message text read back in the confirm prompt
AIM_CAP = 80
READ_CAP = 1500
BRIEF_CAP = 1500
LABEL_CAP = 120
LIST_CAP = 1500

LETTERS = string.ascii_lowercase


# -- running ccc -------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunResult:
    """What one subprocess run returned."""

    returncode: int
    stdout: str
    stderr: str = ""


Runner = Callable[[Sequence[str], "str | None", float], RunResult]


def run_group(argv: Sequence[str], stdin: str | None, timeout: float) -> RunResult:
    """Run ``argv`` in its own process group; on timeout the whole group is killed.

    ``subprocess.run`` kills only the direct child, and a grandchild that still holds the
    output pipes keeps the call waiting long past ``timeout``.
    """
    with subprocess.Popen(  # noqa: S603 - argv list, no shell
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            out, err = proc.communicate(stdin if stdin is not None else "", timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.communicate(timeout=1.0)  # reap; the pipes close with the group
            raise
    return RunResult(proc.returncode, out or "", err or "")


def resolve_ccc(env: Mapping[str, str] | None = None) -> str | None:
    """The ccc executable: ``$MAC_VOICE_CCC_BIN`` → ``$PATH`` → ``~/.local/bin/ccc``."""
    env = os.environ if env is None else env
    override = env.get(CCC_ENV, "").strip()
    if override:
        return override
    found = shutil.which("ccc")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "ccc"
    return str(fallback) if fallback.is_file() else None


@dataclass(frozen=True)
class CccResult:
    """A parsed envelope: ``ok`` + ``data``, or an error ``code`` (data may still be set)."""

    ok: bool
    data: dict[str, Any] | None = None
    code: str = ""
    message: str = field(default="", repr=False)


def _envelope_fault(env: Any) -> tuple[str, str] | None:
    """``(code, why)`` when ``env`` is not a valid section-6 envelope, else None."""
    if not isinstance(env, dict):
        return "bad_envelope", "envelope is not an object"
    version = env.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        return "schema_mismatch", f"schema_version {version!r}"
    ok, data, error = env.get("ok"), env.get("data"), env.get("error")
    if not isinstance(ok, bool) or not (data is None or isinstance(data, dict)):
        return "bad_envelope", "ok/data malformed"
    if ok and data is None:
        return "bad_envelope", "ok without data"
    if not ok and not (isinstance(error, dict) and isinstance(error.get("code"), str)):
        return "bad_envelope", "error malformed"
    return None


def parse_envelope(stdout: str) -> CccResult:
    """The section-6 envelope, strictly: anything else is ``bad_envelope``."""
    try:
        env = json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return CccResult(False, code="bad_envelope", message="stdout is not JSON")
    fault = _envelope_fault(env)
    data = env.get("data") if isinstance(env, dict) else None
    if fault is not None:
        return CccResult(False, data if isinstance(data, dict) else None, *fault)
    if env["ok"]:
        return CccResult(True, data)
    error = env["error"]
    return CccResult(False, data, code=error["code"], message=str(error.get("message", "")))


class CccClient:
    """Runs ``ccc <cmd> … -j`` and parses the envelope (never raises)."""

    def __init__(self, runner: Runner | None = None, ccc: str | None = None) -> None:
        self.runner: Runner = runner or run_group
        self._ccc = ccc

    def _exe(self) -> str | None:
        if self._ccc is None:
            self._ccc = resolve_ccc()
        return self._ccc

    def run(self, args: Sequence[str], *, stdin: str | None = None, timeout: float) -> CccResult:
        """``ccc <args> -j`` with ``stdin``; a missing binary / timeout becomes a code.

        With no time left (``timeout`` ≤ 0) ccc is not started at all.
        """
        if timeout <= 0:
            return CccResult(False, code="timeout", message=f"no time left for ccc {args[0]}")
        exe = self._exe()
        if exe is None:
            return CccResult(False, code="ccc_missing", message="ccc not found")
        try:
            res = self.runner([exe, *args, "-j"], stdin, timeout)
        except subprocess.TimeoutExpired:
            return CccResult(False, code="timeout", message=f"ccc {args[0]} timed out")
        except OSError as exc:
            return CccResult(False, code="ccc_missing", message=type(exc).__name__)
        return parse_envelope(res.stdout)

    def sessions(self, timeout: float = READ_TIMEOUT_S) -> CccResult:
        return self.run(["sessions"], timeout=timeout)

    def inspect(
        self, session_id: str, *, llm: bool = False, timeout: float = READ_TIMEOUT_S
    ) -> CccResult:
        args = ["inspect", "-s", session_id] + ([] if llm else ["-N"])
        return self.run(args, timeout=timeout)

    def send(self, session_id: str, text: str, timeout: float = MUTATION_BUDGET_S) -> CccResult:
        return self.run(["send", "-s", session_id], stdin=text, timeout=timeout)

    def answer(
        self, session_id: str, payload: Mapping[str, Any], timeout: float = MUTATION_BUDGET_S
    ) -> CccResult:
        stdin = json.dumps(payload, ensure_ascii=False)
        return self.run(["answer", "-s", session_id], stdin=stdin, timeout=timeout)


# -- speakable reasons ----------------------------------------------------------------------------
REASONS: dict[str, str] = {
    # transport / contract
    "ccc_missing": "the command center is not installed",
    "timeout": "the command center did not answer in time",
    "bad_envelope": "the command center gave an unreadable answer",
    "schema_mismatch": "the command center speaks a newer format",
    "internal": "the command center hit an internal error",
    "usage": "the command center rejected the request",
    "unknown_field": "the command center rejected the request",
    "invalid_json": "the command center rejected the request",
    "stdin_required": "the command center got no message",
    # targets
    "unknown_session": "that session is not known",
    "not_found": "that session was not found",
    "ambiguous_session": "that session id is ambiguous",
    "not_running": "that session is no longer running",
    "background": "it is a background session — I can read it but not message it",
    "account_conflict": "it is listed under both accounts",
    "no_tab": "its terminal tab was not found",
    "stale_tab": "its terminal tab changed",
    "tab_locked": "another message is being typed into it",
    "foreground_not_claude": "its tab is not showing Claude right now",
    "iterm_unreachable": "iTerm cannot be reached",
    # session state
    "waiting": "it is waiting for an answer to a question",
    "blocked": "it is blocked on a permission prompt",
    "picker_pending": "it shows a question picker — answer the decision instead",
    "not_waiting": "it is not waiting for a decision any more",
    "unknown_status": "its state is unknown",
    "transcript_missing": "its transcript was not found",
    "transcript_unknown": "its transcript could not be read",
    # message checks
    "empty_message": "the message is empty",
    "message_too_long": "the message is too long",
    "control_characters": "the message has invalid characters",
    # outcomes
    "delivery_unknown": "sent, not yet confirmed",
    "answer_unknown": "answered, not yet confirmed",
    "delivery_failed": "the message did not reach the session",
    "answer_failed": "the answer did not reach the session",
    "answer_mismatch": "the session recorded a different answer",
    "decision_changed": "the decision changed meanwhile — ask for the briefing again",
    "unsupported_shape": "that kind of question cannot be answered by voice",
    "invalid_answer": "the answer does not fit the question",
    "invalid_answers": "the answer does not fit the question",
    "invalid_decision_id": "the decision changed meanwhile — ask for the briefing again",
}
UNKNOWN_CODES = {"delivery_unknown", "answer_unknown"}
#: During send / answer these say nothing about whether ccc already acted: "not yet
#: confirmed", never "failed", and never retried.
UNCERTAIN_CODES = {"timeout", "bad_envelope", "schema_mismatch"}
#: Failures AFTER something was typed into the session (ccc returns ``data`` with them);
#: only these reach the attention inbox, refusals before acting are only spoken.
POST_ACTION_FAILURES = {"delivery_failed", "answer_failed", "answer_mismatch"}


def reason_for(code: str) -> str:
    """A short speakable reason for a ccc ``error.code`` (never the raw message)."""
    return REASONS.get(code, "the command center refused")


# -- sessions + target resolution ------------------------------------------------------------------
@dataclass(frozen=True)
class Session:
    """One row of ``ccc sessions``."""

    session_id: str
    name: str
    kind: str = "interactive"
    status: str = "unknown"
    aim_short: str = ""

    @property
    def background(self) -> bool:
        """Anything not reported as ``interactive`` is read-only (fail closed)."""
        return self.kind != "interactive"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Session | None:
        sid = row.get("session_id")
        if not isinstance(sid, str) or not sid:
            return None
        name = row.get("name") if isinstance(row.get("name"), str) else ""
        return cls(
            session_id=sid,
            name=(name or sid[:8]).strip(),
            kind=str(row.get("kind") or "background"),
            status=str(row.get("status") or "unknown"),
            aim_short=str(row.get("aim_short") or ""),
        )


def _squash(text: str) -> str:
    return normalise(text).replace(" ", "")


_TARGET_FILLER = {"number", "nummer", "numero", "session", "no", "nr", "the", "die", "la", "le"}


def _target_number(target: str) -> int | None:
    """The list number a target says ("2", "number two", "Nummer zwei"), else None."""
    words = [w for w in normalise(target).split() if w not in _TARGET_FILLER]
    if not words or len(words) > 3 or not all(w.isdigit() or numbers_in(w) for w in words):
        return None
    nums = numbers_in(" ".join(words))
    return next(iter(nums)) if len(nums) == 1 else None


class Resolution:
    """Either a :class:`Session` or a question for the agent to ask."""

    def __init__(self, session: Session | None = None, question: str = "") -> None:
        self.session = session
        self.question = question


def _match_name(target: str, sessions: Sequence[Session]) -> Resolution | None:
    """Exact, case-insensitive, then punctuation-insensitive name match (or id)."""
    tests: tuple[Callable[[Session], bool], ...] = (
        lambda s: target in (s.name, s.session_id),
        lambda s: s.name.casefold() == target.casefold(),
        lambda s: bool(_squash(target)) and _squash(s.name) == _squash(target),
    )
    for test in tests:
        hits = [s for s in sessions if test(s)]
        if len(hits) == 1:
            return Resolution(hits[0])
        if len(hits) > 1:
            names = ", ".join(_say_name(s) for s in hits)
            return Resolution(question=f"Several sessions match {target!r}: {names}. Which one?")
    return None


def _candidates(target: str, sessions: Sequence[Session]) -> list[Session]:
    want = _squash(target)
    by_name = {_squash(s.name): s for s in sessions}
    close = difflib.get_close_matches(want, list(by_name), n=3, cutoff=0.6)
    picks = [by_name[c] for c in close]
    words = set(normalise(target).split())
    for s in sessions:
        if s not in picks and words & set(normalise(s.name).split()):
            picks.append(s)
    return picks[:3]


def _say_name(s: Session) -> str:
    return redact(s.name, 40)


# -- decisions ----------------------------------------------------------------------------------
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "erste": 1, "erster": 1, "erstes": 1, "zweite": 2, "zweiter": 2, "zweites": 2,
    "dritte": 3, "dritter": 3, "drittes": 3, "vierte": 4, "funfte": 5,
    "premier": 1, "premiere": 1, "deuxieme": 2, "troisieme": 3, "quatrieme": 4,
}  # fmt: skip
_CHOICE_FILLER = {
    "option", "options", "answer", "choice", "letter", "the", "please", "take", "pick",
    "choose", "and", "plus", "or", "antwort", "bitte", "nimm", "die", "den", "das", "und",
    "variante", "choix", "reponse", "prends", "la", "le", "l", "et", "ou", "number", "nummer",
    "numero", "i", "we", "let", "go", "going", "with", "for", "would", "will", "prefer",
    "want", "ich", "nehme", "wahle", "je", "choisis",
}  # fmt: skip
#: "the second ONE", "die zweite EINS", "la deuxième UNE": a number word right after an
#: ordinal or a letter is filler, not a second pick.
_ONE_WORDS = {"one", "eins", "ein", "eine", "un", "une"}
#: "I'd", "let's", "we'll" — the clitic would otherwise read as option d / s.
_CLITIC = re.compile(r"\b(i|we|you|let)['’](d|ll|s|m|re|ve)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    """One decision question as ccc inspect reports it."""

    text: str
    multi_select: bool
    labels: tuple[str, ...]
    consequences: tuple[str, ...]


def _questions(decision: Mapping[str, Any]) -> list[Question]:
    out: list[Question] = []
    for q in decision.get("questions") or []:
        if not isinstance(q, Mapping):
            continue
        opts = [o for o in q.get("options") or [] if isinstance(o, Mapping)]
        out.append(
            Question(
                text=str(q.get("text") or ""),
                multi_select=bool(q.get("multi_select")),
                labels=tuple(str(o.get("label") or "") for o in opts),
                consequences=tuple(str(o.get("consequence") or "") for o in opts),
            )
        )
    return out


def _segments(choice: str, count: int) -> list[str] | None:
    """One answer per question: ``;`` / newlines, or ``1) … 2) …`` markers."""
    if count <= 1:
        return [choice]
    parts = [p.strip() for p in re.split(r"[;\n]+", choice) if p.strip()]
    if len(parts) == count:
        return parts
    marked = [p.strip() for p in re.split(r"(?:^|\s)\d\s*[).:]\s*", choice) if p.strip()]
    return marked if len(marked) == count else None


def _choice_words(segment: str) -> list[str]:
    """The normalised words of ``segment`` without filler ("the second one" → second)."""
    words: list[str] = []
    after_pick = False
    for w in normalise(_CLITIC.sub(r"\1", segment)).split():
        filler = w in _CHOICE_FILLER or (after_pick and w in _ONE_WORDS)
        after_pick = not filler and (w in _ORDINALS or len(w) == 1)
        if not filler:
            words.append(w)
    return words


def _index_tokens(segment: str, n: int) -> list[int] | None:
    """Option indices when the segment is ONLY letters / numbers / ordinals (+ filler)."""
    words = _choice_words(segment)
    if not words:
        return None
    picks: list[int] = []
    for w in words:
        if len(w) == 1 and w in LETTERS[:n]:
            idx = LETTERS.index(w)
        elif w.isdigit():
            idx = int(w) - 1
        elif w in _ORDINALS:
            idx = _ORDINALS[w] - 1
        elif len(nums := numbers_in(w)) == 1:
            idx = next(iter(nums)) - 1
        else:
            return None
        if not 0 <= idx < n:
            return None
        if idx not in picks:
            picks.append(idx)
    return picks


def _label_hits(segment: str, labels: Sequence[str]) -> list[int]:
    seg = f" {normalise(segment)} "
    exact = [i for i, lab in enumerate(labels) if normalise(lab) and normalise(lab) == seg.strip()]
    if exact:
        return exact
    return [i for i, lab in enumerate(labels) if normalise(lab) and f" {normalise(lab)} " in seg]


def _fuzzy_label(segment: str, labels: Sequence[str]) -> list[int]:
    norm = [normalise(lab) for lab in labels]
    close = difflib.get_close_matches(normalise(segment), norm, n=2, cutoff=0.75)
    if len(close) == 1:
        return [norm.index(close[0])]
    words = set(normalise(segment).split()) - _CHOICE_FILLER
    overlap = [i for i, lab in enumerate(norm) if words & {w for w in lab.split() if len(w) > 3}]
    return overlap if len(overlap) == 1 else []


@dataclass(frozen=True)
class Pick:
    """A resolved answer to one question: option indices, or free text."""

    indices: tuple[int, ...] = ()
    other: str = ""


def resolve_choice(q: Question, segment: str) -> Pick | str:
    """Indices of the options ``segment`` names, free text, or a question (str) back."""
    n = len(q.labels)
    picks = _label_hits(segment, q.labels) or _index_tokens(segment, n) or []
    if not picks:
        picks = _fuzzy_label(segment, q.labels)
    if not picks:
        other = " ".join(segment.split())[:OTHER_TEXT_MAX]
        if not other:
            return f"Which option for: {q.text}?"
        if q.multi_select:
            return f"Please pick from the options for: {q.text}"
        return Pick(other=other)
    if len(picks) > 1 and not q.multi_select:
        names = " or ".join(f"{LETTERS[i]}) {q.labels[i]}" for i in picks)
        return f"Only one option fits {q.text!r} — {names}?"
    return Pick(indices=tuple(picks))


def _pick_summary(q: Question, pick: Pick) -> str:
    if pick.other:
        return f"your own answer {pick.other!r}"
    return ", ".join(f"{LETTERS[i]}) {q.labels[i]}" for i in pick.indices)


def _picks_summary(questions: Sequence[Question], picks: Sequence[Pick]) -> str:
    return "; ".join(_pick_summary(q, p) for q, p in zip(questions, picks, strict=True))


def _pick_text(q: Question, pick: Pick) -> str:
    if pick.other:
        return pick.other
    return ", ".join(q.labels[i] for i in pick.indices)


# -- the tools ------------------------------------------------------------------------------------
def _clean_message(text: str) -> str:
    """Drop control characters ccc would reject (keeps newline and tab)."""
    return "".join(
        c for c in text if c in "\n\t" or not (ord(c) < 32 or ord(c) == 127 or 128 <= ord(c) < 160)
    ).strip()


def _within(deadline: Deadline | None, cap: float) -> float:
    """``cap`` seconds, or less when the tool call's ``deadline`` is closer."""
    return cap if deadline is None else min(cap, deadline.remaining())


def _tail_reply(text: str) -> str:
    """A tail-kept ``last_reply`` ("…" first) without its cut-off first word."""
    if not text.startswith("…"):
        return text
    rest = text[1:].split(None, 1)
    return "… " + rest[1] if len(rest) > 1 else ""


def _accepts_dedupe_key(fn: Callable[..., Any]) -> bool:
    """True when a sink's ``report`` takes ``dedupe_key`` (the attention inbox does)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "dedupe_key" or p.kind is p.VAR_KEYWORD for p in params)


class SessionTools:  # pylint: disable=too-many-instance-attributes  # tools + per-call caches
    """The sessions client tools + the ``send`` / ``answer`` executors."""

    def __init__(
        self,
        controller: BridgeController,
        client: CccClient | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ctl = controller
        self.client = client or CccClient()
        self.clock = clock
        self._numbered: list[Session] = []
        self._listed: list[Session] = []
        self._listed_at: float | None = None
        self._briefed: dict[str, str] = {}  # session_id → decision_id last briefed
        self._call: object | None = None  # the call (controller.turns) the caches belong to
        self._cached_at: float | None = None
        self._cache_lock = threading.Lock()

    # -- registration ---------------------------------------------------------------------
    def register(self) -> None:
        tools: dict[str, Callable[..., str]] = {
            "list_sessions": self.list_sessions,
            "read_session": self.read_session,
            "brief_decision": self.brief_decision,
            "send_message": self.send_message,
            "answer_decision": self.answer_decision,
        }
        for name, fn in tools.items():
            self.ctl.register_tool(name, _guarded(name, fn, self.ctl))
        self.ctl.register_executor("send", self.execute_send)
        self.ctl.register_executor("answer", self.execute_answer)

    # -- helpers ------------------------------------------------------------------------
    def _sync_call(self) -> None:
        """A new call (or 10 idle minutes) forgets the numbered list and the briefings."""
        now = self.clock()
        with self._cache_lock:
            turns = self.ctl.turns
            stale = self._cached_at is not None and now - self._cached_at >= CACHE_TTL_S
            if turns is not self._call or stale:
                self._call = turns
                self._numbered, self._briefed = [], {}
                self._listed, self._listed_at = [], None
            self._cached_at = now

    def _fetch(self, deadline: Deadline | None = None) -> list[Session] | str:
        res = self.client.sessions(timeout=_within(deadline, READ_TIMEOUT_S))
        if not res.ok or res.data is None:
            return f"Cannot list the sessions: {reason_for(res.code)}."
        rows = res.data.get("sessions")
        if not isinstance(rows, list):
            return f"Cannot list the sessions: {reason_for('bad_envelope')}."
        sessions = [s for r in rows if isinstance(r, Mapping) and (s := Session.from_row(r))]
        self._listed, self._listed_at = sessions, self.clock()
        return sessions

    def _sessions(self, fresh: bool, deadline: Deadline | None) -> list[Session] | str:
        recent = self._listed_at is not None and self.clock() - self._listed_at < LIST_TTL_S
        if not fresh and recent:
            return self._listed
        return self._fetch(deadline)

    def resolve(
        self, target: str, *, fresh: bool = False, deadline: Deadline | None = None
    ) -> Resolution:
        """The session ``target`` names: canonical name, case-insensitive, or list number."""
        self._sync_call()
        target = " ".join(str(target or "").split())
        if not target:
            return Resolution(question="Which session?")
        number = _target_number(target)
        if number is not None:
            return self._by_number(number, fresh, deadline)
        sessions = self._sessions(fresh, deadline)
        if isinstance(sessions, str):
            return Resolution(question=sessions)
        found = _match_name(target, sessions)
        if found is None and not fresh and sessions is self._listed:
            refreshed = self._fetch(deadline)  # the name may be new since the cached list
            if isinstance(refreshed, str):
                return Resolution(question=refreshed)
            sessions = refreshed
            found = _match_name(target, sessions)
        return found or self._unknown(target, sessions)

    def _by_number(self, number: int, fresh: bool, deadline: Deadline | None) -> Resolution:
        if not self._numbered:
            return Resolution(question="I have no numbered list yet — shall I list the sessions?")
        if not 1 <= number <= len(self._numbered):
            return Resolution(question=f"The list has {len(self._numbered)} sessions. Which one?")
        picked = self._numbered[number - 1]
        if not fresh:
            return Resolution(picked)
        sessions = self._fetch(deadline)
        if isinstance(sessions, str):
            return Resolution(question=sessions)
        for s in sessions:
            if s.session_id == picked.session_id:
                return Resolution(s)
        return Resolution(question=f"{_say_name(picked)} is no longer running.")

    @staticmethod
    def _unknown(target: str, sessions: Sequence[Session]) -> Resolution:
        close = _candidates(target, sessions)
        said = redact(target, 40)
        if close:
            names = ", ".join(_say_name(s) for s in close)
            return Resolution(question=f"No session is called {said!r}. Did you mean {names}?")
        if sessions:
            names = ", ".join(_say_name(s) for s in sessions[:5])
            return Resolution(question=f"No session is called {said!r}. Running: {names}.")
        return Resolution(question="No Claude sessions are running.")

    def _done(self, reason: str, text: str) -> str:
        """Every result reached the agent: unused capabilities are void now."""
        self.ctl.note_context(reason)
        return text

    # -- list / read / brief --------------------------------------------------------------
    def list_sessions(self) -> str:
        """Numbered list of every live session (both accounts)."""
        self._sync_call()
        sessions = self._fetch()
        if isinstance(sessions, str):
            log_tool("list_sessions", "failed")
            return self._done("list_sessions", sessions)
        self._numbered = list(sessions)
        log_tool("list_sessions", str(len(sessions)))
        if not sessions:
            return self._done("list_sessions", "No Claude sessions are running.")
        lines = [f"{len(sessions)} sessions:"]
        for i, s in enumerate(sessions, start=1):
            state = f"background, {s.status}" if s.background else s.status
            aim = redact(" ".join(s.aim_short.split()), AIM_CAP)
            lines.append(f"{i}. {_say_name(s)} — {state}" + (f" — {aim}" if aim else ""))
        return self._done("list_sessions", redact("\n".join(lines), LIST_CAP))

    def read_session(self, target: str, what: str = "last_reply") -> str:
        """What the session last said (``last_reply``) or was last asked (``last_prompt``)."""
        what = (what or "last_reply").strip().lower()
        if what not in {"last_reply", "last_prompt"}:
            return self._done("read_session", "I can read the last reply or the last prompt.")
        deadline = Deadline(TOOL_BUDGET_S, self.clock)
        found = self.resolve(target, deadline=deadline)
        if found.session is None:
            return self._done("read_session", found.question)
        s = found.session
        res = self.client.inspect(s.session_id, timeout=_within(deadline, READ_TIMEOUT_S))
        log_tool("read_session", f"{s.name} {what}")
        if not res.ok or res.data is None:
            return self._done(
                "read_session", f"Cannot read {_say_name(s)}: {reason_for(res.code)}."
            )
        text = " ".join(str(res.data.get(what) or "").split())
        if what == "last_reply":
            text = _tail_reply(text)  # before redaction: a cut-off token is no token
        if not text:
            empty = "has no reply yet" if what == "last_reply" else "has no prompt yet"
            return self._done("read_session", f"{_say_name(s)} {empty}.")
        head = "last said" if what == "last_reply" else "was last asked"
        return self._done("read_session", f"{_say_name(s)} {head}: {redact(text, READ_CAP)}")

    def _inspect_decision(
        self, s: Session, *, llm: bool, deadline: Deadline | None = None
    ) -> tuple[dict[str, Any] | None, str]:
        """``(data, "")`` or ``(None, speakable failure)``."""
        res = self.client.inspect(
            s.session_id,
            llm=llm,
            timeout=_within(deadline, BRIEF_TIMEOUT_S if llm else READ_TIMEOUT_S),
        )
        if llm and not res.ok and res.code == "timeout":
            fallback = _within(deadline, BRIEF_FALLBACK_TIMEOUT_S)
            res = self.client.inspect(s.session_id, timeout=fallback)
        if not res.ok or res.data is None:
            return None, f"Cannot read {_say_name(s)}: {reason_for(res.code)}."
        return res.data, ""

    def brief_decision(self, target: str) -> str:
        """The spoken briefing of the decision the session needs (D5), ≤ 10 s in total."""
        deadline = Deadline(TOOL_BUDGET_S, self.clock)
        found = self.resolve(target, deadline=deadline)
        if found.session is None:
            return self._done("brief_decision", found.question)
        s = found.session
        data, failure = self._inspect_decision(s, llm=True, deadline=deadline)
        log_tool("brief_decision", s.name)
        if data is None:
            return self._done("brief_decision", failure)
        decision = data.get("decision")
        if not isinstance(decision, Mapping) or not _questions(decision):
            state = str(data.get("state") or s.status)
            return self._done(
                "brief_decision", f"{_say_name(s)} needs no decision right now (it is {state})."
            )
        self._briefed[s.session_id] = str(decision.get("decision_id") or "")
        return self._done("brief_decision", redact(format_briefing(s, decision), BRIEF_CAP))

    # -- proposals ----------------------------------------------------------------------
    def _mutable(self, target: str, verb: str, deadline: Deadline) -> Session | str:
        found = self.resolve(target, fresh=True, deadline=deadline)
        if found.session is None:
            return found.question
        s = found.session
        if s.background:
            return f"{_say_name(s)} is a background session — I can read it but not {verb} it."
        return s

    def _authorise(self, action: str, target: str, s: Session) -> str | None:
        """Spend the capability of what Albert just said on ``action``; None when allowed.

        The target must come from that transcript: the list number when he said one, else
        the session's name. A read / briefing since then already voided the capability.
        """
        cap = self.ctl.mint()
        if isinstance(cap, Refusal):
            return f"refused: {cap.reason}"
        number = _target_number(" ".join(str(target or "").split()))
        said = str(number) if number is not None else s.name
        refusal = self.ctl.caps.use(cap, action, {"target": said})
        return None if refusal is None else f"refused: {refusal.reason}"

    def send_message(self, target: str, text: str) -> str:
        """Propose sending ``text`` to the session; it runs after ``confirm_action(code)``."""
        deadline = Deadline(TOOL_BUDGET_S, self.clock)
        s = self._mutable(target, "message", deadline)
        if isinstance(s, str):
            return self._done("send_message", s)
        message = _clean_message(str(text or ""))
        log_tool("send_message", s.name)
        if not message:
            return self._done("send_message", "What should I send?")
        if len(message) > MESSAGE_MAX:
            return self._done("send_message", f"That message is too long for {_say_name(s)}.")
        if s.status in {"waiting", "blocked"}:
            why = REASONS["picker_pending"] if s.status == "waiting" else REASONS["blocked"]
            return self._done("send_message", f"Cannot message {_say_name(s)}: {why}.")
        refused = self._authorise("send_message", target, s)
        if refused is not None:
            return self._done("send_message", refused)
        return self._done("send_message", self._propose_send(s, message))

    def _propose_send(self, s: Session, message: str, summary: str = "") -> str:
        pending = self.ctl.propose(
            "send", {"session_id": s.session_id, "name": s.name, "text": message}
        )
        if summary:
            return f"Say confirm {pending.code} to answer {_say_name(s)} with {summary}."
        said = redact(" ".join(message.split()), PROMPT_TEXT_CAP)
        return f"Say confirm {pending.code} to send to {_say_name(s)}: {said}"

    def answer_decision(self, target: str, choice: str) -> str:
        """Propose answering the session's decision with ``choice`` (letter, label, number)."""
        deadline = Deadline(TOOL_BUDGET_S, self.clock)
        s = self._mutable(target, "answer", deadline)
        if isinstance(s, str):
            return self._done("answer_decision", s)
        log_tool("answer_decision", s.name)
        refused = self._authorise("answer_decision", target, s)
        if refused is not None:
            return self._done("answer_decision", refused)
        proposal = self._propose_decision(s, str(choice or ""), deadline)
        return self._done("answer_decision", proposal)

    def _open_decision(
        self, s: Session, deadline: Deadline
    ) -> tuple[Mapping[str, Any], list[Question]] | str:
        """The session's current decision + questions, or why there is none to answer."""
        data, failure = self._inspect_decision(s, llm=False, deadline=deadline)
        if data is None:
            return failure
        decision = data.get("decision")
        questions = _questions(decision) if isinstance(decision, Mapping) else []
        if not isinstance(decision, Mapping) or not questions:
            return f"{_say_name(s)} has no open decision."
        briefed = self._briefed.get(s.session_id)
        if briefed is not None and briefed != str(decision.get("decision_id") or ""):
            return f"{_say_name(s)}: {REASONS['decision_changed']}."
        return decision, questions

    def _propose_decision(self, s: Session, choice: str, deadline: Deadline) -> str:
        opened = self._open_decision(s, deadline)
        if isinstance(opened, str):
            return opened
        decision, questions = opened
        picks = self._resolve_picks(questions, choice)
        if isinstance(picks, str):
            return picks
        if decision.get("source") == "todo_line":  # answered with a message (send)
            # one line: the session reads the answers as one prompt
            text = " ".join(
                f"{q.text} → {_pick_text(q, p)}" for q, p in zip(questions, picks, strict=True)
            )
            said = redact(_picks_summary(questions, picks), LABEL_CAP * 2)
            return self._propose_send(s, _clean_message(text)[:MESSAGE_MAX], said)
        did = str(decision.get("decision_id") or "")
        return self._propose_answer(s, did, questions, picks)

    @staticmethod
    def _resolve_picks(questions: Sequence[Question], choice: str) -> list[Pick] | str:
        segments = _segments(choice, len(questions))
        if segments is None:
            return (
                f"That decision has {len(questions)} questions — "
                "give one answer per question, separated by semicolons."
            )
        picks: list[Pick] = []
        for q, seg in zip(questions, segments, strict=True):
            pick = resolve_choice(q, seg)
            if isinstance(pick, str):
                return redact(pick, LABEL_CAP * 3)
            picks.append(pick)
        return picks

    def _propose_answer(
        self, s: Session, did: str, questions: Sequence[Question], picks: Sequence[Pick]
    ) -> str:
        answers: list[dict[str, Any]] = []
        for i, p in enumerate(picks):
            if p.other:
                answers.append({"question_index": i, "other_text": p.other})
            else:
                answers.append({"question_index": i, "option_indices": list(p.indices)})
        pending = self.ctl.propose(
            "answer",
            {"session_id": s.session_id, "name": s.name, "decision_id": did, "answers": answers},
        )
        said = redact(_picks_summary(questions, picks), LABEL_CAP * 2)
        return f"Say confirm {pending.code} to answer {_say_name(s)} with {said}."

    # -- executors (run by confirm_action) -------------------------------------------------
    def execute_send(self, pending: PendingAction) -> str:
        """Run a confirmed ``send`` proposal: ``ccc send -s ID -j`` (text on stdin)."""
        args = pending.args
        return self._mutate(
            "send",
            pending,
            lambda budget: self.client.send(str(args["session_id"]), str(args["text"]), budget),
        )

    def execute_answer(self, pending: PendingAction) -> str:
        """Run a confirmed ``answer`` proposal: ``ccc answer -s ID -j`` (JSON on stdin)."""
        args = pending.args
        payload = {"decision_id": args["decision_id"], "answers": args["answers"]}
        return self._mutate(
            "answer",
            pending,
            lambda budget: self.client.answer(str(args["session_id"]), payload, budget),
        )

    def _mutate(self, kind: str, pending: PendingAction, act: Callable[[float], CccResult]) -> str:
        name = str(pending.args.get("name") or "the session")
        session_id = str(pending.args.get("session_id") or "")
        deadline = Deadline(MUTATION_BUDGET_S, self.ctl.clock)
        lock_wait = min(LOCK_TIMEOUT_S, deadline.remaining())
        if not self.ctl.mutation_lock.acquire(timeout=max(lock_wait, 0.0)):
            return self._done(kind, "refused: another action is still running")
        try:
            refusal = before_mutation(self.ctl.cancel, deadline, None, Mutation(kind, pending.args))
            if refusal is not None:
                return self._done(kind, f"refused: {refusal.reason}")
            result = act(deadline.remaining())
        finally:
            self.ctl.mutation_lock.release()
        return self._done(kind, self._outcome(kind, session_id, name, result))

    def _outcome(self, kind: str, session_id: str, name: str, result: CccResult) -> str:
        """Speak the result; follow accepted / unknown deliveries, report real failures."""
        data = result.data or {}
        outcome = str(data.get("outcome") or "")
        verb = "sent" if kind == "send" else "answered"
        if result.ok and outcome in {"", "accepted"}:
            state = "accepted"
        elif (
            result.code in UNKNOWN_CODES
            or result.code in UNCERTAIN_CODES
            or (result.ok and outcome == "unknown")
        ):
            state = "unknown"  # ccc may well have acted: never "failed", never retried
        else:
            return self._failed(kind, session_id, name, result)
        _log_delivery(kind, name, f"{state} ({result.code})" if result.code else state)
        delivery_id = data.get("delivery_id")
        if isinstance(delivery_id, str) and delivery_id:
            self._track(delivery_id, session_id, name, state)
        return verb if state == "accepted" else f"{verb}, not yet confirmed"

    def _failed(self, kind: str, session_id: str, name: str, result: CccResult) -> str:
        outcome = str((result.data or {}).get("outcome") or "")
        fallback = "delivery_failed" if kind == "send" else "answer_failed"
        code = result.code or (fallback if outcome == "failed" else outcome or fallback)
        reason = reason_for(code)
        _log_delivery(kind, name, f"failed ({code})")
        if code in POST_ACTION_FAILURES and (result.data is not None or result.ok):
            self._report(Problem(fallback, name, reason), session_id)
        return f"failed: {redact(name, 40)}: {reason}"

    def _track(self, delivery_id: str, session_id: str, name: str, outcome: str) -> None:
        """Hand the delivery to the attention monitor; a tracker error never masks the send."""
        try:
            self.ctl.deliveries.track_delivery(delivery_id, session_id, name, outcome)
        except Exception:  # pylint: disable=broad-exception-caught  # the send already happened
            log.warning("⚠️  delivery tracking failed", exc_info=True)

    def _report(self, problem: Problem, session_id: str) -> None:
        """One inbox item per failing session (the monitor's ``failed:<session_id>`` key)."""
        report: Callable[..., None] = self.ctl.problems.report
        try:
            if _accepts_dedupe_key(report):
                report(problem, dedupe_key=f"failed:{session_id}")
            else:
                report(problem)
        except Exception:  # pylint: disable=broad-exception-caught  # the outcome stands
            log.warning("⚠️  problem report failed", exc_info=True)


def format_briefing(s: Session, decision: Mapping[str, Any]) -> str:
    """D5: "The session X, which works on Y, needs a decision: Q Options: a) … It recommends …".

    The ccc contract carries no separate "benefit" field: the AIM says what the session
    works on, and ``recent_summary`` follows as "Recently: …" after the recommendation.
    """
    raw_ctx = decision.get("context")
    ctx: Mapping[str, Any] = raw_ctx if isinstance(raw_ctx, Mapping) else {}
    aim = " ".join(str(ctx.get("aim") or s.aim_short or "").split()).rstrip(".")
    ticket = str(ctx.get("ticket") or "")
    head = f"The session {s.name}"
    if aim:
        head += f", which works on {redact(aim, 200)}"
        if ticket and ticket not in aim:
            head += f" ({ticket})"
    questions = _questions(decision)
    if len(questions) == 1:
        parts = [f"{head}, needs a decision: {_say_question(questions[0])}"]
    else:
        parts = [f"{head}, needs {len(questions)} decisions."]
        parts += [f"{i}) {_say_question(q)}" for i, q in enumerate(questions, start=1)]
    parts.append(_recommendation(questions, decision))
    summary = " ".join(str(ctx.get("recent_summary") or "").split())
    if summary:
        parts.append(f"Recently: {redact(summary, 500)}")
    return " ".join(parts)


def _say_question(q: Question) -> str:
    text = redact(q.text, 300)
    if text and text[-1] not in ".?!":
        text += "."
    if q.multi_select:
        text += " Several can be picked."
    opts = "; ".join(
        f"{LETTERS[j]}) {redact(lab, LABEL_CAP)}" + (f" — {redact(con, LABEL_CAP)}" if con else "")
        for j, (lab, con) in enumerate(zip(q.labels, q.consequences, strict=True))
    )
    return f"{text} Options: {opts}." if opts else text


def _recommendation(questions: Sequence[Question], decision: Mapping[str, Any]) -> str:
    rec = " ".join(str(decision.get("recommendation") or "").split())
    why = " ".join(str(decision.get("recommendation_reason") or "").split())
    if not rec:
        return "It gives no recommendation."
    if len(questions) == 1:
        labels = [normalise(lab) for lab in questions[0].labels]
        if normalise(rec) in labels:
            idx = labels.index(normalise(rec))
            rec = f"{LETTERS[idx]}) {questions[0].labels[idx]}"
    text = f"It recommends {redact(rec, 300)}"
    return text + (f" because {redact(why, 300)}." if why else ".")


def _log_delivery(kind: str, name: str, outcome: str) -> None:
    arrow = "📨 send" if kind == "send" else "📨 answer"
    log.info("%s → %s: %s", arrow, redact(" ".join(name.split()), 40), outcome)


def _guarded(name: str, fn: Callable[..., str], ctl: BridgeController) -> Callable[..., str]:
    """A tool that never raises into the SDK's thread pool."""

    signature = inspect.signature(fn)

    def tool(*args: Any, **kwargs: Any) -> str:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            log.warning("⚠️  %s: bad arguments", name)
            ctl.note_context(name)
            return f"{name}: missing or unexpected arguments"
        try:
            return fn(*args, **kwargs)
        except Exception:  # pylint: disable=broad-exception-caught  # tools never raise
            log.warning("⚠️  %s failed", name, exc_info=True)
            ctl.note_context(name)
            return f"{name} failed: internal error"

    tool.__name__ = name
    tool.__doc__ = fn.__doc__
    return tool


def register(
    controller: BridgeController,
    client: CccClient | None = None,
    env: Mapping[str, str] | None = None,
) -> SessionTools | None:
    """Add the sessions tools + executors when ``MAC_VOICE_CLAUDE_BRIDGE=1`` (else None)."""
    env = os.environ if env is None else env
    if env.get(FLAG, "").strip() != "1":
        return None
    tools = SessionTools(controller, client, clock=controller.clock)
    tools.register()
    return tools
