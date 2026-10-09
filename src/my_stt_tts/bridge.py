"""Bridge between the voice agent and the Mac / Claude Code sessions: who may act, and how.

Built once in ``voice_control.daemon_main`` when one of the opt-in flags
(``MAC_VOICE_CLAUDE_BRIDGE``, ``MAC_VOICE_MAC_CONTROL``, ``MAC_VOICE_OPERATOR``) is ``1``.
:class:`BridgeController` lives as long as the daemon; each call runs between
:meth:`~BridgeController.begin_call` and :meth:`~BridgeController.end_call`. The
``VoiceSession`` feeds it the call's mic frames (segmented by
:class:`~my_stt_tts.turns.TurnSource`) and every user transcript with a sequence number
and the monotonic time it arrived.

Authorisation (only Albert's voice may act, fail closed):

* :class:`Authoriser` — the utterance a transcript binds to must match the profile named
  by ``MAC_VOICE_AUTHORIZED`` (its call-domain profile ``enroll/call/<name>.npy`` when there
  is one, else the wake profile) with cosine ≥ 0.35. No such variable, no profile, a model
  failure, no / stale binding, an utterance under 0.4 s → refused. With several candidate
  utterances every one must pass. ``[system notice]`` messages never pass.
* :class:`CapabilityStore` — an authorised transcript mints at most ONE capability; any
  read, briefing, tool result or contextual update invalidates the unused ones (and the
  transcripts behind them). A capability allows one action whose arguments come from the
  transcript (:func:`~my_stt_tts.bridge_text.derivable`).
* :class:`PendingActions` — risky actions become a proposal with a random 2-digit code,
  valid 60 s, single use, confirmed only by a NEW authorised transcript saying the code.

Every mutation ends with :func:`before_mutation` (cancellation, deadline, capability)
right before it acts. Logging is one line per event and never carries content.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .bridge_text import derivable, numbers_in, redact
from .turns import MIN_UTTERANCE_S, Ambiguous, Binding, SpeechDetector, TurnSource, Utterance

log = logging.getLogger("my_stt_tts.bridge")

FLAGS = ("MAC_VOICE_CLAUDE_BRIDGE", "MAC_VOICE_MAC_CONTROL", "MAC_VOICE_OPERATOR")
AUTH_ENV = "MAC_VOICE_AUTHORIZED"
AUTH_THRESHOLD = 0.35
CONFIRM_TTL_S = 60.0
NOTICE_PREFIX = "[system notice]"
BRIEFING_CAP = 600
NOTICE_CAP = 300
LOG_DETAIL_CAP = 80


def bridge_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when any bridge flag is ``1`` in ``env`` (default: the process environment)."""
    env = os.environ if env is None else env
    return any(env.get(flag, "").strip() == "1" for flag in FLAGS)


# -- results ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Refusal:
    """Why something was not allowed: ``code`` for programs, ``reason`` short and speakable."""

    code: str
    reason: str


VOICE_NOT_VERIFIED = "voice not verified"


def _refuse(code: str, reason: str = VOICE_NOT_VERIFIED) -> Refusal:
    return Refusal(code, reason)


@dataclass(frozen=True)
class Transcript:
    """One transcript as the controller received it (``received_at`` monotonic seconds)."""

    seq: int
    text: str = field(repr=False)
    received_at: float
    injected: bool = False
    binding: Binding = field(default=None, repr=False)


# -- logging (one line, never content) -----------------------------------------------------
def _detail(text: str) -> str:
    return redact(" ".join(str(text).split()), LOG_DETAIL_CAP)


def log_tool(name: str, detail: str = "") -> str:
    """``🛠️ open_url youtube`` — ``detail`` is a short label, never message content."""
    line = f"🛠️ {_detail(name)} {_detail(detail)}".rstrip()
    log.info(line)
    return line


def log_confirm(summary: str) -> str:
    """``🔒 confirm needed: close 3 tabs``."""
    line = f"🔒 confirm needed: {_detail(summary)}"
    log.info(line)
    return line


def log_refused(reason: str = VOICE_NOT_VERIFIED) -> str:
    """``🚫 refused: voice not verified``."""
    line = f"🚫 refused: {_detail(reason)}"
    log.info(line)
    return line


# -- identity ------------------------------------------------------------------------------
class SpeakerScorer(Protocol):
    """Cosine of a clip against one enrolled profile (``voice_gate.VoiceGate``)."""

    def score_against(self, audio: Any, name: str, *, timeout: float = ...) -> float | None: ...


class Authoriser:
    """Decide whether a transcript was spoken by the authorised person (fail closed)."""

    def __init__(
        self,
        scorer: SpeakerScorer | None,
        authorized: str | None,
        *,
        threshold: float = AUTH_THRESHOLD,
        min_utterance_s: float = MIN_UTTERANCE_S,
    ) -> None:
        self.scorer = scorer
        self.authorized = (authorized or "").strip() or None
        self.threshold = threshold
        self.min_utterance_s = min_utterance_s

    @classmethod
    def from_env(
        cls, scorer: SpeakerScorer | None, env: Mapping[str, str] | None = None
    ) -> Authoriser:
        env = os.environ if env is None else env
        return cls(scorer, env.get(AUTH_ENV))

    def verify(self, transcript: Transcript) -> Refusal | None:
        """None when the transcript may act, else the :class:`Refusal`."""
        refusal = self._precheck(transcript)
        notes: list[str] = []
        if refusal is None:
            for utt in _candidates(transcript.binding):
                refusal = self._check_utterance(utt, notes)
                if refusal is not None:
                    break
        if refusal is not None:
            ambiguous = isinstance(transcript.binding, Ambiguous)
            diag = ", ".join([refusal.code, *(["ambiguous"] if ambiguous else []), *notes])
            log_refused(f"{refusal.reason} ({diag})")
            return refusal
        for note in notes:
            log.info("🔓 voice verified (%s)", note)
        return None

    def _precheck(self, transcript: Transcript) -> Refusal | None:
        if transcript.injected:
            return _refuse("injected", "injected message")
        if self.authorized is None:
            return _refuse("not_configured")
        if self.scorer is None:
            return _refuse("no_profile")
        if transcript.binding is None:
            return _refuse("no_utterance")
        return None

    def _check_utterance(self, utt: Utterance, notes: list[str]) -> Refusal | None:
        """Check one utterance; append ``<duration>s[ score[ kind]]`` to ``notes`` (no content).

        ``kind`` (``call`` / ``wake``) is the profile the scorer used, when it can tell
        (``VoiceGate.profile_kind``); scorers without that method just omit it.
        """
        notes.append(f"{utt.duration:.2f}s")
        if utt.duration < self.min_utterance_s:
            return _refuse("too_short")
        assert self.scorer is not None and self.authorized is not None
        try:
            score = self.scorer.score_against(utt.pcm, self.authorized)
        except Exception:  # pylint: disable=broad-exception-caught  # fail closed
            log.warning("⚠️  voice check failed", exc_info=True)
            return _refuse("model_failure")
        if score is None:
            return _refuse("no_profile")
        notes[-1] += f" {score:.2f}"
        kind = self._profile_kind()
        if kind:
            notes[-1] += f" {kind}"
        if score < self.threshold:
            return _refuse("other_speaker")
        return None

    def _profile_kind(self) -> str | None:
        kind_of = getattr(self.scorer, "profile_kind", None)
        if not callable(kind_of) or self.authorized is None:
            return None
        try:
            kind = kind_of(self.authorized)
        except Exception:  # pylint: disable=broad-exception-caught  # diagnostics only
            return None
        return kind if isinstance(kind, str) else None


def _candidates(binding: Binding) -> tuple[Utterance, ...]:
    if isinstance(binding, Ambiguous):
        return binding.candidates
    return () if binding is None else (binding,)


# -- capabilities --------------------------------------------------------------------------
@dataclass(frozen=True)
class Capability:
    """The right to ONE action derived from one authorised transcript."""

    id: str
    transcript_seq: int
    text: str = field(repr=False)
    minted_at: float


class CapabilityStore:
    """Mint, check, use and invalidate capabilities (see module docstring)."""

    def __init__(self, authoriser: Authoriser, clock: Callable[[], float] = time.monotonic):
        self.authoriser = authoriser
        self.clock = clock
        self._live: dict[str, Capability] = {}
        self._minted: set[int] = set()
        self._burned_through = -1  # transcripts up to this seq can no longer mint
        self._lock = threading.Lock()

    def mint(self, transcript: Transcript) -> Capability | Refusal:
        """At most one capability per authorised user transcript."""
        with self._lock:
            refusal = self._mint_precheck(transcript)
            if refusal is None:
                self._minted.add(transcript.seq)  # one attempt per transcript, pass or fail
        if refusal is not None:
            log_refused(refusal.reason)
            return refusal
        refusal = self.authoriser.verify(transcript)
        if refusal is not None:
            return refusal
        cap = Capability(secrets.token_hex(8), transcript.seq, transcript.text, self.clock())
        with self._lock:
            if transcript.seq <= self._burned_through:  # a read arrived while verifying
                return _refuse("invalidated", "context changed — say it again")
            self._live[cap.id] = cap
        return cap

    def _mint_precheck(self, transcript: Transcript) -> Refusal | None:
        if transcript.injected:
            return _refuse("injected", "injected message")
        if transcript.seq in self._minted:
            return _refuse("already_used", "already used — say it again")
        if transcript.seq <= self._burned_through:
            return _refuse("invalidated", "context changed — say it again")
        return None

    def consume_transcript(self, seq: int) -> None:
        """Mark a transcript as spent (e.g. it confirmed a code) so it mints nothing."""
        with self._lock:
            self._minted.add(seq)

    def check(self, cap: Capability, action: str, args: Mapping[str, Any]) -> Refusal | None:
        """Would :meth:`use` allow this? (does not consume)."""
        with self._lock:
            return self._check(cap, action, args)

    def use(self, cap: Capability, action: str, args: Mapping[str, Any]) -> Refusal | None:
        """Consume ``cap`` for ``action(args)``; None when allowed, else the Refusal."""
        with self._lock:
            refusal = self._check(cap, action, args)
            if refusal is None:
                del self._live[cap.id]
        if refusal is not None:
            log_refused(refusal.reason)
        return refusal

    def _check(self, cap: Capability, action: str, args: Mapping[str, Any]) -> Refusal | None:
        if self._live.get(cap.id) != cap:
            return _refuse("no_capability", "no fresh instruction — say it again")
        for key, value in args.items():
            if not derivable(value, cap.text, key):
                return _refuse("not_in_transcript", f"{action}: argument not in what you said")
        return None

    def invalidate_all(self, reason: str, *, through_seq: int | None = None) -> int:
        """Drop every unused capability; transcripts up to ``through_seq`` mint nothing more."""
        with self._lock:
            dropped = len(self._live)
            self._live.clear()
            if through_seq is not None:
                self._burned_through = max(self._burned_through, through_seq)
        if dropped:
            log.debug("capabilities invalidated (%s): %d", reason, dropped)
        return dropped


# -- proposals + confirmation codes ---------------------------------------------------------
def _norm_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    if isinstance(value, Mapping):
        return {str(k): _norm_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_norm_value(v) for v in value]
    return value


def args_digest(kind: str, args: Mapping[str, Any]) -> str:
    """sha256 of ``kind`` + normalised arguments (whitespace, Unicode NFC, key order)."""
    blob = json.dumps(
        {"kind": kind, "args": _norm_value(args)},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True)
class PendingAction:  # pylint: disable=too-many-instance-attributes  # a plain record
    """A risky action waiting for its spoken code."""

    id: str
    kind: str
    args: Mapping[str, Any] = field(repr=False)
    sha256: str
    code: str
    expires: float
    created_at: float
    source_seq: int | None = None

    def matches(self, kind: str, args: Mapping[str, Any]) -> bool:
        """True only for exactly the proposed action (no replacement arguments)."""
        return secrets.compare_digest(args_digest(kind, args), self.sha256)


class PendingActions:
    """Proposals confirmed by a random 2-digit code said in a NEW authorised transcript."""

    def __init__(
        self,
        authoriser: Authoriser,
        clock: Callable[[], float] = time.monotonic,
        ttl: float = CONFIRM_TTL_S,
    ) -> None:
        self.authoriser = authoriser
        self.clock = clock
        self.ttl = ttl
        self._by_code: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def propose(
        self, kind: str, args: Mapping[str, Any], *, source_seq: int | None = None
    ) -> PendingAction:
        """Store a proposal; ``source_seq`` is the transcript that asked (it cannot confirm)."""
        now = self.clock()
        with self._lock:
            self._expire(now)
            code = self._fresh_code()
            pending = PendingAction(
                id=secrets.token_hex(8),
                kind=kind,
                args=dict(args),  # executed as given; the digest is over the normalised form
                sha256=args_digest(kind, args),
                code=code,
                expires=now + self.ttl,
                created_at=now,
                source_seq=source_seq,
            )
            self._by_code[code] = pending
        log_confirm(kind)
        return pending

    def _fresh_code(self) -> str:
        while True:
            code = str(secrets.randbelow(90) + 10)  # 10–99: always two spoken digits
            if code not in self._by_code:
                return code

    def _expire(self, now: float) -> None:
        for code in [c for c, p in self._by_code.items() if p.expires <= now]:
            del self._by_code[code]

    def confirm(self, code: str | int, transcript: Transcript) -> PendingAction | Refusal:
        """The proposal ``code`` names, once a new authorised transcript says the code."""
        refusal = self.authoriser.verify(transcript)
        if refusal is not None:
            return refusal
        with self._lock:
            result = self._take(_code_key(code), transcript)
        if isinstance(result, Refusal):
            log_refused(result.reason)
        return result

    def _take(self, key: str, transcript: Transcript) -> PendingAction | Refusal:
        pending = self._by_code.get(key)
        if pending is None:
            return _refuse("wrong_code", "wrong code")
        if pending.expires <= self.clock():
            del self._by_code[key]
            return _refuse("expired", "code expired")
        if not _is_newer(transcript, pending):
            return _refuse("not_new", "say the code in a new sentence")
        if int(key) not in numbers_in(transcript.text):
            return _refuse("code_not_said", "code not heard")
        if not CONFIRM_WORD.search(transcript.text):
            return _refuse("no_confirm_word", "say confirm and the code")
        del self._by_code[key]
        return pending

    def clear(self) -> None:
        with self._lock:
            self._by_code.clear()

    def live(self) -> list[PendingAction]:
        with self._lock:
            self._expire(self.clock())
            return list(self._by_code.values())


#: The code only counts next to a confirm word (en/de/fr), so steering Albert into saying the
#: number in another sentence ("set the volume to 35") confirms nothing.
CONFIRM_WORD = re.compile(r"\b(confirm\w*|bestätig\w*|bestaetig\w*)", re.IGNORECASE)


def _code_key(code: str | int) -> str:
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return str(int(digits)) if digits else ""


def _is_newer(transcript: Transcript, pending: PendingAction) -> bool:
    if pending.source_seq is not None and transcript.seq <= pending.source_seq:
        return False
    return transcript.received_at > pending.created_at


# -- cancellation + deadlines ----------------------------------------------------------------
class CancellationToken:
    """Set once (call end, "stop", daemon shutdown); mutations check it right before acting."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "cancelled") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


class Deadline:
    """A monotonic deadline: ``remaining()`` feeds subprocess timeouts."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.at = clock() + seconds

    def remaining(self) -> float:
        return max(0.0, self.at - self.clock())

    @property
    def expired(self) -> bool:
        return self.clock() >= self.at


@dataclass(frozen=True)
class Mutation:
    """What :func:`before_mutation` checks: the action, its arguments and its capability."""

    action: str
    args: Mapping[str, Any]
    cap: Capability | None = None


def before_mutation(
    token: CancellationToken,
    deadline: Deadline,
    caps: CapabilityStore | None,
    mutation: Mutation,
) -> Refusal | None:
    """The final gate right before acting: not cancelled, in time, capability consumed.

    With ``caps`` None (a code-confirmed :class:`PendingAction`) only cancellation and the
    deadline are checked.
    """
    refusal: Refusal | None = None
    if token.cancelled:
        refusal = _refuse("cancelled", token.reason or "cancelled")
    elif deadline.expired:
        refusal = _refuse("deadline", "took too long")
    if refusal is not None:
        log_refused(refusal.reason)
        return refusal
    if caps is None:
        return None
    if mutation.cap is None:
        log_refused("no fresh instruction")
        return _refuse("no_capability", "no fresh instruction — say it again")
    return caps.use(mutation.cap, mutation.action, mutation.args)


# -- problems + briefings ------------------------------------------------------------------
@dataclass(frozen=True)
class Problem:
    """A problem worth telling Albert about (no message text in ``reason``)."""

    kind: str
    subject: str
    reason: str
    severity: str = "error"


class ProblemSink(Protocol):
    """Where problems go; ``resolve`` closes an open one once later evidence settled it."""

    def report(self, problem: Problem) -> None: ...

    def resolve(self, kind: str, subject: str) -> bool: ...


class BriefingProvider(Protocol):
    def briefing(self) -> str | None: ...


class MemoryProblemSink:
    """Keeps problems in a list (Phase 6 replaces it with the attention inbox)."""

    def __init__(self) -> None:
        self.problems: list[Problem] = []
        self._lock = threading.Lock()

    def report(self, problem: Problem) -> None:
        with self._lock:
            self.problems.append(problem)
        log.info("⚠️  problem: %s · %s", _detail(problem.kind), _detail(problem.subject))

    def resolve(self, kind: str, subject: str) -> bool:
        """Drop every kept problem ``kind`` · ``subject``; True when there was one."""
        with self._lock:
            kept = [p for p in self.problems if (p.kind, p.subject) != (kind, subject)]
            dropped = len(self.problems) - len(kept)
            self.problems[:] = kept
        if dropped:
            log.info("✅ problem resolved: %s · %s", _detail(kind), _detail(subject))
        return dropped > 0


class DeliveryTracker(Protocol):
    """Who follows a ccc delivery after the voice sent it (Phase 6's attention monitor)."""

    def track_delivery(
        self, delivery_id: str, session_id: str, subject: str, outcome: str
    ) -> None: ...


class MemoryDeliveryTracker:
    """In-memory :class:`DeliveryTracker` (tests, and until the attention store is wired)."""

    def __init__(self) -> None:
        self.tracked: list[tuple[str, str, str, str]] = []

    def track_delivery(self, delivery_id: str, session_id: str, subject: str, outcome: str) -> None:
        self.tracked.append((delivery_id, session_id, subject, outcome))


class MemoryBriefingProvider:
    """A fixed briefing (or none), redacted and capped at 600 characters."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def briefing(self) -> str | None:
        return redact(self.text, BRIEFING_CAP) if self.text else None


# -- the controller --------------------------------------------------------------------------
def _default_vad() -> SpeechDetector:
    from .vad import SileroVad  # pylint: disable=import-outside-toplevel  # torch is heavy

    return SileroVad()


class BridgeController:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Owns authorisation state for the daemon's lifetime; one call at a time."""

    def __init__(
        self,
        authoriser: Authoriser,
        *,
        vad_factory: Callable[[], SpeechDetector] = _default_vad,
        problems: ProblemSink | None = None,
        deliveries: DeliveryTracker | None = None,
        briefings: BriefingProvider | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.authoriser = authoriser
        self.clock = clock
        self.caps = CapabilityStore(authoriser, clock)
        self.pending = PendingActions(authoriser, clock)
        self.problems: ProblemSink = problems or MemoryProblemSink()
        self.briefings: BriefingProvider = briefings or MemoryBriefingProvider()
        self.deliveries: DeliveryTracker = deliveries or MemoryDeliveryTracker()
        self.tools: dict[str, Callable[..., Any]] = {"confirm_action": self.confirm_action}
        self.executors: dict[str, Callable[[PendingAction], str]] = {}
        self.mutation_lock = threading.Lock()  # one mutation at a time (7.7)
        self.turns: TurnSource | None = None
        self.cancel = CancellationToken()
        self._vad_factory = vad_factory
        self._vad: SpeechDetector | None = None
        self._latest: Transcript | None = None
        self._shutdown_hooks: list[Callable[[], None]] = []
        self._feed_failed = False
        self._notice_sink: Callable[[str], Any] | None = None
        self._in_flight: dict[str, int] = {}
        #: False once the agent refused the ``first_message`` override (its platform setting
        #: is off — ``scripts/eleven_agent_config.py -a`` turns it on): no more briefings at
        #: call start for this daemon's lifetime, so a refused override cannot break every call.
        self.first_message_ok = True
        self.components: dict[str, Any] = {}  # what bridge_wiring.wire registered (by name)
        self._lock = threading.Lock()

    # -- registries ---------------------------------------------------------------------
    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Client tools later phases add (name → callable)."""
        self.tools[name] = fn

    def register_executor(self, kind: str, fn: Callable[[PendingAction], str]) -> None:
        """What runs a confirmed :class:`PendingAction` of ``kind`` (returns speakable text)."""
        self.executors[kind] = fn

    def on_shutdown(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` at daemon shutdown (monitor, operator)."""
        self._shutdown_hooks.append(fn)

    # -- call lifecycle -----------------------------------------------------------------
    def preload(self) -> None:
        """Load the VAD model in the background so the first call's audio is not delayed."""

        def _load() -> None:
            with contextlib.suppress(Exception):
                self._ensure_vad().is_speech(np.zeros(512, dtype=np.float32))

        threading.Thread(target=_load, name="bridge-vad-load", daemon=True).start()

    def _ensure_vad(self) -> SpeechDetector:
        with self._lock:
            if self._vad is None:
                self._vad = self._vad_factory()
            return self._vad

    def begin_call(self) -> None:
        """A call starts: fresh turn ring, no capabilities or proposals from before."""
        turns = TurnSource(self._ensure_vad(), clock=self.clock)
        self._reset("call started")
        with self._lock:
            self.turns, self.cancel = turns, CancellationToken()

    def end_call(self) -> None:
        """The call ended: stop listening, cancel running work, drop every right."""
        with self._lock:
            self.turns = None
            cancel = self.cancel
        cancel.cancel("call ended")
        self._reset("call ended")

    def _reset(self, reason: str) -> None:
        latest = self._latest
        self.caps.invalidate_all(reason, through_seq=latest.seq if latest else None)
        self.pending.clear()
        with self._lock:
            self._latest = None

    def shutdown(self) -> None:
        """Daemon shutdown: end the call, then run the registered shutdown hooks."""
        self.end_call()
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception:  # pylint: disable=broad-exception-caught
                log.warning("⚠️  bridge shutdown hook failed", exc_info=True)

    # -- inputs from the VoiceSession -----------------------------------------------------
    def feed_audio(self, frame: Any) -> None:
        """One 16 kHz float32 mic frame of the current call (never raises)."""
        turns = self.turns
        if turns is None:
            return
        try:
            turns.feed(frame)
        except Exception:  # pylint: disable=broad-exception-caught  # never break the audio
            if not self._feed_failed:
                self._feed_failed = True
                log.warning("⚠️  turn segmentation failed", exc_info=True)

    def on_transcript(self, seq: int, text: str, received_at: float) -> Transcript:
        """A user transcript: bind it to its utterance and make it the latest."""
        if text.lstrip().casefold().startswith(NOTICE_PREFIX):
            return self.on_injected(text, seq=seq, received_at=received_at)
        turns = self.turns
        binding = turns.bind(seq, text, received_at) if turns is not None else None
        transcript = Transcript(seq, text, received_at, binding=binding)
        with self._lock:
            self._latest = transcript
        return transcript

    def on_injected(
        self, text: str, *, seq: int = -1, received_at: float | None = None
    ) -> Transcript:
        """An injected ``[system notice]``: never mints, and voids unused rights."""
        at = self.clock() if received_at is None else received_at
        self.note_context("injected notice")
        return Transcript(seq, text, at, injected=True)

    def note_context(self, reason: str) -> None:
        """A read / briefing / tool result / contextual update reached the agent."""
        latest = self._latest
        self.caps.invalidate_all(reason, through_seq=latest.seq if latest else None)

    # -- in-call notices ------------------------------------------------------------------
    def set_notice_sink(self, sink: Callable[[str], Any] | None) -> None:
        """The live call's way to make the agent speak (``send_user_message``); None = no call."""
        with self._lock:
            self._notice_sink = sink

    def notify_in_call(self, text: str) -> bool:
        """Speak ``text`` in the running call as ``[system notice] <text>``; False outside one.

        The notice is recorded as injected (:meth:`on_injected`) BEFORE it is sent, so it
        voids unused capabilities and can never mint one. Outside a call nothing is sent —
        the attention inbox keeps the problem for the next "voice on".
        """
        with self._lock:
            sink = self._notice_sink if self.turns is not None else None
        if sink is None:
            return False
        body = redact(" ".join(str(text).split()), NOTICE_CAP)
        if not body:
            return False
        notice = f"{NOTICE_PREFIX} {body}"
        self.on_injected(notice)
        try:
            sink(notice)
        except Exception:  # pylint: disable=broad-exception-caught  # the inbox keeps it
            log.warning("⚠️  in-call notice failed", exc_info=True)
            return False
        log.info("📣 notice sent to the call")
        return True

    # -- client tools in flight ------------------------------------------------------------
    def tool_started(self, name: str) -> None:
        with self._lock:
            self._in_flight[name] = self._in_flight.get(name, 0) + 1

    def tool_finished(self, name: str) -> None:
        with self._lock:
            left = self._in_flight.get(name, 0) - 1
            if left > 0:
                self._in_flight[name] = left
            else:
                self._in_flight.pop(name, None)

    def tool_in_flight(self, name: str) -> bool:
        """True while a call of the client tool ``name`` has not returned to the agent."""
        with self._lock:
            return self._in_flight.get(name, 0) > 0

    # -- authorisation entry points for tools ----------------------------------------------
    @property
    def latest(self) -> Transcript | None:
        return self._latest

    def mint(self) -> Capability | Refusal:
        """A capability from the latest user transcript (at most one per transcript)."""
        latest = self._latest
        if latest is None:
            log_refused("nothing said")
            return _refuse("no_transcript", "nothing said yet")
        return self.caps.mint(latest)

    def propose(self, kind: str, args: Mapping[str, Any]) -> PendingAction:
        """A risky action waiting for its code; the asking transcript cannot confirm it."""
        latest = self._latest
        return self.pending.propose(kind, args, source_seq=latest.seq if latest else None)

    def confirm(self, code: str | int) -> PendingAction | Refusal:
        """Confirm with the latest transcript, which is then spent (mints nothing)."""
        latest = self._latest
        if latest is None:
            log_refused("nothing said")
            return _refuse("no_transcript", "nothing said yet")
        self.caps.consume_transcript(latest.seq)
        return self.pending.confirm(code, latest)

    def confirm_action(self, code: str) -> str:
        """The ``confirm_action`` client tool: run the proposal ``code`` names, once."""
        result = self.confirm(code)
        if isinstance(result, Refusal):
            return f"refused: {result.reason}"
        executor = self.executors.get(result.kind)
        if executor is None:
            log_refused(f"no executor for {result.kind}")
            return "refused: nothing to confirm"
        return executor(result)


def build_bridge(
    scorer: SpeakerScorer | None,
    env: Mapping[str, str] | None = None,
    *,
    wire: bool = True,
) -> BridgeController:
    """The daemon's controller: identity from ``MAC_VOICE_AUTHORIZED`` + the voice gate.

    With ``wire`` the components register themselves, each gated by its own flag
    (:func:`my_stt_tts.bridge_wiring.wire`): attention inbox + monitor, Claude sessions,
    Mac control and the Mac operator.
    """
    authoriser = Authoriser.from_env(scorer, env)
    if authoriser.authorized is None:
        log.warning("⚠️  %s unset — every bridge action will be refused", AUTH_ENV)
    controller = BridgeController(authoriser)
    if wire:
        # pylint: disable-next=import-outside-toplevel  # it imports every component
        from .bridge_wiring import wire as wire_components

        wire_components(controller, env)
    controller.preload()
    return controller
