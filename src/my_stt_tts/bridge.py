"""Bridge between the voice agent and the Mac / Claude Code sessions: who may act, and how.

Built once in ``voice_control.daemon_main`` when one of the opt-in flags
(``MAC_VOICE_CLAUDE_BRIDGE``, ``MAC_VOICE_MAC_CONTROL``, ``MAC_VOICE_OPERATOR``) is ``1``.
:class:`BridgeController` lives as long as the daemon; each call runs between
:meth:`~BridgeController.begin_call` and :meth:`~BridgeController.end_call`. The
``VoiceSession`` feeds it the call's mic frames (segmented by
:class:`~my_stt_tts.turns.TurnSource`) and every user transcript with a sequence number
and the monotonic time it arrived.

Authorisation (two risk tiers, fail closed):

* Full voice check — Claude session sends / answers / acknowledgements, ``do_on_mac``,
  ``confirm_action`` and every other risky action. :class:`Authoriser`: the utterance a
  transcript binds to must match the profile named by ``MAC_VOICE_AUTHORIZED`` (its
  call-domain profile ``enroll/call/<name>.npy`` when there is one, else the wake profile)
  with cosine ≥ 0.35. No such variable, no profile, a model failure, no / stale binding,
  an utterance under 0.4 s → refused. With several candidate utterances every one must
  pass. ``[system notice]`` messages never pass.
* Low-risk, no voice check — the reversible Mac fast-path tools
  (:mod:`~my_stt_tts.mac_control`: open a site or app, volume, brightness, media) mint
  with ``verify_voice=False``. Speaker scores on short utterances (a one-word "Play.") are
  unreliable and refused Albert himself, and a call only starts after a verified wake
  word. Everything else still holds: ``MAC_VOICE_AUTHORIZED`` must be set, injected
  notices never mint, at most three capabilities per transcript, invalidation, arguments
  derivable from the transcript. Residual risk: another voice in the room can change the
  volume.
* :class:`CapabilityStore` — an authorised transcript mints up to
  :data:`MAX_ACTIONS_PER_TRANSCRIPT` capabilities ("open YouTube, then louder"); its voice
  check runs once and is cached. External content reaching the agent — a session read, a
  briefing, an operator result, an injected notice, any other tool result — invalidates
  the unused ones (and the transcripts behind them); the results of the bridge's own Mac
  fast-path tools and of ``acknowledge_problem`` do not (:meth:`BridgeController.note_tool_result`).
  A capability allows one action whose arguments come from the transcript
  (:func:`~my_stt_tts.bridge_text.derivable`).
* :class:`PendingActions` — a risky action becomes THE pending proposal (a newer one
  replaces it), valid 60 s, single use, confirmed only by a NEW transcript that passes the
  full voice check and says yes ("yes, do it" — a bare "yes" is too short to verify); a
  new transcript saying no / cancel / stop drops it.

Problems (:mod:`~my_stt_tts.bridge_problems`, re-exported here): a failure the agent
already heard in the running call is reported with ``spoken=True``
(:meth:`BridgeController.report_problem`), so the next "voice on" never briefs it again.

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

from .bridge_problems import (
    LOG_DETAIL_CAP,
    BriefingProvider,
    DeliveryTracker,
    MemoryBriefingProvider,
    MemoryDeliveryTracker,
    MemoryProblemSink,
    Problem,
    ProblemSink,
    report_problem,
)
from .bridge_text import derivable, normalise, redact
from .turns import MIN_UTTERANCE_S, Ambiguous, Binding, SpeechDetector, TurnSource, Utterance

log = logging.getLogger("my_stt_tts.bridge")

FLAGS = ("MAC_VOICE_CLAUDE_BRIDGE", "MAC_VOICE_MAC_CONTROL", "MAC_VOICE_OPERATOR")
AUTH_ENV = "MAC_VOICE_AUTHORIZED"
AUTH_THRESHOLD = 0.35
CONFIRM_TTL_S = 60.0
#: How many actions one authorised sentence may start ("open YouTube and make it louder").
MAX_ACTIONS_PER_TRANSCRIPT = 3
#: What Albert is asked to say to confirm (two words: long enough for the voice check).
CONFIRM_PHRASE = "yes, do it"
NOTICE_PREFIX = "[system notice]"
NOTICE_CAP = 300


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
    """``🔒 confirm needed: send a message``."""
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


class CapabilityStore:  # pylint: disable=too-many-instance-attributes  # per-call state
    """Mint, check, use and invalidate capabilities (see module docstring)."""

    def __init__(self, authoriser: Authoriser, clock: Callable[[], float] = time.monotonic):
        self.authoriser = authoriser
        self.clock = clock
        self._live: dict[str, Capability] = {}
        self._minted: dict[int, int] = {}  # transcript seq → mint attempts
        self._spent: set[int] = set()  # transcripts that confirmed a proposal
        self._voice: dict[int, Refusal | None] = {}  # transcript seq → cached voice check
        self._burned_through = -1  # transcripts up to this seq can no longer mint
        self._lock = threading.Lock()
        self._verify_lock = threading.Lock()  # one voice check per transcript, not one per mint

    def reset(self) -> None:
        """A call starts or ends: forget every capability and transcript (seqs restart)."""
        with self._lock:
            self._live.clear()
            self._minted.clear()
            self._spent.clear()
            self._voice.clear()
            self._burned_through = -1

    def mint(self, transcript: Transcript, *, verify_voice: bool = True) -> Capability | Refusal:
        """Up to :data:`MAX_ACTIONS_PER_TRANSCRIPT` capabilities per user transcript.

        ``verify_voice=False`` (low-risk tier) skips only the speaker check; an
        unconfigured ``MAC_VOICE_AUTHORIZED`` still refuses. The speaker check of a
        transcript runs once; later mints of the same transcript reuse its verdict.
        """
        with self._lock:
            refusal = self._mint_precheck(transcript)
            if refusal is None and not verify_voice and self.authoriser.authorized is None:
                refusal = _refuse("not_configured")
            if refusal is None:  # every attempt counts, pass or fail
                self._minted[transcript.seq] = self._minted.get(transcript.seq, 0) + 1
        if refusal is not None:
            log_refused(refusal.reason)
            return refusal
        if verify_voice:
            refusal = self._verify(transcript)
            if refusal is not None:
                return refusal
        cap = Capability(secrets.token_hex(8), transcript.seq, transcript.text, self.clock())
        with self._lock:
            if transcript.seq <= self._burned_through:  # a read arrived while verifying
                return _refuse("invalidated", "context changed — say it again")
            self._live[cap.id] = cap
        if not verify_voice:
            log.info("🔓 low-risk action, no voice check")
        return cap

    def _verify(self, transcript: Transcript) -> Refusal | None:
        """The transcript's voice check, run once and cached by its seq."""
        with self._verify_lock:
            with self._lock:
                if transcript.seq in self._voice:
                    refusal = self._voice[transcript.seq]
                    if refusal is not None:
                        log_refused(refusal.reason)
                    return refusal
            refusal = self.authoriser.verify(transcript)
            with self._lock:
                self._voice[transcript.seq] = refusal
            return refusal

    def _mint_precheck(self, transcript: Transcript) -> Refusal | None:
        if transcript.injected:
            return _refuse("injected", "injected message")
        if transcript.seq in self._spent:
            return _refuse("already_used", "already used — say it again")
        if transcript.seq <= self._burned_through:
            return _refuse("invalidated", "context changed — say it again")
        if self._minted.get(transcript.seq, 0) >= MAX_ACTIONS_PER_TRANSCRIPT:
            return _refuse("too_many", "too many actions in one sentence — say the rest again")
        return None

    def consume_transcript(self, seq: int) -> None:
        """Mark a transcript as spent (it confirmed a proposal) so it mints nothing."""
        with self._lock:
            self._spent.add(seq)

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


# -- proposals + spoken confirmation -------------------------------------------------------
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
    """A risky action waiting for Albert's "yes, do it"."""

    id: str
    kind: str
    args: Mapping[str, Any] = field(repr=False)
    sha256: str
    expires: float
    created_at: float
    source_seq: int | None = None

    def matches(self, kind: str, args: Mapping[str, Any]) -> bool:
        """True only for exactly the proposed action (no replacement arguments)."""
        return secrets.compare_digest(args_digest(kind, args), self.sha256)


#: A confirmation (normalised text, en/de/fr): "yes, do it", "ja, mach das", "oui, vas-y".
AFFIRMATIVE = re.compile(
    r"\b(?:yes|yeah|yep|ok|okay|confirm\w*|go ahead|do it|ja|jawohl|bestatig\w*"
    r"|bestaetig\w*|mach (?:es|das)|oui|vas y)\b"
)
#: A refusal; it wins over an affirmative ("no, don't do it"), so it is checked first.
NEGATION = re.compile(r"\b(?:no|nein|non|cancel|abbrechen|annuler|stop|don t|dont|do not|nicht)\b")


def affirmative(text: str) -> bool:
    """True when ``text`` says yes (and no negation)."""
    norm = normalise(text)
    return bool(AFFIRMATIVE.search(norm)) and not NEGATION.search(norm)


def declined(text: str) -> bool:
    """True when ``text`` says no / cancel / stop."""
    return bool(NEGATION.search(normalise(text)))


class PendingActions:
    """At most ONE proposal, confirmed by a NEW verified transcript that says yes."""

    def __init__(
        self,
        authoriser: Authoriser,
        clock: Callable[[], float] = time.monotonic,
        ttl: float = CONFIRM_TTL_S,
    ) -> None:
        self.authoriser = authoriser
        self.clock = clock
        self.ttl = ttl
        self._current: PendingAction | None = None
        self._declined_by: int | None = None  # seq of the transcript that said no
        self._lock = threading.Lock()

    def propose(
        self, kind: str, args: Mapping[str, Any], *, source_seq: int | None = None
    ) -> PendingAction:
        """Store THE proposal (replacing any other); ``source_seq`` asked, it cannot confirm."""
        now = self.clock()
        pending = PendingAction(
            id=secrets.token_hex(8),
            kind=kind,
            args=dict(args),  # executed as given; the digest is over the normalised form
            sha256=args_digest(kind, args),
            expires=now + self.ttl,
            created_at=now,
            source_seq=source_seq,
        )
        with self._lock:
            old = self._current
            if old is not None and old.expires <= now:
                old = None
            self._current = pending
            self._declined_by = None
        if old is not None:
            log.info("🔁 pending action replaced: %s → %s", _detail(old.kind), _detail(kind))
        log_confirm(kind)
        return pending

    def decline(self, transcript: Transcript) -> PendingAction | None:
        """Drop the proposal when a new transcript says no; returns what was dropped."""
        if transcript.injected or not declined(transcript.text):
            return None
        with self._lock:
            pending = self._current
            if pending is None or pending.expires <= self.clock():
                return None
            if not _is_newer(transcript, pending):
                return None
            self._current = None
            self._declined_by = transcript.seq
        log.info("🚫 pending action cancelled: %s", _detail(pending.kind))
        return pending

    def confirm(self, transcript: Transcript) -> PendingAction | Refusal:
        """The pending proposal, once a new transcript says yes in Albert's voice."""
        if self.decline(transcript) is not None or self._declined_by == transcript.seq:
            return _refused(_refuse("cancelled", "cancelled — nothing was done"))
        with self._lock:
            result = self._precheck(transcript)
        if isinstance(result, Refusal):
            return _refused(result)
        voice = self.authoriser.verify(transcript)  # logs its own refusal
        if voice is not None and voice.code == "too_short":
            return _refuse("too_short", f"too short to recognise your voice — say {CONFIRM_PHRASE}")
        if voice is not None:
            return voice
        with self._lock:
            replaced = self._current is not result  # replaced or dropped while verifying
            if not replaced:
                self._current = None
        if replaced:
            return _refused(_refuse("replaced", "the proposal changed — listen again"))
        return result

    def _precheck(self, transcript: Transcript) -> PendingAction | Refusal:
        pending = self._current
        if pending is None:
            return _refuse("nothing_pending", "nothing to confirm")
        if pending.expires <= self.clock():
            self._current = None
            return _refuse("expired", "too late — ask again")
        if transcript.injected:
            return _refuse("injected", "injected message")
        if not _is_newer(transcript, pending):
            return _refuse("not_new", f"say {CONFIRM_PHRASE} in a new sentence")
        if not affirmative(transcript.text):
            return _refuse("no_yes", f"say {CONFIRM_PHRASE}")
        return pending

    def clear(self) -> None:
        with self._lock:
            self._current = None
            self._declined_by = None

    def live(self) -> list[PendingAction]:
        with self._lock:
            if self._current is not None and self._current.expires <= self.clock():
                self._current = None
            return [] if self._current is None else [self._current]


def _refused(refusal: Refusal) -> Refusal:
    log_refused(refusal.reason)
    return refusal


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

    With ``caps`` None (a confirmed :class:`PendingAction`) only cancellation and the
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
        #: Tools whose result is only the bridge's own outcome (see :meth:`note_tool_result`).
        self.own_outcome_tools: set[str] = set()
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
    def register_tool(
        self, name: str, fn: Callable[..., Any], *, own_outcome: bool = False
    ) -> None:
        """Client tools later phases add (name → callable).

        ``own_outcome``: the tool's result carries no external content (its result does
        not invalidate the sentence's remaining capabilities).
        """
        self.tools[name] = fn
        if own_outcome:
            self.own_outcome_tools.add(name)
        else:
            self.own_outcome_tools.discard(name)

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
        # Transcript seqs restart with every call: a call must not inherit the last one's
        # spent or burned seqs (that refused the first sentences of the next call).
        dropped = self.caps.invalidate_all(reason)
        self.caps.reset()
        self.pending.clear()
        if dropped:
            log.debug("call reset (%s)", reason)
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
        self.pending.decline(transcript)  # "no" / "cancel" drops a waiting proposal
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

    def note_tool_result(self, name: str) -> None:
        """A client tool answered: external content invalidates, the bridge's own outcome not.

        Results of the tools in :attr:`own_outcome_tools` (Mac fast path,
        ``acknowledge_problem``) only say what the bridge itself did, so the rest of the
        sentence ("open YouTube and make it louder") may still act. Every other result
        (session reads, briefings, operator results, confirmations) invalidates.
        """
        if name in self.own_outcome_tools:
            return
        self.note_context("tool result")

    # -- problems ---------------------------------------------------------------------------
    @property
    def in_call(self) -> bool:
        """True while a call runs (between :meth:`begin_call` and :meth:`end_call`)."""
        return self.turns is not None

    def report_problem(
        self, problem: Problem, *, spoken: bool = False, dedupe_key: str | None = None
    ) -> None:
        """Report ``problem``; ``spoken`` (it reached the agent) counts only inside a call."""
        report_problem(
            self.problems, problem, spoken=spoken and self.in_call, dedupe_key=dedupe_key
        )

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

    def mint(self, *, verify_voice: bool = True) -> Capability | Refusal:
        """A capability from the latest user transcript (up to three per transcript).

        ``verify_voice=False`` is the low-risk tier (Mac fast path): no speaker check.
        """
        latest = self._latest
        if latest is None:
            log_refused("nothing said")
            return _refuse("no_transcript", "nothing said yet")
        return self.caps.mint(latest, verify_voice=verify_voice)

    def propose(self, kind: str, args: Mapping[str, Any]) -> PendingAction:
        """THE risky action waiting for "yes, do it"; the asking transcript cannot confirm it."""
        latest = self._latest
        return self.pending.propose(kind, args, source_seq=latest.seq if latest else None)

    def confirm(self) -> PendingAction | Refusal:
        """Confirm with the latest transcript, which is then spent (mints nothing)."""
        latest = self._latest
        if latest is None:
            log_refused("nothing said")
            return _refuse("no_transcript", "nothing said yet")
        self.caps.consume_transcript(latest.seq)
        return self.pending.confirm(latest)

    def confirm_action(self, code: str | None = None) -> str:
        """The ``confirm_action`` client tool: run the pending proposal once Albert said yes.

        ``code`` is ignored (agents configured for the old two-digit codes still pass one).
        """
        del code
        result = self.confirm()
        if isinstance(result, Refusal):
            if result.code == "cancelled":
                return "cancelled: nothing was done"
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
