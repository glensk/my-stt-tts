"""Follow-up window: the agent hears the room only right after it spoke (tp#1234).

An ElevenLabs call streams the mic to the agent for the whole conversation, so the agent
also answered Albert when he talked to his son. :class:`FollowUpGate` decides per mic frame
whether the agent hears it:

* **open** — while the agent speaks, and for ``window_s`` seconds (``MAC_VOICE_FOLLOWUP_S``,
  default 8) after it stopped or after the mic opened. A sentence Albert started inside the
  window keeps it open until he pauses (:data:`SPEECH_GAP_S`), at most
  :data:`MAX_SENTENCE_S`.
* **closed** — the agent gets silence. The same frames go to the daemon's wake detector
  ("hey jarvis" / "voice on"); a fire whose voice matches an enrolled profile opens the
  window again and plays a short tick (``MAC_VOICE_FOLLOWUP_SOUND``). The agent speaking
  again (a ``[system notice]``, a late answer) opens it too.
* **paused** — "pause" / "wait" (:func:`is_pause`, matched on the transcript): closed, and
  only the wake word opens it again; agent audio does not.

Gated frames are silence for the agent and for the bridge's turn source, and they do not
count as the user's activity, so the daemon's idle timeout still hangs up a call nobody
talks to. The agent's turn handling cannot do this on its own: it would have to hear (and
pay for) everything to decide what to ignore, and its decision comes after it already
started answering. The trade-off: after the window Albert has to say the wake word and wait
for the tick before a new request; the start of a sentence said in the same breath as the
wake word may be cut.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.follow_up")

WINDOW_ENV = "MAC_VOICE_FOLLOWUP_S"
DEFAULT_WINDOW_S = 8.0
SOUND_ENV = "MAC_VOICE_FOLLOWUP_SOUND"
DEFAULT_SOUND = "/System/Library/Sounds/Tink.aiff"
VOICE_LEVEL = 600.0 / 32768.0  # float RMS (~-35 dBFS) above which someone counts as speaking
SPEECH_GAP_S = 0.6  # a pause this long ends the sentence that holds the window open
MAX_SENTENCE_S = 30.0  # a sentence started inside the window holds it at most this long
WAKE_FRAME = 1280  # openWakeWord / KWS frame: 80 ms at 16 kHz
WAKE_CLIP_S = 2.0  # audio around a fire that the voice check judges
SAMPLE_RATE = 16000

OPEN, CLOSED, PAUSED = "open", "closed", "paused"

# "pause" / "wait" on its own (with a polite word around it), English and German.
_FILLER = frozenset({"please", "bitte", "mal", "kurz", "ok", "okay", "hey", "jarvis", "now"})
_PAUSE = frozenset(
    {
        "pause",
        "wait",
        "warte",
        "wart",
        "warten",
        "moment",
        "one moment",
        "einen moment",
        "a moment",
        "wait a moment",
        "wait a second",
        "wait a minute",
        "hold on",
        "pause listening",
    }
)


def is_pause(text: str) -> bool:
    """True when a user transcript is only the pause command ("Pause.", "Warte mal.")."""
    words = re.sub(r"[^\w\s]", " ", str(text).casefold()).split()
    while words and words[0] in _FILLER:
        words.pop(0)
    while words and words[-1] in _FILLER:
        words.pop()
    return " ".join(words) in _PAUSE


def window_from_env(env: Mapping[str, str] | None = None) -> float:
    """``MAC_VOICE_FOLLOWUP_S`` in seconds; 0 switches the window off, junk → the default."""
    env = os.environ if env is None else env
    raw = (env.get(WINDOW_ENV) or "").strip()
    if not raw:
        return DEFAULT_WINDOW_S
    try:
        value = float(raw)
    except ValueError:
        log.warning("⚠️  %s=%r is not a number; using %.0f s", WINDOW_ENV, raw, DEFAULT_WINDOW_S)
        return DEFAULT_WINDOW_S
    return max(0.0, value)


def reopen_sound(env: Mapping[str, str] | None = None) -> str | None:
    """The tick played when the wake word opens the window; None when switched off."""
    env = os.environ if env is None else env
    value = env.get(SOUND_ENV)
    if value is None:
        return DEFAULT_SOUND
    value = value.strip()
    return None if value.casefold() in {"", "0", "off"} else value


def play_sound(path: str | None, popen: Callable[..., Any] = subprocess.Popen) -> None:
    """Play ``path`` with ``afplay`` without waiting; failures only log."""
    if not path:
        return
    try:
        popen(
            ["/usr/bin/afplay", path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # pylint: disable=broad-exception-caught  # a tick never matters
        log.debug("follow-up sound failed", exc_info=True)


class FollowUpGate:  # pylint: disable=too-many-instance-attributes  # one call's gate state
    """Per-frame decision: does the agent hear this mic frame (see module docstring)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        window_s: float,
        wake: Any,
        *,
        gain: float = 1.0,
        voice: Any = None,
        clock: Callable[[], float] = time.monotonic,
        on_reopen: Callable[[], None] | None = None,
    ) -> None:
        self.window_s = window_s
        self.wake = wake  # the daemon's wake detector: detect(frame) -> bool, reset()
        self.gain = gain
        self.voice = voice  # voice_gate.VoiceGate: only enrolled voices reopen
        self.clock = clock
        self.on_reopen = on_reopen
        self._state = OPEN
        self._lock = threading.Lock()
        self._last_event: float | None = None  # agent audio, mic open or a reopen
        self._speech_start: float | None = None
        self._speech_last: float | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._ring: deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._broken = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    # -- the per-frame decision ---------------------------------------------------------
    def admit(self, frame: Any, agent_speaking: bool) -> bool:
        """True when the agent may hear ``frame`` (16 kHz float32 mono)."""
        if self._broken:
            return True
        audio = np.asarray(frame, dtype=np.float32).ravel()
        now = self.clock()
        with self._lock:
            if self._last_event is None:
                self._last_event = now  # the mic just opened: the first window starts
            if agent_speaking and self._state != PAUSED:
                self._last_event = now
                if self._state == CLOSED:
                    self._open_locked(now, "the agent speaks")
            if self._state == OPEN:
                self._track_speech(audio, now)
                if not self._expired(now):
                    return True
                self._close_locked(CLOSED)
            fired = self._listen(audio)
            if fired:
                self._open_locked(now, "wake word")
        if fired and self.on_reopen is not None:
            self.on_reopen()
        return False

    def pause(self) -> None:
        """ "pause" / "wait": closed until the wake word; agent audio does not reopen it."""
        with self._lock:
            if self._state != PAUSED:
                self._close_locked(PAUSED)

    # -- internals ----------------------------------------------------------------------
    def _track_speech(self, audio: np.ndarray, now: float) -> None:
        if not audio.size or float(np.sqrt(np.mean(audio**2))) <= VOICE_LEVEL:
            return
        if self._speech_last is None or now - self._speech_last > SPEECH_GAP_S:
            self._speech_start = now
        self._speech_last = now

    def _expired(self, now: float) -> bool:
        assert self._last_event is not None
        deadline = self._last_event + self.window_s
        if now < deadline:
            return False
        if self._speech_last is None or self._speech_start is None:
            return True
        talking = now - self._speech_last < SPEECH_GAP_S
        started_in_window = self._speech_start <= deadline
        return not (talking and started_in_window and now - self._speech_start < MAX_SENTENCE_S)

    def _close_locked(self, state: str) -> None:
        self._state = state
        self._speech_start = self._speech_last = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._ring.clear()
        self._ring_samples = 0
        with contextlib.suppress(Exception):
            self.wake.reset()
        if state == PAUSED:
            log.info("⏸️  paused — say the wake word to go on")
        else:
            log.info("🔇 follow-up window over — say the wake word for a new request")

    def _open_locked(self, now: float, why: str) -> None:
        self._state = OPEN
        self._last_event = now
        self._speech_start = self._speech_last = None
        log.info("👂 listening again (%s)", why)

    def _remember(self, audio: np.ndarray) -> None:
        self._ring.append(audio)
        self._ring_samples += audio.size
        limit = int(WAKE_CLIP_S * SAMPLE_RATE)
        while self._ring and self._ring_samples - self._ring[0].size >= limit:
            self._ring_samples -= self._ring.popleft().size

    def _listen(self, audio: np.ndarray) -> bool:
        """Feed the wake detector; True on a fire the voice check accepts."""
        if not audio.size:
            return False
        self._remember(audio)
        self._pending = np.concatenate([self._pending, audio])
        fired = False
        try:
            while self._pending.size >= WAKE_FRAME:
                chunk, self._pending = self._pending[:WAKE_FRAME], self._pending[WAKE_FRAME:]
                if self.gain != 1.0:
                    chunk = np.clip(chunk * self.gain, -1.0, 1.0)
                if self.wake.detect(chunk):
                    fired = True
                    break
        except Exception:  # pylint: disable=broad-exception-caught  # never lock Albert out
            log.warning("⚠️  in-call wake word failed; the mic stays open", exc_info=True)
            self._broken = True
            self._state = OPEN
            return False
        if not fired:
            return False
        self._pending = np.zeros(0, dtype=np.float32)
        return self._voice_ok()

    def _voice_ok(self) -> bool:
        """Does the voice behind the fire match an enrolled profile (everyone, with none)?"""
        if self.voice is None or not getattr(self.voice, "active", False):
            return True
        clip = np.concatenate(list(self._ring)) if self._ring else np.zeros(0, np.float32)
        try:
            ok, name, sim = self._judge(clip)
        except Exception:  # pylint: disable=broad-exception-caught
            log.warning("⚠️  voice check failed; wake word ignored", exc_info=True)
            ok, name, sim = False, "", 0.0
        if ok:
            log.info("🗣️  wake word in the call: %s (voice match %.2f)", name, sim)
        else:
            log.info("🚫 wake word in the call ignored: voice not enrolled (%s %.2f)", name, sim)
            with contextlib.suppress(Exception):
                self.wake.reset()
        return bool(ok)

    def _judge(self, clip: np.ndarray) -> tuple[bool, str, float]:
        """The best enrolled match, scored on the call-domain profile where there is one.

        In-call audio has gone through VoiceProcessingIO; a raw-mic wake profile scores it
        too low (``voice_gate``), so ``score_against`` (call profile first) decides per
        person, and :meth:`VoiceGate.check` (wake profiles) is the fallback.
        """
        voice = self.voice
        names = sorted(set(getattr(voice, "call_profiles", {}) or {}) | set(voice.profiles or {}))
        if not callable(getattr(voice, "score_against", None)) or not names:
            ok, name, sim = voice.check(clip)
            return bool(ok), str(name), float(sim)
        scores = [(n, voice.score_against(clip, n)) for n in names]
        judged = [(n, float(sc)) for n, sc in scores if sc is not None]
        if not judged:
            return False, "", 0.0
        name, sim = max(judged, key=lambda pair: pair[1])
        return sim >= float(voice.threshold), name, sim


def gate_factory(
    kit: Any, env: Mapping[str, str] | None = None
) -> Callable[[], FollowUpGate] | None:
    """A fresh :class:`FollowUpGate` per call from the daemon's wake kit; None when off.

    Off when ``MAC_VOICE_FOLLOWUP_S`` is 0 or there is no wake detector (nothing could open
    the window again).
    """
    window = window_from_env(env)
    detector = getattr(kit, "detector", None)
    if window <= 0:
        log.info("👂 follow-up window off (%s=0): the agent hears the whole call", WINDOW_ENV)
        return None
    if detector is None:
        log.info("👂 follow-up window off: no wake word to reopen it")
        return None
    sound = reopen_sound(env)
    log.info("👂 follow-up window %.0f s, then the wake word opens it again", window)

    def make() -> FollowUpGate:
        return FollowUpGate(
            window,
            detector,
            gain=float(getattr(kit, "gain", 1.0) or 1.0),
            voice=getattr(kit, "gate", None),
            on_reopen=lambda: play_sound(sound),
        )

    return make


__all__ = [
    "DEFAULT_WINDOW_S",
    "FollowUpGate",
    "gate_factory",
    "is_pause",
    "reopen_sound",
    "window_from_env",
]
