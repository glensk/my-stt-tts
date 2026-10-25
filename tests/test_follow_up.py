"""The follow-up window (tp#1234): fake clock, fake wake detector, synthetic frames only."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from my_stt_tts import follow_up
from my_stt_tts.follow_up import CLOSED, OPEN, PAUSED, FollowUpGate, is_pause

FRAME = 4000  # 250 ms, what the VoiceProcessingIO capture delivers
QUIET = np.zeros(FRAME, dtype=np.float32)
LOUD = np.full(FRAME, 0.1, dtype=np.float32)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Wake:
    """Fires on the n-th 80 ms frame it is fed (None: never)."""

    def __init__(self, fire_at: int | None = None, fail: bool = False) -> None:
        self.fire_at = fire_at
        self.fail = fail
        self.frames: list[np.ndarray] = []
        self.resets = 0

    def detect(self, frame: np.ndarray) -> bool:
        if self.fail:
            raise RuntimeError("model gone")
        self.frames.append(frame)
        return self.fire_at is not None and len(self.frames) == self.fire_at

    def reset(self) -> None:
        self.resets += 1


class Voice:
    """A VoiceGate with only wake profiles: ``check`` decides."""

    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.active = True
        self.profiles: dict[str, Any] = {}
        self.clips: list[np.ndarray] = []

    def check(self, clip: Any) -> tuple[bool, str, float]:
        self.clips.append(np.asarray(clip))
        return self.ok, "albert", 0.6 if self.ok else 0.1


class CallVoice:
    """A VoiceGate with call-domain profiles: ``score_against`` decides per person."""

    active = True
    threshold = 0.35

    def __init__(self, scores: dict[str, float | None]) -> None:
        self.scores = scores
        self.profiles = {"albert": None}
        self.call_profiles = {name: None for name in scores}

    def check(self, clip: Any) -> tuple[bool, str, float]:
        raise AssertionError("the wake-profile check is only the fallback")

    def score_against(self, clip: Any, name: str) -> float | None:
        del clip
        return self.scores.get(name)


def _gate(wake: Wake | None = None, **kwargs: Any) -> tuple[FollowUpGate, Clock, list[str]]:
    clock = Clock()
    reopened: list[str] = []
    gate = FollowUpGate(
        8.0, wake or Wake(), clock=clock, on_reopen=lambda: reopened.append("tick"), **kwargs
    )
    return gate, clock, reopened


def _run(gate: FollowUpGate, clock: Clock, seconds: float, frame: Any = QUIET,
         speaking: bool = False) -> list[bool]:  # fmt: skip
    """Feed ``seconds`` of 250 ms frames; the verdict of each."""
    out = []
    for _ in range(round(seconds / 0.25)):
        out.append(gate.admit(frame, speaking))
        clock.now += 0.25
    return out


def test_the_window_closes_after_the_agent_has_been_quiet_for_window_s() -> None:
    gate, clock, _ = _gate()
    assert all(_run(gate, clock, 7.75))  # the mic just opened: first window
    assert _run(gate, clock, 0.5) == [True, False]
    assert gate.state == CLOSED
    assert not any(_run(gate, clock, 3.0, LOUD))  # the son talking: the agent hears silence


def test_agent_speech_keeps_the_window_open_and_restarts_it() -> None:
    gate, clock, _ = _gate()
    _run(gate, clock, 6.0)
    assert all(_run(gate, clock, 20.0, speaking=True))
    assert all(_run(gate, clock, 7.75))  # counted from the end of the agent's speech
    assert not _run(gate, clock, 0.5)[-1]


def test_agent_speech_reopens_a_closed_window_without_a_tick() -> None:
    gate, clock, reopened = _gate()
    _run(gate, clock, 9.0)
    assert gate.state == CLOSED
    assert gate.admit(QUIET, True) is True  # a [system notice] read aloud: he may answer
    assert gate.state == OPEN and not reopened


def test_a_sentence_started_in_the_window_is_not_cut_off() -> None:
    gate, clock, _ = _gate()
    _run(gate, clock, 7.0)
    assert all(_run(gate, clock, 4.0, LOUD))  # started at 7 s, runs past the 8 s deadline
    assert _run(gate, clock, 1.0) == [True, True, False, False]  # closes after a 0.6 s pause


def test_a_sentence_holds_the_window_at_most_max_sentence_s() -> None:
    gate, clock, _ = _gate()
    _run(gate, clock, 7.0)
    verdicts = _run(gate, clock, 40.0, LOUD)  # the TV never stops
    assert verdicts[: int(29 / 0.25)] == [True] * int(29 / 0.25)
    assert not verdicts[-1]


def test_speech_after_the_deadline_does_not_reopen() -> None:
    gate, clock, _ = _gate()
    _run(gate, clock, 8.5)
    assert not any(_run(gate, clock, 2.0, LOUD))


def test_the_wake_word_reopens_with_a_tick() -> None:
    wake = Wake()
    gate, clock, reopened = _gate(wake)
    _run(gate, clock, 8.5)
    assert gate.state == CLOSED and wake.resets == 1
    wake.fire_at = len(wake.frames) + 4
    assert gate.admit(LOUD, False) is False  # 4000 samples = 3 wake frames, no fire yet
    assert gate.state == CLOSED
    assert gate.admit(LOUD, False) is False  # the frame holding the wake word stays silent
    assert gate.state == OPEN and reopened == ["tick"]
    assert all(wake_frame.size == follow_up.WAKE_FRAME for wake_frame in wake.frames)
    assert all(_run(gate, clock, 7.75, LOUD))  # the request after the tick is heard


def test_the_wake_frames_get_the_gain() -> None:
    wake = Wake()
    gate, clock, _ = _gate(wake, gain=3.0)
    _run(gate, clock, 8.5)
    gate.admit(np.full(FRAME, 0.5, dtype=np.float32), False)
    assert wake.frames and float(wake.frames[-1].max()) == 1.0  # clipped at full scale


def test_a_wake_word_from_another_voice_is_ignored() -> None:
    voice = Voice(ok=False)
    wake = Wake()
    gate, clock, reopened = _gate(wake, voice=voice)
    _run(gate, clock, 8.5)
    wake.fire_at = len(wake.frames) + 1
    gate.admit(LOUD, False)
    assert gate.state == CLOSED and not reopened
    assert voice.clips and voice.clips[0].size <= follow_up.WAKE_CLIP_S * 16000 + FRAME


@pytest.mark.parametrize(
    ("scores", "opens"),
    [({"albert": 0.52, "susi": 0.2}, True), ({"albert": 0.3}, False), ({"albert": None}, False)],
)
def test_a_wake_word_in_the_call_is_judged_on_the_call_profile(
    scores: dict[str, float | None], opens: bool
) -> None:
    wake = Wake()
    gate, clock, reopened = _gate(wake, voice=CallVoice(scores))
    _run(gate, clock, 8.5)
    wake.fire_at = len(wake.frames) + 1
    gate.admit(LOUD, False)
    assert (gate.state == OPEN) is opens and bool(reopened) is opens


def test_pause_closes_until_the_wake_word_and_agent_audio_does_not_reopen() -> None:
    wake = Wake()
    gate, clock, reopened = _gate(wake)
    _run(gate, clock, 2.0)
    gate.pause()
    assert gate.state == PAUSED
    assert not any(_run(gate, clock, 20.0, speaking=True))  # its reply does not reopen
    assert gate.state == PAUSED
    wake.fire_at = len(wake.frames) + 1
    gate.admit(QUIET, True)
    assert gate.state == OPEN and reopened == ["tick"]  # the wake word does
    assert gate.admit(QUIET, True)


def test_a_broken_wake_detector_never_locks_albert_out() -> None:
    gate, clock, _ = _gate(Wake(fail=True))
    assert _run(gate, clock, 8.5)[-2:] == [False, True]  # closed, the detector failed: open
    assert all(_run(gate, clock, 20.0))  # failed open: the agent hears the call again


@pytest.mark.parametrize(
    ("text", "paused"),
    [
        ("Pause.", True),
        ("Wait.", True),
        ("Wait a second!", True),
        ("Warte mal.", True),
        ("Moment bitte.", True),
        ("Hold on", True),
        ("OK, pause please.", True),
        ("Wait, what did the second option say?", False),
        ("Pause the video", False),
        ("Ich warte auf das Essen", False),
        ("", False),
    ],
)
def test_pause_phrase(text: str, paused: bool) -> None:
    assert is_pause(text) is paused


def test_settings_from_the_environment() -> None:
    assert follow_up.window_from_env({}) == follow_up.DEFAULT_WINDOW_S == 8.0
    assert follow_up.window_from_env({follow_up.WINDOW_ENV: "12"}) == 12.0
    assert follow_up.window_from_env({follow_up.WINDOW_ENV: "0"}) == 0.0
    assert follow_up.window_from_env({follow_up.WINDOW_ENV: "-3"}) == 0.0
    assert follow_up.window_from_env({follow_up.WINDOW_ENV: "soon"}) == 8.0
    assert follow_up.reopen_sound({}) == follow_up.DEFAULT_SOUND
    assert follow_up.reopen_sound({follow_up.SOUND_ENV: "off"}) is None
    assert follow_up.reopen_sound({}) == "/System/Library/Sounds/Glass.aiff"  # = ready sound
    assert follow_up.reopen_volume({}) == follow_up.DEFAULT_VOLUME
    assert follow_up.reopen_volume({follow_up.VOLUME_ENV: " 3.5 "}) == 3.5
    for junk in ("loud", "0", "-1", "inf", "nan"):
        assert follow_up.reopen_volume({follow_up.VOLUME_ENV: junk}) == follow_up.DEFAULT_VOLUME


class Kit:
    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.gain = 2.0
        self.gate = None


def test_gate_factory_needs_a_window_and_a_wake_detector() -> None:
    assert follow_up.gate_factory(Kit(Wake()), {follow_up.WINDOW_ENV: "0"}) is None
    assert follow_up.gate_factory(Kit(None), {}) is None
    make = follow_up.gate_factory(Kit(Wake()), {follow_up.SOUND_ENV: "off"})
    assert make is not None
    first, second = make(), make()
    assert first is not second  # a fresh gate per call
    assert first.window_s == 8.0 and first.gain == 2.0


def test_play_sound_never_raises() -> None:
    calls: list[list[str]] = []

    def popen(argv: list[str], **_kw: Any) -> None:
        calls.append(argv)
        raise FileNotFoundError(argv[0])

    follow_up.play_sound("/x.aiff", popen=popen)
    follow_up.play_sound(None, popen=popen)
    assert calls == [["/usr/bin/afplay", "-v", "2", "/x.aiff"]]
    calls.clear()
    follow_up.play_sound("/x.aiff", popen=popen, volume=1.5)
    assert calls == [["/usr/bin/afplay", "-v", "1.5", "/x.aiff"]]
