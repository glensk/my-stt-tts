"""TurnSource: VAD segmentation, the 20 s ring buffer and transcript binding (fake VAD + clock)."""

from __future__ import annotations

import numpy as np

from my_stt_tts.turns import BIND_WINDOW_S, Ambiguous, TurnSource, Utterance

FRAME = 1600  # 0.1 s at 16 kHz


class FakeVad:
    """Speech = any sample louder than 0.05."""

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def feed(src: TurnSource, clock: Clock, seconds: float, amp: float) -> None:
    """Feed ``seconds`` of constant-amplitude audio in 0.1 s frames, advancing the clock."""
    for _ in range(round(seconds * 10)):
        clock.now += 0.1
        src.feed(np.full(FRAME, amp, dtype=np.float32))


def make() -> tuple[TurnSource, Clock]:
    clock = Clock()
    return TurnSource(FakeVad(), clock=clock), clock


def test_speech_then_silence_makes_one_trimmed_utterance() -> None:
    src, clock = make()
    feed(src, clock, 0.3, 0.0)
    feed(src, clock, 1.0, 0.5)
    feed(src, clock, 0.6, 0.0)
    [utt] = src.utterances()
    assert utt.seq == 1 and utt.finished
    assert abs(utt.start - 100.3) < 1e-6
    assert abs(utt.end - 101.3) < 1e-6
    assert utt.pcm.size == 10 * FRAME  # trailing silence not included


def test_short_pause_stays_inside_the_utterance() -> None:
    src, clock = make()
    feed(src, clock, 0.5, 0.5)
    feed(src, clock, 0.3, 0.0)  # below the 0.5 s endpointer
    feed(src, clock, 0.5, 0.5)
    feed(src, clock, 0.6, 0.0)
    [utt] = src.utterances()
    assert abs(utt.duration - 1.3) < 1e-6
    assert utt.pcm.size == 13 * FRAME


def test_no_utterance_until_the_endpointer_closes_it() -> None:
    src, clock = make()
    feed(src, clock, 1.0, 0.5)
    feed(src, clock, 0.3, 0.0)
    assert not src.utterances()


def test_empty_frames_are_ignored() -> None:
    src, _ = make()
    assert src.feed(np.zeros(0, dtype=np.float32)) is None


def test_ring_buffer_keeps_twenty_seconds() -> None:
    src, clock = make()
    feed(src, clock, 0.5, 0.5)
    feed(src, clock, 0.6, 0.0)
    feed(src, clock, 19.0, 0.0)
    assert len(src.utterances()) == 1
    feed(src, clock, 1.5, 0.0)
    assert not src.utterances()


def test_bind_picks_the_utterance_that_just_ended() -> None:
    src, clock = make()
    feed(src, clock, 1.0, 0.5)
    feed(src, clock, 0.6, 0.0)
    bound = src.bind(1, "open youtube", clock.now + 1.5)
    assert isinstance(bound, Utterance) and bound.seq == 1


def test_bind_refuses_a_stale_utterance() -> None:
    src, clock = make()
    feed(src, clock, 1.0, 0.5)
    feed(src, clock, 0.6, 0.0)
    utt = src.utterances()[0]
    assert src.bind(1, "x", utt.end + BIND_WINDOW_S + 0.1) is None
    assert src.bind(1, "x", utt.end + BIND_WINDOW_S - 0.1) == utt


def test_two_utterances_in_the_window_are_ambiguous() -> None:
    src, clock = make()
    feed(src, clock, 0.6, 0.5)
    feed(src, clock, 0.6, 0.0)
    feed(src, clock, 0.6, 0.5)
    feed(src, clock, 0.6, 0.0)
    bound = src.bind(1, "x", clock.now + 0.5)
    assert isinstance(bound, Ambiguous)
    assert [u.seq for u in bound.candidates] == [1, 2]


def test_utterance_running_at_arrival_makes_it_ambiguous() -> None:
    src, clock = make()
    feed(src, clock, 0.6, 0.5)
    feed(src, clock, 0.6, 0.0)
    feed(src, clock, 0.5, 0.5)  # still talking when the transcript arrives
    bound = src.bind(1, "x", clock.now)
    assert isinstance(bound, Ambiguous)
    assert [u.finished for u in bound.candidates] == [True, False]


def test_transcript_before_the_endpoint_binds_the_running_utterance() -> None:
    src, clock = make()
    feed(src, clock, 0.5, 0.5)  # the ASR finalised before the endpointer closed it
    bound = src.bind(1, "x", clock.now)
    assert isinstance(bound, Utterance)
    assert not bound.finished
    assert bound.duration >= 0.4


def test_no_speech_binds_nothing() -> None:
    src, clock = make()
    feed(src, clock, 0.5, 0.0)
    assert src.bind(1, "x", clock.now) is None
