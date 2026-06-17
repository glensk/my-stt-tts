"""Tests for the threaded spine, rate limiter, and VAD endpointer."""
# pylint: disable=missing-function-docstring

from my_stt_tts.spine import SESSION_END, Pipeline, Stage
from my_stt_tts.util import RateLimiter
from my_stt_tts.vad import SilenceEndpointer


def test_pipeline_transforms_in_order():
    pipe = Pipeline([Stage("double", lambda x: x * 2), Stage("inc", lambda x: x + 1)])
    pipe.start()
    pipe.feed(3)
    assert pipe.output.get(timeout=2) == 7
    pipe.shutdown()


def test_generator_stage_fans_out():
    pipe = Pipeline([Stage("split", lambda s: iter(s.split()), generator=True)])
    pipe.start()
    pipe.feed("a b c")
    assert [pipe.output.get(timeout=2) for _ in range(3)] == ["a", "b", "c"]
    pipe.shutdown()


def test_session_end_forwarded():
    pipe = Pipeline([Stage("identity", lambda x: x)])
    pipe.start()
    pipe.feed(SESSION_END)
    assert pipe.output.get(timeout=2) is SESSION_END
    pipe.shutdown()


def test_rate_limiter_sliding_window():
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.acquire() is True
    assert limiter.acquire() is True
    assert limiter.acquire() is False
    now[0] = 61.0
    assert limiter.acquire() is True


def test_endpointer_ends_after_silence():
    ep = SilenceEndpointer(silence_seconds=0.3, frame_seconds=0.1)
    assert ep.update(True) is False
    assert ep.update(False) is False
    assert ep.update(False) is False
    assert ep.update(False) is True


def test_endpointer_ignores_leading_silence():
    ep = SilenceEndpointer(silence_seconds=0.2, frame_seconds=0.1)
    assert ep.update(False) is False
    assert ep.update(False) is False
