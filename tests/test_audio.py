"""Tests for the pre-roll buffer, mic gate, and chime generation."""
# pylint: disable=missing-function-docstring

import time

import numpy as np

from my_stt_tts import chimes
from my_stt_tts.audio import MicGate, PreRollBuffer


def test_preroll_caps_to_max_length():
    buf = PreRollBuffer(sample_rate=10, seconds=1.0)  # maxlen = 10
    buf.push(np.ones(7, dtype=np.float32))
    buf.push(np.ones(7, dtype=np.float32))
    assert buf.get().shape[0] == 10


def test_preroll_clear():
    buf = PreRollBuffer(sample_rate=10, seconds=1.0)
    buf.push(np.ones(5, dtype=np.float32))
    buf.clear()
    assert buf.get().shape[0] == 0


def test_micgate_gate_then_release():
    gate = MicGate(tail_seconds=0.01)
    assert gate.open
    gate.gate()
    assert not gate.open
    gate.release()
    time.sleep(0.05)
    assert gate.open


def test_tone_shape_dtype_and_volume():
    samples = chimes.tone([440.0], duration=0.1, sample_rate=1000, volume=0.3)
    assert samples.dtype == np.float32
    assert abs(len(samples) - 100) <= 2
    assert np.max(np.abs(samples)) <= 0.31


def test_chimes_nonempty():
    assert len(chimes.chime_listening()) > 0
    assert len(chimes.chime_done()) > 0
    assert len(chimes.chime_error()) > 0
