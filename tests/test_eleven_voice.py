"""Tests for the ElevenLabs Mac audio bridge (no real audio device, no network)."""

from __future__ import annotations

import time
from typing import Any, ClassVar

import pytest

from my_stt_tts import eleven_voice

pytest.importorskip("elevenlabs")


class _FakeStream:
    """Stands in for sounddevice Raw*Stream: records the callback, never opens a device."""

    created: ClassVar[list[_FakeStream]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.callback = kwargs["callback"]
        self.started = False
        _FakeStream.created.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        pass


class _FakeSd:
    RawOutputStream = _FakeStream
    RawInputStream = _FakeStream


@pytest.fixture(autouse=True)
def _fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeStream.created.clear()
    monkeypatch.setattr(eleven_voice, "_sd", lambda: _FakeSd)


def _play(n: int) -> bytes:
    """Pull ``n`` bytes from the playback stream (always the first stream created)."""
    out = bytearray(n)
    _FakeStream.created[0].callback(out, n // 2, None, None)
    return bytes(out)


def _mic() -> Any:
    """The mic stream's callback (created by ``start``, after the playback stream)."""
    return _FakeStream.created[1].callback


def test_output_is_played_then_padded_with_silence() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.output(b"\x01\x02\x03\x04")
    assert _play(8) == b"\x01\x02\x03\x04" + b"\x00" * 4
    assert _play(4) == b"\x00" * 4


def test_interrupt_drops_buffered_audio() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.output(b"\x05" * 100)
    iface.interrupt()
    assert _play(10) == b"\x00" * 10


def test_speakers_mode_mutes_mic_while_agent_speaks() -> None:
    sent: list[bytes] = []
    iface = eleven_voice.make_audio_interface("speakers", None, None)
    iface.start(sent.append)
    mic = _mic()
    iface.output(b"\x07" * 64)
    mic(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x00" * 8  # agent audio queued -> mic gated
    iface.interrupt()
    iface.playback.last_audio = time.monotonic() - 1.0
    mic(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x11" * 8  # agent silent -> mic passes through


def test_headphones_mode_never_gates_the_mic() -> None:
    sent: list[bytes] = []
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.start(sent.append)
    iface.output(b"\x07" * 64)
    _mic()(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x11" * 8


def test_missing_credentials_exit_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eleven_voice, "_load_env", lambda: None)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    assert eleven_voice.main([]) == 2


class _FakeDuplex:
    """Stands in for aec.VoiceProcessingDuplex (no CoreAudio)."""

    starts_ok = True
    last: _FakeDuplex | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.played: list[bytes] = []
        self.flushed = 0
        _FakeDuplex.last = self

    def start(self) -> bool:
        return self.starts_ok

    def mic_frames(self) -> Any:
        return iter(())

    def play(self, pcm: bytes) -> None:
        self.played.append(pcm)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        pass


def test_aec_mode_routes_playback_and_interrupt_through_voice_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("my_stt_tts.aec.VoiceProcessingDuplex", _FakeDuplex)
    iface = eleven_voice.make_audio_interface("aec", None, None)
    iface.start(lambda _pcm: None)
    iface.output(b"\x01\x02")
    iface.interrupt()
    vp = _FakeDuplex.last
    assert vp is not None
    assert vp.played == [b"\x01\x02"]
    assert vp.flushed == 1
    assert len(_FakeStream.created) == 1  # only the (unstarted) sounddevice playback object


def test_aec_mode_falls_back_to_gated_speakers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeDuplex, "starts_ok", False)
    monkeypatch.setattr("my_stt_tts.aec.VoiceProcessingDuplex", _FakeDuplex)
    sent: list[bytes] = []
    iface = eleven_voice.make_audio_interface("aec", None, None)
    iface.start(sent.append)
    assert iface.mode == "speakers"
    iface.output(b"\x07" * 64)
    _mic()(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x00" * 8
