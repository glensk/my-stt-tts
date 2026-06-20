"""Tests for G8 — cross-platform (brain off-Mac).

Covers: platform detection (+ override), the cross-platform WAV-player selection
(afplay on macOS, aplay/paplay on Linux), the non-MLX STT backend selection
(whisper.cpp / faster-whisper) and fallback, and the Linux WebRTC-APM AEC
selection + graceful fallback to NLMS. Everything is faked — no device, no
subprocess, no native module — so it runs identically on any host.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,import-outside-toplevel

import numpy as np

from my_stt_tts import platform as plat
from my_stt_tts.config import Config


class _Cfg:
    """A tiny config stand-in with just the platform/playback fields."""

    def __init__(self, platform="auto", playback_backend="auto", sample_rate=16000):
        self.platform = platform
        self.playback_backend = playback_backend
        self.sample_rate = sample_rate


# ---------------------------------------------------------------------------
# platform detection
# ---------------------------------------------------------------------------


def test_detect_platform_override_wins(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "darwin")
    assert plat.detect_platform(_Cfg(platform="linux")) == "linux"
    assert plat.detect_platform(_Cfg(platform="macos")) == "macos"


def test_detect_platform_auto_macos(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "darwin")
    assert plat.detect_platform(_Cfg(platform="auto")) == "macos"
    assert plat.is_macos(_Cfg())


def test_detect_platform_auto_linux(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "linux")
    assert plat.detect_platform(_Cfg(platform="auto")) == "linux"
    assert not plat.is_macos(_Cfg())


# ---------------------------------------------------------------------------
# player selection
# ---------------------------------------------------------------------------


def test_select_player_macos_prefers_afplay(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "darwin")
    which = lambda name: name if name == "afplay" else None  # noqa: E731
    assert plat.select_player(_Cfg(), which=which) == ("afplay",)


def test_select_player_linux_prefers_aplay(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "linux")
    which = lambda name: name if name in ("aplay", "paplay") else None  # noqa: E731
    assert plat.select_player(_Cfg(), which=which) == ("aplay", "-q")


def test_select_player_linux_falls_to_paplay(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "linux")
    which = lambda name: name if name == "paplay" else None  # noqa: E731
    assert plat.select_player(_Cfg(), which=which) == ("paplay",)


def test_select_player_pinned_backend(monkeypatch):
    monkeypatch.setattr(plat.sys, "platform", "linux")
    which = lambda name: name  # noqa: E731 — everything available
    assert plat.select_player(_Cfg(playback_backend="aplay"), which=which) == ("aplay", "-q")


def test_select_player_none_available():
    assert plat.select_player(_Cfg(), which=lambda name: None) is None


# ---------------------------------------------------------------------------
# play_array — sounddevice preferred, CLI fallback
# ---------------------------------------------------------------------------


def test_play_array_uses_sounddevice(monkeypatch):
    calls = {}

    class _FakeSd:
        def play(self, samples, samplerate):
            calls["played"] = (len(samples), samplerate)

        def wait(self):
            calls["waited"] = True

    from my_stt_tts import audio

    monkeypatch.setattr(audio, "_sd", lambda: _FakeSd())
    plat.play_array(np.zeros(100, dtype=np.float32), 16000, _Cfg())
    assert calls["played"] == (100, 16000)
    assert calls["waited"]


def test_play_array_cli_fallback_when_no_sounddevice(monkeypatch):
    import subprocess

    from my_stt_tts import audio

    def _boom():
        raise RuntimeError("no PortAudio")

    monkeypatch.setattr(audio, "_sd", _boom)
    monkeypatch.setattr(plat, "select_player", lambda cfg: ("aplay", "-q"))
    ran = {}

    def _fake_run(cmd, check=False):  # noqa: ARG001
        ran["cmd"] = cmd

    monkeypatch.setattr(subprocess, "run", _fake_run)
    plat.play_array(np.zeros(50, dtype=np.float32), 16000, _Cfg())
    assert ran["cmd"][0] == "aplay"


# ---------------------------------------------------------------------------
# cross-platform STT backend selection
# ---------------------------------------------------------------------------


def test_select_whispercpp_backend():
    from my_stt_tts.registry import select_transcriber
    from my_stt_tts.stt import WhisperCppSTT

    cfg = Config()
    cfg.stt_backend = "whispercpp"
    assert isinstance(select_transcriber(cfg), WhisperCppSTT)


def test_select_faster_whisper_backend():
    from my_stt_tts.registry import select_transcriber
    from my_stt_tts.stt import FasterWhisperSTT

    cfg = Config()
    cfg.stt_backend = "faster-whisper"
    cfg.faster_whisper_compute = "float16"
    eng = select_transcriber(cfg)
    assert isinstance(eng, FasterWhisperSTT)
    assert eng.compute_type == "float16"


def test_whispercpp_transcribe_with_fake_model():
    from my_stt_tts.stt import WhisperCppSTT

    class _Seg:
        def __init__(self, text):
            self.text = text

    class _Model:
        detected_language = "de"

        def transcribe(self, audio, **kwargs):  # noqa: ARG002
            return [_Seg("hallo"), _Seg("welt")]

    eng = WhisperCppSTT("tiny")
    eng._model = _Model()
    res = eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert res.text == "hallo welt"
    assert res.language == "de"


def test_faster_whisper_transcribe_with_fake_model():
    from my_stt_tts.stt import FasterWhisperSTT

    class _Seg:
        def __init__(self, text):
            self.text = text

    class _Info:
        language = "fr"

    class _Model:
        def transcribe(self, audio, language=None):  # noqa: ARG002
            return [_Seg("bonjour"), _Seg("monde")], _Info()

    eng = FasterWhisperSTT("tiny")
    eng._model = _Model()
    res = eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert res.text == "bonjour monde"
    assert res.language == "fr"


# ---------------------------------------------------------------------------
# Linux WebRTC-APM AEC selection + fallback
# ---------------------------------------------------------------------------


def test_make_echo_canceller_webrtc_falls_back_to_nlms_when_unavailable(monkeypatch):
    from my_stt_tts import aec

    monkeypatch.setattr(aec.WebRtcApmEchoCanceller, "available", staticmethod(lambda: False))
    cfg = Config()
    cfg.aec_mode = "webrtc"
    ec = aec.make_echo_canceller(cfg)
    assert isinstance(ec, aec.NlmsEchoCanceller)


def test_make_echo_canceller_webrtc_used_when_active(monkeypatch):
    from my_stt_tts import aec

    monkeypatch.setattr(aec.WebRtcApmEchoCanceller, "available", staticmethod(lambda: True))
    monkeypatch.setattr(aec.WebRtcApmEchoCanceller, "_build", lambda self: object())
    cfg = Config()
    cfg.aec_mode = "webrtc"
    ec = aec.make_echo_canceller(cfg)
    assert isinstance(ec, aec.WebRtcApmEchoCanceller)
    assert ec.active


def test_webrtc_apm_passthrough_without_native_module():
    from my_stt_tts.aec import WebRtcApmEchoCanceller

    ec = WebRtcApmEchoCanceller(16000)  # native module absent on the test host
    assert not ec.active
    frame = np.linspace(-0.3, 0.3, 160, dtype=np.float32)
    out = ec.process(frame)
    assert np.allclose(out, frame)  # identity pass-through when inactive
    ec.push_reference(frame)  # no-op, must not raise
    ec.reset()


def test_webrtc_apm_processes_with_fake_module():
    from my_stt_tts.aec import WebRtcApmEchoCanceller

    class _FakeApm:
        def process_stream(self, data):
            # Return the same int16 bytes (a fake "cancelled" frame).
            return data

        def process_reverse_stream(self, data):
            pass

    ec = WebRtcApmEchoCanceller(16000)
    ec._apm = _FakeApm()
    ec.active = True
    # Feed exactly one 10 ms frame (160 samples @ 16 kHz).
    frame = np.full(160, 0.1, dtype=np.float32)
    out = ec.process(frame)
    assert out.size == 160
    ec.push_reference(frame)  # exercises the reverse-stream drain


def test_config_accepts_webrtc_aec_mode():
    cfg = Config(anthropic_api_key="x")
    cfg.aec_mode = "webrtc"
    cfg.validate()  # must not raise


# ---------------------------------------------------------------------------
# host_app_name — friendly name of the app the SERVER runs in (TERM_PROGRAM)
# ---------------------------------------------------------------------------


def test_host_app_name_maps_known_term_programs():
    # The exact TERM_PROGRAM values the common emulators set -> friendly names.
    cases = {
        "iTerm.app": "iTerm",
        "Apple_Terminal": "Terminal",
        "vscode": "VS Code",
        "ghostty": "Ghostty",
        "WezTerm": "WezTerm",
        "Hyper": "Hyper",
        "Tabby": "Tabby",
        "alacritty": "Alacritty",
        "WarpTerminal": "Warp",
    }
    for term_program, friendly in cases.items():
        assert plat.host_app_name({"TERM_PROGRAM": term_program}) == friendly


def test_host_app_name_is_case_insensitive():
    assert plat.host_app_name({"TERM_PROGRAM": "ITERM.APP"}) == "iTerm"
    assert plat.host_app_name({"TERM_PROGRAM": "APPLE_TERMINAL"}) == "Terminal"


def test_host_app_name_unset_falls_back_to_generic():
    assert plat.host_app_name({}) == "your terminal app"
    assert plat.host_app_name({"TERM_PROGRAM": ""}) == "your terminal app"
    assert plat.host_app_name({"TERM_PROGRAM": "   "}) == "your terminal app"


def test_host_app_name_unknown_value_is_titlecased_best_guess():
    # An emulator we don't have in the table is still named (an app IS set), with a
    # trailing ".app" stripped, rather than the generic fallback.
    assert plat.host_app_name({"TERM_PROGRAM": "SomeNewTerm"}) == "SomeNewTerm"
    assert plat.host_app_name({"TERM_PROGRAM": "fancyterm.app"}) == "Fancyterm"


def test_host_app_name_reads_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert plat.host_app_name() == "Ghostty"
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert plat.host_app_name() == "your terminal app"
