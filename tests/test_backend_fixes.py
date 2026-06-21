"""Backend functional-fix coverage (branch ``backend-fixes``).

Pure / mocked tests for the "nothing recorded" fixes and the new instruments:

* sample-rate resample + frame reframing (the 48 kHz device → 16 kHz models path),
* the VAD reframe + low-threshold fix (a quiet voice is not dropped, no raise on a
  wrong-size frame),
* current-local-time injection into the assembled system prompt (every brain),
* cross-platform mic-permission verdicts (macOS / Linux / Windows, mocked platform),
* the ``voice_test`` action handler (TTS the selected voice, mocked router),
* the ``bus.debug`` event + the audio debug instrument,
* the ``model`` field on the response event.

No real audio, network, model, or device is touched.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio, platform
from my_stt_tts.config import Config, current_time_line, timezone_for_location
from my_stt_tts.events import EventBus
from my_stt_tts.vad import SilenceEndpointer

# --------------------------------------------------------------------------- #
# (1) resample + reframe — the 48 kHz device → 16 kHz models path             #
# --------------------------------------------------------------------------- #


def test_resample_downsamples_48k_to_16k() -> None:
    src = np.ones(48_000, dtype=np.float32)  # 1 s at 48 kHz
    out = audio.resample_to(src, 48_000, 16_000)
    assert out.size == 16_000  # 1 s at 16 kHz
    assert out.dtype == np.float32


def test_resample_is_identity_when_rates_match() -> None:
    src = np.arange(1000, dtype=np.float32)
    out = audio.resample_to(src, 16_000, 16_000)
    assert np.array_equal(out, src)


def test_resample_empty_is_safe() -> None:
    assert audio.resample_to(np.zeros(0, dtype=np.float32), 48_000, 16_000).size == 0


def test_resample_preserves_a_tone_frequency() -> None:
    # A 1 kHz tone sampled at 48 kHz, resampled to 16 kHz, still peaks at 1 kHz.
    t = np.arange(48_000) / 48_000
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    out = audio.resample_to(tone, 48_000, 16_000)
    spec = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(out.size, 1 / 16_000)[int(np.argmax(spec))]
    assert abs(peak_hz - 1000) < 50  # within 50 Hz of 1 kHz


def test_reframe_pads_last_frame_to_size() -> None:
    frames = audio.reframe(np.arange(1, 1301, dtype=np.float32), 512)  # 1..1300, no zeros
    assert [f.size for f in frames] == [512, 512, 512]  # 1300 -> 3 frames, padded
    # The 3rd frame holds the last 1300-1024 = 276 real samples, then zero-padding.
    real = 1300 - 1024
    assert frames[-1][real - 1] != 0.0  # last real sample present
    assert frames[-1][real] == 0.0  # first padded sample
    assert frames[-1][-1] == 0.0  # padded tail


def test_reframe_empty_or_bad_size() -> None:
    assert audio.reframe(np.zeros(0, dtype=np.float32), 512) == []
    assert audio.reframe(np.arange(10, dtype=np.float32), 0) == []


def test_supported_capture_rate_falls_back_to_native_when_16k_rejected() -> None:
    sd = MagicMock(name="sounddevice")
    sd.check_input_settings.side_effect = ValueError("16k not supported")
    sd.query_devices.return_value = {"default_samplerate": 48000.0}
    assert audio._supported_capture_rate(sd, 16_000) == 48_000


def test_supported_capture_rate_uses_requested_when_supported() -> None:
    sd = MagicMock(name="sounddevice")
    sd.check_input_settings.return_value = None
    assert audio._supported_capture_rate(sd, 16_000) == 16_000


# --------------------------------------------------------------------------- #
# (1) record_until_silence — no stdin, resamples, ends on a pause             #
# --------------------------------------------------------------------------- #


class _FakeSd:
    """Minimal sounddevice stand-in delivering a fixed list of capture blocks."""

    def __init__(self, blocks: list[np.ndarray], device_rate: int = 16_000) -> None:
        self._blocks = blocks
        self._device_rate = device_rate

    def check_input_settings(self, **_kw: object) -> None:
        return None

    def query_devices(self, **_kw: object) -> dict:
        return {"default_samplerate": float(self._device_rate)}

    def InputStream(self, *, callback, **_kw):  # noqa: N802, ANN001 — sd API name
        for block in self._blocks:
            callback(block.reshape(-1, 1), len(block), None, None)
        return _NullStream()


class _NullStream:
    def __enter__(self):
        return self

    def __exit__(self, *a: object) -> None:
        return None


class _ScriptedVad:
    """VAD that returns a scripted speech flag per 512-sample frame."""

    def __init__(self, flags: list[bool]) -> None:
        self._flags = list(flags)
        self.last_prob = 0.0

    def is_speech(self, frame: np.ndarray) -> bool:  # noqa: ARG002
        val = self._flags.pop(0) if self._flags else False
        self.last_prob = 0.9 if val else 0.0
        return val


def test_record_until_silence_returns_empty_on_pure_silence() -> None:
    blocks = [np.zeros(512, dtype=np.float32) for _ in range(5)]
    vad = _ScriptedVad([False] * 10)
    ep = SilenceEndpointer(0.2, frame_seconds=512 / 16000)
    with patch.object(audio, "_sd", return_value=_FakeSd(blocks)):
        clip = audio.record_until_silence(16_000, vad, ep, max_seconds=5.0)
    assert clip.size == 0  # nothing said -> empty, not a hang on stdin


def test_record_until_silence_captures_speech_then_ends_on_silence() -> None:
    blocks = [np.ones(512, dtype=np.float32) for _ in range(6)]
    # speech for two frames, then silence long enough to end (>=0.2 s = ~7 frames)
    vad = _ScriptedVad([True, True, False, False, False, False, False, False, False])
    ep = SilenceEndpointer(0.05, frame_seconds=512 / 16000)
    with patch.object(audio, "_sd", return_value=_FakeSd(blocks)):
        clip = audio.record_until_silence(16_000, vad, ep, max_seconds=5.0)
    assert clip.size > 0  # the utterance was captured (not discarded)


def test_record_until_silence_resamples_48k_device_to_16k() -> None:
    # A 48 kHz device block (1024 samples) is resampled before framing/return.
    blocks = [np.ones(1024, dtype=np.float32) for _ in range(4)]
    vad = _ScriptedVad([True, False, False, False, False, False, False, False])
    ep = SilenceEndpointer(0.05, frame_seconds=512 / 16000)
    fake = _FakeSd(blocks, device_rate=48_000)
    fake.check_input_settings = MagicMock(side_effect=ValueError("16k unsupported"))  # type: ignore[method-assign]
    with patch.object(audio, "_sd", return_value=fake):
        clip = audio.record_until_silence(16_000, vad, ep, max_seconds=5.0)
    # 4 blocks * 1024 @48k -> ~1/3 the samples @16k; just assert it shrank, not raised.
    assert 0 < clip.size < 4 * 1024


def test_record_until_silence_emits_debug() -> None:
    blocks = [np.ones(512, dtype=np.float32) for _ in range(3)]
    vad = _ScriptedVad([True, False, False, False, False, False])
    ep = SilenceEndpointer(0.05, frame_seconds=512 / 16000)
    seen: list[str] = []
    with patch.object(audio, "_sd", return_value=_FakeSd(blocks)):
        audio.record_until_silence(
            16_000, vad, ep, max_seconds=5.0, on_debug=lambda stage, **_k: seen.append(stage)
        )
    assert "capture_start" in seen


# --------------------------------------------------------------------------- #
# (1) VAD reframe + low-threshold fix                                          #
# --------------------------------------------------------------------------- #


class _StubSileroModel:
    """A Silero-shaped callable that only accepts the exact 512-chunk (else raises)."""

    def __init__(self, prob: float) -> None:
        self._prob = prob

    def __call__(self, tensor, _sr):  # noqa: ANN001
        import torch

        if tensor.shape[-1] != 512:  # the real model raises on a wrong size
            raise ValueError("Input audio chunk is too short")
        return torch.tensor(self._prob)


def _silero_with(prob: float):
    from my_stt_tts.vad import SileroVad

    v = SileroVad(16_000, threshold=0.3)
    v._model = _StubSileroModel(prob)  # inject so no model download
    return v


def test_vad_reframes_a_1280_frame_without_raising() -> None:
    import pytest

    pytest.importorskip("torch")  # real scoring needs the 'vad' extra (torch)
    v = _silero_with(0.9)
    # 1280 samples (the wake/PTT block size) would crash the raw model; reframed it
    # scores fine — this is the "push-to-talk records nothing" frame-size fix.
    assert v.is_speech(np.ones(1280, dtype=np.float32)) is True


def test_vad_low_threshold_keeps_quiet_speech() -> None:
    import pytest

    pytest.importorskip("torch")  # real scoring needs the 'vad' extra (torch)
    # A borderline 0.35 probability is SILENCE at the old 0.5 threshold but SPEECH
    # at the new 0.3 default — so a quiet, ~10%-level utterance is no longer dropped.
    v = _silero_with(0.35)
    assert v.is_speech(np.ones(512, dtype=np.float32)) is True
    v.threshold = 0.5
    assert v.is_speech(np.ones(512, dtype=np.float32)) is False


def test_vad_never_raises_on_model_failure() -> None:
    from my_stt_tts.vad import SileroVad

    v = SileroVad(16_000)
    v._model = MagicMock(side_effect=RuntimeError("boom"))
    # A model failure must read as "no speech", not crash the capture loop.
    assert v.is_speech(np.ones(512, dtype=np.float32)) is False
    assert v.last_prob == 0.0


def test_vad_empty_frame_is_silence() -> None:
    v = _silero_with(0.9)
    assert v.is_speech(np.zeros(0, dtype=np.float32)) is False


def test_config_default_vad_threshold_is_low() -> None:
    assert Config().vad_threshold == 0.3


def test_config_rejects_out_of_range_vad_threshold() -> None:
    import pytest

    from my_stt_tts.config import ConfigError

    cfg = Config(anthropic_api_key="x", vad_threshold=1.5)
    with pytest.raises(ConfigError, match="vad_threshold"):
        cfg.validate()


# --------------------------------------------------------------------------- #
# (3) current-time injection into the assembled system prompt                 #
# --------------------------------------------------------------------------- #


def test_current_time_line_uses_location_timezone() -> None:
    fixed = dt.datetime(2026, 6, 19, 22, 51, tzinfo=ZoneInfo("Europe/Zurich"))
    line = current_time_line("Lausanne, Switzerland", now=fixed)
    assert line == "Current local time: 2026-06-19 22:51 (Europe/Zurich)."


def test_timezone_for_known_and_unknown_location() -> None:
    assert str(timezone_for_location("Lausanne")) == "Europe/Zurich"
    assert timezone_for_location("Atlantis") is None


def test_current_time_line_falls_back_to_local_tz_for_unknown_place() -> None:
    line = current_time_line("Nowhere City")
    assert line.startswith("Current local time:")


def test_assembled_system_prompt_contains_current_time() -> None:
    # The single choke point used by EVERY brain (claude-cli, codex, anthropic,
    # openai) so time-awareness works even without tool access.
    from my_stt_tts.brain import Brain

    cfg = Config(llm_provider="claude-cli", location="Lausanne, Switzerland")
    brain = Brain(cfg)
    prompt = brain._system_prompt()
    assert "Current local time:" in prompt
    assert "(Europe/Zurich)" in prompt


# --------------------------------------------------------------------------- #
# (4) cross-platform mic-permission verdicts                                  #
# --------------------------------------------------------------------------- #


def test_mic_permission_macos_uses_tcc() -> None:
    with (
        patch.object(platform, "detect_platform", return_value=platform.MACOS),
        patch.object(platform, "_macos_mic_permission", return_value="authorized"),
    ):
        assert platform.mic_permission_status() == "authorized"


def test_mic_permission_linux_reports_na_with_device() -> None:
    with (
        patch.object(platform, "detect_platform", return_value=platform.LINUX),
        patch.object(platform, "_input_device_present", return_value=True),
    ):
        assert platform.mic_permission_status() == "n/a"


def test_mic_permission_linux_unavailable_without_device() -> None:
    with (
        patch.object(platform, "detect_platform", return_value=platform.LINUX),
        patch.object(platform, "_input_device_present", return_value=False),
    ):
        assert platform.mic_permission_status() == "unavailable"


def test_mic_permission_windows_reads_privacy_toggle() -> None:
    with (
        patch.object(platform, "detect_platform", return_value=platform.OTHER),
        patch.object(platform, "_windows_mic_privacy", return_value="denied"),
    ):
        assert platform.mic_permission_status() == "denied"


def test_mic_permission_windows_falls_back_to_device_check() -> None:
    with (
        patch.object(platform, "detect_platform", return_value=platform.OTHER),
        patch.object(platform, "_windows_mic_privacy", return_value=None),
        patch.object(platform, "_input_device_present", return_value=True),
    ):
        assert platform.mic_permission_status() == "n/a"


def test_audio_mic_permission_delegates_to_platform() -> None:
    with patch("my_stt_tts.platform.mic_permission_status", return_value="n/a") as p:
        assert audio.mic_permission_status() == "n/a"
    p.assert_called_once()


def test_silent_capture_with_na_permission_reads_as_device_issue() -> None:
    # On Linux/Windows ("n/a") a silent capture is a device/level issue, not a
    # permission one — the verdict must say "granted but no audio".
    r = audio.mic_test_verdict(captured=True, rms=0.0, peak=0.0, permission="n/a")
    assert r.verdict == "silent"
    assert "no audio" in r.message.lower()


# --------------------------------------------------------------------------- #
# (5) voice_test action handler                                               #
# --------------------------------------------------------------------------- #


def test_voice_test_speaks_selected_voice() -> None:
    cfg = Config()
    cfg.tts_voices["en"] = "en_US-amy-medium"
    tts = MagicMock(name="tts")
    with patch.object(main_mod.bus, "log"), patch.object(main_mod.bus, "state"):
        main_mod._voice_test(cfg, tts, {})
    tts.speak.assert_called_once()
    spoken = tts.speak.call_args.args[0]
    assert "amy" in spoken  # the friendly preset name appears in the sample line


def test_voice_test_applies_per_request_override() -> None:
    cfg = Config()
    tts = MagicMock(name="tts")
    with patch.object(main_mod.bus, "log"), patch.object(main_mod.bus, "state"):
        main_mod._voice_test(cfg, tts, {"voice_en": "en_GB-alan-medium"})
    assert cfg.tts_voices["en"] == "en_GB-alan-medium"
    assert "alan" in tts.speak.call_args.args[0]


def test_voice_test_never_raises_on_tts_failure() -> None:
    cfg = Config()
    tts = MagicMock(name="tts")
    tts.speak.side_effect = RuntimeError("no synth")
    with (
        patch.object(main_mod.bus, "log") as buslog,
        patch.object(main_mod.bus, "state"),
    ):
        main_mod._voice_test(cfg, tts, {})  # must not raise
    assert any("voice test error" in str(c.args[0]) for c in buslog.call_args_list)


def test_voice_preset_name_maps_id_back() -> None:
    assert main_mod._voice_preset_name("en_US-lessac-medium") == "lessac"
    assert main_mod._voice_preset_name("") == "default"


# --------------------------------------------------------------------------- #
# (2) debug instrument + bus.debug                                            #
# --------------------------------------------------------------------------- #


def test_bus_debug_event_carries_fields() -> None:
    import json

    b = EventBus()
    sub = b.subscribe()
    b.debug("captured", sample_rate=16000, samples=24000, rms=0.04)
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["type"] == "debug"
    assert evt["sample_rate"] == 16000
    assert evt["message"] == "captured"


def test_audio_debug_disabled_is_noop() -> None:
    dbg = main_mod._AudioDebug(False)
    with patch.object(main_mod.bus, "debug") as busdebug:
        dbg("capture_start", sample_rate=16000)
        dbg.action("ptt")
    busdebug.assert_not_called()


def test_audio_debug_enabled_publishes_and_logs(capsys) -> None:  # noqa: ANN001
    dbg = main_mod._AudioDebug(True)
    with patch.object(main_mod.bus, "debug") as busdebug:
        dbg("captured", sample_rate=16000, rms=0.04)
        dbg.action("voice_test", voice="amy")
    assert busdebug.call_count == 2
    err = capsys.readouterr().err
    assert "[audio:captured]" in err
    # The structured stage is still carried for machine use…
    assert "[audio:action:voice_test]" in err
    # …but an unmapped action now reads with a friendly "clicked <NAME>" message.
    assert "clicked VOICE_TEST" in err
    # The friendly label is what the GUI EVENT LOG renders (the bus.debug message).
    assert busdebug.call_args_list[1].args[0].startswith("clicked VOICE_TEST")


def test_debug_audio_enabled_auto_on_for_browser() -> None:
    cfg = Config()  # debug_audio defaults to None (auto)
    assert main_mod.debug_audio_enabled(cfg, browser=True) is True
    assert main_mod.debug_audio_enabled(cfg, browser=False) is False


def test_debug_audio_explicit_env_overrides_browser_auto() -> None:
    cfg = Config(debug_audio=False)
    assert main_mod.debug_audio_enabled(cfg, browser=True) is False
    cfg2 = Config(debug_audio=True)
    assert main_mod.debug_audio_enabled(cfg2, browser=False) is True


def test_audio_debug_vad_frames_are_rate_limited(capsys) -> None:  # noqa: ANN001
    dbg = main_mod._AudioDebug(True)
    with patch.object(main_mod.bus, "debug") as busdebug:
        for _ in range(50):
            dbg("vad_frame", is_speech=False)
    # 50 frames -> sampled 1-in-25 -> only ~2 lines, not a wall.
    assert busdebug.call_count <= 3


# --------------------------------------------------------------------------- #
# (6) model field on the response event                                       #
# --------------------------------------------------------------------------- #


def test_response_event_carries_model() -> None:
    import json

    b = EventBus()
    sub = b.subscribe()
    b.response("", final=True, model="claude-cli / claude-haiku-4-5")
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["type"] == "response"
    assert evt["model"] == "claude-cli / claude-haiku-4-5"


def test_response_event_omits_model_when_blank() -> None:
    import json

    b = EventBus()
    sub = b.subscribe()
    b.response("hi", final=False)
    evt = json.loads(sub.get(timeout=1.0))
    assert "model" not in evt


def test_model_label_format() -> None:
    # The label is the EXACT model + effort/size tier (GUI contract): the version id
    # maps to its marketing version and claude-cli appends the tier it runs at
    # (`xlarge`, space-separated — not `· think`). See config.model_label.
    cfg = Config(llm_provider="claude-cli", llm_model="claude-haiku-4-5")
    assert main_mod._model_label(cfg) == "claude-cli / haiku-4.5 xlarge"
    # The default brain (opus-sub) renders EXACTLY as the contract specifies.
    opus = Config(llm_provider="claude-cli", llm_model="opus")
    assert main_mod._model_label(opus) == "claude-cli / opus-4.8 xlarge"


# --------------------------------------------------------------------------- #
# wake-score plumbing (debug instrument input)                                #
# --------------------------------------------------------------------------- #


def test_wake_detect_tracks_last_score() -> None:
    from my_stt_tts.wake import WakeWord

    w = WakeWord("wakewords/maziko.onnx", threshold=0.5)  # default phases=1 -> one model
    # Pre-seed the (single) phase model + its buffer so _ensure() skips construction
    # (no openwakeword wheel needed in core-only test runs).
    mock = MagicMock()
    w._models = [mock]
    w._reset_pending()
    mock.predict.return_value = {"maziko": 0.72}
    assert w.detect(np.ones(1280, dtype=np.float32)) is True
    assert w.last_score == 0.72
    mock.predict.return_value = {"maziko": 0.10}
    assert w.detect(np.ones(1280, dtype=np.float32)) is False
    assert w.last_score == 0.10


# --------------------------------------------------------------------------- #
# record-and-replay sample-rate fix — record rate == play rate (no speed-up)  #
# --------------------------------------------------------------------------- #


class _FixedSd:
    """sounddevice stand-in that delivers ``seconds`` of audio at ``device_rate``."""

    def __init__(self, device_rate: int, *, honour_requested: bool = False) -> None:
        self._device_rate = device_rate
        self._honour = honour_requested

    def check_input_settings(self, **_kw: object) -> None:
        if not self._honour:
            raise ValueError("requested rate not supported")

    def query_devices(self, **_kw: object) -> dict:
        return {"default_samplerate": float(self._device_rate)}

    def InputStream(self, *, callback, samplerate, blocksize, **_kw):  # noqa: N802, ANN001
        # One ~0.1s block at the (device) rate it was opened with so the timed loop
        # collects something before the deadline.
        n = max(1, int(samplerate * 0.1))
        callback(np.full((n, 1), 0.5, dtype=np.float32), n, None, None)
        return _NullStream()


def test_record_fixed_returns_raw_at_device_rate_no_resample() -> None:
    # 48 kHz device, 16 kHz requested: the RAW clip stays at 48 kHz (NOT resampled to
    # 16 kHz). Resampling to 16 kHz here is what made the human replay sped-up.
    fake = _FixedSd(48_000, honour_requested=False)
    with patch.object(audio, "_sd", return_value=fake):
        clip, device_rate = audio.record_fixed(16_000, seconds=0.3)
    assert device_rate == 48_000
    # The clip length matches the DEVICE rate, not the 16 kHz pipeline rate.
    assert clip.size >= int(48_000 * 0.1)  # at least the one ~0.1s @48k block
    assert clip.size > int(16_000 * 0.1)  # would be smaller if it had been resampled


def test_record_fixed_uses_requested_rate_when_device_honours_it() -> None:
    fake = _FixedSd(16_000, honour_requested=True)
    with patch.object(audio, "_sd", return_value=fake):
        clip, device_rate = audio.record_fixed(16_000, seconds=0.3)
    assert device_rate == 16_000
    assert clip.size >= int(16_000 * 0.1)


def test_record_replay_round_trip_plays_at_record_rate() -> None:
    # End-to-end: a 48 kHz capture is replayed at 48 kHz (record rate == play rate),
    # so a 3 s recording replays as 3 s with faithful pitch — not 1.5×/3× too fast.
    device_rate = 48_000
    clip = (np.sin(np.linspace(0, 200, device_rate * 3)) * 0.5).astype(np.float32)
    played: list[tuple[int, int]] = []  # (num_samples, play_rate)

    def fake_play(samples, sample_rate, *_a, **_k):  # noqa: ANN001
        played.append((np.asarray(samples).size, sample_rate))

    with (
        patch.object(audio, "record_fixed", return_value=(clip, device_rate)),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
        patch.object(main_mod.audio, "play", side_effect=fake_play),
        patch.object(main_mod.bus, "log"),
        patch.object(main_mod.bus, "state"),
        patch.object(main_mod.bus, "mic_result"),
    ):
        main_mod._run_mic_record_replay(Config(anthropic_api_key="x"), seconds=3.0)

    assert played, "the recording was never played back"
    samples_played, play_rate = played[0]
    assert play_rate == device_rate  # plays at the rate it recorded at (the fix)
    assert samples_played == clip.size  # whole clip, un-resampled
    # Duration is preserved: samples / play_rate ≈ the captured 3 s.
    assert abs(samples_played / play_rate - 3.0) < 0.01


def test_record_replay_reports_duration_at_device_rate() -> None:
    # The human-facing duration must use the DEVICE rate (samples/device_rate). With
    # the old bug it divided by 16 kHz and reported 3× too long for a 48 kHz clip.
    device_rate = 48_000
    clip = (np.sin(np.linspace(0, 90, device_rate * 2)) * 0.5).astype(np.float32)  # 2 s @48k
    logs: list[str] = []

    with (
        patch.object(audio, "record_fixed", return_value=(clip, device_rate)),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
        patch.object(main_mod.audio, "play"),
        patch.object(main_mod.bus, "log", side_effect=lambda msg, *_a, **_k: logs.append(str(msg))),
        patch.object(main_mod.bus, "state"),
        patch.object(main_mod.bus, "mic_result"),
    ):
        main_mod._run_mic_record_replay(Config(anthropic_api_key="x"), seconds=2.0)

    assert any("2.0s" in m and "48000 Hz" in m for m in logs), logs


# --------------------------------------------------------------------------- #
# (1 CRITICAL) single-STT-thread marshalling — the PTT MLX-affinity fix        #
# --------------------------------------------------------------------------- #


class _AffinityEngine:
    """A ParakeetSTT stand-in whose MLX `_ensure`/decode record the calling thread.

    Mimics the MLX thread-affinity constraint: it remembers the thread that first
    "loaded the model" and ASSERTS every later decode runs on that SAME thread —
    raising the exact runtime error MLX raises otherwise. The real fix routes all
    calls through one worker thread, so this never raises."""

    def __init__(self) -> None:
        from my_stt_tts.stt import ParakeetSTT, STTResult

        self._stt = ParakeetSTT("fake-model")
        self._stt._model = self  # the "MLX model" is this object
        self._STTResult = STTResult
        self._load_thread: int | None = None
        self.call_threads: list[int] = []

    def transcribe(self, _wav_path):  # noqa: ANN001 — mirrors parakeet_mlx.Model.transcribe
        import threading as _t

        ident = _t.get_ident()
        if self._load_thread is None:
            self._load_thread = ident  # the GPU stream is created on THIS thread
        elif ident != self._load_thread:
            raise RuntimeError("There is no Stream(gpu, 0) in current thread")
        self.call_threads.append(ident)

        class _R:
            text = "ok"
            language = "en"

        return _R()


def test_stt_calls_from_two_threads_share_one_worker_thread() -> None:
    import threading

    eng = _AffinityEngine()
    clip = np.ones(1600, dtype=np.float32)
    results: list[str] = []

    def call() -> None:
        results.append(eng._stt.transcribe(clip, 16000).text)

    # Two DIFFERENT caller threads (as the GUI spawns per PTT/mic/wake action).
    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # Both transcriptions succeeded (no MLX cross-thread crash)…
    assert results == ["ok", "ok"]
    # …and the underlying MLX engine was ONLY ever touched from ONE consistent
    # thread (the affinity invariant), even though two distinct callers asked.
    assert len(set(eng.call_threads)) == 1
    # …and that one thread is the dedicated STT worker, not either caller's thread.
    from my_stt_tts.stt import stt_worker

    assert eng.call_threads[0] == stt_worker().thread_ident


def test_stt_worker_reraises_engine_error_on_caller_thread() -> None:
    import pytest

    from my_stt_tts.stt import stt_worker

    def boom() -> None:
        raise ValueError("decode failed")

    # An error inside the worker is re-raised on the caller's thread (not swallowed).
    with pytest.raises(ValueError, match="decode failed"):
        stt_worker().submit(boom)


def test_ptt_capture_routes_transcription_through_worker() -> None:
    # The PTT capture path (_capture_ptt) must transcribe via the same engine surface,
    # which now marshals onto the worker — so a fresh-thread PTT click no longer
    # crashes. Drive it with a tiny captured clip + the affinity engine.
    import threading

    eng = _AffinityEngine()
    clip = np.ones(1600, dtype=np.float32)

    with (
        patch.object(main_mod, "_capture_ptt_clip", create=True),
        patch.object(main_mod.audio, "record_until_silence", return_value=clip),
        patch.object(main_mod, "_play"),
        patch.object(main_mod.bus, "state"),
        patch.object(main_mod, "_signal_mic_confirmed"),
    ):
        out: dict[str, str] = {}

        def run() -> None:
            cap = main_mod._capture_ptt(Config(), eng._stt, main_mod.audio.MicGate(0.0))
            out["text"] = cap.text

        # Run on a FRESH thread (the bug's trigger) — must not raise.
        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5.0)

    assert out["text"] == "ok"


# --------------------------------------------------------------------------- #
# (4) mstt-side mpv preflight — halt before opening the browser if mpv missing #
# --------------------------------------------------------------------------- #


def test_mpv_preflight_halts_when_missing() -> None:
    cfg = Config()  # music_enabled defaults True
    with (
        patch.object(main_mod.shutil, "which", return_value=None),
        patch.dict("os.environ", {}, clear=False),
        patch.object(main_mod.bus, "log"),
    ):
        import os

        os.environ.pop("MSTT_SKIP_MPV_CHECK", None)
        rc = main_mod._mpv_preflight_gate(cfg)
    assert rc == 4  # non-zero -> main returns it BEFORE opening the browser


def test_mpv_preflight_passes_when_present() -> None:
    cfg = Config()
    with patch.object(main_mod.shutil, "which", return_value="/usr/local/bin/mpv"):
        assert main_mod._mpv_preflight_gate(cfg) is None


def test_mpv_preflight_skippable_via_env() -> None:
    cfg = Config()
    with (
        patch.object(main_mod.shutil, "which", return_value=None),
        patch.dict("os.environ", {"MSTT_SKIP_MPV_CHECK": "1"}),
    ):
        assert main_mod._mpv_preflight_gate(cfg) is None  # opted out -> no halt


def test_mpv_preflight_noop_when_music_disabled() -> None:
    cfg = Config(music_enabled=False)
    with patch.object(main_mod.shutil, "which", return_value=None):
        assert main_mod._mpv_preflight_gate(cfg) is None  # no music -> mpv irrelevant


def test_main_browser_halts_early_when_mpv_missing() -> None:
    # A direct `./mstt --browser` with no mpv halts BEFORE building the brain / opening
    # the GUI (the quickstart check only covers the quickstart path).
    with (
        patch.object(main_mod.shutil, "which", return_value=None),
        patch.object(main_mod, "_run_browser") as run_browser,
        patch.object(main_mod, "Brain") as brain_cls,
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}),
    ):
        import os

        os.environ.pop("MSTT_SKIP_MPV_CHECK", None)
        rc = main_mod.main(["--browser", "--type"])
    assert rc == 4
    run_browser.assert_not_called()  # the browser was never opened
    brain_cls.assert_not_called()  # not even the heavy components were built


# --------------------------------------------------------------------------- #
# (5) score-clip / wake-test ALWAYS classify a detailed reason + detail        #
# --------------------------------------------------------------------------- #


def test_classify_wake_outcome_level_too_low() -> None:
    # A quiet capture (int16 peak below the wake-usable floor) reads level_too_low,
    # NOT a recall miss — the dominant "move closer" failure.
    reason, detail = main_mod._classify_wake_outcome(
        "maziko", fired=False, confidence=0.05, int16_peak=1882, threshold=0.4
    )
    assert reason == "level_too_low"
    assert "maziko" in detail and "1882" in detail


def test_classify_wake_outcome_not_detected_when_level_ok() -> None:
    # Loud enough but still sub-threshold -> a genuine recall miss (level OK).
    reason, detail = main_mod._classify_wake_outcome(
        "hey_jarvis", fired=False, confidence=0.2, int16_peak=12000, threshold=0.4
    )
    assert reason == "not_detected"
    assert "hey_jarvis" in detail and "level OK" in detail


def test_classify_wake_outcome_fired() -> None:
    reason, detail = main_mod._classify_wake_outcome(
        "hey_jarvis", fired=True, confidence=0.8, int16_peak=12000, threshold=0.4
    )
    assert reason == "fired"
    assert "hey_jarvis" in detail


def test_classify_wake_outcome_unavailable() -> None:
    reason, detail = main_mod._classify_wake_outcome(
        "nope", fired=False, confidence=0.0, int16_peak=0, threshold=0.4, available=False
    )
    assert reason == "unavailable"
    assert "unavailable" in detail


def test_score_clip_emits_level_too_low_reason(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    import json

    # A quiet saved clip: score_clip must emit reason=level_too_low + a naming detail.
    quiet = (np.ones(16000, dtype=np.float32) * 0.01).astype(np.float32)  # very low level
    events: list[dict] = []
    b = EventBus()
    sub = b.subscribe()
    cfg = Config(anthropic_api_key="sk-test", wake_threshold=0.5, wake_phases=1)
    with (
        patch.object(main_mod, "bus", b),
        patch.object(main_mod.os.path, "isfile", return_value=True),
        patch.object(
            main_mod, "_load_saved_wake_clip", return_value=(quiet, "abcd1234", "clip.wav")
        ),
        patch("my_stt_tts.wake.score_wake_clip", return_value=(0.02, False)),
    ):
        main_mod._run_score_clip(cfg, "maziko", "abcd1234")
    while True:
        try:
            events.append(json.loads(sub.get(timeout=0.5)))
        except Exception:  # noqa: BLE001 — queue drained
            break
    result = next(e for e in events if e["type"] == "score_clip_result")
    assert result["reason"] == "level_too_low"
    assert "maziko" in result["detail"]


def test_score_clip_emits_not_detected_when_level_ok(monkeypatch) -> None:  # noqa: ANN001
    import json

    loud = (np.ones(16000, dtype=np.float32) * 0.5).astype(np.float32)  # int16 peak ~16k
    events: list[dict] = []
    b = EventBus()
    sub = b.subscribe()
    cfg = Config(anthropic_api_key="sk-test", wake_threshold=0.5, wake_phases=1)
    with (
        patch.object(main_mod, "bus", b),
        patch.object(main_mod.os.path, "isfile", return_value=True),
        patch.object(
            main_mod, "_load_saved_wake_clip", return_value=(loud, "c0ffee99", "clip.wav")
        ),
        patch("my_stt_tts.wake.score_wake_clip", return_value=(0.2, False)),
    ):
        main_mod._run_score_clip(cfg, "hey_jarvis", "c0ffee99")
    while True:
        try:
            events.append(json.loads(sub.get(timeout=0.5)))
        except Exception:  # noqa: BLE001
            break
    result = next(e for e in events if e["type"] == "score_clip_result")
    assert result["reason"] == "not_detected"
    assert "hey_jarvis" in result["detail"]


def test_wake_test_result_carries_reason_and_detail() -> None:
    import json

    b = EventBus()
    sub = b.subscribe()
    b.wake_test_result(
        word="maziko",
        source="server",
        confidence=0.05,
        fired=False,
        message="x",
        reason="level_too_low",
        detail="maziko: level too low (pk 1882) — move closer",
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["reason"] == "level_too_low"
    assert evt["detail"] == "maziko: level too low (pk 1882) — move closer"


def test_score_clip_result_carries_reason_and_detail() -> None:
    import json

    b = EventBus()
    sub = b.subscribe()
    b.score_clip_result(
        word="hey_jarvis",
        hash="abcd1234",
        confidence=0.2,
        fired=False,
        int16_peak=12000,
        message="x",
        reason="not_detected",
        detail="hey_jarvis: wake word not detected (level OK)",
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["reason"] == "not_detected"
    assert "level OK" in evt["detail"]
