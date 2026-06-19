"""Tests for G1 — pluggable backend registry + real cloud adapters.

Covers: the :class:`ServiceRegistry` (register / lookup / namespacing / errors),
the local-first key-gated selection (cloud falls back to local without a key),
and the REAL cloud adapters (Deepgram STT, ElevenLabs + Cartesia TTS) against
*mocked* SDK / HTTP responses — no live key, no socket. Config validation now
cross-checks backend names against the registry.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,import-outside-toplevel

import numpy as np
import pytest

from my_stt_tts.config import Config, ConfigError
from my_stt_tts.registry import (
    ServiceRegistry,
    globals_reg,
    select_transcriber,
    select_tts_backend,
)

# ---------------------------------------------------------------------------
# ServiceRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_build():
    reg = ServiceRegistry()
    reg.register("stt", "fake", lambda cfg: ("built", cfg))
    assert reg.has("stt", "fake")
    out = reg.build("stt", "fake", "CFG")
    assert out == ("built", "CFG")


def test_registry_namespacing_is_isolated_per_kind():
    reg = ServiceRegistry()
    reg.register("stt", "x", lambda cfg: "stt-x")
    reg.register("tts", "x", lambda cfg: "tts-x")
    assert reg.build("stt", "x", None) == "stt-x"
    assert reg.build("tts", "x", None) == "tts-x"
    assert not reg.has("llm", "x")


def test_registry_unknown_kind_and_name_raise():
    reg = ServiceRegistry()
    with pytest.raises(ValueError):
        reg.register("nope", "x", lambda cfg: None)
    with pytest.raises(KeyError):
        reg.build("stt", "missing", None)


def test_registry_duplicate_registration_raises_unless_replace():
    reg = ServiceRegistry()
    reg.register("tts", "x", lambda cfg: 1)
    with pytest.raises(ValueError):
        reg.register("tts", "x", lambda cfg: 2)
    reg.register("tts", "x", lambda cfg: 2, replace=True)
    assert reg.build("tts", "x", None) == 2


def test_builtin_registry_has_all_backends():
    reg = globals_reg()
    assert {"local", "whispercpp", "faster-whisper", "deepgram", "cloud"} <= set(reg.names("stt"))
    assert {"local", "elevenlabs", "cartesia", "cloud"} <= set(reg.names("tts")) | {"local"}
    assert "anthropic" in reg.names("llm")


# ---------------------------------------------------------------------------
# Local-first selection / graceful fallback
# ---------------------------------------------------------------------------


def test_select_transcriber_local_default():
    cfg = Config()
    from my_stt_tts.stt import ParakeetSTT

    assert isinstance(select_transcriber(cfg), ParakeetSTT)


def test_select_transcriber_deepgram_without_key_falls_back_to_local():
    cfg = Config()
    cfg.stt_backend = "deepgram"
    cfg.deepgram_api_key = None
    from my_stt_tts.stt import ParakeetSTT

    # No key -> available() is False -> selector returns the local Parakeet engine.
    assert isinstance(select_transcriber(cfg), ParakeetSTT)


def test_select_transcriber_deepgram_with_key_used():
    cfg = Config()
    cfg.stt_backend = "deepgram"
    cfg.deepgram_api_key = "dg-key"
    from my_stt_tts.stt_cloud import DeepgramSTT

    assert isinstance(select_transcriber(cfg), DeepgramSTT)


def test_select_tts_backend_local_is_none():
    assert select_tts_backend(Config()) is None


def test_select_tts_backend_elevenlabs_without_key_is_none():
    cfg = Config()
    cfg.tts_backend = "elevenlabs"
    cfg.elevenlabs_api_key = None
    assert select_tts_backend(cfg) is None  # falls back to local Piper / say


def test_select_tts_backend_elevenlabs_with_key():
    cfg = Config()
    cfg.tts_backend = "elevenlabs"
    cfg.elevenlabs_api_key = "el-key"
    from my_stt_tts.tts_cloud import ElevenLabsTTS

    assert isinstance(select_tts_backend(cfg), ElevenLabsTTS)


# ---------------------------------------------------------------------------
# Config validation against the registry
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_backends():
    cfg = Config(anthropic_api_key="x")
    cfg.stt_backend = "bogus"
    with pytest.raises(ConfigError):
        cfg.validate()


def test_validate_accepts_registered_backends():
    cfg = Config(anthropic_api_key="x")
    cfg.stt_backend = "deepgram"
    cfg.tts_backend = "elevenlabs"
    cfg.validate()  # should not raise


# ---------------------------------------------------------------------------
# Deepgram STT adapter — mocked
# ---------------------------------------------------------------------------


def _deepgram_payload(text: str, lang: str = "en") -> dict:
    return {
        "results": {
            "channels": [{"alternatives": [{"transcript": text}], "detected_language": lang}]
        }
    }


def test_deepgram_parse_response():
    from my_stt_tts.stt_cloud import parse_deepgram_response

    res = parse_deepgram_response(_deepgram_payload("hello world", "de"))
    assert res.text == "hello world"
    assert res.language == "de"


def test_deepgram_parse_response_malformed_is_empty():
    from my_stt_tts.stt_cloud import parse_deepgram_response

    assert parse_deepgram_response({}).text == ""
    assert parse_deepgram_response({"results": {"channels": []}}).text == ""


def test_deepgram_available_gated_on_key():
    from my_stt_tts.stt_cloud import DeepgramSTT

    assert not DeepgramSTT(api_key=None).available()
    assert DeepgramSTT(api_key="dg").available()


def test_deepgram_transcribe_roundtrip_mocked_http(monkeypatch):
    from my_stt_tts import stt_cloud

    captured = {}

    class _Resp:
        def __init__(self, payload):
            import json

            self._data = json.dumps(payload).encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=30):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        return _Resp(_deepgram_payload("transcribed text", "fr"))

    dg = stt_cloud.DeepgramSTT(api_key="dg-key", model="nova-3")
    monkeypatch.setattr(dg, "_has_sdk", lambda: False)
    monkeypatch.setattr(stt_cloud.urllib.request, "urlopen", _fake_urlopen)
    res = dg.transcribe(np.zeros(16000, dtype=np.float32))
    assert res.text == "transcribed text"
    assert res.language == "fr"
    assert "nova-3" in captured["url"]
    assert captured["auth"] == "Token dg-key"


def test_deepgram_transcribe_no_key_returns_empty():
    from my_stt_tts.stt_cloud import DeepgramSTT

    res = DeepgramSTT(api_key=None).transcribe(np.zeros(16000, dtype=np.float32))
    assert res.text == ""


def test_deepgram_transcribe_network_error_returns_empty(monkeypatch):
    from my_stt_tts import stt_cloud

    dg = stt_cloud.DeepgramSTT(api_key="dg-key")
    monkeypatch.setattr(dg, "_has_sdk", lambda: False)

    def _boom(*a, **k):  # noqa: ARG001
        raise OSError("network down")

    monkeypatch.setattr(stt_cloud.urllib.request, "urlopen", _boom)
    assert dg.transcribe(np.zeros(16000, dtype=np.float32)).text == ""


def test_deepgram_transcribe_via_sdk_to_dict(monkeypatch):
    from my_stt_tts import stt_cloud

    dg = stt_cloud.DeepgramSTT(api_key="dg-key")
    monkeypatch.setattr(dg, "_has_sdk", lambda: True)
    monkeypatch.setattr(dg, "_post_sdk", lambda body: _deepgram_payload("sdk text", "en"))
    res = dg.transcribe(np.zeros(8000, dtype=np.float32))
    assert res.text == "sdk text"


# ---------------------------------------------------------------------------
# ElevenLabs TTS adapter — mocked
# ---------------------------------------------------------------------------


def test_elevenlabs_available_gated_on_key():
    from my_stt_tts.tts_cloud import ElevenLabsTTS

    assert not ElevenLabsTTS(api_key=None).available()
    assert ElevenLabsTTS(api_key="el").available()


def test_elevenlabs_render_roundtrip_mocked_http(monkeypatch):
    from my_stt_tts import tts_cloud

    captured = {}
    pcm = np.linspace(-0.5, 0.5, 2205, dtype=np.float32)
    raw = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()

    class _Resp:
        def read(self):
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=30):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["key"] = req.headers.get("Xi-api-key") or req.headers.get("xi-api-key")
        return _Resp()

    el = tts_cloud.ElevenLabsTTS(api_key="el-key", voice_id="Rachel")
    monkeypatch.setattr(el, "_has_sdk", lambda: False)
    monkeypatch.setattr(tts_cloud.urllib.request, "urlopen", _fake_urlopen)
    out_pcm, sr = el.render("Hallo Welt")
    assert out_pcm is not None and out_pcm.size == 2205
    assert sr == 22050
    assert "Rachel" in captured["url"]


def test_elevenlabs_render_no_key_is_none():
    from my_stt_tts.tts_cloud import ElevenLabsTTS

    assert ElevenLabsTTS(api_key=None).render("hi") == (None, None)


def test_elevenlabs_render_failure_is_none(monkeypatch):
    from my_stt_tts import tts_cloud

    el = tts_cloud.ElevenLabsTTS(api_key="el-key")
    monkeypatch.setattr(el, "_has_sdk", lambda: False)
    monkeypatch.setattr(
        tts_cloud.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError())
    )
    assert el.render("hi") == (None, None)


# ---------------------------------------------------------------------------
# Cartesia TTS adapter — mocked
# ---------------------------------------------------------------------------


def test_cartesia_available_needs_key_and_voice():
    from my_stt_tts.tts_cloud import CartesiaTTS

    assert not CartesiaTTS(api_key="c", voice_id="").available()  # voice required
    assert not CartesiaTTS(api_key=None, voice_id="v").available()
    assert CartesiaTTS(api_key="c", voice_id="v").available()


def test_cartesia_render_roundtrip_mocked_http(monkeypatch):
    from my_stt_tts import tts_cloud

    pcm = np.full(1000, 0.25, dtype=np.float32)
    raw = (pcm * 32767).astype("<i2").tobytes()

    class _Resp:
        def read(self):
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ca = tts_cloud.CartesiaTTS(api_key="c-key", voice_id="voice-1", sample_rate=24000)
    monkeypatch.setattr(ca, "_has_sdk", lambda: False)
    monkeypatch.setattr(tts_cloud.urllib.request, "urlopen", lambda *a, **k: _Resp())
    out_pcm, sr = ca.render("Hello there")
    assert out_pcm is not None and out_pcm.size == 1000
    assert sr == 24000


def test_cartesia_render_no_key_is_none():
    from my_stt_tts.tts_cloud import CartesiaTTS

    assert CartesiaTTS(api_key=None, voice_id="v").render("hi") == (None, None)


# ---------------------------------------------------------------------------
# TTSRouter wiring through the registry
# ---------------------------------------------------------------------------


def test_ttsrouter_uses_cloud_backend_when_keyed():
    from my_stt_tts.tts import TTSRouter

    cfg = Config()
    cfg.tts_backend = "elevenlabs"
    cfg.elevenlabs_api_key = "el-key"
    router = TTSRouter(cfg)
    from my_stt_tts.tts_cloud import ElevenLabsTTS

    assert isinstance(router._cloud, ElevenLabsTTS)


def test_ttsrouter_local_when_no_key():
    from my_stt_tts.tts import TTSRouter

    cfg = Config()
    cfg.tts_backend = "cartesia"  # no key/voice -> not available
    assert TTSRouter(cfg)._cloud is None


def test_ttsrouter_synth_pcm_uses_cloud_render(monkeypatch):
    from my_stt_tts.tts import TTSRouter

    cfg = Config()
    cfg.tts_backend = "elevenlabs"
    cfg.elevenlabs_api_key = "el-key"
    router = TTSRouter(cfg)
    fake_pcm = np.full(100, 0.1, dtype=np.float32)
    monkeypatch.setattr(router._cloud, "render", lambda text: (fake_pcm, 22050))
    pcm, sr = router.synth_pcm("hello")
    assert sr == 22050
    assert np.allclose(pcm, fake_pcm)
