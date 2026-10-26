"""Google Cloud Text-to-Speech adapter (tp#1286) — network faked, no key needed."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from my_stt_tts import tts_cloud
from my_stt_tts.config import TTS_BACKENDS, Config
from my_stt_tts.registry import select_tts_backend
from my_stt_tts.tts import TTSRouter
from my_stt_tts.tts_cloud import GoogleTTS, split_for_google
from my_stt_tts.util import wav_bytes_from_int16


def _wav(samples: int, rate: int = 24000) -> bytes:
    pcm = (np.linspace(-0.3, 0.3, samples) * 32767).astype("<i2").tobytes()
    return wav_bytes_from_int16(pcm, rate)


class _Fake:
    """Records every request; replies with a LINEAR16 WAV of ``samples`` samples."""

    def __init__(self, samples: int = 240) -> None:
        self.samples = samples
        self.requests: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float = 30) -> Any:
        del timeout
        self.requests.append(
            {
                "url": req.full_url,
                "headers": {k.lower(): v for k, v in req.header_items()},
                "body": json.loads(req.data),
            }
        )
        payload = json.dumps({"audioContent": base64.b64encode(_wav(self.samples)).decode()})

        class _Resp:
            def read(self) -> bytes:
                return payload.encode()

            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_a: object) -> None:
                return None

        return _Resp()


@pytest.fixture(name="fake")
def _fake(monkeypatch: pytest.MonkeyPatch) -> _Fake:
    fake = _Fake()
    monkeypatch.setattr(tts_cloud.urllib.request, "urlopen", fake)
    return fake


def test_available_gated_on_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_TTS_API_KEY", raising=False)
    assert not GoogleTTS().available()
    assert GoogleTTS().render("hallo") == (None, None)
    assert GoogleTTS(api_key="fake").available()


def test_render_request_shape_and_header_auth(fake: _Fake) -> None:
    pcm, rate = GoogleTTS(api_key="fake-secret").render("Guten Abend.", lang="de")
    assert rate == 24000 and pcm is not None and pcm.size == 240
    req = fake.requests[0]
    assert req["url"] == tts_cloud.GOOGLE_TTS_URL  # the key is never in the URL
    assert "fake-secret" not in req["url"]
    assert req["headers"]["x-goog-api-key"] == "fake-secret"
    assert req["body"]["voice"] == {"languageCode": "de-DE", "name": "de-DE-Chirp3-HD-Kore"}
    assert req["body"]["audioConfig"] == {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}
    assert req["body"]["input"] == {"text": "Guten Abend."}


def test_english_voice_and_override(fake: _Fake) -> None:
    GoogleTTS(api_key="fake").render("Good evening.", lang="en")
    GoogleTTS(api_key="fake", voices={"en": "en-GB-Neural2-B", "de": ""}).render("Hi.", lang="en")
    assert fake.requests[0]["body"]["voice"]["name"] == "en-US-Chirp3-HD-Charon"
    assert fake.requests[1]["body"]["voice"] == {"languageCode": "en-GB", "name": "en-GB-Neural2-B"}


def test_language_code_follows_the_voice_not_the_text(fake: _Fake) -> None:
    """An unknown language uses the default voice AND that voice's own language code."""
    GoogleTTS(api_key="fake", default_language="de").render("Bom dia.", lang="pt")
    assert fake.requests[0]["body"]["voice"] == {
        "languageCode": "de-DE",
        "name": "de-DE-Chirp3-HD-Kore",
    }


def test_no_lang_uses_the_default_language(fake: _Fake) -> None:
    GoogleTTS(api_key="fake", default_language="de").render("Das ist ein Satz.")
    assert fake.requests[0]["body"]["voice"]["languageCode"] == "de-DE"


def test_router_detects_once_and_passes_the_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("my_stt_tts.tts.detect_language", lambda _t, _d: "de")
    router = TTSRouter(Config())
    seen: list[str | None] = []

    class _Stub:
        def render(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
            del text
            seen.append(lang)
            return np.full(16, 0.1, dtype=np.float32), 24000

    router._cloud = _Stub()  # pylint: disable=protected-access
    router.synth_pcm("Das ist ein Satz.")
    assert seen == ["de"]


def test_long_text_is_split_under_the_byte_limit(fake: _Fake) -> None:
    text = " ".join(["Größenänderung"] * 400)  # ~6.4 kB of UTF-8, over the 5000-byte cap
    pcm, _ = GoogleTTS(api_key="fake").render(text, lang="de")
    assert len(fake.requests) == 2
    assert all(len(r["body"]["input"]["text"].encode()) <= 4500 for r in fake.requests)
    assert pcm is not None and pcm.size == 2 * 240


def test_split_keeps_every_word_and_cuts_giant_words() -> None:
    text = "kurz " + "x" * 10_000 + " ende"
    parts = split_for_google(text, limit=4500)
    assert all(len(p.encode()) <= 4500 for p in parts)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")
    assert split_for_google("  a b  ") == ["a b"]


def test_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise OSError("down")

    monkeypatch.setattr(tts_cloud.urllib.request, "urlopen", _boom)
    assert GoogleTTS(api_key="fake").render("hi", lang="en") == (None, None)


def test_registry_selects_google_and_falls_back_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_TTS_API_KEY", raising=False)
    assert "google" in TTS_BACKENDS
    cfg = Config(tts_backend="google")
    assert select_tts_backend(cfg) is None  # no key → local Piper / say
    cfg = Config(
        tts_backend="google", google_tts_api_key="fake", google_tts_voices={"de": "de-DE-Neural2-G"}
    )
    backend = select_tts_backend(cfg)
    assert isinstance(backend, GoogleTTS)
    assert backend.voice_for("de") == ("de-DE", "de-DE-Neural2-G")


def test_router_passes_one_language_to_every_streamed_clause() -> None:
    """A German reply split into clauses keeps one voice (no per-clause detection)."""
    router = TTSRouter(Config())
    seen: list[str | None] = []

    class _Stub:
        def render(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
            del text
            seen.append(lang)
            return np.full(16, 0.1, dtype=np.float32), 24000

    router._cloud = _Stub()  # pylint: disable=protected-access
    chunks = list(router.synth_pcm_stream("Ja, klar. OK, Server läuft. Alles gut, danke.", "de"))
    assert chunks and set(seen) == {"de"}


def test_cli_writes_a_wav(fake: _Fake, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    del fake
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "k")
    out = tmp_path / "de.wav"
    assert tts_cloud.main(["-b", "google", "-t", "Guten Abend.", "-l", "de", "-o", str(out)]) == 0
    pcm, rate = tts_cloud._decode_wav_bytes(out.read_bytes())  # pylint: disable=protected-access
    assert rate == 24000 and pcm is not None and pcm.size == 240


def test_cli_without_key_fails_cleanly(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_TTS_API_KEY", raising=False)
    assert tts_cloud.main(["-t", "hi", "-o", str(tmp_path / "x.wav")]) == 2
