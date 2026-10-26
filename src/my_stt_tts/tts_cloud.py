"""Real cloud TTS adapters behind the TTS ``render`` seam (G1).

Three neural voices: **ElevenLabs**, **Cartesia** (Sonic) and **Google Cloud
Text-to-Speech** (Chirp 3 HD / Neural2). All expose the same
``render(text) -> (pcm, sample_rate)`` surface as
:class:`~my_stt_tts.tts.CloudTTS`, so :class:`~my_stt_tts.tts.TTSRouter` can sink
them through the existing cancellable :class:`~my_stt_tts.tts.Playback` /
:class:`~my_stt_tts.tts.StreamingPlayback` paths unchanged.

All adapters are **key-gated** (``available()`` is true only with an API key) and
the router falls back to local Piper / ``say`` when no key is set — a missing key
never hard-fails the pipeline. SDKs are lazy-imported from optional extras with a
dependency-light ``urllib`` HTTP fallback to the same REST endpoint, so the
adapter works with just the core package + a key. Every render returns
``(None, None)`` on any failure (never raises into the loop). Network is faked in
tests — no live key is ever required.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.tts_cloud")

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-11-13"
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
GOOGLE_SAMPLE_RATE = 24000  # Chirp 3 HD / Neural2 native rate
GOOGLE_MAX_BYTES = 4500  # the API refuses input over 5000 bytes per request
# Default voices per language (Chirp 3 HD: $30 per 1M characters; a Neural2 name such as
# de-DE-Neural2-G costs $16). Override with GOOGLE_TTS_VOICE_<LANG>.
GOOGLE_VOICES = {
    "de": "de-DE-Chirp3-HD-Kore",
    "en": "en-US-Chirp3-HD-Charon",
    "fr": "fr-FR-Chirp3-HD-Leda",
}


def _decode_wav_bytes(data: bytes) -> tuple[np.ndarray | None, int | None]:
    """Decode a (mono/stereo) PCM WAV byte string to float32 mono PCM + sample rate.

    Returns ``(None, None)`` if the bytes are not a readable WAV — the caller then
    treats it as a failed render and falls back to local TTS. Never raises.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            sr = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error, EOFError):
        return None, None
    if width != 2:
        return None, None
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sr


def _decode_pcm_s16le(data: bytes, sample_rate: int) -> tuple[np.ndarray | None, int | None]:
    """Decode raw little-endian int16 PCM (no header) to float32 mono + sample rate."""
    if not data:
        return None, None
    if len(data) % 2:
        data = data[:-1]
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return pcm, sample_rate


class ElevenLabsTTS:
    """ElevenLabs cloud TTS adapter (key-gated, graceful fallback) — G1.

    Renders text to PCM via ElevenLabs' ``/v1/text-to-speech/{voice_id}`` endpoint.
    Requests WAV (``output_format=pcm_*`` would be raw PCM; we ask for a WAV
    container so the sample rate is self-describing) and reads it back as float32
    mono. Prefers the ``elevenlabs`` SDK (the ``elevenlabs`` extra), falling back
    to a plain ``urllib`` POST. ``eleven_multilingual_v2`` handles DE/FR/EN — a
    strong cloud German voice, the local weak spot.
    """

    def __init__(
        self,
        *,
        voice_id: str = "Rachel",
        model: str = "eleven_multilingual_v2",
        api_key: str | None = None,
        url: str = ELEVENLABS_URL,
    ) -> None:
        self.voice_id = voice_id
        self.model = model
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.url = url
        self._client: Any = None

    def available(self) -> bool:
        """True when an ElevenLabs API key is configured."""
        return bool(self.api_key)

    def render(self, text: str, lang: str | None = None) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to float32 mono PCM + sample rate; ``(None, None)`` on failure."""
        del lang  # the seam passes it; this backend has one voice for every language
        if not self.available() or not text.strip():
            return None, None
        try:
            data = self._render_sdk(text) if self._has_sdk() else self._render_http(text)
        except Exception:  # never break the loop on a cloud hiccup
            log.warning("ElevenLabs TTS render failed", exc_info=True)
            return None, None
        return _decode_wav_bytes(data)

    def _has_sdk(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("elevenlabs") is not None
        except Exception:
            return False

    def _render_sdk(self, text: str) -> bytes:
        from elevenlabs.client import ElevenLabs

        if self._client is None:
            self._client = ElevenLabs(api_key=self.api_key)
        stream = self._client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id=self.model,
            text=text,
            output_format="pcm_22050",
        )
        raw = b"".join(stream) if not isinstance(stream, (bytes, bytearray)) else bytes(stream)
        # SDK pcm_* is headerless raw PCM; wrap it so the decoder reads the rate.
        return _wrap_pcm_wav(raw, 22050)

    def _render_http(self, text: str) -> bytes:
        req = urllib.request.Request(  # noqa: S310 — pinned HTTPS ElevenLabs endpoint
            f"{self.url}/{self.voice_id}?output_format=pcm_22050",
            data=_json_body({"text": text, "model_id": self.model}),
            method="POST",
            headers={"xi-api-key": self.api_key or "", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — pinned HTTPS
            return _wrap_pcm_wav(resp.read(), 22050)


class CartesiaTTS:
    """Cartesia (Sonic) cloud TTS adapter (key-gated, graceful fallback) — G1.

    Renders text to PCM via Cartesia's ``/tts/bytes`` endpoint requesting raw
    little-endian int16 PCM at a known rate (so no container parsing is needed).
    Prefers the ``cartesia`` SDK (the ``cartesia`` extra), falling back to a plain
    ``urllib`` POST. Sonic is a very-low-latency neural voice — a good fit for the
    interruptible, conversational use-case.
    """

    def __init__(
        self,
        *,
        voice_id: str = "",
        model: str = "sonic-2",
        api_key: str | None = None,
        sample_rate: int = 22050,
        url: str = CARTESIA_URL,
    ) -> None:
        self.voice_id = voice_id
        self.model = model
        self.api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        self.sample_rate = sample_rate
        self.url = url
        self._client: Any = None

    def available(self) -> bool:
        """True when a Cartesia API key (and a voice id) is configured."""
        return bool(self.api_key and self.voice_id)

    def render(self, text: str, lang: str | None = None) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to float32 mono PCM + sample rate; ``(None, None)`` on failure."""
        del lang  # the seam passes it; this backend has one voice for every language
        if not self.available() or not text.strip():
            return None, None
        try:
            data = self._render_sdk(text) if self._has_sdk() else self._render_http(text)
        except Exception:  # never break the loop on a cloud hiccup
            log.warning("Cartesia TTS render failed", exc_info=True)
            return None, None
        return _decode_pcm_s16le(data, self.sample_rate)

    def _has_sdk(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("cartesia") is not None
        except Exception:
            return False

    def _output_format(self) -> dict[str, Any]:
        return {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }

    def _render_sdk(self, text: str) -> bytes:
        from cartesia import Cartesia

        if self._client is None:
            self._client = Cartesia(api_key=self.api_key)
        chunks = self._client.tts.bytes(
            model_id=self.model,
            transcript=text,
            voice={"mode": "id", "id": self.voice_id},
            output_format=self._output_format(),
        )
        if isinstance(chunks, (bytes, bytearray)):
            return bytes(chunks)
        return b"".join(chunks)

    def _render_http(self, text: str) -> bytes:
        body = {
            "model_id": self.model,
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "output_format": self._output_format(),
        }
        req = urllib.request.Request(  # noqa: S310 — pinned HTTPS Cartesia endpoint
            self.url,
            data=_json_body(body),
            method="POST",
            headers={
                "X-API-Key": self.api_key or "",
                "Cartesia-Version": CARTESIA_VERSION,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — pinned HTTPS
            return bytes(resp.read())


class GoogleTTS:
    """Google Cloud Text-to-Speech adapter (key-gated, graceful fallback) — tp#1286.

    Renders text via the REST ``v1/text:synthesize`` endpoint with an API key sent in
    the ``X-Goog-Api-Key`` header (never in the URL, so it cannot leak into logs or
    tracebacks). The voice is picked per language (German / English / French, see
    :data:`GOOGLE_VOICES`); :class:`~my_stt_tts.tts.TTSRouter` passes the language it
    detected once per reply, so streamed clauses keep one voice; no ``lang`` means
    ``default_language``. ``LINEAR16`` replies are WAV files, decoded by the shared WAV reader.
    Text longer than :data:`GOOGLE_MAX_BYTES` is split at word boundaries and the
    parts are joined. No SDK: plain ``urllib``, so only the key is needed.
    """

    def __init__(
        self,
        *,
        voices: dict[str, str] | None = None,
        default_language: str = "en",
        api_key: str | None = None,
        sample_rate: int = GOOGLE_SAMPLE_RATE,
        url: str = GOOGLE_TTS_URL,
    ) -> None:
        self.voices = {**GOOGLE_VOICES, **{k: v for k, v in (voices or {}).items() if v}}
        self.default_language = default_language
        self.api_key = api_key or os.environ.get("GOOGLE_TTS_API_KEY")
        self.sample_rate = sample_rate
        self.url = url

    def available(self) -> bool:
        """True when a Google Cloud API key is configured."""
        return bool(self.api_key)

    def voice_for(self, lang: str) -> tuple[str, str]:
        """``(language_code, voice_name)`` for ``lang``; unknown languages use the default."""
        name = self.voices.get(lang) or self.voices.get(self.default_language) or ""
        if not name:
            name = GOOGLE_VOICES["en"]
        return "-".join(name.split("-")[:2]), name

    def render(self, text: str, lang: str | None = None) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to float32 mono PCM + sample rate; ``(None, None)`` on failure."""
        if not self.available() or not text.strip():
            return None, None
        code, name = self.voice_for(lang or self.default_language)
        parts: list[np.ndarray] = []
        rate: int | None = None
        try:
            for chunk in split_for_google(text):
                pcm, rate = _decode_wav_bytes(self._synthesize(chunk, code, name))
                if pcm is None or rate is None:
                    return None, None
                parts.append(pcm)
        except Exception:  # pylint: disable=broad-exception-caught  # never break the loop
            log.warning("Google TTS render failed (voice %s)", name)  # no request echo
            return None, None
        return np.concatenate(parts), rate

    def _synthesize(self, text: str, language_code: str, voice: str) -> bytes:
        body = {
            "input": {"text": text},
            "voice": {"languageCode": language_code, "name": voice},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": self.sample_rate},
        }
        req = urllib.request.Request(  # noqa: S310 — pinned HTTPS Google endpoint
            self.url,
            data=_json_body(body),
            method="POST",
            headers={"X-Goog-Api-Key": self.api_key or "", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — pinned HTTPS
            reply = json.load(resp)
        return base64.b64decode(reply["audioContent"])


def split_for_google(text: str, limit: int = GOOGLE_MAX_BYTES) -> list[str]:
    """Split ``text`` into parts of at most ``limit`` UTF-8 bytes, at word boundaries."""
    words = text.split()
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate.encode("utf-8")) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(word.encode("utf-8")) > limit:  # one giant "word": hard cut
            cut = word.encode("utf-8")[:limit].decode("utf-8", "ignore")
            parts.append(cut)
            word = word[len(cut) :]
        current = word
    if current:
        parts.append(current)
    return parts


def _json_body(obj: dict[str, Any]) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _wrap_pcm_wav(raw: bytes, sample_rate: int) -> bytes:
    """Wrap headerless int16-LE PCM bytes in a minimal WAV container for decoding."""
    from .util import wav_bytes_from_int16

    return wav_bytes_from_int16(raw, sample_rate)


ADAPTERS: dict[str, Any] = {
    "google": GoogleTTS,
    "elevenlabs": ElevenLabsTTS,
    "cartesia": CartesiaTTS,
}


def _file_link(path: Path) -> str:
    """``path`` as an OSC 8 hyperlink (plain text where the terminal ignores OSC 8)."""
    uri = path.resolve().as_uri()
    return f"\x1b]8;;{uri}\x1b\\{path.resolve()}\x1b]8;;\x1b\\"


def main(argv: list[str] | None = None) -> int:
    """Render one text with a cloud voice into a WAV file (plays nothing)."""
    parser = argparse.ArgumentParser(
        prog="python -m my_stt_tts.tts_cloud",
        description="Render TEXT with a cloud TTS backend into a WAV file (nothing is played).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  secret-run -e GOOGLE_TTS_API_KEY=google-tts-api -- \\\n"
            "    uv run python -m my_stt_tts.tts_cloud -b google -t 'Guten Abend.' -l de \\\n"
            "    -o de.wav\n"
            "  uv run python -m my_stt_tts.tts_cloud -b google -t 'Good evening.' -o en.wav\n"
            "\nThe key comes from the environment (GOOGLE_TTS_API_KEY, ELEVENLABS_API_KEY,\n"
            "CARTESIA_API_KEY); voices from GOOGLE_TTS_VOICE_DE/EN/FR."
        ),
    )
    parser.add_argument("-b", "--backend", choices=sorted(ADAPTERS), default="google")
    parser.add_argument("-t", "--text", required=True, help="text to speak")
    parser.add_argument("-l", "--lang", default="en", help="language: de, en or fr (default en)")
    parser.add_argument("-o", "--out", required=True, type=Path, help="WAV file to write")
    args = parser.parse_args(argv)
    if args.backend == "google":
        env_voices = {
            lang: os.environ.get(f"GOOGLE_TTS_VOICE_{lang.upper()}", "") for lang in GOOGLE_VOICES
        }
        adapter = GoogleTTS(voices=env_voices)
    else:
        adapter = ADAPTERS[args.backend]()
    if not adapter.available():
        print(f"❌ no API key for {args.backend} in the environment", file=sys.stderr)
        return 2
    pcm, rate = adapter.render(args.text, lang=args.lang)
    if pcm is None or rate is None or not pcm.size:
        print(f"❌ {args.backend} render failed (see the warning above)", file=sys.stderr)
        return 1
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    args.out.write_bytes(_wrap_pcm_wav(pcm16, rate))
    print(f"✅ {pcm.size / rate:.2f} s at {rate} Hz → {_file_link(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
