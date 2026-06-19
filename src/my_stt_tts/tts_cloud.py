"""Real cloud TTS adapters behind the TTS ``render`` seam (G1).

Two premium neural voices: **ElevenLabs** and **Cartesia** (Sonic). Both expose
the same ``render(text) -> (pcm, sample_rate)`` surface as
:class:`~my_stt_tts.tts.CloudTTS`, so :class:`~my_stt_tts.tts.TTSRouter` can sink
them through the existing cancellable :class:`~my_stt_tts.tts.Playback` /
:class:`~my_stt_tts.tts.StreamingPlayback` paths unchanged.

Both adapters are **key-gated** (``available()`` is true only with an API key) and
the router falls back to local Piper / ``say`` when no key is set — a missing key
never hard-fails the pipeline. SDKs are lazy-imported from optional extras with a
dependency-light ``urllib`` HTTP fallback to the same REST endpoint, so the
adapter works with just the core package + a key. Every render returns
``(None, None)`` on any failure (never raises into the loop). Network is faked in
tests — no live key is ever required.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
import wave
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.tts_cloud")

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-11-13"


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

    def render(self, text: str) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to float32 mono PCM + sample rate; ``(None, None)`` on failure."""
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

    def render(self, text: str) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to float32 mono PCM + sample rate; ``(None, None)`` on failure."""
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


def _json_body(obj: dict[str, Any]) -> bytes:
    import json

    return json.dumps(obj).encode("utf-8")


def _wrap_pcm_wav(raw: bytes, sample_rate: int) -> bytes:
    """Wrap headerless int16-LE PCM bytes in a minimal WAV container for decoding."""
    from .util import wav_bytes_from_int16

    return wav_bytes_from_int16(raw, sample_rate)
