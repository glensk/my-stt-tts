"""Real cloud STT adapters behind the :class:`~my_stt_tts.stt.Transcriber` seam (G1).

Currently: **Deepgram** (streaming-capable, low-latency multilingual STT). The
adapter speaks Deepgram's actual prerecorded/streaming HTTP API, so it works
against the real service — but it is **key-gated** (``available()`` is true only
when an API key is configured) and the registry falls back to local parakeet-mlx
when no key is present, so a missing key never hard-fails the pipeline.

The Deepgram SDK is lazy-imported from the optional ``deepgram`` extra; if it is
unavailable we fall back to a dependency-light ``urllib`` POST to the same REST
endpoint (so the adapter still works with just the core package + a key). All
network is faked in tests — no live key is ever required.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

import numpy as np

from .stt import STTResult
from .util import wav_bytes_from_float

log = logging.getLogger("my_stt_tts.stt_cloud")

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def parse_deepgram_response(payload: dict[str, Any]) -> STTResult:
    """Extract ``(transcript, language)`` from a Deepgram JSON response.

    Defensive against the nested ``results.channels[0].alternatives[0]`` shape so a
    partial/odd response degrades to an empty transcript rather than raising. The
    detected language is read from the channel when Deepgram's ``detect_language``
    is on; otherwise it is ``None``.
    """
    try:
        channel = payload["results"]["channels"][0]
        alt = channel["alternatives"][0]
        text = str(alt.get("transcript", "") or "").strip()
    except (KeyError, IndexError, TypeError):
        return STTResult(text="")
    language = None
    with _suppress():
        language = channel.get("detected_language") or payload["results"].get("language")
    return STTResult(text=text, language=language)


class _suppress:  # noqa: N801 — tiny best-effort guard
    def __enter__(self) -> _suppress:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return True


class DeepgramSTT:
    """Deepgram cloud STT adapter (key-gated, graceful fallback) — G1.

    Sends the clip to Deepgram's ``/v1/listen`` endpoint and returns the best
    alternative's transcript (+ detected language when language detection is on).
    Prefers the official ``deepgram-sdk`` (the ``deepgram`` extra) and falls back
    to a plain ``urllib`` POST when the SDK is not installed, so it works either
    way. Never raises into the loop: a transport/parse failure yields empty text.
    """

    def __init__(
        self,
        *,
        model: str = "nova-3",
        api_key: str | None = None,
        language: str | None = None,
        url: str = DEEPGRAM_URL,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        self.language = language
        self.url = url
        self._client: Any = None

    def available(self) -> bool:
        """True when a Deepgram API key is configured (so cloud STT can be used)."""
        return bool(self.api_key)

    def _query(self) -> str:
        params = [f"model={self.model}", "smart_format=true", "punctuate=true"]
        params.append(f"language={self.language}" if self.language else "detect_language=true")
        return self.url + "?" + "&".join(params)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """Transcribe a float32 mono clip via Deepgram; empty text on any failure."""
        if not self.available():
            log.info("DeepgramSTT.transcribe called without an API key; returning empty.")
            return STTResult(text="")
        body = wav_bytes_from_float(audio, sample_rate)
        try:
            payload = self._post_sdk(body) if self._has_sdk() else self._post_http(body)
        except Exception:  # network / SDK failure must not break the loop
            log.warning("Deepgram STT request failed", exc_info=True)
            return STTResult(text="")
        return parse_deepgram_response(payload)

    def _has_sdk(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("deepgram") is not None
        except Exception:
            return False

    def _post_sdk(self, body: bytes) -> dict[str, Any]:
        """POST via the official deepgram-sdk; returns the response as a plain dict."""
        from deepgram import DeepgramClient, PrerecordedOptions

        if self._client is None:
            self._client = DeepgramClient(self.api_key)
        options = PrerecordedOptions(
            model=self.model,
            smart_format=True,
            punctuate=True,
            language=self.language,
            detect_language=self.language is None,
        )
        source = {"buffer": body, "mimetype": "audio/wav"}
        response = self._client.listen.rest.v("1").transcribe_file(source, options)
        to_dict = getattr(response, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return json.loads(str(response))

    def _post_http(self, body: bytes) -> dict[str, Any]:
        """POST via plain urllib (no SDK needed); returns the parsed JSON dict."""
        req = urllib.request.Request(  # noqa: S310 — pinned HTTPS Deepgram endpoint
            self._query(),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — pinned HTTPS
            return dict(json.loads(resp.read().decode("utf-8")))
