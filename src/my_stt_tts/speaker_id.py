"""Speaker identification: enrollment centroids + cosine match with rejection.

The matching math is pure and unit-tested. The ECAPA-TDNN embedder
(``speechbrain``) is lazy-imported from the ``speaker`` extra and runs in
parallel with STT on the same audio clip, so it adds ~no wall-clock latency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.speaker_id")

UNKNOWN = "unknown"
AMBIGUOUS = "ambiguous"


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0.0 else vec / norm


def match_speaker(
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
    *,
    threshold: float,
    margin: float,
) -> tuple[str, float]:
    """Match an utterance embedding against enrolled per-person centroids.

    Returns ``(name, score)``. Biases toward :data:`UNKNOWN` (absolute gate) and
    :data:`AMBIGUOUS` (top-two within ``margin``) rather than risk misattribution
    — important for children's voices and short commands.
    """
    if not centroids:
        return UNKNOWN, 0.0
    emb = _l2(np.asarray(embedding, dtype=np.float32))
    sims = {
        name: float(np.dot(emb, _l2(np.asarray(c, dtype=np.float32))))
        for name, c in centroids.items()
    }
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    if best < threshold:
        return UNKNOWN, best
    if best - second < margin:
        return AMBIGUOUS, best
    return best_name, best


class EcapaEmbedder:
    """Lazy wrapper around SpeechBrain's ECAPA-TDNN speaker embedder."""

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb") -> None:
        self.source = source
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(source=self.source)

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return a 1-D L2-normalizable speaker embedding for ``audio``."""
        import torch

        self._ensure()
        if sample_rate != 16000:
            log.warning("ECAPA expects 16 kHz; got %d Hz", sample_rate)
        signal = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        emb = self._model.encode_batch(signal)
        return emb.squeeze().detach().cpu().numpy().astype(np.float32)
