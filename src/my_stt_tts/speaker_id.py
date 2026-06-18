"""Speaker identification: enrollment centroids + cosine match with rejection.

The matching + calibration math is pure and unit-tested. The ECAPA-TDNN embedder
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


def calibrate_threshold(
    centroids: dict[str, np.ndarray],
    labeled: dict[str, list[np.ndarray]],
    *,
    thresholds: list[float] | None = None,
    margin: float = 0.06,
) -> tuple[float, list[tuple[float, float, float]]]:
    """Sweep the absolute match threshold over labeled held-out embeddings.

    ``labeled`` maps each true speaker name (or :data:`UNKNOWN` for impostor /
    guest clips) to a list of test embeddings. Returns
    ``(recommended_threshold, rows)`` where each row is
    ``(threshold, accuracy, impostor_accept_rate)``. The recommendation maximises
    accuracy while preferring thresholds that accept zero impostors — i.e. it errs
    toward rejecting strangers, which is what you want for a home assistant.
    """
    if thresholds is None:
        thresholds = [round(0.30 + 0.02 * i, 2) for i in range(21)]  # 0.30 .. 0.70
    rows: list[tuple[float, float, float]] = []
    for thr in thresholds:
        correct = total = impostors = impostor_accepts = 0
        for true_name, embeddings in labeled.items():
            is_impostor = true_name == UNKNOWN
            for emb in embeddings:
                name, _ = match_speaker(emb, centroids, threshold=thr, margin=margin)
                total += 1
                if is_impostor:
                    impostors += 1
                    if name in {UNKNOWN, AMBIGUOUS}:
                        correct += 1
                    else:
                        impostor_accepts += 1
                elif name == true_name:
                    correct += 1
        accuracy = correct / total if total else 0.0
        far = impostor_accepts / impostors if impostors else 0.0
        rows.append((thr, round(accuracy, 3), round(far, 3)))

    zero_far = [r for r in rows if r[2] == 0.0]
    best = max(zero_far or rows, key=lambda r: r[1])
    return best[0], rows


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
