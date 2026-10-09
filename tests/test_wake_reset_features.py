"""Regression: WakeWord.reset() must clear openWakeWord's audio-feature buffers.

openWakeWord 0.4.0's Model.reset() only clears predictions; the preprocessor kept the last
wake word's audio/embeddings, so the next listen session re-fired on pure silence (~1.0).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from my_stt_tts.wake import WakeWord, _reset_oww_features


class _FakePre:
    def __init__(self) -> None:
        self.raw_data_buffer: deque[int] = deque([1, 2, 3], maxlen=10)
        self.melspectrogram_buffer = np.full((90, 32), 7.0)
        self.accumulated_samples = 640
        self.feature_buffer = np.full((120, 96), 5.0)  # "still holds the wake word"
        self.embedding_calls = 0

    def _get_embeddings(self, audio: np.ndarray) -> np.ndarray:
        self.embedding_calls += 1
        assert audio.size == 160000 and not audio.any()
        return np.zeros((41, 96))


class _FakeModel:
    def __init__(self) -> None:
        self.preprocessor = _FakePre()
        self.predictions_cleared = False

    def reset(self) -> None:
        self.predictions_cleared = True


def test_reset_restores_blank_feature_buffers() -> None:
    model = _FakeModel()
    _reset_oww_features(model)
    pre = model.preprocessor
    assert not pre.raw_data_buffer and pre.accumulated_samples == 0
    assert pre.melspectrogram_buffer.shape == (76, 32) and (pre.melspectrogram_buffer == 1).all()
    assert pre.feature_buffer.shape == (41, 96) and not pre.feature_buffer.any()


def test_blank_embeddings_are_computed_once() -> None:
    model = _FakeModel()
    _reset_oww_features(model)
    model.preprocessor.feature_buffer[:] = 9.0  # new audio arrives
    _reset_oww_features(model)
    assert model.preprocessor.embedding_calls == 1
    assert not model.preprocessor.feature_buffer.any()  # the cached block was copied


def test_wakeword_reset_clears_predictions_and_features() -> None:
    word = WakeWord("unused.onnx")
    model = _FakeModel()
    word._models = [model]  # pylint: disable=protected-access
    word.reset()
    assert model.predictions_cleared
    assert not model.preprocessor.feature_buffer.any()
