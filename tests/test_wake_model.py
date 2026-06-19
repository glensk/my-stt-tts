"""Version-tolerant openWakeWord ``Model`` construction + the wake loop's
fail-once-then-stop behaviour.

openWakeWord's ``Model.__init__`` changed shape across releases: modern builds
take ``wakeword_models=[...]`` + ``inference_framework="onnx"``; ``0.4.0`` (the
arm64-pinned version) takes ``wakeword_model_paths=[...]`` and rejects the modern
kwargs (they leak into ``AudioFeatures`` → ``TypeError``). :class:`WakeWord` must
try the modern signature first and fall back to the 0.4.0 one. These tests use a
**fake** ``openwakeword.model`` module so both branches are exercised without the
real wheel (the real-model load is verified manually against wakewords/maziko.onnx).
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from my_stt_tts.wake import WakeUnavailable, WakeWord


def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    """Register a fake ``openwakeword.model`` module exposing ``Model``."""
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


# --------------------------------------------------------------------------- #
# Modern API path: accepts wakeword_models= + inference_framework=             #
# --------------------------------------------------------------------------- #
def test_build_model_uses_modern_api_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class ModernModel:
        def __init__(self, *, wakeword_models: list[str], inference_framework: str) -> None:
            seen["paths"] = wakeword_models
            seen["framework"] = inference_framework

    _install_fake_openwakeword(monkeypatch, ModernModel)
    w = WakeWord("wakewords/maziko.onnx")
    w._ensure()
    assert isinstance(w._model, ModernModel)
    assert seen == {"paths": ["wakewords/maziko.onnx"], "framework": "onnx"}


# --------------------------------------------------------------------------- #
# 0.4.0 API path: modern kwargs raise TypeError → fall back to *_paths         #
# --------------------------------------------------------------------------- #
def test_build_model_falls_back_to_0_4_0_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class LegacyModel:
        # Mirrors openwakeword==0.4.0: no inference_framework, **kwargs would leak
        # into AudioFeatures and raise — so we reject anything but the legacy arg.
        def __init__(self, wakeword_model_paths: list[str] | None = None, **kwargs: Any) -> None:
            if kwargs:  # the modern call passes unknown kwargs → TypeError (like 0.4.0)
                raise TypeError(
                    "AudioFeatures.__init__() got an unexpected keyword argument "
                    f"{next(iter(kwargs))!r}"
                )
            seen["paths"] = wakeword_model_paths

    _install_fake_openwakeword(monkeypatch, LegacyModel)
    w = WakeWord("wakewords/maziko.onnx")
    w._ensure()
    assert isinstance(w._model, LegacyModel)
    # Fell back to the 0.4.0 kwarg name, no inference_framework passed.
    assert seen == {"paths": ["wakewords/maziko.onnx"]}


# --------------------------------------------------------------------------- #
# detect() reads score VALUES (key is the model stem on 0.4.0)                 #
# --------------------------------------------------------------------------- #
def test_detect_reads_score_values_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    class StemKeyModel:
        # 0.4.0 keys the score dict by the model-file STEM (e.g. "maziko").
        def __init__(self, **_kwargs: Any) -> None:
            self.score = 0.0

        def predict(self, _frame: np.ndarray) -> dict[str, float]:
            return {"maziko": self.score}

    _install_fake_openwakeword(monkeypatch, StemKeyModel)
    w = WakeWord("wakewords/maziko.onnx", threshold=0.5)
    frame = np.zeros(1280, dtype=np.float32)
    assert w.detect(frame) is False  # 0.0 < 0.5
    w._model.score = 0.9
    assert w.detect(frame) is True  # value read regardless of the "maziko" key


# --------------------------------------------------------------------------- #
# Unrecoverable failures raise WakeUnavailable ONCE and stay sticky (no spin)  #
# --------------------------------------------------------------------------- #
def test_construction_failure_raises_wakeunavailable_and_is_sticky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    class ExplodingModel:
        def __init__(self, **_kwargs: Any) -> None:
            calls["n"] += 1
            raise RuntimeError("onnx load failed")

    _install_fake_openwakeword(monkeypatch, ExplodingModel)
    w = WakeWord("wakewords/maziko.onnx")
    frame = np.zeros(1280, dtype=np.float32)
    with pytest.raises(WakeUnavailable):
        w.detect(frame)
    # Second call must NOT retry construction (sticky _broken) → no error spin.
    with pytest.raises(WakeUnavailable):
        w.detect(frame)
    assert calls["n"] == 1


def test_predict_failure_raises_wakeunavailable_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"predict": 0}

    class BadPredictModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, _frame: np.ndarray) -> dict[str, float]:
            calls["predict"] += 1
            raise RuntimeError("inference blew up")

    _install_fake_openwakeword(monkeypatch, BadPredictModel)
    w = WakeWord("wakewords/maziko.onnx")
    frame = np.zeros(1280, dtype=np.float32)
    with pytest.raises(WakeUnavailable):
        w.detect(frame)
    with pytest.raises(WakeUnavailable):
        w.detect(frame)
    # Predict ran exactly once; the second detect short-circuited on _broken.
    assert calls["predict"] == 1
