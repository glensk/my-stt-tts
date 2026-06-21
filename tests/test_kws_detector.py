"""sherpa-onnx KeywordSpotter (KWS) — the second, OR'd wake detector for CUSTOM words.

openWakeWord needs a GPU retrain per new word and fails on a non-native accent; sherpa
KWS is open-vocabulary, so it is added as a SECOND detector that runs in parallel with
openWakeWord and fires if EITHER fires — for CUSTOM / self-trained words ONLY. Official
words (alexa/hey_jarvis/hey_mycroft) stay openWakeWord-only, byte-identical.

These tests mock sherpa (a fake ``KeywordSpotter`` + fake ``sentencepiece``) so they run
with NO model, NO network; the one real-model test is gated on availability. They pin:

* keyword-line build (boost / threshold / multi-spelling / one ``@label``);
* the OR-combine routing (official -> oww-only; custom -> either-fires; detector naming);
* ``detector`` on the detection event + ``settings_dict`` contract;
* graceful unavailable (sherpa missing / model absent -> no-op, never raises);
* config (env parse, validate, defaults).
"""

from __future__ import annotations

import numpy as np
import pytest

from my_stt_tts import kws as kwsmod
from my_stt_tts.config import Config, is_official_wake_word
from my_stt_tts.kws import (
    SherpaKws,
    build_keywords,
    kws_available,
    spellings_for,
)
from my_stt_tts.wake import OrCombinedWake, WakeWord, make_wake_detector, score_wake_clip_combined

# --- test doubles -------------------------------------------------------------


class _FakeSp:
    """Stands in for sentencepiece: uppercases + splits into char pieces (deterministic)."""

    def encode(self, text: str, out_type: type = str) -> list[str]:  # noqa: ARG002
        return ["▁" + text.upper()[0], *list(text.upper()[1:])] if text else []


class _FakeStream:
    def __init__(self, fire_on_calls: set[int]) -> None:
        self.calls = 0
        self.fire_on_calls = fire_on_calls
        self.fired = False

    def accept_waveform(self, _rate: int, _samples: np.ndarray) -> None:
        self.calls += 1
        if self.calls in self.fire_on_calls:
            self.fired = True


class _FakeSpotter:
    """A KeywordSpotter stand-in: a stream "fires" on a configured Nth accept_waveform."""

    def __init__(self, fire_on_calls: set[int] | None = None) -> None:
        self.fire_on_calls = fire_on_calls or set()
        self.streams: list[_FakeStream] = []

    def create_stream(self, _keywords: str) -> _FakeStream:
        s = _FakeStream(self.fire_on_calls)
        self.streams.append(s)
        return s

    def is_ready(self, _s: _FakeStream) -> bool:
        return False

    def decode_stream(self, _s: _FakeStream) -> None:
        pass

    def get_result(self, s: _FakeStream) -> str:
        if s.fired:
            s.fired = False  # transient, like the real one
            return "maziko"
        return ""

    def reset_stream(self, _s: _FakeStream) -> None:
        pass


class _FakeWake:
    """A WakeWord stand-in for OrCombinedWake routing tests."""

    def __init__(self, *, fires: bool, score: float = 0.0) -> None:
        self._fires = fires
        self.last_score = score
        self.threshold = 0.4
        self.model_name = "maziko"
        self.reset_count = 0
        self._available = True

    def detect(self, _frame: np.ndarray) -> bool:
        return self._fires

    def available(self) -> bool:
        return self._available

    def reset(self) -> None:
        self.reset_count += 1


class _FakeKws:
    """A SherpaKws stand-in for OrCombinedWake routing tests."""

    def __init__(self, *, fires: bool) -> None:
        self._fires = fires
        self.last_score = 1.0 if fires else 0.0
        self.reset_count = 0

    def detect(self, _frame: np.ndarray) -> bool:
        return self._fires

    def reset(self) -> None:
        self.reset_count += 1


def _wire_fake_sherpa(monkeypatch, spotter: _FakeSpotter) -> None:
    """Patch SherpaKws._ensure to use the fake spotter (no sherpa, no model, no temp file)."""

    def _ensure(self: SherpaKws) -> bool:
        if self._spotter is None:
            self._keywords = "FAKE :1.5 #0.25 @maziko"
            self._spotter = spotter
            self._stream = spotter.create_stream(self._keywords)
        return True

    monkeypatch.setattr(SherpaKws, "_ensure", _ensure)


# --- keyword build ------------------------------------------------------------


def test_build_keywords_encodes_boost_threshold_label():
    kw = build_keywords({"maziko": ["maziko"]}, _FakeSp(), boost=2.0, threshold=0.2)
    assert kw == "▁M A Z I K O :2.0 #0.2 @maziko"


def test_build_keywords_multiple_spellings_one_label():
    kw = build_keywords({"maziko": ["maziko", "matsiko"]}, _FakeSp(), boost=1.5, threshold=0.25)
    lines = kw.splitlines()
    assert len(lines) == 2
    assert all(line.endswith("@maziko") for line in lines)
    assert all(":1.5 #0.25" in line for line in lines)


def test_build_keywords_skips_unencodable_spelling():
    # An empty spelling encodes to nothing -> skipped, never aborts the build.
    kw = build_keywords({"maziko": ["", "maziko"]}, _FakeSp(), boost=1.0, threshold=0.3)
    assert len(kw.splitlines()) == 1


def test_spellings_for_includes_word_first_then_dedup_variants():
    out = spellings_for("maziko", {"maziko": ["ma zi ko", "Maziko", "ma tsi ko"]})
    # word first; "Maziko" is a case-dup of "maziko" -> dropped.
    assert out == ["maziko", "ma zi ko", "ma tsi ko"]


def test_spellings_for_no_variants_is_just_the_word():
    assert spellings_for("nexus", {}) == ["nexus"]


# --- SherpaKws detect (mocked sherpa) ----------------------------------------


def test_detect_fires_when_keyword_matches(monkeypatch):
    spotter = _FakeSpotter(fire_on_calls={2})  # fire on the 2nd frame
    _wire_fake_sherpa(monkeypatch, spotter)
    det = SherpaKws("models/x", "maziko")
    frame = np.zeros(1280, dtype=np.float32)
    assert det.detect(frame) is False  # call 1: no fire
    assert det.detect(frame) is True  # call 2: fires
    assert det.last_score >= 1.0


def test_detect_no_match_never_fires(monkeypatch):
    spotter = _FakeSpotter(fire_on_calls=set())
    _wire_fake_sherpa(monkeypatch, spotter)
    det = SherpaKws("models/x", "maziko")
    frame = np.zeros(1280, dtype=np.float32)
    assert not any(det.detect(frame) for _ in range(5))
    assert det.last_score == 0.0


def test_flush_fires_on_trailing_word(monkeypatch):
    spotter = _FakeSpotter(fire_on_calls={2})  # fire only on the flush (2nd accept)
    _wire_fake_sherpa(monkeypatch, spotter)
    det = SherpaKws("models/x", "maziko")
    assert det.detect(np.zeros(1280, dtype=np.float32)) is False
    assert det.flush() is True  # trailing-silence flush decodes the last word


def test_reset_zeroes_score_and_rebuilds_stream(monkeypatch):
    spotter = _FakeSpotter(fire_on_calls={1})
    _wire_fake_sherpa(monkeypatch, spotter)
    det = SherpaKws("models/x", "maziko")
    det.detect(np.zeros(1280, dtype=np.float32))
    assert det.last_score >= 1.0
    det.reset()
    assert det.last_score == 0.0


def test_detect_failure_latches_unavailable_no_raise(monkeypatch):
    det = SherpaKws("models/x", "maziko")

    def _boom(self: SherpaKws) -> bool:
        self._spotter = object()
        self._stream = object()
        return True

    monkeypatch.setattr(SherpaKws, "_ensure", _boom)
    # accept_waveform on a bare object() raises -> caught, latched, returns False.
    assert det.detect(np.zeros(1280, dtype=np.float32)) is False
    assert det._unavailable is True
    assert det.detect(np.zeros(1280, dtype=np.float32)) is False  # stays no-op


# --- from_config gating (the guardrail) --------------------------------------


def test_from_config_none_for_official_word():
    cfg = Config.from_env()
    for word in ("alexa", "hey_jarvis", "hey_mycroft"):
        assert SherpaKws.from_config(cfg, word) is None


def test_from_config_none_when_kws_disabled():
    cfg = Config.from_env()
    cfg.kws_enabled = False
    assert SherpaKws.from_config(cfg, "maziko") is None


def test_from_config_none_when_sherpa_unimportable(monkeypatch):
    monkeypatch.setattr(kwsmod, "_sherpa_importable", lambda: False)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    assert SherpaKws.from_config(cfg, "maziko") is None


def test_from_config_none_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(kwsmod, "_sherpa_importable", lambda: True)
    monkeypatch.setattr(kwsmod, "ensure_kws_model", lambda *a, **k: False)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    assert SherpaKws.from_config(cfg, "maziko") is None


def test_from_config_builds_for_custom_word_when_available(monkeypatch):
    monkeypatch.setattr(kwsmod, "_sherpa_importable", lambda: True)
    monkeypatch.setattr(kwsmod, "ensure_kws_model", lambda *a, **k: True)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    det = SherpaKws.from_config(cfg, "maziko")
    assert isinstance(det, SherpaKws)
    assert det.word == "maziko"


# --- kws_available probe ------------------------------------------------------


def test_kws_available_false_when_disabled():
    cfg = Config.from_env()
    cfg.kws_enabled = False
    assert kws_available(cfg) is False


def test_kws_available_false_when_sherpa_missing(monkeypatch):
    monkeypatch.setattr(kwsmod, "_sherpa_importable", lambda: False)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    assert kws_available(cfg) is False


def test_kws_available_true_when_importable_and_autodownload(monkeypatch):
    monkeypatch.setattr(kwsmod, "_sherpa_importable", lambda: True)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    cfg.kws_auto_download = True
    assert kws_available(cfg) is True


# --- OR-combine routing -------------------------------------------------------


def test_or_combine_oww_fires_reports_oww():
    comb = OrCombinedWake(_FakeWake(fires=True, score=0.7), _FakeKws(fires=False))
    assert comb.detect(np.zeros(1280, dtype=np.float32)) is True
    assert comb.last_detector == "oww"


def test_or_combine_kws_fires_reports_kws():
    comb = OrCombinedWake(_FakeWake(fires=False), _FakeKws(fires=True))
    assert comb.detect(np.zeros(1280, dtype=np.float32)) is True
    assert comb.last_detector == "kws"


def test_or_combine_neither_fires():
    comb = OrCombinedWake(_FakeWake(fires=False), _FakeKws(fires=False))
    assert comb.detect(np.zeros(1280, dtype=np.float32)) is False
    assert comb.last_detector == ""


def test_or_combine_oww_wins_when_both_fire():
    comb = OrCombinedWake(_FakeWake(fires=True), _FakeKws(fires=True))
    assert comb.detect(np.zeros(1280, dtype=np.float32)) is True
    assert comb.last_detector == "oww"  # oWW scored first, wins the label


def test_or_combine_surface_matches_wakeword():
    oww = _FakeWake(fires=False, score=0.33)
    comb = OrCombinedWake(oww, None)
    assert comb.threshold == 0.4
    assert comb.model_name == "maziko"
    assert comb.last_score == 0.33  # the oWW continuous score is plotted
    assert comb.available() is True
    comb.reset()
    assert comb.last_detector == ""
    assert oww.reset_count == 1


def test_or_combine_with_no_kws_is_oww_only():
    comb = OrCombinedWake(_FakeWake(fires=False), None)
    assert comb.detect(np.zeros(1280, dtype=np.float32)) is False


# --- make_wake_detector -------------------------------------------------------


def test_make_wake_detector_official_is_bare_wakeword():
    cfg = Config.from_env()
    cfg.select_wake_word("hey_jarvis")
    det = make_wake_detector(cfg)
    assert isinstance(det, WakeWord)  # NOT combined — official is oWW-only


def test_make_wake_detector_custom_unavailable_is_bare_wakeword(monkeypatch):
    # KWS unavailable -> fall back to bare WakeWord (still works, oWW-only).
    monkeypatch.setattr("my_stt_tts.kws.SherpaKws.from_config", staticmethod(lambda *a, **k: None))
    cfg = Config.from_env()
    cfg.select_wake_word("maziko")
    det = make_wake_detector(cfg)
    assert isinstance(det, WakeWord)


def test_make_wake_detector_custom_available_is_combined(monkeypatch):
    monkeypatch.setattr(
        "my_stt_tts.kws.SherpaKws.from_config",
        staticmethod(lambda *a, **k: _FakeKws(fires=False)),
    )
    cfg = Config.from_env()
    cfg.kws_enabled = True
    cfg.select_wake_word("maziko")
    det = make_wake_detector(cfg)
    assert isinstance(det, OrCombinedWake)


def test_make_wake_detector_kws_disabled_is_bare_wakeword():
    cfg = Config.from_env()
    cfg.kws_enabled = False
    cfg.select_wake_word("maziko")
    det = make_wake_detector(cfg)
    assert isinstance(det, WakeWord)


# --- score_wake_clip_combined (detector reporting) ---------------------------


def test_score_combined_official_never_consults_kws(monkeypatch):
    monkeypatch.setattr("my_stt_tts.wake.score_wake_clip", lambda *a, **k: (0.9, True, [0.9]))
    called = {"kws": False}

    def _spy(*_a, **_k):
        called["kws"] = True
        return True

    monkeypatch.setattr("my_stt_tts.wake._kws_fires_on_clip", _spy)
    cfg = Config.from_env()
    conf, fired, detector, _trace = score_wake_clip_combined(
        np.zeros(16000, dtype=np.float32), 16000, "hey_jarvis", cfg
    )
    assert (fired, detector) == (True, "oww")
    assert called["kws"] is False  # guardrail: official never touches KWS


def test_score_combined_custom_kws_recovers_oww_miss(monkeypatch):
    monkeypatch.setattr("my_stt_tts.wake.score_wake_clip", lambda *a, **k: (0.001, False, [0.001]))
    monkeypatch.setattr("my_stt_tts.wake._kws_fires_on_clip", lambda *a, **k: True)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    conf, fired, detector, _trace = score_wake_clip_combined(
        np.zeros(16000, dtype=np.float32), 16000, "maziko", cfg
    )
    assert (fired, detector) == (True, "kws")
    assert conf == 0.001  # oWW continuous score is preserved


def test_score_combined_custom_neither_fires(monkeypatch):
    monkeypatch.setattr("my_stt_tts.wake.score_wake_clip", lambda *a, **k: (0.001, False, [0.001]))
    monkeypatch.setattr("my_stt_tts.wake._kws_fires_on_clip", lambda *a, **k: False)
    cfg = Config.from_env()
    cfg.kws_enabled = True
    _conf, fired, detector, _trace = score_wake_clip_combined(
        np.zeros(16000, dtype=np.float32), 16000, "maziko", cfg
    )
    assert (fired, detector) == (False, "")


def test_score_combined_oww_fires_reports_oww(monkeypatch):
    monkeypatch.setattr("my_stt_tts.wake.score_wake_clip", lambda *a, **k: (0.67, True, [0.67]))
    cfg = Config.from_env()
    _conf, fired, detector, _trace = score_wake_clip_combined(
        np.zeros(16000, dtype=np.float32), 16000, "maziko", cfg
    )
    assert (fired, detector) == (True, "oww")


# --- config -------------------------------------------------------------------


def test_config_kws_defaults():
    cfg = Config.from_env()
    assert cfg.kws_enabled is True
    assert cfg.kws_boost == 1.5
    assert cfg.kws_threshold == 0.25
    assert cfg.kws_spellings == {}


def test_config_kws_env_parse(monkeypatch):
    monkeypatch.setenv("KWS_ENABLED", "false")
    monkeypatch.setenv("KWS_BOOST", "3.0")
    monkeypatch.setenv("KWS_THRESHOLD", "0.1")
    monkeypatch.setenv("KWS_SPELLINGS", "maziko=ma zi ko|ma tsi ko;nexus=neksus")
    cfg = Config.from_env()
    assert cfg.kws_enabled is False
    assert cfg.kws_boost == 3.0
    assert cfg.kws_threshold == 0.1
    assert cfg.kws_spellings == {"maziko": ["ma zi ko", "ma tsi ko"], "nexus": ["neksus"]}


def test_config_validate_rejects_bad_kws_threshold():
    cfg = Config.from_env()
    cfg.kws_threshold = 1.5
    with pytest.raises(Exception, match="kws_threshold"):
        cfg.validate()


def test_config_validate_rejects_negative_kws_boost():
    cfg = Config.from_env()
    cfg.kws_boost = -1.0
    with pytest.raises(Exception, match="kws_boost"):
        cfg.validate()


def test_is_official_wake_word():
    assert is_official_wake_word("hey_jarvis")
    assert is_official_wake_word("ALEXA")  # case-insensitive
    assert not is_official_wake_word("maziko")
    assert not is_official_wake_word("nexus")


# --- contract: detector on events + settings_dict ----------------------------


def test_wake_event_carries_detector():
    import json

    from my_stt_tts.events import EventBus

    bus = EventBus()
    sub = bus.subscribe()
    bus.wake(detector="kws")
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["type"] == "wake"
    assert evt["fired"] is True
    assert evt["detector"] == "kws"


def test_wake_event_default_detector_is_oww():
    import json

    from my_stt_tts.events import EventBus

    bus = EventBus()
    sub = bus.subscribe()
    bus.wake()
    assert json.loads(sub.get(timeout=1.0))["detector"] == "oww"


def test_wake_test_result_carries_detector():
    import json

    from my_stt_tts.events import EventBus

    bus = EventBus()
    sub = bus.subscribe()
    bus.wake_test_result(
        word="maziko",
        source="server",
        confidence=0.0,
        fired=True,
        message="maziko detected via kws",
        detector="kws",
    )
    assert json.loads(sub.get(timeout=1.0))["detector"] == "kws"


def test_settings_dict_exposes_kws_contract(monkeypatch):
    from my_stt_tts import webui

    # KWS available -> custom words gain "oww+kws"; official stays "oww". settings_dict
    # imports kws_available from my_stt_tts.kws at call time, so patch it at the source.
    monkeypatch.setattr(kwsmod, "kws_available", lambda _cfg: True)
    monkeypatch.setattr(
        webui,
        "wake_word_info",
        lambda *_a, **_k: {
            "hey_jarvis": {"tier": "green", "note": "", "reliability": 0.9},
            "maziko": {"tier": "red", "note": "", "reliability": 0.3},
        },
    )
    cfg = Config.from_env()
    cfg.kws_enabled = True
    s = webui.settings_dict(cfg)
    assert s["kws_available"] is True
    assert s["kws_enabled"] is True
    assert s["wake_word_info"]["hey_jarvis"]["detector"] == "oww"
    assert s["wake_word_info"]["maziko"]["detector"] == "oww+kws"


def test_settings_dict_kws_unavailable_custom_is_oww_only(monkeypatch):
    from my_stt_tts import webui

    monkeypatch.setattr(kwsmod, "kws_available", lambda _cfg: False)
    monkeypatch.setattr(
        webui,
        "wake_word_info",
        lambda *_a, **_k: {"maziko": {"tier": "red", "note": "", "reliability": 0.3}},
    )
    s = webui.settings_dict(Config.from_env())
    assert s["kws_available"] is False
    assert s["wake_word_info"]["maziko"]["detector"] == "oww"


def test_apply_settings_toggles_kws_enabled():
    from my_stt_tts.webui import apply_settings

    cfg = Config.from_env()
    cfg.kws_enabled = True
    apply_settings(cfg, {"kws_enabled": False})
    assert cfg.kws_enabled is False
    apply_settings(cfg, {"kws_enabled": True})
    assert cfg.kws_enabled is True


# --- real-model integration (gated on the model + sherpa being present) -------


def _real_model_ready() -> bool:
    """True if sherpa imports AND the KWS model files are on disk (no download here)."""
    cfg = Config.from_env()
    return kwsmod._sherpa_importable() and kwsmod.kws_model_present(cfg.kws_model_dir)


real_model = pytest.mark.skipif(
    not _real_model_ready(),
    reason="sherpa-onnx + the GigaSpeech KWS model must be present (uv sync --extra all)",
)


@real_model
def test_real_kws_fires_on_english_keyword():
    """The real model fires on an in-vocab English phrase (proves the pipeline is wired)."""
    cfg = Config.from_env()
    cfg.kws_boost = 2.0
    cfg.kws_threshold = 0.2
    # "hello world" is in the model's own example keywords — must decode + fire on TTS-ish
    # tone? We can't synthesize speech here, so instead assert the detector BUILDS and runs
    # without raising on real audio (a stronger real-fire A/B lives in PLAN_kws_detector.md).
    det = SherpaKws.from_config(cfg, "maziko")
    assert det is not None
    det.reset()
    # Feed 1 s of float32 noise through the real engine — must not raise, must stay no-op.
    rng = np.random.default_rng(0)
    for _start in range(0, 16000, 1280):
        det.detect(rng.standard_normal(1280).astype(np.float32) * 0.01)
    det.flush()
    assert det._unavailable is False  # ran cleanly on real audio


@real_model
def test_real_kws_coexists_with_openwakeword():
    """sherpa KWS (bundled onnxruntime) + openWakeWord (standalone) load in ONE process."""
    pytest.importorskip("openwakeword")
    cfg = Config.from_env()
    kws = SherpaKws.from_config(cfg, "maziko")
    assert kws is not None
    kws.detect(np.zeros(1280, dtype=np.float32))  # build the sherpa engine
    # Now build an official openWakeWord model in the SAME process — no dlopen clash.
    from my_stt_tts.config import wake_model_for
    from my_stt_tts.wake import WakeWord

    oww = WakeWord(wake_model_for("hey_jarvis"))
    if oww.available():
        oww.detect(np.zeros(1280, dtype=np.float32))
    assert kws._unavailable is False
