"""Few-shot ENROLLED wake detector (EfficientWord-Net's idea, oWW-embedding adapted).

Enroll a handful of the user's own clips of a word -> mean-pooled openWakeWord embeddings
saved to models/wake_embeddings/<word>.npz; live audio fires on the MAX cosine similarity
to those references, OR'd with openWakeWord + sherpa-KWS for CUSTOM words only. Official
words (alexa/hey_jarvis/hey_mycroft) are NEVER enrolled and stay openWakeWord-only,
byte-identical.

These tests mock openWakeWord's embedding front-end with a deterministic fake (no model, no
network) for the math/wiring paths, and gate the one real-model round-trip on the wheel. They
pin:

* enrollment store/load round-trip (per-clip refs, NOT a centroid) + the >= N-clips gate;
* max-cosine detect (a window matching a reference fires; an orthogonal one does not);
* the rolling-window + patience de-bounce (N consecutive hits required);
* OR-routing: official word -> oWW-only (no fewshot branch); custom word -> fewshot OR'd;
* the detector contract ("fewshot") on the combined clip path + settings_dict;
* threshold tuned against NEGATIVES (a low threshold accepts a negative; a high one rejects);
* config (env parse, validate, defaults) + graceful degradation (oWW absent -> no-op).
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,import-outside-toplevel,redefined-outer-name

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import enrolled_wake as ew
from my_stt_tts.config import Config, is_official_wake_word
from my_stt_tts.enrolled_wake import (
    EnrolledWake,
    embeddings_path,
    enroll_word,
    load_references,
    score_clip_enrolled,
    score_window,
)
from my_stt_tts.wake import OrCombinedWake, WakeWord, make_wake_detector, score_wake_clip_combined

# --------------------------------------------------------------------------- #
# Fake openWakeWord embedding front-end
# --------------------------------------------------------------------------- #
# A deterministic stand-in for openwakeword.utils.AudioFeatures._get_embeddings: it maps a
# PCM window to a fixed 96-d embedding chosen by the DC-offset sign of the audio, so a test
# can craft "this clip embeds to vector A" vs "vector B". One embedding row per ~0.8 s of
# audio (so a >= MIN-window clip yields >= 1 row, mirroring the real model).

_EMBED_A = np.zeros(96, dtype=np.float32)
_EMBED_A[0] = 1.0  # unit vector along axis 0
_EMBED_B = np.zeros(96, dtype=np.float32)
_EMBED_B[1] = 1.0  # orthogonal unit vector along axis 1


class _FakeAudioFeatures:
    """Embeds to _EMBED_A when the clip's mean is >= 0, else _EMBED_B (deterministic)."""

    def _get_embeddings(self, x: np.ndarray, **_kw: Any) -> np.ndarray:  # noqa: ANN401
        arr = np.asarray(x, dtype=np.float32)
        # >= 3 rows for a >= ~0.8 s clip (so it clears wake_verifier._MIN_EMBED_ROWS),
        # scaling with length to mimic the real ~80 ms hop.
        rows = max(3, arr.size // 1600)
        vec = _EMBED_A if float(arr.mean()) >= 0.0 else _EMBED_B
        return np.tile(vec, (rows, 1)).astype(np.float32)


def _install_fake_oww(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make wake_verifier._embedding_model() return the deterministic fake."""
    from my_stt_tts import wake_verifier

    monkeypatch.setattr(wake_verifier, "_embedding_model", lambda: _FakeAudioFeatures())


def _pos_clip(seconds: float = 2.0) -> np.ndarray:
    """A clip that embeds to _EMBED_A (positive: mean >= 0)."""
    return np.full(int(seconds * 16000), 0.2, dtype=np.float32)


def _neg_clip(seconds: float = 2.0) -> np.ndarray:
    """A clip that embeds to _EMBED_B (negative: mean < 0)."""
    return np.full(int(seconds * 16000), -0.2, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Enrollment store / load
# --------------------------------------------------------------------------- #
def test_enroll_stores_per_clip_refs_and_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_fake_oww(monkeypatch)
    clips = [_pos_clip(), _pos_clip(), _pos_clip()]
    out = str(tmp_path / "emb")
    res = enroll_word("maziko", clips=clips, embeddings_dir=out)
    assert res["enrolled"] is True
    assert res["n_refs"] == 3
    assert res["path"] == embeddings_path("maziko", embeddings_dir=out)
    refs = load_references("maziko", embeddings_dir=out)
    assert refs is not None
    # Per-clip refs (3 rows), NOT a single averaged centroid; each L2-normalized to _EMBED_A.
    assert refs.shape == (3, 96)
    np.testing.assert_allclose(refs[0], _EMBED_A, atol=1e-5)


def test_enroll_needs_minimum_clips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_fake_oww(monkeypatch)
    res = enroll_word("maziko", clips=[_pos_clip(), _pos_clip()], embeddings_dir=str(tmp_path))
    assert res["enrolled"] is False
    assert res["n_refs"] == 2
    assert "need >= 3" in res["message"]


def test_load_references_absent_returns_none(tmp_path: Path):
    assert load_references("nope", embeddings_dir=str(tmp_path)) is None


def test_enroll_without_openwakeword_degrades(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from my_stt_tts import wake_verifier

    monkeypatch.setattr(wake_verifier, "_embedding_model", lambda: None)
    res = enroll_word("maziko", clips=[_pos_clip()] * 4, embeddings_dir=str(tmp_path))
    assert res["enrolled"] is False
    assert "wake` extra" in res["message"]


# --------------------------------------------------------------------------- #
# Max-cosine scoring
# --------------------------------------------------------------------------- #
def test_score_window_max_cosine(monkeypatch: pytest.MonkeyPatch):
    _install_fake_oww(monkeypatch)
    refs = np.vstack([_EMBED_A, _EMBED_B])  # two orthogonal references
    feats = _FakeAudioFeatures()
    # A positive clip (-> _EMBED_A) has cosine 1.0 to ref A, 0.0 to ref B -> max 1.0.
    assert score_window(_pos_clip(), refs, features=feats) == pytest.approx(1.0, abs=1e-5)


def test_score_clip_enrolled_fires_on_match(monkeypatch: pytest.MonkeyPatch):
    _install_fake_oww(monkeypatch)
    refs = np.vstack([_EMBED_A])  # enrolled on the positive word
    conf, fired = score_clip_enrolled(_pos_clip(), 16000, refs, threshold=0.9, patience=1)
    assert conf == pytest.approx(1.0, abs=1e-5)
    assert fired is True
    # An orthogonal (negative) clip scores ~0 and does NOT fire.
    conf_n, fired_n = score_clip_enrolled(_neg_clip(), 16000, refs, threshold=0.9, patience=1)
    assert conf_n == pytest.approx(0.0, abs=1e-5)
    assert fired_n is False


# --------------------------------------------------------------------------- #
# Rolling-window patience de-bounce
# --------------------------------------------------------------------------- #
def test_detect_patience_requires_consecutive_windows(monkeypatch: pytest.MonkeyPatch):
    _install_fake_oww(monkeypatch)
    refs = np.vstack([_EMBED_A])
    det = EnrolledWake(refs, "maziko", threshold=0.9, patience=2)
    # Feed a long positive stream in loop-sized frames; with patience 2 it needs two
    # consecutive scored windows above threshold before it fires.
    clip = _pos_clip(4.0)
    fires = [det.detect(clip[s : s + 1280]) for s in range(0, clip.size, 1280)]
    assert any(fires), "patience-2 detector should eventually fire on a sustained match"
    # patience 1 fires earlier than patience 2 (first scored window).
    det1 = EnrolledWake(refs, "maziko", threshold=0.9, patience=1)
    first_p1 = next(
        i
        for i, _ in enumerate(range(0, clip.size, 1280))
        if det1.detect(clip[i * 1280 : i * 1280 + 1280])
    )
    det2 = EnrolledWake(refs, "maziko", threshold=0.9, patience=2)
    fires2 = [det2.detect(clip[s : s + 1280]) for s in range(0, clip.size, 1280)]
    first_p2 = fires2.index(True)
    assert first_p2 >= first_p1


def test_detect_resets_state(monkeypatch: pytest.MonkeyPatch):
    _install_fake_oww(monkeypatch)
    det = EnrolledWake(np.vstack([_EMBED_A]), "maziko", threshold=0.9, patience=1)
    clip = _pos_clip()
    for s in range(0, clip.size, 1280):
        det.detect(clip[s : s + 1280])
    det.reset()
    assert det.last_score == 0.0
    assert det._buffer.size == 0
    assert det._consecutive == 0


# --------------------------------------------------------------------------- #
# Threshold tuned against NEGATIVES
# --------------------------------------------------------------------------- #
def test_threshold_separates_positive_from_negative(monkeypatch: pytest.MonkeyPatch):
    _install_fake_oww(monkeypatch)
    refs = np.vstack([_EMBED_A])
    # Positive scores 1.0, negative scores 0.0. A threshold between them accepts only the
    # positive (the threshold-from-negatives requirement: a too-low threshold leaks the neg).
    _, pos_fire = score_clip_enrolled(_pos_clip(), 16000, refs, threshold=0.5, patience=1)
    _, neg_fire = score_clip_enrolled(_neg_clip(), 16000, refs, threshold=0.5, patience=1)
    assert pos_fire is True and neg_fire is False
    # A threshold BELOW the negative's score would falsely accept it.
    _, neg_leak = score_clip_enrolled(_neg_clip(), 16000, refs, threshold=-0.1, patience=1)
    assert neg_leak is True  # proves the threshold is what gates the negative out


# --------------------------------------------------------------------------- #
# OR-routing — official words stay openWakeWord-only (byte-identical)
# --------------------------------------------------------------------------- #
def test_official_word_never_builds_fewshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_fake_oww(monkeypatch)
    # Even WITH enrolled refs on disk for the (hypothetical) official name, from_config refuses.
    enroll_word("hey_jarvis", clips=[_pos_clip()] * 4, embeddings_dir=str(tmp_path))
    monkeypatch.setattr(ew, "EMBEDDINGS_DIR", str(tmp_path))
    cfg = Config(wake_phrase="hey_jarvis")
    assert is_official_wake_word("hey_jarvis")
    assert EnrolledWake.from_config(cfg) is None


def test_make_wake_detector_official_is_bare_wakeword(monkeypatch: pytest.MonkeyPatch):
    # An official word -> bare WakeWord, no OrCombinedWake (KWS + fewshot both skipped).
    cfg = Config(wake_phrase="hey_jarvis", kws_enabled=True, fewshot_wake_enabled=True)
    det = make_wake_detector(cfg)
    assert isinstance(det, WakeWord)
    assert not isinstance(det, OrCombinedWake)


def test_or_combined_fires_on_fewshot_branch(monkeypatch: pytest.MonkeyPatch):
    _install_fake_oww(monkeypatch)

    class _DeadOww:
        last_score = 0.0
        threshold = 0.5
        model_name = "maziko"

        def detect(self, _frame: np.ndarray) -> bool:
            return False

        def reset(self) -> None:
            pass

        def available(self) -> bool:
            return True

    fewshot = EnrolledWake(np.vstack([_EMBED_A]), "maziko", threshold=0.9, patience=1)
    # _DeadOww duck-types the WakeWord surface OrCombinedWake actually uses (detect/reset/...).
    combined = OrCombinedWake(_DeadOww(), kws=None, fewshot=fewshot)  # type: ignore[arg-type]
    clip = _pos_clip(4.0)
    fired = any(combined.detect(clip[s : s + 1280]) for s in range(0, clip.size, 1280))
    assert fired is True
    assert combined.last_detector == "fewshot"
    combined.reset()
    assert combined.last_detector == ""


def test_combined_clip_path_reports_fewshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_fake_oww(monkeypatch)
    enroll_word("maziko", clips=[_pos_clip()] * 4, embeddings_dir=str(tmp_path))
    # _fewshot_fires_on_clip loads references via ew.load_references with the DEFAULT dir;
    # bind it to the tmp enrollment so this exercises the real combined clip path.
    real_load = ew.load_references
    monkeypatch.setattr(
        ew, "load_references", lambda w, **_k: real_load(w, embeddings_dir=str(tmp_path))
    )
    # oWW has no model for maziko here, so score_wake_clip returns 0 (no fire); the few-shot
    # branch should then fire on a positive clip and the detector string is "fewshot".
    cfg = Config(wake_phrase="maziko", kws_enabled=False, fewshot_wake_enabled=True)
    _conf, fired, detector, _trace = score_wake_clip_combined(_pos_clip(), 16000, "maziko", cfg)
    assert fired is True
    assert detector == "fewshot"


# --------------------------------------------------------------------------- #
# Config — env parse, validate, defaults
# --------------------------------------------------------------------------- #
def test_config_defaults():
    cfg = Config()
    assert cfg.fewshot_wake_enabled is True
    assert cfg.fewshot_threshold == pytest.approx(0.96)
    assert cfg.fewshot_patience == 2


def test_config_env_parse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("FEWSHOT_WAKE_ENABLED=false\nFEWSHOT_THRESHOLD=0.97\nFEWSHOT_PATIENCE=3\n")
    cfg = Config.from_env(env)
    assert cfg.fewshot_wake_enabled is False
    assert cfg.fewshot_threshold == pytest.approx(0.97)
    assert cfg.fewshot_patience == 3


def test_config_validate_rejects_bad_values():
    import pytest as _pytest

    from my_stt_tts.config import ConfigError

    cfg = Config(fewshot_threshold=1.5)
    with _pytest.raises(ConfigError):
        cfg.validate()
    cfg2 = Config(fewshot_patience=0)
    with _pytest.raises(ConfigError):
        cfg2.validate()


def test_settings_dict_exposes_fewshot(monkeypatch: pytest.MonkeyPatch):
    from my_stt_tts.webui import settings_dict

    cfg = Config(wake_phrase="maziko")
    sd = settings_dict(cfg)
    assert sd["fewshot_wake_enabled"] is True
    assert sd["fewshot_threshold"] == pytest.approx(0.96)
    assert sd["fewshot_patience"] == 2


def test_apply_settings_fewshot():
    from my_stt_tts.webui import apply_settings

    cfg = Config()
    apply_settings(
        cfg, {"fewshot_wake_enabled": False, "fewshot_threshold": 0.5, "fewshot_patience": 4}
    )
    assert cfg.fewshot_wake_enabled is False
    assert cfg.fewshot_threshold == pytest.approx(0.5)
    assert cfg.fewshot_patience == 4
    # Clamps: a hand-crafted out-of-range POST is bounded.
    apply_settings(cfg, {"fewshot_threshold": 5.0, "fewshot_patience": 0})
    assert cfg.fewshot_threshold == pytest.approx(1.0)
    assert cfg.fewshot_patience == 1


# --------------------------------------------------------------------------- #
# Graceful degradation when openWakeWord is absent
# --------------------------------------------------------------------------- #
def test_detect_without_openwakeword_is_noop(monkeypatch: pytest.MonkeyPatch):
    from my_stt_tts import wake_verifier

    monkeypatch.setattr(wake_verifier, "_embedding_model", lambda: None)
    det = EnrolledWake(np.vstack([_EMBED_A]), "maziko", threshold=0.9, patience=1)
    clip = _pos_clip(4.0)
    assert not any(det.detect(clip[s : s + 1280]) for s in range(0, clip.size, 1280))


# --------------------------------------------------------------------------- #
# Real-model round-trip (gated on the openWakeWord wheel)
# --------------------------------------------------------------------------- #
def _have_openwakeword() -> bool:
    return importlib.util.find_spec("openwakeword") is not None


@pytest.mark.skipif(not _have_openwakeword(), reason="needs the openWakeWord `wake` extra")
def test_real_embedding_enroll_and_detect(tmp_path: Path):
    """With the real oWW embedding, enroll on 3 copies of a tone and confirm the same tone
    scores high while white noise scores lower — proves the real front-end is wired."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2.0, 32000, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.3 * rng.standard_normal(32000)).astype(np.float32)
    out = str(tmp_path / "emb")
    res = enroll_word("tone", clips=[tone, tone, tone], embeddings_dir=out)
    assert res["enrolled"] is True
    refs = load_references("tone", embeddings_dir=out)
    assert refs is not None and refs.shape[1] == 96
    tone_conf, _ = score_clip_enrolled(tone, 16000, refs, threshold=0.99, patience=1)
    noise_conf, _ = score_clip_enrolled(noise, 16000, refs, threshold=0.99, patience=1)
    assert tone_conf > noise_conf  # the enrolled tone matches itself better than noise
