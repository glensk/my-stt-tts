"""Within-turn speaker diarization (G7+).

A single captured turn that holds multiple voices + TV is split into per-speaker
segments via sherpa-onnx, each NAMED via the EXISTING SpeechBrain ECAPA path, instead
of collapsing into one unattributed transcript. These tests mock the sherpa diarizer,
the ECAPA embedder, and the bus — no models, no audio, no network — and pin:

* the diarization stage + checksum-verified model fetch (and its graceful fallbacks);
* per-segment ECAPA matching reusing the real :class:`SpeakerIdentifier`;
* the labelled transcript + one ``bus.speaker`` per segment;
* unknown / background-TV rejection (-> dropped for command routing);
* the graceful single-segment fallback (diarizer off / unavailable / short clip);
* gating (off by default; on only with diarize enabled + sherpa + models + centroids).
"""

from __future__ import annotations

import numpy as np
import pytest

from my_stt_tts import diarize as diar
from my_stt_tts.config import Config
from my_stt_tts.diarize import SherpaDiarizer, whole_clip_segment
from my_stt_tts.speaker_id import AMBIGUOUS, UNKNOWN, SpeakerIdentifier
from my_stt_tts.speaker_pipeline import (
    SpeakerPipeline,
    TurnSpeakers,
    resolve_turn_speaker,
    route_speaker,
)

# --- test doubles -------------------------------------------------------------


class _Seg:
    """A sherpa diarization result segment stand-in (.start/.end/.speaker)."""

    def __init__(self, start: float, end: float, speaker: int) -> None:
        self.start = start
        self.end = end
        self.speaker = speaker


class _FakeResult:
    def __init__(self, segs: list[_Seg]) -> None:
        self._segs = segs

    def sort_by_start_time(self) -> list[_Seg]:
        return sorted(self._segs, key=lambda s: s.start)


class _FakeEngine:
    """Stands in for sherpa_onnx.OfflineSpeakerDiarization; returns fixed segments."""

    def __init__(self, segs: list[_Seg], *, raises: bool = False) -> None:
        self._segs = segs
        self._raises = raises
        self.calls: list[np.ndarray] = []

    def process(self, audio: np.ndarray) -> _FakeResult:
        self.calls.append(np.asarray(audio))
        if self._raises:
            raise RuntimeError("sherpa exploded")
        return _FakeResult(self._segs)


def _diarizer_with(engine: _FakeEngine) -> SherpaDiarizer:
    d = SherpaDiarizer("seg.onnx", "emb.onnx", min_segment_s=0.4, sample_rate=16000)
    d._engine = engine  # inject; skip the real sherpa load
    return d


class _PerWindowEmbedder:
    """Returns a distinct embedding per audio slice based on its mean amplitude.

    Lets the real SpeakerIdentifier map a slice to a centroid by its amplitude, so a
    multi-segment clip can be assembled to land each segment on a different person /
    unknown deterministically.
    """

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:  # noqa: ARG002
        self.calls.append(np.asarray(audio))
        m = float(np.mean(audio)) if audio.size else 0.0
        # amplitude ~0.9 -> "alice" axis, ~0.5 -> "bob" axis, else a 3rd axis that is
        # orthogonal to both enrolled centroids -> scores 0 < threshold -> unknown (TV).
        if m > 0.8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if 0.4 < m <= 0.6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


class _FixedIdentifier:
    def __init__(self, name: str) -> None:
        self._name = name
        self.calls = 0

    def identify(self, embedding: np.ndarray) -> str:  # noqa: ARG002
        self.calls += 1
        return self._name


class _BusSpy:
    """Captures bus.speaker / bus.transcript / bus.log calls (duck-typed bus)."""

    def __init__(self) -> None:
        self.speakers: list[str | None] = []
        self.transcripts: list[dict] = []
        self.logs: list[str] = []
        self.states: list[str] = []

    def speaker(self, name: str | None) -> None:
        self.speakers.append(name)

    def transcript(
        self, text: str, *, partial: bool = False, source: str = "", speaker=None
    ) -> None:  # noqa: ANN001
        self.transcripts.append({"text": text, "source": source, "speaker": speaker})

    def log(self, message: str, level: str = "info") -> None:  # noqa: ARG002
        self.logs.append(message)

    def state(self, state: str, detail: str = "") -> None:  # noqa: ARG002
        self.states.append(state)


# --- whole_clip_segment / fallback shape --------------------------------------


def test_whole_clip_segment_spans_the_clip():
    assert whole_clip_segment(np.ones(16000, dtype=np.float32), 16000) == [(0.0, 1.0, 0)]


def test_whole_clip_segment_empty_is_empty():
    assert whole_clip_segment(np.zeros(0, dtype=np.float32), 16000) == []


# --- SherpaDiarizer.segments (mocked sherpa engine) ---------------------------


def test_segments_splits_into_local_speakers_sorted():
    engine = _FakeEngine([_Seg(2.0, 3.0, 1), _Seg(0.0, 1.5, 0)])
    d = _diarizer_with(engine)
    clip = np.full(16000 * 3, 0.3, dtype=np.float32)
    segs = d.segments(clip)
    assert segs == [(0.0, 1.5, 0), (2.0, 3.0, 1)]  # sorted by start
    assert engine.calls and engine.calls[0].dtype == np.float32


def test_segments_drops_sub_min_duration_segments():
    engine = _FakeEngine([_Seg(0.0, 1.0, 0), _Seg(1.0, 1.2, 1)])  # 0.2 s < min 0.4
    d = _diarizer_with(engine)
    segs = d.segments(np.full(16000 * 2, 0.3, dtype=np.float32))
    assert segs == [(0.0, 1.0, 0)]


def test_segments_short_clip_is_single_segment_without_engine():
    engine = _FakeEngine([_Seg(0.0, 0.5, 0), _Seg(0.5, 1.0, 1)])
    d = _diarizer_with(engine)
    clip = np.full(8000, 0.3, dtype=np.float32)  # 0.5 s < 1 s min for diarization
    segs = d.segments(clip)
    assert segs == [(0.0, 0.5, 0)]  # whole-clip fallback
    assert engine.calls == []  # the engine was never consulted on a short clip


def test_segments_engine_failure_falls_back_to_single_segment():
    engine = _FakeEngine([], raises=True)
    d = _diarizer_with(engine)
    clip = np.full(16000 * 2, 0.3, dtype=np.float32)
    segs = d.segments(clip)
    assert segs == [(0.0, 2.0, 0)]  # never crashes; one whole-clip segment


def test_segments_no_segments_found_falls_back_to_single_segment():
    engine = _FakeEngine([])  # sherpa returned nothing
    d = _diarizer_with(engine)
    clip = np.full(16000 * 2, 0.3, dtype=np.float32)
    assert d.segments(clip) == [(0.0, 2.0, 0)]


def test_segments_empty_or_none_clip_is_empty():
    d = _diarizer_with(_FakeEngine([]))
    assert d.segments(None) == []
    assert d.segments(np.zeros(0, dtype=np.float32)) == []


def test_segments_unavailable_engine_falls_back(monkeypatch):
    # _ensure_engine returns None (sherpa not loadable) -> whole-clip fallback.
    d = SherpaDiarizer("seg.onnx", "emb.onnx", sample_rate=16000)
    monkeypatch.setattr(d, "_ensure_engine", lambda: None)
    clip = np.full(16000 * 2, 0.3, dtype=np.float32)
    assert d.segments(clip) == [(0.0, 2.0, 0)]


# --- SpeakerPipeline.identify_segments (per-segment ECAPA naming) -------------


def _centroids() -> dict[str, np.ndarray]:
    # 3-D so the unknown/TV embedding can sit on a third axis orthogonal to both.
    return {
        "alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }


def test_identify_segments_names_each_via_existing_ecapa_path():
    # alice speaks 0-1 s (amp 0.9), bob 1-2 s (amp 0.5), TV 2-3 s (amp 0.1 -> unknown).
    clip = np.concatenate(
        [
            np.full(16000, 0.9, dtype=np.float32),
            np.full(16000, 0.5, dtype=np.float32),
            np.full(16000, 0.1, dtype=np.float32),
        ]
    )
    engine = _FakeEngine([_Seg(0.0, 1.0, 0), _Seg(1.0, 2.0, 1), _Seg(2.0, 3.0, 2)])
    identifier = SpeakerIdentifier(_centroids(), threshold=0.5, margin=0.0)
    pipe = SpeakerPipeline(
        identifier, embedder=_PerWindowEmbedder(), diarizer=_diarizer_with(engine)
    )
    named = pipe.identify_segments(clip)
    assert [n for _s, _e, n in named] == ["alice", "bob", None]  # TV -> unknown -> None
    assert pipe.diarize_active is True


def test_identify_segments_unknown_and_ambiguous_become_none():
    clip = np.full(16000 * 2, 0.3, dtype=np.float32)
    engine = _FakeEngine([_Seg(0.0, 1.0, 0), _Seg(1.0, 2.0, 1)])
    for verdict in (UNKNOWN, AMBIGUOUS):
        pipe = SpeakerPipeline(
            _FixedIdentifier(verdict),  # type: ignore[arg-type]
            embedder=_PerWindowEmbedder(),
            diarizer=_diarizer_with(engine),
        )
        named = pipe.identify_segments(clip)
        assert [n for _s, _e, n in named] == [None, None]


def test_identify_segments_no_diarizer_is_single_whole_clip_segment():
    # No diarizer wired -> one segment spanning the clip, named by the legacy path.
    pipe = SpeakerPipeline(_FixedIdentifier("alice"), embedder=_PerWindowEmbedder())  # type: ignore[arg-type]
    clip = np.full(16000, 0.9, dtype=np.float32)
    named = pipe.identify_segments(clip)
    assert named == [(0.0, 1.0, "alice")]
    assert pipe.diarize_active is False


def test_identify_segments_empty_clip_is_empty():
    pipe = SpeakerPipeline(_FixedIdentifier("alice"), embedder=_PerWindowEmbedder())  # type: ignore[arg-type]
    assert pipe.identify_segments(None) == []
    assert pipe.identify_segments(np.zeros(0, dtype=np.float32)) == []


# --- route_speaker (command-routing policy) -----------------------------------


def test_route_speaker_picks_the_longest_known_speaker():
    segs = [(0.0, 1.0, "alice"), (1.0, 4.0, "bob"), (4.0, 5.0, None)]
    routed, drop = route_speaker(segs)
    assert routed == "bob"  # bob spoke longest (3 s)
    assert drop is False


def test_route_speaker_drops_when_all_unknown():
    segs = [(0.0, 1.0, None), (1.0, 2.0, None)]
    routed, drop = route_speaker(segs)
    assert routed is None
    assert drop is True  # only TV / unknown -> drop the command


def test_route_speaker_empty_is_guest_not_dropped():
    assert route_speaker([]) == (None, False)


# --- resolve_turn_speaker (bus wiring) ----------------------------------------


def test_resolve_turn_speaker_no_pipeline_is_guest():
    bus = _BusSpy()
    out = resolve_turn_speaker(None, np.ones(8000, dtype=np.float32), bus=bus)
    assert out == TurnSpeakers(routed=None, should_drop=False, diarized=False)
    assert bus.speakers == [None]


def test_resolve_turn_speaker_single_speaker_emits_one_speaker():
    bus = _BusSpy()
    pipe = SpeakerPipeline(_FixedIdentifier("alice"), embedder=_PerWindowEmbedder())  # type: ignore[arg-type]
    out = resolve_turn_speaker(pipe, np.full(16000, 0.9, dtype=np.float32), bus=bus)
    assert out.routed == "alice"
    assert out.diarized is False
    assert bus.speakers == ["alice"]


def test_resolve_turn_speaker_multi_emits_per_segment_speaker_and_routes():
    bus = _BusSpy()
    clip = np.concatenate(
        [
            np.full(16000, 0.9, dtype=np.float32),  # alice
            np.full(32000, 0.5, dtype=np.float32),  # bob (longer)
            np.full(16000, 0.1, dtype=np.float32),  # TV -> unknown
        ]
    )
    engine = _FakeEngine([_Seg(0.0, 1.0, 0), _Seg(1.0, 3.0, 1), _Seg(3.0, 4.0, 2)])
    pipe = SpeakerPipeline(
        SpeakerIdentifier(_centroids(), threshold=0.5, margin=0.0),
        embedder=_PerWindowEmbedder(),
        diarizer=_diarizer_with(engine),
    )
    out = resolve_turn_speaker(pipe, clip, bus=bus)
    assert out.diarized is True
    assert out.routed == "bob"  # bob spoke longest among the known segments
    assert out.should_drop is False
    # one speaker event per segment (alice, bob, unknown=None) for the UI
    assert bus.speakers == ["alice", "bob", None]


def test_resolve_turn_speaker_all_unknown_drops():
    bus = _BusSpy()
    clip = np.full(16000 * 2, 0.1, dtype=np.float32)  # all TV -> unknown
    engine = _FakeEngine([_Seg(0.0, 1.0, 0), _Seg(1.0, 2.0, 1)])
    pipe = SpeakerPipeline(
        SpeakerIdentifier(_centroids(), threshold=0.5, margin=0.0),
        embedder=_PerWindowEmbedder(),
        diarizer=_diarizer_with(engine),
    )
    out = resolve_turn_speaker(pipe, clip, bus=bus)
    assert out.should_drop is True
    assert out.routed is None
    assert bus.speakers == [None, None]


# --- TurnSpeakers.emit_transcripts (labelled transcript) ----------------------


def test_emit_transcripts_one_per_known_segment_with_speaker_label():
    bus = _BusSpy()
    ts = TurnSpeakers(
        diarized=True,
        segments=[(0.0, 1.0, "alice"), (1.0, 2.0, "bob"), (2.0, 3.0, None)],
    )
    owned = ts.emit_transcripts("turn on the light", bus=bus)
    assert owned is True
    # one labelled transcript per KNOWN segment; the unknown one is skipped
    assert [t["speaker"] for t in bus.transcripts] == ["alice", "bob"]
    assert all(t["text"] == "turn on the light" for t in bus.transcripts)


def test_emit_transcripts_single_speaker_does_not_own_transcript():
    bus = _BusSpy()
    ts = TurnSpeakers(routed="alice", diarized=False)
    assert ts.emit_transcripts("hello", bus=bus) is False
    assert bus.transcripts == []  # caller emits the plain transcript instead


# --- gating: SpeakerPipeline.from_config wires a diarizer only when usable -----


def test_from_config_no_diarizer_when_diarize_disabled(tmp_path, monkeypatch):
    np.save(tmp_path / "alice.npy", np.ones(4, dtype=np.float32))
    cfg = Config(speaker_id_enabled=True, speaker_diarize_enabled=False, enroll_dir=tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    pipe = SpeakerPipeline.from_config(cfg)
    assert isinstance(pipe, SpeakerPipeline)
    assert pipe.diarizer is None  # speaker ID on, diarization off


def test_from_config_no_diarizer_when_sherpa_missing(tmp_path, monkeypatch):
    np.save(tmp_path / "alice.npy", np.ones(4, dtype=np.float32))
    cfg = Config(speaker_id_enabled=True, speaker_diarize_enabled=True, enroll_dir=tmp_path)

    # speechbrain "present", sherpa_onnx "absent".
    def _find_spec(name: str):
        return None if name == "sherpa_onnx" else object()

    monkeypatch.setattr("importlib.util.find_spec", _find_spec)
    pipe = SpeakerPipeline.from_config(cfg)
    assert isinstance(pipe, SpeakerPipeline)
    assert pipe.diarizer is None  # diarization wanted but sherpa not installed


def test_from_config_wires_diarizer_when_enabled_and_models_present(tmp_path, monkeypatch):
    np.save(tmp_path / "alice.npy", np.ones(4, dtype=np.float32))
    cfg = Config(speaker_id_enabled=True, speaker_diarize_enabled=True, enroll_dir=tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())  # all present
    monkeypatch.setattr(diar, "_sherpa_importable", lambda: True)  # sherpa loads
    # Pretend the models are already on disk + checksum-verified.
    monkeypatch.setattr(diar, "ensure_segmentation_model", lambda *a, **k: True)
    monkeypatch.setattr(diar, "ensure_embedding_model", lambda *a, **k: True)
    pipe = SpeakerPipeline.from_config(cfg)
    assert isinstance(pipe, SpeakerPipeline)
    assert isinstance(pipe.diarizer, SherpaDiarizer)
    assert pipe.diarize_active is True


def test_diarizer_from_config_none_when_sherpa_unimportable(tmp_path, monkeypatch):
    # find_spec lies (package dir exists) but the native extension fails to dlopen.
    cfg = Config(speaker_id_enabled=True, speaker_diarize_enabled=True, enroll_dir=tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(diar, "_sherpa_importable", lambda: False)
    # Models must NOT be fetched for an unusable install.
    monkeypatch.setattr(
        diar, "ensure_segmentation_model", lambda *a, **k: pytest.fail("should not download")
    )
    assert SherpaDiarizer.from_config(cfg) is None  # graceful: single-speaker


def test_diarizer_from_config_none_when_models_unavailable(tmp_path, monkeypatch):
    cfg = Config(speaker_id_enabled=True, speaker_diarize_enabled=True, enroll_dir=tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(diar, "_sherpa_importable", lambda: True)
    monkeypatch.setattr(diar, "ensure_segmentation_model", lambda *a, **k: False)  # missing
    monkeypatch.setattr(diar, "ensure_embedding_model", lambda *a, **k: True)
    assert SherpaDiarizer.from_config(cfg) is None  # graceful: single-speaker


def test_diarizer_from_config_none_when_disabled():
    assert SherpaDiarizer.from_config(Config(speaker_diarize_enabled=False)) is None


# --- checksum-verified model fetch --------------------------------------------


def test_ensure_embedding_model_rejects_checksum_mismatch(tmp_path, monkeypatch):
    dest = tmp_path / "emb.onnx"
    monkeypatch.setattr(diar, "_http_get", lambda url, timeout=120: b"not the real model")
    ok = diar.ensure_embedding_model(
        str(dest), "https://example/emb.onnx", auto_download=True, expected_sha256="deadbeef" * 8
    )
    assert ok is False
    assert not dest.exists()  # a wrong-hash download is discarded, never installed


def test_ensure_embedding_model_accepts_matching_checksum(tmp_path, monkeypatch):
    import hashlib

    payload = b"a tiny fake onnx blob"
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "emb.onnx"
    monkeypatch.setattr(diar, "_http_get", lambda url, timeout=120: payload)
    ok = diar.ensure_embedding_model(
        str(dest), "https://example/emb.onnx", auto_download=True, expected_sha256=digest
    )
    assert ok is True
    assert dest.read_bytes() == payload


def test_ensure_segmentation_model_no_download_when_off_and_absent(tmp_path):
    dest = tmp_path / "seg" / "model.onnx"
    ok = diar.ensure_segmentation_model(
        str(dest), "https://example/seg.tar.bz2", auto_download=False
    )
    assert ok is False  # absent + auto_download off -> not fetched


def test_ensure_segmentation_model_existing_valid_file_is_kept(tmp_path):
    import hashlib

    dest = tmp_path / "model.onnx"
    dest.write_bytes(b"already here")
    digest = hashlib.sha256(b"already here").hexdigest()
    # No _http_get patched: if it tried to download, urlopen would fail the test.
    ok = diar.ensure_segmentation_model(
        str(dest), "https://example/seg.tar.bz2", auto_download=True, expected_sha256=digest
    )
    assert ok is True


def test_ensure_segmentation_model_extracts_inner_onnx_from_tar_bz2(tmp_path, monkeypatch):
    import bz2
    import hashlib
    import io
    import tarfile

    inner = b"the pyannote segmentation model bytes"
    # Build a .tar.bz2 with <dir>/model.onnx, exactly like the sherpa release.
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        info = tarfile.TarInfo(name="sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
        info.size = len(inner)
        tar.addfile(info, io.BytesIO(inner))
    archive = bz2.compress(tar_buf.getvalue())

    dest = tmp_path / "model.onnx"
    monkeypatch.setattr(diar, "_http_get", lambda url, timeout=120: archive)
    digest = hashlib.sha256(inner).hexdigest()
    ok = diar.ensure_segmentation_model(
        str(dest), "https://example/seg.tar.bz2", auto_download=True, expected_sha256=digest
    )
    assert ok is True
    assert dest.read_bytes() == inner  # the inner model.onnx was extracted + installed


# --- config gating / validation -----------------------------------------------


def test_speaker_diarize_default_off():
    assert Config().speaker_diarize_enabled is False


def test_speaker_diarize_env_opt_in(monkeypatch):
    monkeypatch.setenv("SPEAKER_DIARIZE", "true")
    monkeypatch.setenv("DIARIZE_NUM_SPEAKERS", "3")
    cfg = Config.from_env()
    assert cfg.speaker_diarize_enabled is True
    assert cfg.diarize_num_speakers == 3


@pytest.mark.parametrize("num", [0, -2])
def test_validate_rejects_bad_num_speakers(num):
    from my_stt_tts.config import ConfigError

    cfg = Config(diarize_num_speakers=num, anthropic_api_key="sk-test")
    with pytest.raises(ConfigError, match="diarize_num_speakers"):
        cfg.validate()


def test_validate_accepts_auto_and_fixed_num_speakers():
    Config(diarize_num_speakers=-1, anthropic_api_key="sk-test").validate()  # auto
    Config(diarize_num_speakers=4, anthropic_api_key="sk-test").validate()  # fixed N
