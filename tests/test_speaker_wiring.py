"""Speaker-ID is wired into the LIVE turn path (G7).

These tests pin the correctness gap that motivated the wiring: the speaker-ID
pieces were unit-tested but NEVER invoked at runtime, so per-speaker memory never
keyed to a real person. Here we drive the actual ``run_turn`` / transport / browser
paths with fake embedder + identifier and assert the resolved name reaches
``brain.set_speaker`` — plus the graceful-skip path when nothing is enrolled /
speechbrain is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from my_stt_tts import __main__ as cli
from my_stt_tts import net_loop
from my_stt_tts.config import Config
from my_stt_tts.events import bus
from my_stt_tts.memory import GUEST_KEY, speaker_key
from my_stt_tts.speaker_id import AMBIGUOUS, UNKNOWN, SpeakerIdentifier
from my_stt_tts.speaker_pipeline import SpeakerPipeline, load_centroids
from my_stt_tts.stt import STTResult

# --- test doubles -------------------------------------------------------------


class _FakeEmbedder:
    """Records the clip it was asked to embed; returns a fixed vector (no model)."""

    def __init__(self, vector: np.ndarray | None = None, *, raises: bool = False) -> None:
        self._vector = vector if vector is not None else np.ones(4, dtype=np.float32)
        self._raises = raises
        self.calls: list[np.ndarray] = []

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:  # noqa: ARG002
        self.calls.append(np.asarray(audio))
        if self._raises:
            raise RuntimeError("model exploded")
        return self._vector


class _FixedIdentifier:
    """A SpeakerIdentifier stand-in that returns a fixed name regardless of input."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.calls = 0

    def identify(self, embedding: np.ndarray) -> str:  # noqa: ARG002
        self.calls += 1
        return self._name


class _RecordingBrain:
    """Records set_speaker + the text streamed (mirrors the real Brain surface)."""

    def __init__(self, parts: list[str] | None = None) -> None:
        self._parts = parts if parts is not None else ["ok"]
        self.speaker_calls: list[str | None] = []
        self.speaker: str | None = None

    def set_speaker(self, name: str | None) -> None:
        self.speaker_calls.append(name)
        self.speaker = name

    def stream(self, text: str):  # noqa: ARG002
        yield from self._parts

    def commit_spoken(self, text: str) -> None:  # pragma: no cover - barge-in only
        pass


class _SilentTTS:
    """A TTSRouter stand-in: speak() is a no-op; synth returns 1 PCM sample."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str, lang=None) -> None:  # noqa: ANN001, ARG002
        self.spoken.append(text)

    def synth_pcm(self, text: str, lang=None):  # noqa: ANN001, ARG002
        self.spoken.append(text)
        return np.full(8, 0.1, dtype=np.float32), 16000


class _StubSTT:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:  # noqa: ARG002
        return STTResult(text=self._text)


class _BusSpy:
    def __init__(self) -> None:
        self._sub = bus.subscribe()

    def drain(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(json.loads(self._sub.get_nowait()))
            except Exception:  # queue.Empty  # noqa: BLE001
                break
        return out

    def close(self) -> None:
        bus.unsubscribe(self._sub)


def _pipeline(name: str, *, raises: bool = False) -> SpeakerPipeline:
    # Duck-typed fakes (no SpeechBrain / no real centroids) stand in for the real
    # identifier + embedder — the pipeline only calls .identify()/.embed().
    return SpeakerPipeline(_FixedIdentifier(name), embedder=_FakeEmbedder(raises=raises))  # type: ignore[arg-type]


# --- SpeakerPipeline.identify (defensive resolution) --------------------------


def test_identify_returns_enrolled_name_and_embeds_clip():
    embedder = _FakeEmbedder()
    pipe = SpeakerPipeline(_FixedIdentifier("alice"), embedder=embedder)  # type: ignore[arg-type]
    clip = np.full(16000, 0.2, dtype=np.float32)
    assert pipe.identify(clip) == "alice"
    assert len(embedder.calls) == 1  # the same clip was embedded


@pytest.mark.parametrize("verdict", [UNKNOWN, AMBIGUOUS])
def test_identify_maps_unknown_and_ambiguous_to_none(verdict: str):
    pipe = SpeakerPipeline(_FixedIdentifier(verdict), embedder=_FakeEmbedder())  # type: ignore[arg-type]
    # None routes to the shared guest bucket in memory (never an enrolled person).
    assert pipe.identify(np.ones(8000, dtype=np.float32)) is None
    assert speaker_key(pipe.identify(np.ones(8000, dtype=np.float32))) == GUEST_KEY


def test_identify_empty_or_none_clip_is_none_without_embedding():
    embedder = _FakeEmbedder()
    pipe = SpeakerPipeline(_FixedIdentifier("bob"), embedder=embedder)
    assert pipe.identify(None) is None
    assert pipe.identify(np.zeros(0, dtype=np.float32)) is None
    assert embedder.calls == []  # nothing embedded on an empty clip


def test_identify_swallows_embed_failure_and_returns_none():
    pipe = SpeakerPipeline(_FixedIdentifier("carol"), embedder=_FakeEmbedder(raises=True))
    # A model blow-up must degrade to guest, never crash the turn.
    assert pipe.identify(np.ones(8000, dtype=np.float32)) is None


# --- SpeakerPipeline.from_config (gating) -------------------------------------


def test_from_config_none_when_disabled(tmp_path):
    cfg = Config(speaker_id_enabled=False, enroll_dir=tmp_path)
    assert SpeakerPipeline.from_config(cfg) is None


def test_from_config_none_when_no_enrollment(tmp_path):
    cfg = Config(speaker_id_enabled=True, enroll_dir=tmp_path)  # empty dir
    assert SpeakerPipeline.from_config(cfg) is None


def test_from_config_none_when_speechbrain_missing(tmp_path, monkeypatch):
    np.save(tmp_path / "alice.npy", np.ones(4, dtype=np.float32))
    cfg = Config(speaker_id_enabled=True, enroll_dir=tmp_path)
    # Simulate speechbrain not installed.
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert SpeakerPipeline.from_config(cfg) is None


def test_from_config_builds_when_enabled_enrolled_and_speechbrain_present(tmp_path, monkeypatch):
    np.save(tmp_path / "alice.npy", np.ones(4, dtype=np.float32))
    np.save(tmp_path / "bob.npy", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    cfg = Config(speaker_id_enabled=True, enroll_dir=tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())  # pretend installed
    pipe = SpeakerPipeline.from_config(cfg)
    assert isinstance(pipe, SpeakerPipeline)
    assert isinstance(pipe.identifier, SpeakerIdentifier)
    assert set(pipe.identifier.centroids) == {"alice", "bob"}


# --- load_centroids -----------------------------------------------------------


def test_load_centroids_reads_npy_and_skips_reserved_and_missing(tmp_path):
    np.save(tmp_path / "alice.npy", np.ones(4, dtype=np.float32))
    np.save(tmp_path / f"{UNKNOWN}.npy", np.ones(4, dtype=np.float32))  # reserved -> skip
    np.save(tmp_path / f"{GUEST_KEY}.npy", np.ones(4, dtype=np.float32))  # reserved -> skip
    (tmp_path / "garbage.npy").write_bytes(b"not a numpy file")  # unreadable -> skipped
    centroids = load_centroids(tmp_path)
    assert set(centroids) == {"alice"}


def test_load_centroids_missing_dir_is_empty(tmp_path):
    assert load_centroids(tmp_path / "does-not-exist") == {}


def test_load_centroids_averages_a_per_person_directory(tmp_path):
    person = tmp_path / "dave"
    person.mkdir()
    np.save(person / "a.npy", np.array([1.0, 0.0], dtype=np.float32))
    np.save(person / "b.npy", np.array([0.0, 1.0], dtype=np.float32))
    centroids = load_centroids(tmp_path)
    assert "dave" in centroids
    np.testing.assert_allclose(centroids["dave"], [0.5, 0.5])


# --- LIVE local turn path (run_turn) ------------------------------------------


def test_run_turn_ptt_identifies_speaker_before_streaming(monkeypatch):
    """The real run_turn must embed the recorded clip and set_speaker(resolved)."""
    cfg = Config()
    brain = _RecordingBrain()
    tts = _SilentTTS()
    clip = np.full(16000, 0.3, dtype=np.float32)

    # Stub the mic capture + chimes so no audio hardware is touched.
    monkeypatch.setattr(cli.audio, "record_push_to_talk", lambda sr, secs: clip)
    monkeypatch.setattr(cli, "_play", lambda *a, **k: None)
    gate = cli.audio.MicGate(0.0)
    stt = _StubSTT("turn on the light")
    embedder = _FakeEmbedder()
    pipe = SpeakerPipeline(_FixedIdentifier("alice"), embedder=embedder)

    spy = _BusSpy()
    try:
        spy.drain()
        cli.run_turn(cfg, brain, tts, gate, stt=stt, speaker_id=pipe)
        events = spy.drain()
    finally:
        spy.close()

    # identify ran on the recorded clip, and the resolved name was set BEFORE stream.
    assert embedder.calls and embedder.calls[0].shape == clip.shape
    assert brain.speaker_calls == ["alice"]
    # And it was surfaced on the bus for the UI.
    speaker_events = [e for e in events if e.get("type") == "speaker"]
    assert speaker_events and speaker_events[0]["name"] == "alice"
    assert speaker_events[0]["known"] is True


def test_run_turn_typed_sets_speaker_none(monkeypatch):
    cfg = Config()
    brain = _RecordingBrain()
    tts = _SilentTTS()
    monkeypatch.setattr(cli, "_play", lambda *a, **k: None)
    gate = cli.audio.MicGate(0.0)
    pipe = SpeakerPipeline(_FixedIdentifier("alice"), embedder=_FakeEmbedder())
    # Typed input has no audio -> guest (None), pipeline is never asked to embed.
    cli.run_turn(cfg, brain, tts, gate, typed_text="hello", speaker_id=pipe)
    assert brain.speaker_calls == [None]
    assert pipe.embedder.calls == []  # type: ignore[attr-defined]


def test_run_turn_graceful_skip_when_no_pipeline(monkeypatch):
    """No pipeline (disabled / no enrollment / no speechbrain) => set_speaker(None), no crash."""
    cfg = Config()
    brain = _RecordingBrain()
    tts = _SilentTTS()
    clip = np.full(8000, 0.3, dtype=np.float32)
    monkeypatch.setattr(cli.audio, "record_push_to_talk", lambda sr, secs: clip)
    monkeypatch.setattr(cli, "_play", lambda *a, **k: None)
    gate = cli.audio.MicGate(0.0)
    stt = _StubSTT("anyone there")
    cli.run_turn(cfg, brain, tts, gate, stt=stt, speaker_id=None)
    assert brain.speaker_calls == [None]  # guest bucket, turn still completed


def test_set_speaker_helper_swallows_pipeline_failure():
    """A pipeline whose embed/identify blows up degrades to guest, never crashes."""
    brain = _RecordingBrain()
    pipe = _pipeline("alice", raises=True)  # embed raises -> identify returns None
    cli._set_speaker(brain, pipe, np.ones(8000, dtype=np.float32))
    assert brain.speaker_calls == [None]


# --- LIVE transport path (respond_over_transport / capture_turn_clip) ---------


class _AllSpeechVad:
    def is_speech(self, frame) -> bool:  # noqa: ANN001, ARG002
        return True


class _NFrameAnalyzer:
    """Ends the turn after ``n`` update() calls (deterministic for tests)."""

    def __init__(self, n: int = 3) -> None:
        self._n = n
        self._seen = 0

    def reset(self) -> None:
        self._seen = 0

    def update(self, frame, is_speech) -> bool:  # noqa: ANN001, ARG002
        self._seen += 1
        return self._seen >= self._n


def test_capture_turn_clip_returns_audio_for_speaker_id():
    from my_stt_tts.transport import encode_frame
    from my_stt_tts.ws_transport import WebSocketTransport

    cfg = Config(sample_rate=16000)
    transport = WebSocketTransport(sample_rate=16000)
    for _ in range(3):
        transport.feed_mic(encode_frame(np.full(512, 0.3, dtype=np.float32)))
    transport.end_mic()
    text, clip = net_loop.capture_turn_clip(
        transport, cfg, _AllSpeechVad(), _NFrameAnalyzer(3), _StubSTT("hi over the wire")
    )
    assert text == "hi over the wire"
    assert clip.size > 0  # the audio is preserved for embedding (not discarded)


def test_respond_over_transport_sets_speaker_from_clip():
    from my_stt_tts.ws_transport import WebSocketTransport

    cfg = Config(sample_rate=16000)
    transport = WebSocketTransport(sample_rate=16000)
    brain = _RecordingBrain(["Hello. "])
    tts = _SilentTTS()
    clip = np.full(16000, 0.25, dtype=np.float32)
    embedder = _FakeEmbedder()
    pipe = SpeakerPipeline(_FixedIdentifier("bob"), embedder=embedder)

    spy = _BusSpy()
    try:
        spy.drain()
        net_loop.respond_over_transport(
            transport, cfg, brain, tts, "hi", speaker_id=pipe, clip=clip
        )
        events = spy.drain()
    finally:
        spy.close()

    assert brain.speaker_calls == ["bob"]  # identified before streaming, over the wire
    assert embedder.calls and embedder.calls[0].shape == clip.shape
    assert any(e.get("type") == "speaker" and e.get("name") == "bob" for e in events)


def test_respond_over_transport_no_pipeline_is_guest():
    from my_stt_tts.ws_transport import WebSocketTransport

    cfg = Config(sample_rate=16000)
    transport = WebSocketTransport(sample_rate=16000)
    brain = _RecordingBrain(["Hi. "])
    net_loop.respond_over_transport(transport, cfg, brain, _SilentTTS(), "hi")
    assert brain.speaker_calls == [None]
