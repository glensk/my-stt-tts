"""Interference capture, expiry and labelling (tp#1235): fake audio, scorer and clock only."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import interference as itf
from my_stt_tts.bridge import Authoriser, BridgeController, Transcript
from my_stt_tts.eleven_voice import VoiceSession
from my_stt_tts.turns import Ambiguous, Utterance

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "label_clips_under_test", _SCRIPTS / "label_voice_clips.py"
)
assert _spec and _spec.loader
lbl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = lbl
_spec.loader.exec_module(lbl)

NOW = datetime(2026, 10, 25, 20, 0, 0).astimezone()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def utt(start: float, seconds: float = 1.5, level: float = 0.5, seq: int = 1) -> Utterance:
    pcm = np.full(int(seconds * 16000), level, dtype=np.float32)
    return Utterance(seq, start, start + seconds, pcm)


class FakeScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del name, timeout
        self.calls += 1
        return round(float(np.max(np.abs(audio))), 2)

    def profile_kind(self, name: str) -> str:
        del name
        return "call"


@dataclass
class Rig:
    store: itf.ClipStore
    capture: itf.CallCapture
    clock: list[float]

    def transcript(self, seq: int, *utts: Utterance, at: float | None = None) -> Transcript:
        binding: Any = utts[0] if len(utts) == 1 else (Ambiguous(utts) if utts else None)
        t = Transcript(seq, "words", self.clock[0] if at is None else at, binding=binding)
        self.capture.on_transcript(t)
        return t

    def decisions(self) -> list[str]:
        return [c.decision for c in self.store.clips()]


def make_rig(
    tmp_path: Path, *, sample: float = 0.2, rand: float = 0.5, enabled: bool = True
) -> Rig:
    clock = [100.0]
    store = itf.ClipStore(tmp_path / "interference")
    capture = itf.CallCapture(
        store,
        itf.Config(enabled=enabled, sample=sample, root=store.root),
        scorer=FakeScorer(),
        authorized="albert",
        rand=lambda: rand,
        clock=lambda: clock[0],
        background=False,
    )
    capture.begin_call()
    return Rig(store, capture, clock)


# -- the clip folder -------------------------------------------------------------------------
def test_save_writes_private_files_without_any_text(tmp_path: Path) -> None:
    store = itf.ClipStore(tmp_path / "i")
    clip = store.save(
        np.full(16000, 0.4, dtype=np.float32),
        decision="skipped",
        score=0.312345,
        profile="call",
        now=NOW,
    )
    assert clip.wav.parent == tmp_path / "i" / "2026-10"
    assert clip.id.startswith("20261025-200000-skipped-")
    assert _mode(clip.wav) == 0o600 and _mode(clip.sidecar) == 0o600
    assert _mode(clip.wav.parent) == 0o700
    meta = json.loads(clip.sidecar.read_text(encoding="utf-8"))
    assert meta["decision"] == "skipped" and meta["score"] == 0.3123
    assert meta["duration"] == 1.0 and meta["label"] is None and meta["profile"] == "call"
    assert "text" not in meta and "transcript" not in meta


def test_find_label_and_delete(tmp_path: Path) -> None:
    store = itf.ClipStore(tmp_path)
    clip = store.save(np.zeros(8000), decision="verified", score=0.6, profile=None, now=NOW)
    assert store.find(clip.id[:20]) is not None and store.find("nope") is None
    store.set_label(clip, "other", now=NOW)
    again = store.find(clip.id)
    assert again is not None and again.label == "other" and again.meta["labelled"]
    with pytest.raises(ValueError):
        store.set_label(clip, "susi")
    store.set_label(clip, None)
    assert store.clips()[0].label is None
    store.delete(clip)
    assert not store.clips() and not (tmp_path / "2026-10").exists()


def test_find_refuses_an_ambiguous_prefix(tmp_path: Path) -> None:
    store = itf.ClipStore(tmp_path)
    store.save(np.zeros(8000), decision="skipped", score=None, profile=None, now=NOW)
    store.save(np.zeros(8000), decision="skipped", score=None, profile=None, now=NOW)
    assert store.find("20261025") is None


def test_expire_keeps_labelled_and_recent_clips(tmp_path: Path) -> None:
    store = itf.ClipStore(tmp_path)
    old = store.save(
        np.zeros(8000), decision="skipped", score=None, profile=None, now=NOW - timedelta(days=91)
    )
    kept_label = store.save(
        np.zeros(8000),
        decision="other_speaker",
        score=0.2,
        profile=None,
        now=NOW - timedelta(days=200),
    )
    store.set_label(kept_label, "other")
    recent = store.save(
        np.zeros(8000), decision="skipped", score=None, profile=None, now=NOW - timedelta(days=10)
    )
    assert store.expire(90, now=NOW) == 1
    ids = {c.id for c in store.clips()}
    assert ids == {kept_label.id, recent.id} and old.id not in ids


def test_expire_falls_back_to_the_file_time(tmp_path: Path) -> None:
    store = itf.ClipStore(tmp_path)
    clip = store.save(np.zeros(8000), decision="skipped", score=None, profile=None, now=NOW)
    meta = json.loads(clip.sidecar.read_text(encoding="utf-8"))
    del meta["created"]
    clip.sidecar.write_text(json.dumps(meta), encoding="utf-8")
    stamp = (NOW - timedelta(days=100)).timestamp()
    os.utime(clip.wav, (stamp, stamp))
    assert store.expire(90, now=NOW) == 1


def test_config_from_env() -> None:
    assert itf.Config.from_env({}).enabled and itf.Config.from_env({}).sample == 0.2
    assert not itf.Config.from_env({itf.CAPTURE_ENV: "0"}).enabled
    assert not itf.Config.from_env({itf.CAPTURE_ENV: "off"}).enabled
    assert itf.Config.from_env({itf.SAMPLE_ENV: "junk"}).sample == 0.2
    assert itf.Config.from_env({itf.SAMPLE_ENV: "5"}).sample == 1.0
    assert itf.Config.from_env({itf.DAYS_ENV: "30"}).days == 30.0


# -- deciding what to keep -------------------------------------------------------------------
def test_other_speaker_refusals_are_always_kept_with_their_score(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.99)
    u = utt(10.0, level=0.3)
    t = rig.transcript(1, u)
    rig.capture.on_verdict(t, "other_speaker", [(u, 0.21)])
    rig.capture.on_agent_said("Sorry, I could not verify your voice.")
    clips = rig.store.clips()
    assert [c.decision for c in clips] == ["other_speaker"]
    assert clips[0].score == 0.21 and clips[0].meta["profile"] == "call"


def test_verified_utterances_are_sampled(tmp_path: Path) -> None:
    keep = make_rig(tmp_path / "a", rand=0.1)
    u = utt(10.0)
    keep.capture.on_verdict(keep.transcript(1, u), None, [(u, 0.6)])
    assert keep.decisions() == ["verified"]
    drop = make_rig(tmp_path / "b", rand=0.5)
    drop.capture.on_verdict(drop.transcript(1, u), None, [(u, 0.6)])
    drop.capture.on_agent_said("Sent.")
    assert drop.decisions() == []  # decided by the verdict: not sampled again as answered


def test_a_silent_agent_turn_is_kept_as_skipped_and_scored(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.99)
    rig.transcript(1, utt(10.0, level=0.45))
    rig.capture.on_agent_said("...")
    clips = rig.store.clips()
    assert [c.decision for c in clips] == ["skipped"] and clips[0].score == 0.45


def test_a_spoken_reply_answers_and_is_sampled(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.1)
    rig.transcript(1, utt(10.0))
    rig.transcript(2, utt(12.0))  # two quick transcripts, one reply: both answered
    rig.capture.on_agent_said("Here is your plate.")
    assert rig.decisions() == ["answered", "answered"]
    not_sampled = make_rig(tmp_path / "x", rand=0.9)
    not_sampled.transcript(1, utt(10.0))
    not_sampled.capture.on_agent_said("Sure.")
    assert not_sampled.decisions() == []


def test_a_reply_after_the_window_means_the_turn_was_skipped(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.1)
    rig.transcript(1, utt(10.0))
    rig.clock[0] += itf.REPLY_WINDOW_S + 1
    rig.capture.on_agent_said("Anything else?")
    assert rig.decisions() == ["skipped"]


def test_an_unanswered_turn_is_skipped_at_the_next_transcript_or_call_end(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.99)
    rig.transcript(1, utt(10.0))
    rig.clock[0] += itf.REPLY_WINDOW_S + 1
    rig.transcript(2, utt(30.0), at=rig.clock[0])
    assert rig.decisions() == ["skipped"]
    rig.capture.end_call()
    assert rig.decisions() == ["skipped", "skipped"]


def test_short_injected_duplicate_and_disabled_are_not_kept(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.0)
    rig.transcript(1, utt(10.0, seconds=0.3))  # too short
    rig.capture.on_transcript(Transcript(2, "[system notice] x", 100.0, injected=True))
    u = utt(20.0)
    t = rig.transcript(3, u)
    rig.capture.on_verdict(t, "other_speaker", [(u, 0.2)])
    rig.capture.on_verdict(t, "other_speaker", [(u, 0.2)])  # same utterance again
    rig.capture.end_call()
    assert rig.decisions() == ["other_speaker"]
    off = make_rig(tmp_path / "off", rand=0.0, enabled=False)
    v = utt(10.0)
    off.capture.on_verdict(off.transcript(1, v), "other_speaker", [(v, 0.2)])
    off.capture.end_call()
    assert not (tmp_path / "off" / "interference").exists()


def test_at_most_max_clips_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(itf, "MAX_CLIPS_PER_CALL", 3)
    rig = make_rig(tmp_path, rand=0.99)
    for i in range(5):
        u = utt(10.0 * i)
        rig.capture.on_verdict(rig.transcript(i, u), "other_speaker", [(u, 0.2)])
    assert len(rig.store.clips()) == 3
    rig.capture.begin_call()  # a new call may keep clips again
    u = utt(500.0)
    rig.capture.on_verdict(rig.transcript(9, u), "other_speaker", [(u, 0.2)])
    assert len(rig.store.clips()) == 4


def test_ambiguous_bindings_keep_every_candidate(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rand=0.99)
    rig.transcript(1, utt(10.0, seq=1), utt(12.0, seq=2))
    rig.capture.end_call()
    assert rig.decisions() == ["skipped", "skipped"]


def test_begin_call_deletes_expired_clips(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    rig.store.save(
        np.zeros(8000),
        decision="skipped",
        score=None,
        profile=None,
        now=datetime.now().astimezone() - timedelta(days=120),
    )
    rig.capture.begin_call()
    assert not rig.store.clips()


def test_the_background_worker_saves_and_survives_failures(tmp_path: Path) -> None:
    store = itf.ClipStore(tmp_path)

    class Broken(FakeScorer):
        def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
            raise RuntimeError("model crashed")

    capture = itf.CallCapture(
        store, itf.Config(root=tmp_path), scorer=Broken(), authorized="albert", rand=lambda: 0.99
    )
    capture.begin_call()
    capture.on_transcript(Transcript(1, "x", 0.0, binding=utt(1.0)))
    capture.end_call()  # skipped → scored on the worker → the scorer raises: logged, not saved
    u = utt(5.0)
    capture.on_verdict(Transcript(2, "y", 0.0, binding=u), "other_speaker", [(u, 0.3)])
    capture.flush()
    assert [c.decision for c in store.clips()] == ["other_speaker"]
    capture.shutdown()


# -- bridge integration ----------------------------------------------------------------------
class FakeVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class AmplitudeScorer(FakeScorer):
    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del name, timeout
        return 0.6 if float(np.max(np.abs(audio))) > 0.4 else 0.1


def _call(
    tmp_path: Path, env: dict[str, str] | None = None
) -> tuple[BridgeController, list[float]]:
    clock = [1000.0]
    auth = Authoriser(AmplitudeScorer(), "albert", min_utterance_s=1.0)
    ctl = BridgeController(auth, vad_factory=FakeVad, clock=lambda: clock[0])
    env = {"MAC_VOICE_INTERFERENCE_SAMPLE": "0", **(env or {})}
    monkey_root = tmp_path / "interference"
    capture = itf.attach(ctl, auth.scorer, env)
    if capture is not None:
        capture.store = itf.ClipStore(monkey_root)
        capture.background = False
    ctl.begin_call()
    return ctl, clock


def _speak(ctl: BridgeController, clock: list[float], level: float, seconds: float = 1.5) -> None:
    for _ in range(round(seconds * 10)):
        clock[0] += 0.1
        ctl.feed_audio(np.full(1600, level, dtype=np.float32))
    for _ in range(6):
        clock[0] += 0.1
        ctl.feed_audio(np.zeros(1600, dtype=np.float32))


def test_bridge_refusal_and_skipped_turn_end_up_as_clips(tmp_path: Path) -> None:
    ctl, clock = _call(tmp_path)
    _speak(ctl, clock, 0.3)  # someone else asks for an action
    clock[0] += 1.0
    ctl.on_transcript(1, "send that to Peter", clock[0])
    latest = ctl.latest
    assert latest is not None
    assert ctl.authoriser.verify(latest) is not None  # refused: other_speaker
    ctl.note_agent_said("I could not verify your voice.")
    _speak(ctl, clock, 0.3)  # someone else talks to the room; the agent stays silent
    clock[0] += 1.0
    ctl.on_transcript(2, "Kommst du essen?", clock[0])
    ctl.note_agent_said("...")
    ctl.end_call()
    clips = itf.ClipStore(tmp_path / "interference").clips()
    assert sorted(c.decision for c in clips) == ["other_speaker", "skipped"]
    refused = next(c for c in clips if c.decision == "other_speaker")
    assert refused.score == 0.1


def test_attach_is_off_by_env_and_wires_hooks_when_on() -> None:
    auth = Authoriser(AmplitudeScorer(), "albert")
    ctl = BridgeController(auth, vad_factory=FakeVad)
    assert itf.attach(ctl, auth.scorer, {itf.CAPTURE_ENV: "0"}) is None
    assert ctl.capture is None and auth.observer is None
    auth2 = Authoriser(AmplitudeScorer(), "albert")
    ctl2 = BridgeController(auth2, vad_factory=FakeVad)
    capture = itf.attach(ctl2, auth2.scorer, {})
    assert capture is not None
    assert ctl2.capture is capture
    assert auth2.observer == capture.on_verdict  # pylint: disable=comparison-with-callable


def test_a_failing_observer_never_changes_the_decision() -> None:
    auth = Authoriser(AmplitudeScorer(), "albert", min_utterance_s=1.0)

    def boom(*_: Any) -> None:
        raise RuntimeError("disk full")

    auth.observer = boom
    t = Transcript(1, "x", 0.0, binding=utt(1.0, level=0.5))
    assert auth.verify(t) is None


def test_observer_sees_the_code_and_every_scored_utterance() -> None:
    auth = Authoriser(AmplitudeScorer(), "albert", min_utterance_s=1.0)
    seen: list[tuple[str | None, list[float]]] = []
    auth.observer = lambda t, code, scored: seen.append((code, [s for _, s in scored]))
    auth.verify(Transcript(1, "x", 0.0, binding=utt(1.0, level=0.5)))
    auth.verify(Transcript(2, "y", 0.0, binding=utt(5.0, level=0.2)))
    auth.verify(Transcript(3, "z", 0.0, binding=utt(9.0, seconds=0.5)))  # too short: unscored
    assert seen == [(None, [0.6]), ("other_speaker", [0.1]), ("too_short", [])]


def test_agent_replies_reach_the_bridge() -> None:
    said: list[str] = []

    class Session:
        bridge = type("B", (), {"note_agent_said": staticmethod(said.append)})()

    VoiceSession._agent_reply(Session(), "...")  # type: ignore[arg-type]  # pylint: disable=protected-access
    assert said == ["..."]


# -- the labelling CLI -----------------------------------------------------------------------
class Cli:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.lines: list[str] = []
        self.played: list[Path] = []
        self.deps = lbl.Deps(play=self.played.append, ask=self.ask, out=self.lines.append)

    def ask(self, _prompt: str) -> str:
        return self.answers.pop(0) if self.answers else "q"


def _three_clips(root: Path) -> list[itf.Clip]:
    store = itf.ClipStore(root)
    return [
        store.save(
            np.zeros(8000), decision=d, score=0.3, profile="call", now=NOW + timedelta(seconds=i)
        )
        for i, d in enumerate(("other_speaker", "skipped", "verified"))
    ]


def test_play_and_label_walks_the_unlabelled_clips(tmp_path: Path) -> None:
    clips = _three_clips(tmp_path)
    cli = Cli(["r", "o", "a", "d"])
    assert lbl.main(["-p", "-D", str(tmp_path)], cli.deps) == 0
    labels = {c.id: c.label for c in itf.ClipStore(tmp_path).clips()}
    assert labels == {clips[0].id: "other", clips[1].id: "albert"}  # the third was deleted
    assert len(cli.played) == 4  # replayed once
    assert "2 clip(s): albert 1, other 1" in cli.lines[-1]


def test_list_filters_and_set_by_prefix(tmp_path: Path) -> None:
    clips = _three_clips(tmp_path)
    cli = Cli([])
    assert lbl.main(["-l", "-d", "skipped", "-D", str(tmp_path)], cli.deps) == 0
    assert len(cli.lines) == 2 and clips[1].id in cli.lines[0]
    assert lbl.main(["-s", clips[1].id, "other", "-D", str(tmp_path)], cli.deps) == 0
    assert lbl.main(["-s", clips[1].id, "susi", "-D", str(tmp_path)], cli.deps) == 2
    assert lbl.main(["-s", "nothing", "other", "-D", str(tmp_path)], cli.deps) == 1
    out = Cli([])
    lbl.main(["-l", "-D", str(tmp_path)], out.deps)
    assert all(clips[1].id not in line for line in out.lines[:-1])  # labelled: hidden
    out_all = Cli([])
    lbl.main(["-l", "-A", "-D", str(tmp_path)], out_all.deps)
    assert any(clips[1].id in line and line.endswith("other") for line in out_all.lines)


def test_cli_delete_and_expire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips = _three_clips(tmp_path)
    cli = Cli([])
    assert lbl.main(["-x", clips[0].id, "-D", str(tmp_path)], cli.deps) == 0
    monkeypatch.setenv(itf.DAYS_ENV, "0")
    assert lbl.main(["-e", "-D", str(tmp_path)], cli.deps) == 0
    assert not itf.ClipStore(tmp_path).clips()
    assert "deleted 2 unlabelled clip(s) older than 0 days" in cli.lines[-1]
