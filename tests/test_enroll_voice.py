"""scripts/enroll_voice.py + voice_profiles with fake audio, VAD and embedder (no mic, no model)."""
# pylint: disable=missing-function-docstring,redefined-outer-name

from __future__ import annotations

import importlib.util
import json
import stat
import sys
import threading
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import voice_control as vc
from my_stt_tts import voice_profiles as vp

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # the script imports _bootstrap / enroll_call from there
_spec = importlib.util.spec_from_file_location(
    "enroll_voice_under_test", _SCRIPTS / "enroll_voice.py"
)
assert _spec and _spec.loader
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev  # dataclasses resolve their module while the script loads
_spec.loader.exec_module(ev)


class FakeMic:
    """Streams 250 ms frames at ``level`` until closed (the rig sets the level per take)."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.level = 0.0
        self.closed = threading.Event()

    def start(self) -> bool:
        return self.ok

    def mic_frames(self) -> Iterator[np.ndarray]:
        while not self.closed.is_set():
            time.sleep(0.001)
            yield np.full(ev.FRAME_SAMPLES, self.level, dtype=np.float32)

    def close(self) -> None:
        self.closed.set()


class FakeVad:
    def is_speech(self, frame: Any) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


def fake_embed(clip: np.ndarray) -> np.ndarray:
    """Level → direction: positive = the enrolled voice, ~0.3 = 'raw mic' flavour."""
    level = float(np.mean(clip))
    if level > 0.35:
        return np.array([1.0, 0.0, 0.0])
    if level > 0:
        return np.array([0.9, 0.1, 0.0])
    return np.array([0.0, 0.0, 1.0])


class FakeDaemon:
    def __init__(self, running: bool = False, stop_ok: bool = True, managed: bool = True) -> None:
        self.is_running = running
        self.is_managed = managed
        self.stop_ok = stop_ok
        self.calls: list[str] = []

    def running(self) -> bool:
        return self.is_running

    def managed(self) -> bool:
        return self.is_managed

    def stop_hint(self) -> str:
        return "stop it first (hint)"

    def stop(self) -> bool:
        self.calls.append("stop")
        self.is_running = not self.stop_ok
        return self.stop_ok

    def start(self) -> bool:
        self.calls.append("start")
        return True


class Rig:
    """Deps whose ``ask`` answers from a script: (answer, level of the next take)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        tmp: Path,
        script: list[tuple[str, float]],
        *,
        vpio_ok: bool = True,
        twin_ok: bool = True,
        daemon: FakeDaemon | None = None,
    ) -> None:
        self.duplex = FakeMic(vpio_ok)
        self.twin = FakeMic(twin_ok)
        self.script = list(script)
        self.lines: list[str] = []
        self.daemon = daemon or FakeDaemon()
        self.wake = tmp / "wake"
        self.deps = ev.Deps(
            duplex=lambda: self.duplex,
            raw_mic=lambda: self.twin,
            vad=FakeVad,
            embedder=lambda: fake_embed,
            daemon=self.daemon,
            ask=self.ask,
            out=self.lines.append,
            stamp=lambda: "20261025-200000",
            wake_clips=self.wake,
            env_file=tmp / "absent-config",  # never the live config: its threshold is tuned
            env={},
        )

    def ask(self, prompt: str) -> str:
        answer, level = self.script.pop(0) if self.script else ("q", 0.0)
        if "Enter = record" in prompt:
            self.duplex.level = level
            self.twin.level = level * 0.6  # the plain mic copy: a different "domain"
        return answer

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _run(tmp: Path, rig: Rig, *argv: str) -> int:
    return int(ev.main(["-o", str(tmp / "enroll"), "-s", "1.5", *argv], rig.deps))


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_sentences_cover_both_languages_and_kinds() -> None:
    assert len(ev.SENTENCES) == 25
    assert {s.lang for s in ev.SENTENCES} == {"de", "en"}
    kinds = [s.kind for s in ev.SENTENCES]
    assert kinds.count("short") >= 12 and kinds.count("long") >= 8
    assert ev.record_seconds("Yes.") == 3.5
    assert ev.record_seconds(" ".join(["word"] * 40)) == 10.0


def test_pick_sentences_filters_language_and_count() -> None:
    args = ev.build_parser().parse_args(["-l", "de", "-c", "3"])
    picked = ev.pick_sentences(args)
    assert len(picked) == 3 and all(s.lang == "de" for s in picked)


def test_full_session_keeps_takes_privately_and_rebuilds_both_profiles(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [("", 0.5)] * 4)
    assert _run(tmp_path, rig, "-c", "4") == 0
    enroll = tmp_path / "enroll"
    session = enroll / "albert" / "20261025-200000"
    data = json.loads((session / "session.json").read_text())
    assert data["domain"] == "vpio" and data["plain_mic_twin"] is True
    assert [t["usable"] for t in data["takes"]] == [True] * 4
    assert data["takes"][0]["text"] == ev.SENTENCES[0].text
    for take in data["takes"]:
        for key in ("call_file", "raw_file"):
            assert _mode(session / take[key]) == 0o600
    assert _mode(session) == 0o700 and _mode(session / "call") == 0o700
    assert _mode(session / "session.json") == 0o600
    call, wake = enroll / "call" / "albert.npy", enroll / "albert.npy"
    assert _mode(call) == 0o600 and _mode(wake) == 0o600
    assert np.load(call)[0] > 0.99  # built from the VoiceProcessingIO takes
    assert np.load(wake)[1] > 0.05  # built from the plain-mic copies
    manifest = json.loads((enroll / "albert" / "profiles.json").read_text())
    assert manifest["call"]["sha256"] == vp.sha256_file(call)
    assert len(manifest["call"]["takes"]) == 4
    assert all(t.startswith("albert/20261025-200000/call/") for t in manifest["call"]["takes"])
    assert "| take" in rig.text and "short sentences" in rig.text
    assert "Next: scripts/calibrate_voice.py" in rig.text


def test_a_silent_take_is_kept_but_retried(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [("", 0.0), ("", 0.5), ("", 0.5), ("", 0.5)])
    assert _run(tmp_path, rig, "-c", "3") == 0
    session = tmp_path / "enroll" / "albert" / "20261025-200000"
    takes = json.loads((session / "session.json").read_text())["takes"]
    assert [(t["n"], t["attempt"], t["usable"]) for t in takes] == [
        (1, 1, False),
        (1, 2, True),
        (2, 1, True),
        (3, 1, True),
    ]
    assert (session / "call" / "01-a1.wav").is_file()  # every recording is kept
    assert "only 0.0 s of speech — Enter to try again" in rig.text


def test_old_profiles_are_backed_up_and_scored_before(tmp_path: Path) -> None:
    enroll = tmp_path / "enroll"
    (enroll / "call").mkdir(parents=True)
    np.save(enroll / "call" / "albert.npy", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    rig = Rig(tmp_path, [("", 0.5)] * 3)
    assert _run(tmp_path, rig, "-c", "3") == 0
    backup = enroll / "albert" / "backup" / "20261025-200000-call.npy"
    assert np.load(backup)[1] == 1.0 and _mode(backup) == 0o600
    assert "short sentences ≥ 0.35: before 0/3, after 3/3" in rig.text


def test_without_voice_processing_it_records_the_plain_mic_and_says_so(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [("", 0.5)] * 3, vpio_ok=False)
    rig.twin.ok = True
    assert _run(tmp_path, rig, "-c", "3") == 0
    data = json.loads(
        (tmp_path / "enroll" / "albert" / "20261025-200000" / "session.json").read_text()
    )
    assert data["domain"] == "raw" and all("call_file" not in t for t in data["takes"])
    assert "VoiceProcessingIO did not start" in rig.text
    assert "built from plain-mic takes" in rig.text


def test_without_a_second_stream_the_wake_profile_uses_the_wake_clips(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [("", 0.5)] * 3, twin_ok=False)
    assert _run(tmp_path, rig, "-c", "3") == 0
    assert "no second (plain-mic) stream" in rig.text
    assert "wake profile is rebuilt from the wake-word clips only" in rig.text
    assert not (tmp_path / "enroll" / "albert.npy").exists()  # no wake clips in this test


def test_too_few_usable_takes_keep_the_profiles(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [("", 0.5), ("s", 0.0), ("q", 0.0)])
    assert _run(tmp_path, rig, "-c", "5") == 1
    assert not (tmp_path / "enroll" / "call" / "albert.npy").exists()
    assert "too few usable takes" in rig.text


def test_running_daemon_is_stopped_and_started_again(tmp_path: Path) -> None:
    daemon = FakeDaemon(running=True)
    rig = Rig(tmp_path, [("", 0.0)] + [("", 0.5)] * 3, daemon=daemon)
    assert _run(tmp_path, rig, "-c", "3") == 0
    assert daemon.calls == ["stop", "start"]
    assert "mac-voice started again" in rig.text


def test_running_daemon_and_no_means_nothing_is_recorded(tmp_path: Path) -> None:
    daemon = FakeDaemon(running=True)
    rig = Rig(tmp_path, [("n", 0.0)], daemon=daemon)
    assert _run(tmp_path, rig, "-c", "3") == 2
    assert not daemon.calls and "stop it first" in rig.text
    assert not (tmp_path / "enroll").exists()


def test_daemon_that_cannot_be_stopped_refuses(tmp_path: Path) -> None:
    daemon = FakeDaemon(running=True, stop_ok=False)
    rig = Rig(tmp_path, [], daemon=daemon)
    assert _run(tmp_path, rig, "-y", "-c", "3") == 2
    assert daemon.calls == ["stop"]


def test_hand_started_daemon_is_not_killed_and_gets_the_manual_hint(tmp_path: Path) -> None:
    daemon = FakeDaemon(running=True, managed=False)
    rig = Rig(tmp_path, [], daemon=daemon)
    assert _run(tmp_path, rig, "-y", "-c", "3") == 2
    assert not daemon.calls and "stop it first" in rig.text
    assert not (tmp_path / "enroll").exists()


def test_launchd_daemon_hint_names_the_pid_of_a_hand_started_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vc, "send", lambda *_a, **_k: {"pid": 4242})
    monkeypatch.setattr(vc, "launch_agent_loaded", lambda: False)
    hint = ev.LaunchdDaemon().stop_hint()
    assert "pid 4242" in hint and "Ctrl-C" in hint and "launchctl" not in hint


def test_rebuild_only_uses_every_kept_session(tmp_path: Path) -> None:
    rig = Rig(tmp_path, [("", 0.5)] * 3)
    assert _run(tmp_path, rig, "-c", "3") == 0
    rig3 = Rig(tmp_path, [])
    assert _run(tmp_path, rig3, "-R") == 0
    assert "restart mac-voice" in rig3.text
    assert "| take" in rig3.text


def test_score_table_counts_only_takes_long_enough() -> None:
    rows = [
        ("a", "short", 1.5, 0.30, 0.50),
        ("b", "short", 1.0, 0.10, 0.20),  # too short for actions: not counted
        ("c", "long", 4.0, None, 0.70),
    ]
    lines = ev.score_table(rows, 0.35, 1.2)
    text = "\n".join(lines)
    assert "(too short for actions)" in text
    assert "short sentences ≥ 0.35: before 0/1, after 1/1" in text
    assert "long sentences ≥ 0.35: after 1/1" in text


def test_read_wav_resamples_and_downmixes(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    with wave.Wave_write(str(path)) as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(22050)
        out.writeframes((np.full((22050, 2), 0.25) * 32767).astype("<i2").tobytes())
    pcm = vp.read_wav(path)
    assert abs(pcm.size - 16000) <= 1 and abs(float(pcm.mean()) - 0.25) < 1e-3


def test_leave_one_out_and_best_match() -> None:
    embs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    loo = vp.leave_one_out(embs)
    assert loo[2] == pytest.approx(0.0, abs=1e-6)
    assert loo[0] == pytest.approx(0.7071, abs=1e-3)
    assert vp.best_match(np.array([1.0, 0.0]), None) is None
    assert vp.best_match(np.array([1.0, 0.0]), np.array([[0.0, 1.0], [0.6, 0.8]])) == (
        pytest.approx(0.6)
    )


def test_env_float_reads_only_the_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=secret\nMAC_VOICE_AUTH_THRESHOLD=0.42\n")
    assert vp.env_float("MAC_VOICE_AUTH_THRESHOLD", 0.35, env_file, env={}) == 0.42
    assert (
        vp.env_float(
            "MAC_VOICE_AUTH_THRESHOLD", 0.35, env_file, env={"MAC_VOICE_AUTH_THRESHOLD": "0.5"}
        )
        == 0.5
    )
    assert vp.env_float("MISSING", 0.35, env_file, env={}) == 0.35
