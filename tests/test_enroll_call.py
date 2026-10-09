"""scripts/enroll_call.py with a fake VoiceProcessingIO duplex, VAD and embedder (no audio)."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # the script imports _bootstrap from its own directory
_spec = importlib.util.spec_from_file_location(
    "enroll_call_under_test", _SCRIPTS / "enroll_call.py"
)
assert _spec and _spec.loader
ec = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ec  # dataclasses resolve their module while the script loads
_spec.loader.exec_module(ec)


class FakeDuplex:
    """Streams 250 ms frames at ``level`` until closed (the test changes ``level`` per clip)."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.level = 0.5
        self.started = False
        self.closed = threading.Event()

    def start(self) -> bool:
        self.started = True
        return self.ok

    def mic_frames(self) -> Iterator[np.ndarray]:
        while not self.closed.is_set():
            time.sleep(0.001)
            yield np.full(ec.FRAME_SAMPLES, self.level, dtype=np.float32)

    def close(self) -> None:
        self.closed.set()


class FakeVad:
    def is_speech(self, frame: Any) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class Rig:
    """A :class:`Deps` whose ``ask`` sets the level of the next clip from ``levels``."""

    def __init__(self, levels: list[float], *, daemon: bool = False, ok: bool = True) -> None:
        self.duplex = FakeDuplex(ok)
        self.levels = list(levels)
        self.prompts: list[str] = []
        self.lines: list[str] = []
        self.embedded = 0
        self.deps = ec.Deps(
            duplex=lambda: self.duplex,
            vad=FakeVad,
            embedder=lambda: self.embed,
            daemon_running=lambda: daemon,
            ask=self.ask,
            out=self.lines.append,
        )

    def ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.duplex.level = self.levels.pop(0)
        return ""

    def embed(self, clip: np.ndarray) -> np.ndarray:
        assert clip.size >= ec.SAMPLE_RATE  # only trimmed speech of ≥ 1 s gets here
        self.embedded += 1
        return np.array([1.0, 0.05 * self.embedded], dtype=np.float32)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _wake_profile(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "albert.npy", np.array([1.0, 0.0], dtype=np.float32))


def test_enrolls_and_saves_the_call_centroid(tmp_path: Path) -> None:
    _wake_profile(tmp_path)
    rig = Rig([0.5, 0.0, 0.5, 0.5, 0.5])  # clip 2 is silent
    code = ec.main(["albert", "-c", "5", "-s", "1.5", "-o", str(tmp_path)], rig.deps)
    assert code == 0
    assert rig.prompts[0] == "Clip 1/5 — say any sentence in German, press Enter to start "
    assert "in English" in rig.prompts[1] and "in French" in rig.prompts[2]
    assert "in German" in rig.prompts[3]
    assert "❌ only 0.00 s of speech — skipped" in rig.text
    assert rig.embedded == 4
    saved = np.load(tmp_path / "call" / "albert.npy")
    assert saved.dtype == np.float32 and abs(float(np.linalg.norm(saved)) - 1.0) < 1e-5
    table = [line for line in rig.lines if line.startswith("|")]
    assert len(table) == 2 + 4 and table[3].startswith("|    3 |")  # clip 2 skipped
    assert "leave-one-out: min" in rig.text
    assert "✅ min leave-one-out" in rig.text
    assert rig.duplex.closed.is_set()


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    rig = Rig([0.5, 0.5, 0.5])
    assert ec.main(["albert", "-n", "-c", "3", "-s", "1.2", "-o", str(tmp_path)], rig.deps) == 0
    assert not (tmp_path / "call").exists()
    assert "no wake profile" in rig.text
    assert "|       - |" in rig.text  # no wake profile: "vs wake" shows "-"
    assert "✅ dry run — nothing written" in rig.text


def test_refuses_while_the_daemon_runs(tmp_path: Path) -> None:
    rig = Rig([], daemon=True)
    assert ec.main(["albert", "-o", str(tmp_path)], rig.deps) == 2
    assert rig.text == ec.STOP_HINT and "mac-voice" in rig.text
    assert not rig.duplex.started


def test_too_few_usable_clips_saves_nothing(tmp_path: Path) -> None:
    rig = Rig([0.5, 0.0, 0.0])
    assert ec.main(["albert", "-c", "3", "-s", "1.2", "-o", str(tmp_path)], rig.deps) == 1
    assert "❌ 1 usable clip(s), need 3 — nothing saved" in rig.text
    assert not (tmp_path / "call").exists()


def test_voice_processing_unavailable(tmp_path: Path) -> None:
    rig = Rig([], ok=False)
    assert ec.main(["albert", "-o", str(tmp_path)], rig.deps) == 2
    assert "❌ VoiceProcessingIO did not start" in rig.text


@pytest.mark.parametrize("argv", [["../x"], ["albert", "-c", "0"], ["albert", "-s", "0"]])
def test_bad_arguments(argv: list[str]) -> None:
    assert ec.main(argv, Rig([]).deps) == 2


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        ec.main(["-h"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("-c, --clips", "-s, --seconds", "-o, --out", "-l, --languages", "-n, --dry-run"):
        assert flag.split(",", maxsplit=1)[0] in out and flag.split(", ")[1] in out


def test_calibration_fails_below_the_threshold() -> None:
    lines, ok = ec.calibration_table(["   1 |  2.00s", "   2 |  2.00s"], [0.5, 0.3], [None, 0.4])
    assert not ok
    assert lines[-2] == "leave-one-out: min 0.30, median 0.40"
    assert lines[-1].startswith("❌ min leave-one-out 0.30 is below")


def test_leave_one_out_scores_each_clip_against_the_others() -> None:
    embs = [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    loo = ec.leave_one_out(embs)
    assert loo[2] == pytest.approx(0.0, abs=1e-6)
    assert loo[0] == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_speech_only_trims_silence() -> None:
    clip = np.concatenate([np.zeros(8000), np.full(24000, 0.5), np.zeros(16000)]).astype(np.float32)
    speech, seconds = ec.speech_only(clip, FakeVad())
    assert seconds == pytest.approx(1.5, abs=0.3)
    assert speech.size < clip.size and float(np.min(speech)) == 0.5  # no silence kept
