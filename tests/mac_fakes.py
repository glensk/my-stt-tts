"""Fakes for the Mac fast-path tests (``test_mac_control``, ``test_mac_sites``).

A runner that answers osascript / open / mdls by script and by the ``/*mv:<op>*/``
marker of each Safari snippet, a fake display, a manual clock (``sleep`` advances it)
and a controller in a call with the amplitude-based speaker scorer of
``test_bridge_auth``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from my_stt_tts.bridge import Authoriser, BridgeController, MemoryProblemSink, Transcript
from my_stt_tts.mac_control import (
    MDLS,
    OPEN,
    PRESS_KEY,
    READ_VOLUME,
    SAFARI_JS_PROBE,
    SAFARI_WINDOWS,
    SET_MUTE,
    SET_VOLUME,
    MacControl,
)
from my_stt_tts.mac_sites import (
    FRONT_WINDOW,
    OSASCRIPT,
    RUN_JS,
    SET_URL,
    RunResult,
)

FIXTURES = Path(__file__).parent / "fixtures" / "spikes"
FRAME = 1600
ALBERT, OTHER = 0.5, 0.3
SCORES = {ALBERT: 0.52, OTHER: 0.18}
WIN = 6857
SYSTEM_EVENTS_PROBE = (
    'tell application "System Events" to get name of first process whose frontmost is true'
)


# -- fakes -----------------------------------------------------------------------------------
class FakeVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class FakeScorer:
    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del timeout
        return SCORES[round(float(np.max(np.abs(audio))), 2)] if name == "albert" else None


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeDisplay:
    def __init__(self, builtin: bool = True, values: Sequence[float | None] = (0.5, 0.625)):
        self.builtin = builtin
        self.values = list(values)

    def builtin_is_main(self) -> bool:
        return self.builtin

    def brightness(self) -> float | None:
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


class FakeMac:  # pylint: disable=too-many-instance-attributes  # one fake Mac
    """Answers the runner's argv like a Mac with Safari, volume and apps would."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.win: int | None = WIN
        self.js: dict[str, Any] = {}
        self.volume = 55
        self.volume_sticks = False
        self.muted = False
        self.fail: dict[str, RunResult] = {}
        self.open_rc = 0
        self.mdls = "com.example.fallback"
        self.safari_windows = "1"

    def __call__(self, argv: Sequence[str], timeout: float) -> RunResult:
        assert timeout > 0
        argv = list(argv)
        assert all(isinstance(a, str) for a in argv)
        self.calls.append(argv)
        if argv[0] == OPEN:
            return RunResult(self.open_rc)
        if argv[0] == MDLS:
            return RunResult(0, self.mdls + "\n")
        assert argv[0] == OSASCRIPT and argv[1] == "-e"
        script, rest = argv[2], argv[3:]
        if script in self.fail:
            return self.fail[script]
        return self._osa(script, rest)

    def _osa(self, script: str, rest: list[str]) -> RunResult:
        if script == FRONT_WINDOW:
            return RunResult(0, f"{self.win or ''}\n")
        if script == RUN_JS:
            return self._js(rest[0], rest[1])
        simple = {
            SET_URL: "",
            READ_VOLUME: f"{self.volume}\n",
            SAFARI_WINDOWS: f"{self.safari_windows}\n",
            SAFARI_JS_PROBE: "2\n",
            SYSTEM_EVENTS_PROBE: "iTerm2\n",
            PRESS_KEY: "",
        }
        if script in simple:
            return RunResult(0, simple[script])
        if script == SET_VOLUME:
            if not self.volume_sticks:
                self.volume = int(rest[0])
            return RunResult(0, f"{self.volume}\n")
        if script == SET_MUTE:
            self.muted = rest[0] == "true"
            return RunResult(0, f"{str(self.muted).lower()}\n")
        raise AssertionError(f"unexpected script {script[:60]!r}")

    def _js(self, code: str, win: str) -> RunResult:
        assert win == str(WIN), "Safari must be targeted by the front window's id"
        match = re.match(r"/\*mv:(\w+)\*/", code)
        assert match, code[:40]
        answer = self.js.get(match.group(1), {"ok": True})
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if isinstance(answer, RunResult):
            return answer
        return RunResult(0, json.dumps(answer) + "\n")

    # -- what was sent -------------------------------------------------------------------
    def ops(self) -> list[str]:
        out = []
        for argv in self.calls:
            if argv[0] == OSASCRIPT and argv[2] == RUN_JS:
                found = re.match(r"/\*mv:(\w+)\*/", argv[3])
                out.append(found.group(1) if found else "?")
        return out

    def js_of(self, op: str) -> list[str]:
        return [
            a[3] for a in self.calls if a[0] == OSASCRIPT and a[2] == RUN_JS and f"mv:{op}*" in a[3]
        ]

    def scripts(self, script: str) -> list[list[str]]:
        return [a[3:] for a in self.calls if a[0] == OSASCRIPT and a[2] == script]

    def opened(self) -> list[list[str]]:
        return [a for a in self.calls if a[0] == OPEN]


class Call:
    """A controller in a call with a MacControl on fakes."""

    def __init__(self, apps: Sequence[Path] = (), real_click: Any = None) -> None:
        self.clock = Clock()
        self.ctl = BridgeController(
            Authoriser(FakeScorer(), "albert"), vad_factory=FakeVad, clock=self.clock
        )
        self.ctl.begin_call()
        self.fake = FakeMac()
        self.display = FakeDisplay()
        self.mac = MacControl(
            self.ctl,
            runner=self.fake,
            display=self.display,
            apps=lambda: list(apps),
            sleep=self.sleep,
            real_click=real_click,
        )
        self.seq = 0

    def sleep(self, seconds: float) -> None:
        self.clock.now += seconds

    def say(self, text: str, level: float = ALBERT) -> Transcript:
        for _ in range(10):
            self.clock.now += 0.1
            self.ctl.feed_audio(np.full(FRAME, level, dtype=np.float32))
        for _ in range(6):
            self.clock.now += 0.1
            self.ctl.feed_audio(np.zeros(FRAME, dtype=np.float32))
        self.clock.now += 1.5
        self.seq += 1
        return self.ctl.on_transcript(self.seq, text, self.clock.now)

    @property
    def problems(self) -> list[Any]:
        sink = self.ctl.problems
        assert isinstance(sink, MemoryProblemSink)
        return sink.problems


def yt_state(**over: Any) -> dict[str, Any]:
    base = {
        "site": "youtube",
        "video": True,
        "id": "lDrAZ1wAyVs",
        "paused": False,
        "t": 5.0,
        "duration": 600.0,
        "fullscreen": False,
    }
    return base | over


def jf_state(**over: Any) -> dict[str, Any]:
    return yt_state(site="jellyfin", id="bfa78bb580de320fe4d1dd15c15c4b6a") | over
