# pylint: disable=too-many-lines  # one test module per bridge component, like the module
"""Claude sessions by voice: ccc envelopes, targets, briefings, proposals and outcomes.

No real ccc: a fake runner returns section-6 envelopes per subcommand and records every
argv / stdin / timeout. The controller is the real :class:`BridgeController` with a fake
VAD, a fake speaker scorer and a manual clock (as in ``test_bridge_auth.py``).
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import bridge, claude_sessions
from my_stt_tts.attention import AttentionStore
from my_stt_tts.attention import register as register_attention
from my_stt_tts.bridge import (
    Authoriser,
    BridgeController,
    Capability,
    MemoryDeliveryTracker,
    MemoryProblemSink,
    Problem,
)
from my_stt_tts.claude_sessions import (
    POST_ACTION_FAILURES,
    REASONS,
    UNCERTAIN_CODES,
    CccClient,
    Question,
    RunResult,
    SessionTools,
    parse_envelope,
    reason_for,
    register,
    resolve_choice,
    run_group,
)

FRAME = 1600
ALBERT = 0.5
CCC = "/fake/bin/ccc"
SECRET = "ghp_" + "A" * 30


# -- fakes -----------------------------------------------------------------------------------
class FakeVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class FakeScorer:
    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del audio, timeout
        return 0.52 if name == "albert" else None


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "ok": True, "data": data, "error": None}


def err(code: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": False,
        "data": data,
        "error": {"code": code, "message": f"internal detail for {code}"},
    }


Reply = dict[str, Any] | str | BaseException | Callable[[list[str], str | None], Any]


class FakeCcc:
    """Answers per subcommand (``sessions``, ``inspect``, ``send``, ``answer``)."""

    def __init__(self) -> None:
        self.replies: dict[str, Reply] = {}
        self.calls: list[tuple[list[str], str | None, float]] = []

    def __call__(self, argv: Sequence[str], stdin: str | None, timeout: float) -> RunResult:
        argv = list(argv)
        self.calls.append((argv, stdin, timeout))
        reply = self.replies.get(argv[1], err("internal"))
        if callable(reply) and not isinstance(reply, dict):
            reply = reply(argv, stdin)
        if isinstance(reply, BaseException):
            raise reply
        out = reply if isinstance(reply, str) else json.dumps(reply)
        return RunResult(0 if isinstance(reply, dict) and reply.get("ok") else 1, out)

    def of(self, cmd: str) -> list[tuple[list[str], str | None, float]]:
        return [c for c in self.calls if c[0][1] == cmd]


SESSIONS = [
    {"session_id": "id-voice", "name": "voice bridge", "kind": "interactive", "status": "busy",
     "aim_short": "build the Claude bridge"},
    {"session_id": "id-idle", "name": "scratch idle", "kind": "interactive", "status": "idle",
     "aim_short": ""},
    {"session_id": "id-bg", "name": "nightly", "kind": "background", "status": "busy",
     "aim_short": "nightly report"},
    {"session_id": "id-wait", "name": "Picker Demo", "kind": "interactive", "status": "waiting",
     "aim_short": "set up the example service"},
]  # fmt: skip

ASK = {
    "decision_id": "d-ask",
    "source": "ask_user_question",
    "questions": [
        {
            "text": "Which storage engine should the example service use?",
            "multi_select": False,
            "options": [
                {"label": "Embedded file", "consequence": "No extra process; fine for one user."},
                {"label": "Local server", "consequence": "Needs a running daemon."},
                {"label": "In-memory only", "consequence": "All data is lost on restart."},
            ],
        }
    ],
    "recommendation": "Embedded file",
    "recommendation_reason": "it needs no extra process",
    "context": {
        "aim": "set up the example service so that demos start in seconds",
        "ticket": "tp#855",
        "recent_summary": "Two settings need your choice before I continue.",
    },
}

MULTI = {
    "text": "Which optional features should be enabled?",
    "multi_select": True,
    "options": [
        {"label": "Metrics", "consequence": "Expose a metrics endpoint."},
        {"label": "Audit log", "consequence": "Write every change to a log file."},
        {"label": "Email alerts", "consequence": "Send a mail when a job fails."},
    ],
}

TODO = {
    "decision_id": "d-todo",
    "source": "todo_line",
    "questions": [
        {
            "text": "Which cache backend should the build use?",
            "multi_select": False,
            "options": [
                {"label": "Local disk", "consequence": "fastest, no setup"},
                {"label": "Shared bucket", "consequence": "survives a machine wipe"},
                {"label": "No cache", "consequence": "slowest but simplest"},
            ],
        },
        {
            "text": "Release the patch today or wait for the second review?",
            "multi_select": False,
            "options": [
                {"label": "Release the patch today", "consequence": None},
                {"label": "Wait for the second review", "consequence": None},
            ],
        },
    ],
    "recommendation": "1) Local disk; 2) waiting for the second review",
    "recommendation_reason": "1) fastest, no setup; 2) the migration touches stored data",
    "context": {"aim": None, "ticket": None, "recent_summary": "The build is faster now."},
}


def inspect_data(decision: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    base = {"state": "idle", "live": True, "last_reply": "", "last_prompt": ""}
    return ok(base | kw | {"decision": decision})


class Rig:
    """A controller in a call + the session tools on a fake ccc."""

    def __init__(self, sessions: list[dict[str, Any]] | None = None) -> None:
        self.clock = Clock()
        self.problems = MemoryProblemSink()
        self.deliveries = MemoryDeliveryTracker()
        auth = Authoriser(FakeScorer(), "albert")
        self.ctl = BridgeController(
            auth,
            vad_factory=FakeVad,
            clock=self.clock,
            problems=self.problems,
            deliveries=self.deliveries,
        )
        self.ctl.begin_call()
        self.ccc = FakeCcc()
        self.ccc.replies["sessions"] = ok({"sessions": sessions or SESSIONS})
        tools = register(
            self.ctl, CccClient(self.ccc, ccc=CCC), env={"MAC_VOICE_CLAUDE_BRIDGE": "1"}
        )
        assert tools is not None
        self.tools: SessionTools = tools
        self.seq = 0

    def say(self, text: str) -> None:
        for _ in range(10):
            self.clock.now += 0.1
            self.ctl.feed_audio(np.full(FRAME, ALBERT, dtype=np.float32))
        for _ in range(6):
            self.clock.now += 0.1
            self.ctl.feed_audio(np.zeros(FRAME, dtype=np.float32))
        self.clock.now += 1.5
        self.seq += 1
        self.ctl.on_transcript(self.seq, text, self.clock.now)

    def tool(self, name: str, *args: Any, **kwargs: Any) -> str:
        result = self.ctl.tools[name](*args, **kwargs)
        assert isinstance(result, str)
        return result


@pytest.fixture(name="rig")
def _rig() -> Rig:
    return Rig()


@pytest.fixture(name="code_35")
def _code_35(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge.secrets, "randbelow", lambda _n: 25)  # 25 + 10 = 35


# -- registration ----------------------------------------------------------------------------
def test_register_only_with_the_flag() -> None:
    auth = Authoriser(FakeScorer(), "albert")
    ctl = BridgeController(auth, vad_factory=FakeVad)
    assert register(ctl, CccClient(FakeCcc(), ccc=CCC), env={}) is None
    assert set(ctl.tools) == {"confirm_action"}
    assert register(ctl, CccClient(FakeCcc(), ccc=CCC), env={"MAC_VOICE_CLAUDE_BRIDGE": "1"})
    assert {
        "list_sessions",
        "read_session",
        "brief_decision",
        "send_message",
        "answer_decision",
    } <= set(ctl.tools)
    assert set(ctl.executors) == {"send", "answer"}


# -- list + targets --------------------------------------------------------------------------
def test_list_is_numbered_marks_background_and_uses_three_seconds(rig: Rig) -> None:
    text = rig.tool("list_sessions")
    assert text.splitlines() == [
        "4 sessions:",
        "1. voice bridge — busy — build the Claude bridge",
        "2. scratch idle — idle",
        "3. nightly — background, busy — nightly report",
        "4. Picker Demo — waiting — set up the example service",
    ]
    argv, stdin, timeout = rig.ccc.calls[-1]
    assert argv == [CCC, "sessions", "-j"]
    assert stdin is None
    assert timeout == 3.0


@pytest.mark.parametrize(
    ("target", "session_id"),
    [
        ("voice bridge", "id-voice"),
        ("Voice Bridge", "id-voice"),
        ("voice-bridge", "id-voice"),
        ("picker demo", "id-wait"),
        ("id-idle", "id-idle"),
    ],
)
def test_target_by_name(rig: Rig, target: str, session_id: str) -> None:
    found = rig.tools.resolve(target)
    assert found.session is not None and found.session.session_id == session_id


@pytest.mark.parametrize("target", ["2", "number two", "Nummer zwei", "numéro deux"])
def test_target_by_number_from_the_latest_list(rig: Rig, target: str) -> None:
    rig.tool("list_sessions")
    found = rig.tools.resolve(target)
    assert found.session is not None and found.session.name == "scratch idle"


def test_number_without_a_list_or_out_of_range_asks(rig: Rig) -> None:
    assert "no numbered list" in rig.tools.resolve("2").question
    rig.tool("list_sessions")
    assert "The list has 4 sessions" in rig.tools.resolve("9").question


def test_unknown_target_offers_close_candidates(rig: Rig) -> None:
    found = rig.tools.resolve("voice bridg")
    assert found.session is None
    assert "Did you mean voice bridge" in found.question
    unknown = rig.tools.resolve("zebra")
    assert unknown.session is None and "Running: voice bridge, scratch idle" in unknown.question


def test_ambiguous_target_lists_the_matches() -> None:
    rows = [
        {"session_id": "a", "name": "voice-bridge", "kind": "interactive", "status": "idle"},
        {"session_id": "b", "name": "voice_bridge", "kind": "interactive", "status": "idle"},
    ]
    rig = Rig(rows)
    found = rig.tools.resolve("voice bridge")
    assert found.session is None
    assert "Several sessions match" in found.question and "voice_bridge" in found.question


def test_names_resolve_against_a_fresh_list_once_and_refetch_on_a_miss(rig: Rig) -> None:
    rig.tool("list_sessions")
    rig.tools.resolve("voice bridge")
    assert len(rig.ccc.of("sessions")) == 1  # cached list reused
    rig.ccc.replies["sessions"] = ok(
        {"sessions": [*SESSIONS, {"session_id": "id-new", "name": "fresh one"}]}
    )
    found = rig.tools.resolve("fresh one")
    assert found.session is not None and found.session.session_id == "id-new"
    assert len(rig.ccc.of("sessions")) == 2


def test_list_failure_is_spoken(rig: Rig) -> None:
    rig.ccc.replies["sessions"] = subprocess.TimeoutExpired("ccc", 3)
    assert rig.tool("list_sessions") == (
        "Cannot list the sessions: the command center did not answer in time."
    )


# -- read --------------------------------------------------------------------------------------
def test_read_last_reply_is_redacted_and_capped(rig: Rig) -> None:
    long_reply = f"token {SECRET} and key sk-{'x' * 20} " + "word " * 600
    rig.ccc.replies["inspect"] = inspect_data(last_reply=long_reply)
    text = rig.tool("read_session", "voice bridge", "last_reply")
    assert text.startswith("voice bridge last said: token [redacted] and key [redacted]")
    assert SECRET not in text and len(text) <= 1500 + 40
    argv, stdin, timeout = rig.ccc.calls[-1]
    assert argv == [CCC, "inspect", "-s", "id-voice", "-N", "-j"]
    assert stdin is None and timeout == 3.0


def test_read_last_prompt_and_empty(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(last_prompt="run the tests")
    assert rig.tool("read_session", "scratch idle", "last_prompt") == (
        "scratch idle was last asked: run the tests"
    )
    assert rig.tool("read_session", "scratch idle") == "scratch idle has no reply yet."
    assert "last reply or the last prompt" in rig.tool("read_session", "scratch idle", "aim")


def test_read_failure_maps_the_error_code(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = err("transcript_unknown")
    text = rig.tool("read_session", "voice bridge")
    assert text == "Cannot read voice bridge: its transcript could not be read."
    assert "internal detail" not in text


def test_a_read_invalidates_a_capability_minted_before_it(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(last_reply="hello")
    rig.say("open youtube")
    cap = rig.ctl.mint()
    assert isinstance(cap, Capability)
    rig.tool("read_session", "voice bridge")
    refusal = rig.ctl.caps.use(cap, "open_url", {"target": "youtube"})
    assert refusal is not None and refusal.code == "no_capability"


@pytest.mark.parametrize("tool", ["brief_decision", "list_sessions"])
def test_briefing_and_list_also_invalidate(rig: Rig, tool: str) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.say("open youtube")
    cap = rig.ctl.mint()
    assert isinstance(cap, Capability)
    rig.tool(tool, *(["picker demo"] if tool == "brief_decision" else []))
    assert rig.ctl.caps.use(cap, "open_url", {"target": "youtube"}) is not None


# -- briefings ---------------------------------------------------------------------------------
def test_briefing_single_question_follows_d5(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    text = rig.tool("brief_decision", "Picker Demo")
    assert text == (
        "The session Picker Demo, which works on set up the example service so that demos "
        "start in seconds (tp#855), needs a decision: Which storage engine should the "
        "example service use? Options: a) Embedded file — No extra process; fine for one "
        "user.; b) Local server — Needs a running daemon.; c) In-memory only — All data is "
        "lost on restart.. It recommends a) Embedded file because it needs no extra process. "
        "Recently: Two settings need your choice before I continue."
    )
    argv, _stdin, timeout = rig.ccc.calls[-1]
    assert argv == [CCC, "inspect", "-s", "id-wait", "-j"]  # with the LLM summary
    assert timeout == claude_sessions.BRIEF_TIMEOUT_S


def test_briefing_several_questions_and_multi_select(rig: Rig) -> None:
    decision = dict(ASK, questions=[*ASK["questions"], MULTI], recommendation="1) Embedded file")
    rig.ccc.replies["inspect"] = inspect_data(decision, state="waiting")
    text = rig.tool("brief_decision", "picker demo")
    assert "needs 2 decisions. 1) Which storage engine" in text
    assert "2) Which optional features should be enabled? Several can be picked. Options:" in text
    assert "It recommends 1) Embedded file because" in text


def test_briefing_todo_line_without_aim_falls_back_to_the_list_aim(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(TODO)
    text = rig.tool("brief_decision", "voice bridge")
    assert text.startswith(
        "The session voice bridge, which works on build the Claude bridge, needs 2 decisions."
    )
    assert "It recommends 1) Local disk; 2) waiting for the second review because 1)" in text


def test_briefing_is_redacted(rig: Rig) -> None:
    decision = dict(ASK, context={"aim": f"rotate {SECRET}", "recent_summary": SECRET})
    rig.ccc.replies["inspect"] = inspect_data(decision, state="waiting")
    text = rig.tool("brief_decision", "picker demo")
    assert SECRET not in text and "[redacted]" in text


def test_no_decision_is_said_plainly(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(None, state="busy")
    assert rig.tool("brief_decision", "voice bridge") == (
        "voice bridge needs no decision right now (it is busy)."
    )


def test_briefing_llm_timeout_falls_back_to_no_llm(rig: Rig) -> None:
    def reply(argv: list[str], _stdin: str | None) -> Any:
        if "-N" not in argv:
            return subprocess.TimeoutExpired("ccc", 6)
        return inspect_data(ASK, state="waiting")

    rig.ccc.replies["inspect"] = reply
    assert rig.tool("brief_decision", "picker demo").startswith("The session Picker Demo")
    timeouts = [(c[0][-2], c[2]) for c in rig.ccc.of("inspect")]
    assert timeouts == [("id-wait", 6.0), ("-N", 3.0)]


# -- background sessions (D4) ----------------------------------------------------------------
def test_background_session_is_read_only(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, last_reply="nightly done")
    assert rig.tool("read_session", "nightly") == "nightly last said: nightly done"
    assert rig.tool("brief_decision", "nightly").startswith("The session nightly")
    for result in (
        rig.tool("send_message", "nightly", "hi"),
        rig.tool("answer_decision", "nightly", "a"),
    ):
        assert "background session" in result
    assert rig.ctl.pending.live() == []
    assert not rig.ccc.of("send") and not rig.ccc.of("answer")


# -- send: proposal → confirm → executor -------------------------------------------------------
@pytest.mark.usefixtures("code_35")
def test_send_is_proposed_then_runs_with_text_on_stdin_only(rig: Rig) -> None:
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted", "channel": "api"})
    rig.say("tell voice bridge to run the tests")
    assert rig.tool("send_message", "voice bridge", "run the tests") == (
        "Say confirm 35 to send to voice bridge: run the tests"
    )
    assert not rig.ccc.of("send")  # nothing sent before the code
    rig.say("confirm 35")
    assert rig.tool("confirm_action", "35") == "sent"
    [(argv, stdin, timeout)] = rig.ccc.of("send")
    assert argv == [CCC, "send", "-s", "id-voice", "-j"]
    assert stdin == "run the tests"
    assert 0 < timeout <= 20.0
    assert not rig.problems.problems
    assert rig.deliveries.tracked == [("d1", "id-voice", "voice bridge", "accepted")]


@pytest.mark.usefixtures("code_35")
def test_send_logs_one_line_without_content(rig: Rig, caplog: pytest.LogCaptureFixture) -> None:
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted", "channel": "api"})
    caplog.set_level("INFO")
    rig.say("tell voice bridge the secret plan")
    prompt = rig.tool("send_message", "voice bridge", "the secret plan")
    assert prompt.endswith(": the secret plan")  # the agent reads the text back
    rig.say("confirm 35")
    rig.tool("confirm_action", "35")
    assert "🔒 confirm needed: send" in caplog.text
    assert "📨 send → voice bridge: accepted" in caplog.text
    assert "secret plan" not in caplog.text


@pytest.mark.usefixtures("code_35")
def test_send_text_is_cleaned_and_checked(rig: Rig) -> None:
    rig.say("tell voice bridge something")
    assert rig.tool("send_message", "voice bridge", "  \x1b\x00 ") == "What should I send?"
    rig.say("tell voice bridge something long")
    assert "too long" in rig.tool("send_message", "voice bridge", "x" * 4001)
    rig.say("tell voice bridge hi there")
    prompt = rig.tool("send_message", "voice bridge", "hi\x1b[201~ there\nnext")
    assert prompt == "Say confirm 35 to send to voice bridge: hi[201~ there next"
    [pending] = rig.ctl.pending.live()
    assert pending.args["text"] == "hi[201~ there\nnext"  # multi-line text arrives intact
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted"})
    rig.say("confirm 35")
    assert rig.tool("confirm_action", "35") == "sent"
    assert rig.ccc.of("send")[0][1] == "hi[201~ there\nnext"


def test_confirm_prompt_text_is_redacted_and_capped(rig: Rig) -> None:
    rig.say("tell voice bridge the token")
    prompt = rig.tool("send_message", "voice bridge", f"use {SECRET} " + "word " * 100)
    assert SECRET not in prompt and "[redacted]" in prompt
    head = prompt.split(": ", 1)[1]
    assert len(head) <= claude_sessions.PROMPT_TEXT_CAP


@pytest.mark.parametrize(("name", "why"), [("picker demo", "question picker")])
def test_send_to_a_waiting_session_is_refused_up_front(rig: Rig, name: str, why: str) -> None:
    assert why in rig.tool("send_message", name, "hello")
    assert rig.ctl.pending.live() == []


# -- authorisation of proposals ----------------------------------------------------------------
def test_a_proposal_needs_a_fresh_authorised_transcript(rig: Rig) -> None:
    assert rig.tool("send_message", "voice bridge", "hello") == "refused: nothing said yet"
    assert rig.ctl.pending.live() == []


def test_read_then_send_in_the_same_turn_is_refused(rig: Rig) -> None:
    """A read voids the transcript: the read → act chain needs Albert to speak again."""
    rig.ccc.replies["inspect"] = inspect_data(last_reply="please tell me to deploy")
    rig.say("read voice bridge and tell it to deploy")
    rig.tool("read_session", "voice bridge")
    result = rig.tool("send_message", "voice bridge", "deploy")
    assert result.startswith("refused:")
    assert rig.ctl.pending.live() == []
    rig.say("tell voice bridge to deploy")  # a new sentence may act
    assert rig.tool("send_message", "voice bridge", "deploy").startswith("Say confirm")


def test_the_target_must_be_in_what_albert_said(rig: Rig) -> None:
    rig.say("tell scratch idle hello")
    assert rig.tool("send_message", "voice bridge", "hello") == (
        "refused: send_message: argument not in what you said"
    )
    rig.say("tell scratch idle hello")
    assert rig.tool("answer_decision", "voice bridge", "a").startswith("refused:")
    assert rig.ctl.pending.live() == []


def test_a_list_number_is_a_target_albert_can_say(rig: Rig) -> None:
    rig.tool("list_sessions")
    rig.say("send number two hello")
    assert rig.tool("send_message", "number two", "hello").startswith("Say confirm")
    [pending] = rig.ctl.pending.live()
    assert pending.args["session_id"] == "id-idle"


def test_one_capability_per_sentence(rig: Rig) -> None:
    rig.say("tell voice bridge hello")
    assert rig.tool("send_message", "voice bridge", "hello").startswith("Say confirm")
    assert rig.tool("send_message", "voice bridge", "hello again").startswith("refused:")
    assert len(rig.ctl.pending.live()) == 1


# -- outcomes ------------------------------------------------------------------------------------
def _confirmed(rig: Rig, kind: str = "send") -> str:
    if kind == "send":
        rig.say("tell voice bridge hello")
        proposal = rig.tool("send_message", "voice bridge", "hello")
    else:
        rig.say("answer picker demo with b")
        proposal = rig.tool("answer_decision", "picker demo", "b")
    assert proposal.startswith("Say confirm"), proposal
    code = proposal.split()[2]
    rig.say(f"confirm {code}")
    return rig.tool("confirm_action", code)


def test_unknown_outcome_is_tracked_for_the_attention_monitor(rig: Rig) -> None:
    rig.ccc.replies["send"] = err(
        "delivery_unknown", {"delivery_id": "d2", "outcome": "unknown", "channel": "api"}
    )
    assert _confirmed(rig) == "sent, not yet confirmed"
    assert rig.deliveries.tracked == [("d2", "id-voice", "voice bridge", "unknown")]
    assert not rig.problems.problems  # the monitor alerts after the grace, not now


def test_failed_delivery_is_spoken_and_reported(rig: Rig) -> None:
    rig.ccc.replies["send"] = err(
        "delivery_failed", {"delivery_id": "d3", "outcome": "failed", "channel": "api"}
    )
    assert _confirmed(rig) == "failed: voice bridge: the message did not reach the session"
    [problem] = rig.problems.problems
    assert problem.kind == "delivery_failed" and "hello" not in problem.reason
    assert not rig.deliveries.tracked


class KeyedSink:
    """A problem sink that takes a dedupe key (like the attention inbox)."""

    def __init__(self) -> None:
        self.reports: list[tuple[Problem, str | None]] = []

    def report(self, problem: Problem, *, dedupe_key: str | None = None) -> None:
        self.reports.append((problem, dedupe_key))


@pytest.mark.parametrize("code", sorted(POST_ACTION_FAILURES))
def test_post_action_failures_reach_the_inbox_once_per_session(rig: Rig, code: str) -> None:
    sink = KeyedSink()
    rig.ctl.problems = sink
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    reply = err(code, {"delivery_id": "dx", "outcome": "failed"})
    rig.ccc.replies["send"] = rig.ccc.replies["answer"] = reply
    kind, name, session_id = (
        ("send", "voice bridge", "id-voice")
        if code == "delivery_failed"
        else ("answer", "Picker Demo", "id-wait")
    )
    assert _confirmed(rig, kind) == f"failed: {name}: {REASONS[code]}"
    assert len(sink.reports) == 1
    problem, key = sink.reports[0]
    assert key == f"failed:{session_id}"
    assert problem.reason == REASONS[code]


PRE_ACTION = sorted(set(REASONS) - UNCERTAIN_CODES - {"delivery_unknown", "answer_unknown"})


@pytest.mark.parametrize("code", PRE_ACTION)
def test_every_error_code_maps_to_a_speakable_failure(rig: Rig, code: str) -> None:
    rig.ccc.replies["send"] = err(code)
    result = _confirmed(rig)
    assert result == f"failed: voice bridge: {REASONS[code]}"
    assert "internal detail" not in result
    assert not rig.problems.problems  # refused before acting: spoken only


@pytest.mark.parametrize(
    "reply",
    [subprocess.TimeoutExpired("ccc", 20), "garbage", json.dumps({"schema_version": 2})],
)
def test_send_without_a_readable_answer_is_not_yet_confirmed(rig: Rig, reply: Any) -> None:
    rig.ccc.replies["send"] = reply
    assert _confirmed(rig) == "sent, not yet confirmed"
    assert len(rig.ccc.of("send")) == 1  # never retried
    assert not rig.problems.problems


def test_answer_without_a_readable_answer_is_not_yet_confirmed(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.ccc.replies["answer"] = subprocess.TimeoutExpired("ccc", 20)
    assert _confirmed(rig, kind="answer") == "answered, not yet confirmed"
    assert len(rig.ccc.of("answer")) == 1


def test_a_sink_error_never_masks_the_outcome(rig: Rig) -> None:
    class Broken:
        def report(self, problem: Problem) -> None:
            raise RuntimeError(problem.kind)

        def track_delivery(self, *args: str) -> None:
            raise RuntimeError(args[0])

    rig.ctl.problems = rig.ctl.deliveries = Broken()
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted"})
    assert _confirmed(rig) == "sent"
    rig.ccc.replies["send"] = err("delivery_failed", {"delivery_id": "d2", "outcome": "failed"})
    assert _confirmed(rig) == "failed: voice bridge: the message did not reach the session"


def test_unmapped_code_gets_a_generic_reason() -> None:
    assert reason_for("brand_new_code") == "the command center refused"


def test_cancel_before_the_mutation_sends_nothing(rig: Rig) -> None:
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted"})
    rig.say("tell voice bridge hello")
    code = rig.tool("send_message", "voice bridge", "hello").split()[2]
    rig.say(f"confirm {code}")
    rig.ctl.cancel.cancel("stopped")
    assert rig.tool("confirm_action", code) == "refused: stopped"
    assert not rig.ccc.of("send")


def test_one_mutation_at_a_time(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_sessions, "LOCK_TIMEOUT_S", 0.01)
    rig.say("tell voice bridge hello")
    code = rig.tool("send_message", "voice bridge", "hello").split()[2]
    rig.say(f"confirm {code}")
    with rig.ctl.mutation_lock:
        assert rig.tool("confirm_action", code) == "refused: another action is still running"
    assert not rig.ccc.of("send")


def test_the_lock_wait_leaves_the_budget_to_ccc() -> None:
    assert claude_sessions.LOCK_TIMEOUT_S <= 2.0 < claude_sessions.MUTATION_BUDGET_S


# -- end to end with the attention inbox ---------------------------------------------------------
def test_accepted_send_then_stop_failure_alerts(rig: Rig, tmp_path: Path) -> None:
    store = AttentionStore(tmp_path / "attention.db")
    events: list[dict[str, Any]] = []

    def monitor_ccc(argv: Sequence[str], timeout: float) -> tuple[int, str]:
        del timeout
        if argv[1] == "events":
            return 0, json.dumps(ok({"events": events, "next_cursor": len(events)}))
        return 0, json.dumps(ok({"delivery_id": "d1", "state": "accepted"}))

    monitor = register_attention(
        rig.ctl, store, monitor_ccc, env={"MAC_VOICE_CLAUDE_BRIDGE": "1"}, start=False
    )
    assert monitor is not None
    monitor.ccc = ("ccc",)
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted"})
    assert _confirmed(rig) == "sent"
    [delivery] = store.open_deliveries()
    assert (delivery.delivery_id, delivery.session_id, delivery.subject) == (
        "d1",
        "id-voice",
        "voice bridge",
    )
    at = datetime.fromtimestamp(time.time() + 5, UTC).isoformat()
    events.append(
        {"cursor": 1, "kind": "stop_failure", "session_id": "id-voice", "at": at, "detail": {}}
    )
    monitor.banner = False
    monitor.poll_once()
    [item] = store.open_items()
    assert (item.kind, item.subject) == ("stop_failure", "voice bridge")
    store.close()


# -- answer_decision --------------------------------------------------------------------------
def _q(multi: bool = False) -> Question:
    labels = ("Embedded file", "Local server", "In-memory only")
    return Question("Which engine?", multi, labels, ("", "", ""))


@pytest.mark.parametrize(
    ("choice", "indices"),
    [
        ("b", (1,)),
        ("option B please", (1,)),
        ("Local server", (1,)),
        ("the local server", (1,)),
        ("in memory", (2,)),
        ("3", (2,)),
        ("two", (1,)),
        ("zwei", (1,)),
        ("the second", (1,)),
        ("la deuxième", (1,)),
        ("embeded file", (0,)),
        ("the second one", (1,)),
        ("die zweite", (1,)),
        ("le deuxième", (1,)),
        ("la deuxième une", (1,)),
        ("the first one please", (0,)),
        ("b one", (1,)),
        ("I'd go with b", (1,)),
        ("let's take c", (2,)),
        ("I'd go with the local server", (1,)),
    ],
)
def test_choice_resolution_single_select(choice: str, indices: tuple[int, ...]) -> None:
    pick = resolve_choice(_q(), choice)
    assert not isinstance(pick, str), pick
    assert pick.indices == indices and not pick.other


@pytest.mark.parametrize(
    ("choice", "indices"),
    [("a and c", (0, 2)), ("Embedded file and in-memory only", (0, 2)), ("1, 2", (0, 1))],
)
def test_choice_resolution_multi_select(choice: str, indices: tuple[int, ...]) -> None:
    pick = resolve_choice(_q(multi=True), choice)
    assert not isinstance(pick, str) and pick.indices == indices


def test_choice_other_text_and_ambiguity() -> None:
    pick = resolve_choice(_q(), "use postgres instead")
    assert not isinstance(pick, str) and pick.other == "use postgres instead"
    long = resolve_choice(_q(), "postgres " * 100)
    assert not isinstance(long, str) and len(long.other) == claude_sessions.OTHER_TEXT_MAX
    assert isinstance(resolve_choice(_q(), "a and b"), str)  # single-select, two options
    assert isinstance(resolve_choice(_q(multi=True), "use postgres"), str)


@pytest.mark.usefixtures("code_35")
def test_picker_answer_proposes_then_runs_ccc_answer_with_json_stdin(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.ccc.replies["answer"] = ok({"delivery_id": "a1", "outcome": "accepted"})
    rig.say("answer picker demo with b")
    assert rig.tool("answer_decision", "picker demo", "b") == (
        "Say confirm 35 to answer Picker Demo with b) Local server."
    )
    rig.say("confirm 35")
    assert rig.tool("confirm_action", "35") == "answered"
    [(argv, stdin, timeout)] = rig.ccc.of("answer")
    assert argv == [CCC, "answer", "-s", "id-wait", "-j"]
    assert stdin is not None
    assert json.loads(stdin) == {
        "decision_id": "d-ask",
        "answers": [{"question_index": 0, "option_indices": [1]}],
    }
    assert 0 < timeout <= 20.0


def test_picker_answer_multi_question_and_other_text(rig: Rig) -> None:
    decision = dict(ASK, questions=[*ASK["questions"], MULTI])
    rig.ccc.replies["inspect"] = inspect_data(decision, state="waiting")
    rig.say("answer picker demo: use sqlite; metrics and email alerts")
    rig.tool("answer_decision", "picker demo", "use sqlite; metrics and email alerts")
    [pending] = rig.ctl.pending.live()
    assert pending.kind == "answer"
    assert pending.args["answers"] == [
        {"question_index": 0, "other_text": "use sqlite"},
        {"question_index": 1, "option_indices": [0, 2]},
    ]
    rig.say("answer picker demo with a")
    assert "one answer per question" in rig.tool("answer_decision", "picker demo", "a")


@pytest.mark.usefixtures("code_35")
def test_todo_line_decision_is_answered_with_a_message(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(TODO)
    rig.ccc.replies["send"] = ok({"delivery_id": "d1", "outcome": "accepted"})
    rig.say("answer scratch idle with b and wait for the second review")
    proposal = rig.tool("answer_decision", "scratch idle", "b; wait for the second review")
    assert proposal == (
        "Say confirm 35 to answer scratch idle with b) Shared bucket; "
        "b) Wait for the second review."
    )
    rig.say("confirm 35")
    assert rig.tool("confirm_action", "35") == "sent"
    [(argv, stdin, _timeout)] = rig.ccc.of("send")
    assert argv == [CCC, "send", "-s", "id-idle", "-j"]
    assert stdin == (
        "Which cache backend should the build use? → Shared bucket "
        "Release the patch today or wait for the second review? → Wait for the second review"
    )
    assert not rig.ccc.of("answer")


def test_no_open_decision(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(None)
    rig.say("answer scratch idle with a")
    assert rig.tool("answer_decision", "scratch idle", "a") == "scratch idle has no open decision."


def test_decision_changed_after_the_briefing_is_caught_before_proposing(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.tool("brief_decision", "picker demo")
    rig.ccc.replies["inspect"] = inspect_data(dict(ASK, decision_id="d-new"), state="waiting")
    rig.say("answer picker demo with a")
    assert "decision changed" in rig.tool("answer_decision", "picker demo", "a")
    assert rig.ctl.pending.live() == []


def test_decision_changed_between_brief_and_confirm(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.tool("brief_decision", "picker demo")
    rig.ccc.replies["answer"] = err("decision_changed")
    result = _confirmed(rig, kind="answer")
    assert result == (
        "failed: Picker Demo: the decision changed meanwhile — ask for the briefing again"
    )
    assert not rig.problems.problems  # refused before any key was typed: spoken only


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (ok({"delivery_id": "a", "outcome": "accepted"}), "answered"),
        (err("answer_unknown", {"outcome": "unknown"}), "answered, not yet confirmed"),
        (
            err("answer_mismatch", {"outcome": "failed", "mismatched_questions": [0]}),
            "failed: Picker Demo: the session recorded a different answer",
        ),
        (
            err("unsupported_shape"),
            "failed: Picker Demo: that kind of question cannot be answered by voice",
        ),
    ],
)
def test_answer_outcomes(rig: Rig, reply: dict[str, Any], expected: str) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.ccc.replies["answer"] = reply
    assert _confirmed(rig, kind="answer") == expected


# -- envelope ------------------------------------------------------------------------------------
def env(version: Any = 1, okay: Any = True, data: Any = None, error: Any = None) -> str:
    return json.dumps({"schema_version": version, "ok": okay, "data": data, "error": error})


@pytest.mark.parametrize(
    ("stdout", "code"),
    [
        (env(2, data={}), "schema_mismatch"),
        (env(True, data={}), "schema_mismatch"),
        (json.dumps({"ok": True, "data": {}, "error": None}), "schema_mismatch"),
        ("not json", "bad_envelope"),
        ("", "bad_envelope"),
        ("[1, 2]", "bad_envelope"),
        (env(1, True, None), "bad_envelope"),
        (env(1, "yes", {}), "bad_envelope"),
        (env(1, False, None, None), "bad_envelope"),
        (env(1, False, None, {"message": "no code"}), "bad_envelope"),
    ],
)
def test_envelope_is_parsed_strictly(stdout: str, code: str) -> None:
    result = parse_envelope(stdout)
    assert not result.ok and result.code == code


def test_failed_envelope_keeps_its_data() -> None:
    result = parse_envelope(json.dumps(err("delivery_unknown", {"delivery_id": "d"})))
    assert result.code == "delivery_unknown" and result.data == {"delivery_id": "d"}


def test_schema_mismatch_is_spoken(rig: Rig) -> None:
    rig.ccc.replies["sessions"] = json.dumps(
        {"schema_version": 2, "ok": True, "data": {"sessions": []}, "error": None}
    )
    assert "newer format" in rig.tool("list_sessions")


def test_missing_ccc_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_sessions, "resolve_ccc", lambda: None)
    assert CccClient(FakeCcc()).sessions().code == "ccc_missing"


def test_resolve_ccc_prefers_the_env_override() -> None:
    assert claude_sessions.resolve_ccc({"MAC_VOICE_CCC_BIN": "/opt/ccc"}) == "/opt/ccc"


def test_tools_never_raise(rig: Rig) -> None:
    def boom(_argv: list[str], _stdin: str | None) -> Any:
        raise RuntimeError("bug")

    rig.ccc.replies["sessions"] = boom
    assert rig.tool("list_sessions") == "list_sessions failed: internal error"
    assert "missing or unexpected" in rig.tool("read_session")


# -- budgets, caches, fail-closed defaults -------------------------------------------------------
def test_brief_decision_shares_one_ten_second_budget(rig: Rig) -> None:
    def slow_list(_argv: list[str], _stdin: str | None) -> Any:
        rig.clock.now += 3.0
        return ok({"sessions": SESSIONS})

    def slow_inspect(argv: list[str], _stdin: str | None) -> Any:
        if "-N" not in argv:
            rig.clock.now += 6.0
            return subprocess.TimeoutExpired("ccc", 6)
        return inspect_data(ASK, state="waiting")

    rig.ccc.replies["sessions"] = slow_list
    rig.ccc.replies["inspect"] = slow_inspect
    assert rig.tool("brief_decision", "picker demo").startswith("The session Picker Demo")
    timeouts = [c[2] for c in rig.ccc.of("inspect")]
    assert timeouts == [6.0, pytest.approx(1.0)]  # 10 s − 3 s list − 6 s LLM inspect


def test_no_time_left_starts_no_ccc(rig: Rig) -> None:
    def very_slow_list(_argv: list[str], _stdin: str | None) -> Any:
        rig.clock.now += 11.0
        return ok({"sessions": SESSIONS})

    rig.ccc.replies["sessions"] = very_slow_list
    text = rig.tool("brief_decision", "picker demo")
    assert text == "Cannot read Picker Demo: the command center did not answer in time."
    assert not rig.ccc.of("inspect")


def test_run_group_kills_the_whole_process_group() -> None:
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        # the grandchild keeps stdout open: subprocess.run would wait for it
        run_group(["sh", "-c", "sleep 30 & sleep 30"], None, 0.3)
    assert time.monotonic() - started < 5.0
    assert run_group(["sh", "-c", "cat; echo done"], "in\n", 5.0) == RunResult(0, "in\ndone\n")


def test_a_session_without_kind_is_read_only() -> None:
    rig = Rig([{"session_id": "id-x", "name": "mystery", "status": "idle"}])
    rig.say("tell mystery hello")
    assert "background session" in rig.tool("send_message", "mystery", "hello")
    assert "background" in rig.tool("list_sessions")


def test_the_numbered_list_is_forgotten_at_a_new_call(rig: Rig) -> None:
    rig.tool("list_sessions")
    assert rig.tools.resolve("2").session is not None
    rig.ctl.end_call()
    rig.ctl.begin_call()
    assert "no numbered list" in rig.tools.resolve("2").question


def test_the_numbered_list_expires_after_ten_minutes(rig: Rig) -> None:
    rig.tool("list_sessions")
    rig.clock.now += claude_sessions.CACHE_TTL_S + 1
    assert "no numbered list" in rig.tools.resolve("2").question


def test_briefings_are_forgotten_at_a_new_call(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(ASK, state="waiting")
    rig.tool("brief_decision", "picker demo")
    rig.ctl.end_call()
    rig.ctl.begin_call()
    rig.ccc.replies["inspect"] = inspect_data(dict(ASK, decision_id="d-new"), state="waiting")
    rig.say("answer picker demo with a")
    assert rig.tool("answer_decision", "picker demo", "a").startswith("Say confirm")


def test_a_tail_cut_reply_loses_its_partial_first_word(rig: Rig) -> None:
    rig.ccc.replies["inspect"] = inspect_data(last_reply="…p_" + "A" * 30 + " and the rest")
    assert rig.tool("read_session", "voice bridge") == "voice bridge last said: … and the rest"
