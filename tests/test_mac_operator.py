"""Mac operator lifecycle: argv, environment, process group, states, confirmation, audit.

No screen and no ``claude``: a fake ``claude`` script (a small Python program written per
test) plays the operator — it records what it saw, writes the broker's files into its cwd
(the call dir) and then behaves as the test needs (finish, block, flood stdout, hang).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import mac_broker, mac_operator
from my_stt_tts.bridge import Authoriser, BridgeController, MemoryProblemSink
from my_stt_tts.mac_operator import (
    CANCELLED,
    DONE,
    FAILED,
    NEEDS_CONFIRMATION,
    RUNNING,
    UNKNOWN_PARTIAL,
    MacOperator,
    OperatorResult,
    kill_group,
    operator_argv,
    operator_env,
    register,
)

FRAME = 1600
ALBERT, OTHER = 0.5, 0.3
SCORES = {ALBERT: 0.52, OTHER: 0.18}


# -- an authorised call (same fakes as test_bridge_auth) -----------------------------------
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


class Call:
    def __init__(self) -> None:
        self.clock = Clock()
        self.problems = MemoryProblemSink()
        auth = Authoriser(FakeScorer(), "albert")
        self.ctl = BridgeController(
            auth, vad_factory=FakeVad, clock=self.clock, problems=self.problems
        )
        self.ctl.begin_call()
        self.seq = 0

    def say(self, text: str, level: float = ALBERT) -> None:
        for _ in range(10):
            self.clock.now += 0.1
            self.ctl.feed_audio(np.full(FRAME, level, dtype=np.float32))
        for _ in range(6):
            self.clock.now += 0.1
            self.ctl.feed_audio(np.zeros(FRAME, dtype=np.float32))
        self.clock.now += 1.5
        self.seq += 1
        self.ctl.on_transcript(self.seq, text, self.clock.now)


# -- the fake claude -------------------------------------------------------------------------
_FAKE_HEAD = """#!{python}
import json, os, signal, subprocess, sys, time
from pathlib import Path
here = Path.cwd()
seen = {{
    "argv": sys.argv[1:],
    "env": dict(os.environ),
    "cwd": str(here),
    "env_file_visible": os.path.exists(".env"),
    "stdin_len": len(sys.stdin.read()),
    "pgid_is_own": os.getpgid(0) == os.getpid(),
}}
(here / "fake_seen.json").write_text(json.dumps(seen))
"""

MODES = {
    "finish": """
(here / "finish.json").write_text(json.dumps(
    {"status": "done", "requested_status": "done", "reason": "", "summary": "Calculator shows 4."}))
time.sleep(60)  # the operator must not wait for this
""",
    "hang": """
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(120)
""",
    "flood": """
sys.stdout.write("x" * 3_000_000)
sys.stdout.flush()
""",
    "output": """
print(json.dumps({"structured_output":
    {"status": "done", "summary": "said done", "evidence_obs_id": None}}))
""",
    "evidence": """
(here / "evidence.json").write_text(json.dumps(EVIDENCE))
print(json.dumps({"structured_output": STRUCTURED}))
""",
    "block": """
sha = "b" * 64
nonce = "n0nce"
(here / f"blocked-{sha}.json").write_text(json.dumps(
    {"tool": "key", "sha256": sha, "summary": "press cmd+w in Safari", "nonce": nonce}))
allow = here / f"allow-{sha}.json"
for _ in range(400):
    if allow.exists():
        data = json.loads(allow.read_text())
        ok = data.get("nonce") == nonce
        allow.unlink()
        (here / f"blocked-{sha}.json").unlink()
        status = "done" if ok else "failed"
        (here / "finish.json").write_text(json.dumps(
            {"status": status, "reason": "", "summary": "Closed the tab."}))
        time.sleep(60)
    time.sleep(0.05)
(here / f"blocked-{sha}.json").unlink()
(here / "finish.json").write_text(json.dumps(
    {"status": "needs_confirmation", "reason": "", "summary": "Not confirmed."}))
time.sleep(60)
""",
}


def fake_claude(tmp_path: Path, mode: str, prelude: str = "") -> Path:
    path = tmp_path / f"fake-claude-{mode}"
    head = _FAKE_HEAD.format(python=sys.executable)
    path.write_text(head + prelude + textwrap.dedent(MODES[mode]))
    path.chmod(0o755)
    return path


def make_operator(
    tmp_path: Path, call: Call, mode: str, prelude: str = "", **kw: Any
) -> MacOperator:
    kw.setdefault("start_wait", 2.0)
    kw.setdefault("kill_grace", 0.5)
    kw.setdefault("poll", 0.05)
    claude = fake_claude(tmp_path, mode, prelude)
    op = MacOperator(call.ctl, root=tmp_path / "operator", claude=claude, **kw)
    call.ctl.register_executor("operator_confirm", op.confirm_executor)
    return op


def wait_done(op: MacOperator, timeout: float = 10.0) -> OperatorResult:
    call = op.current
    assert call is not None
    assert call.done.wait(timeout), f"operator still {call.state}"
    assert call.result is not None
    return call.result


def seen(op: MacOperator) -> dict[str, Any]:
    assert op.current is not None
    return json.loads((op.current.call_dir / "fake_seen.json").read_text())


def started(call: Call, op: MacOperator, request: str = "open Calculator and compute 2+2") -> str:
    call.say(request)
    return op.do_on_mac(request)


# -- argv + environment ----------------------------------------------------------------------
def test_exact_argv_with_inline_schema_and_no_persistence(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    started(call, op)
    wait_done(op)
    argv = seen(op)["argv"]
    assert op.current is not None
    broker_json = str(op.current.call_dir / "broker.json")
    schema = mac_operator.inline_schema()
    assert argv == [
        "-p",
        "--model",
        "sonnet",
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        broker_json,
        "--permission-mode",
        "bypassPermissions",
        "--setting-sources",
        "",
        "--max-turns",
        "30",
        "--output-format",
        "json",
        "--json-schema",
        schema,
        "--no-session-persistence",
    ]
    parsed = json.loads(schema)
    assert parsed["$schema"].startswith("http://json-schema.org/draft-07")
    assert "\n" not in schema and not schema.endswith(".json")


def test_broker_config_launches_this_broker_isolated(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    started(call, op)
    wait_done(op)
    assert op.current is not None
    cfg_path = op.current.call_dir / "broker.json"
    assert cfg_path.stat().st_mode & 0o777 == 0o600
    server = json.loads(cfg_path.read_text())["mcpServers"]["mac"]
    assert server["command"] == sys.executable
    assert server["args"][:2] == ["-I", str(Path(mac_broker.__file__).resolve())]
    assert server["args"][2:] == [
        "-c",
        str(op.current.call_dir),
        "-r",
        str(tmp_path / "operator"),
    ]


def test_env_is_built_from_scratch_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-should-never-leak")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-nope")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/elsewhere")
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    started(call, op)
    wait_done(op)
    env = seen(op)["env"]
    env.pop("__CF_USER_TEXT_ENCODING", None)  # added by macOS to every process
    home = str(Path.home())
    assert set(env) == {"PATH", "HOME", "LANG", "USER", "CLAUDE_CONFIG_DIR", "AI_NO_AUTOCOMMIT"}
    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["HOME"] == home and env["USER"]
    assert env["CLAUDE_CONFIG_DIR"] == f"{home}/.claude-work"
    assert env["AI_NO_AUTOCOMMIT"] == "1"
    assert not any(k.startswith("ELEVENLABS") for k in env)
    assert "sk-" not in json.dumps(env)


def test_operator_env_needs_user_even_without_it() -> None:
    env = operator_env(Path("/Users/x"), base={})
    assert env["USER"] and env["LANG"] == "en_US.UTF-8"
    assert env["CLAUDE_CONFIG_DIR"] == "/Users/x/.claude-work"


def test_cwd_is_a_fresh_private_dir_without_env_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ELEVENLABS_API_KEY=secret\n")  # a repo-like parent .env
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    started(call, op)
    wait_done(op)
    info = seen(op)
    cwd = Path(info["cwd"]).resolve()
    root = (tmp_path / "operator").resolve()
    assert cwd.parent == root and cwd.name.startswith("op-")
    assert cwd.stat().st_mode & 0o777 == 0o700 and root.stat().st_mode & 0o777 == 0o700
    assert info["env_file_visible"] is False
    with pytest.raises(ValueError):  # the broker refuses a call dir outside its root
        mac_broker.check_call_dir(tmp_path, root)


# -- states ----------------------------------------------------------------------------------
def test_finish_json_returns_early_and_kills_the_run(tmp_path: Path) -> None:
    call = Call()
    results: list[OperatorResult] = []
    op = make_operator(tmp_path, call, "finish", on_result=results.append)
    t0 = time.monotonic()
    answer = started(call, op)
    result = wait_done(op)
    assert time.monotonic() - t0 < 5  # the fake sleeps 60 s after writing finish.json
    assert answer == "done: Calculator shows 4."
    assert result.status == DONE and results == [result]
    assert op.current is not None and op.current.proc is not None
    assert op.current.proc.poll() is not None  # terminated
    assert seen(op)["pgid_is_own"] is True  # own process group
    assert not call.problems.problems  # success is silent


def test_states_needs_confirmation_then_done_after_confirm(tmp_path: Path) -> None:
    call = Call()
    notices: list[str] = []
    op = make_operator(tmp_path, call, "block", notify=notices.append)
    answer = started(call, op, "close this tab")
    assert answer.startswith("needs confirmation: press cmd+w in Safari — say confirm ")
    assert op.current is not None and op.current.state == NEEDS_CONFIRMATION
    deadline = time.monotonic() + 2.0  # the watcher notifies right after the answer is ready
    while not notices and time.monotonic() < deadline:
        time.sleep(0.01)
    assert notices and notices[0] in answer
    code = answer.rsplit(" ", 1)[1]
    call.say(f"confirm {code}")
    assert call.ctl.confirm_action(code) == "confirmed"
    result = wait_done(op)
    assert result.status == DONE and result.summary == "Closed the tab."


def test_confirm_needs_a_new_authorised_transcript(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "block")
    answer = started(call, op, "close this tab")
    code = answer.rsplit(" ", 1)[1]
    call.say(f"confirm {code}", level=OTHER)  # someone else says the code
    assert call.ctl.confirm_action(code).startswith("refused")
    assert op.current is not None and op.current.state == NEEDS_CONFIRMATION
    op.cancel("test over")
    assert wait_done(op).status == CANCELLED


def test_returns_started_after_the_wait(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "hang", start_wait=0.5)
    assert started(call, op).startswith("started")
    assert op.current is not None and op.current.state == RUNNING
    op.cancel()
    assert wait_done(op).status == CANCELLED


def test_hard_limit_kills_and_reports_a_problem(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "hang", start_wait=0.2, hard_limit=1.0)
    started(call, op)
    result = wait_done(op)
    assert result.status == FAILED and result.reason == "timeout"
    assert [(p.kind, p.reason) for p in call.problems.problems] == [("operator", "timeout")]
    assert mac_operator.HARD_LIMIT_S == 300.0


def test_call_end_cancels_a_foreground_run(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "hang", start_wait=0.2)
    started(call, op)
    call.ctl.end_call()
    result = wait_done(op)
    assert result.status == CANCELLED and result.reason == "call ended"
    assert call.problems.problems[0].severity == "warning"


def test_background_run_survives_the_hang_up_only_when_said(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "hang", start_wait=0.2)
    call.say("download the report, do it even if I hang up")
    op.do_on_mac("download the report", background=True)
    assert op.current is not None and op.current.background is True
    call.ctl.end_call()
    time.sleep(0.4)
    assert not op.current.done.is_set()
    op.shutdown()
    assert op.current.result is not None and op.current.result.status == CANCELLED
    # the flag alone (not said) does not make it a background run
    call2 = Call()
    (tmp_path / "b").mkdir()
    op2 = make_operator(tmp_path / "b", call2, "hang", start_wait=0.2)
    call2.say("download the report")
    op2.do_on_mac("download the report", background=True)
    assert op2.current is not None and op2.current.background is False
    op2.shutdown()


def test_done_without_broker_finish_is_unknown_partial(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "output")
    started(call, op)
    result = wait_done(op)
    assert result.status == UNKNOWN_PARTIAL
    assert call.problems.problems and call.problems.problems[0].kind == "operator"


def _evidence_run(
    tmp_path: Path, evidence: dict[str, Any] | None, obs_id: str | None
) -> OperatorResult:
    """Structured ``done`` without finish.json, plus the given broker evidence.json."""
    structured = {"status": "done", "summary": "Calculator\nshows 42.", "evidence_obs_id": obs_id}
    prelude = f"EVIDENCE = {evidence!r}\nSTRUCTURED = {structured!r}\n"
    mode = "evidence" if evidence is not None else "output"
    call = Call()
    op = make_operator(tmp_path, call, mode, prelude=prelude)
    started(call, op)
    return wait_done(op)


def test_structured_done_with_evidence_after_the_last_action_is_done(tmp_path: Path) -> None:
    evidence = {"last_action_seq": 3, "observations": {"obs-2-aa": 2, "obs-4-bb": 4}}
    result = _evidence_run(tmp_path, evidence, "obs-4-bb")
    assert result.status == DONE and result.reason == ""
    assert result.summary == "Calculator shows 42."


def test_structured_done_with_evidence_older_than_the_last_action(tmp_path: Path) -> None:
    evidence = {"last_action_seq": 3, "observations": {"obs-2-aa": 2}}
    result = _evidence_run(tmp_path, evidence, "obs-2-aa")
    assert result.status == UNKNOWN_PARTIAL and result.reason == "unverified"
    assert result.summary == ""


def test_structured_done_with_unknown_obs_id(tmp_path: Path) -> None:
    evidence = {"last_action_seq": 3, "observations": {"obs-4-bb": 4}}
    result = _evidence_run(tmp_path, evidence, "obs-9-zz")
    assert result.status == UNKNOWN_PARTIAL and result.reason == "unverified"


def test_structured_done_without_evidence_json(tmp_path: Path) -> None:
    result = _evidence_run(tmp_path, None, "obs-4-bb")
    assert result.status == UNKNOWN_PARTIAL and result.reason == "unverified"


def test_stdout_is_capped(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "flood", stdout_cap=1000)
    started(call, op)
    result = wait_done(op)
    assert op.current is not None
    assert len(op.current.stdout) == 1000 and op.current.stdout_truncated
    assert result.status == FAILED and result.reason == "output truncated"
    assert mac_operator.STDOUT_CAP == 1024 * 1024


def test_daemon_shutdown_stops_the_run_via_the_controller(tmp_path: Path) -> None:
    call = Call()
    op = register(
        call.ctl,
        env={"MAC_VOICE_OPERATOR": "1"},
        root=tmp_path / "operator",
        claude=fake_claude(tmp_path, "hang"),
        start_wait=0.2,
        kill_grace=0.5,
        poll=0.05,
    )
    assert op is not None
    call.say("open Calculator")
    call.ctl.tools["do_on_mac"]("open Calculator")
    call.ctl.shutdown()
    assert op.current is not None and op.current.done.is_set()
    assert op.current.result is not None and op.current.result.status == CANCELLED
    assert op.current.proc is not None and op.current.proc.poll() is not None


# -- authorisation ---------------------------------------------------------------------------
def test_needs_a_capability_from_albert(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    assert op.do_on_mac("open Calculator").startswith("refused")  # nothing said yet
    call.say("open Calculator", level=OTHER)
    assert op.do_on_mac("open Calculator") == "refused: voice not verified"
    assert op.current is None


def test_capability_is_single_use_and_one_task_at_a_time(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "hang", start_wait=0.2)
    started(call, op)
    assert op.do_on_mac("open Notes") == "refused: another Mac task is still running"
    op.cancel()
    wait_done(op)
    assert op.do_on_mac("open Notes").startswith("refused")  # the transcript's capability is used


def test_executor_refuses_when_nothing_waits(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    pending = call.ctl.propose("operator_confirm", {"call_id": "op-x", "sha": "a" * 64})
    assert op.confirm_executor(pending).startswith("refused")


# -- process group kill ----------------------------------------------------------------------
_STUBBORN = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"])
open(sys.argv[1], "w").write(str(child.pid))
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_kill_group_terms_then_kills_after_grace(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    proc = subprocess.Popen(  # pylint: disable=consider-using-with  # killed by the test
        [sys.executable, "-c", _STUBBORN, str(pid_file)], start_new_session=True
    )
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        time.sleep(0.05)
    grandchild = int(pid_file.read_text())
    t0 = time.monotonic()
    assert kill_group(proc, grace=0.6) is True  # TERM ignored → KILL needed
    assert time.monotonic() - t0 >= 0.6
    assert proc.poll() == -signal.SIGKILL
    for _ in range(50):
        if not _alive(grandchild):
            break
        time.sleep(0.05)
    assert not _alive(grandchild)
    assert mac_operator.KILL_GRACE_S == 3.0


def test_kill_group_term_is_enough_for_a_polite_child() -> None:
    proc = subprocess.Popen(  # pylint: disable=consider-using-with  # killed by the test
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    time.sleep(0.2)
    t0 = time.monotonic()
    assert kill_group(proc, grace=3.0) is False
    assert time.monotonic() - t0 < 2.0 and proc.poll() == -signal.SIGTERM


# -- audit + registration ----------------------------------------------------------------------
def test_audit_has_no_content_and_mode_0600(tmp_path: Path) -> None:
    call = Call()
    op = make_operator(tmp_path, call, "finish")
    request = "open Calculator and compute 2+2 secret-marker"
    call.say(request)
    op.do_on_mac(request)
    wait_done(op)
    audit = op.audit_path
    assert audit.stat().st_mode & 0o777 == 0o600
    raw = audit.read_text()
    assert "secret-marker" not in raw and "Calculator" not in raw
    rec = json.loads(raw.splitlines()[-1])
    assert rec["request_len"] == len(request) and len(rec["request_sha256"]) == 64
    assert rec["status"] == DONE and rec["duration_s"] >= 0
    assert op.current is not None
    assert not list(op.current.call_dir.glob("*.jpg"))


def test_register_only_with_the_flag(tmp_path: Path) -> None:
    call = Call()
    assert register(call.ctl, env={}) is None
    assert register(call.ctl, env={"MAC_VOICE_OPERATOR": "0"}) is None
    assert "do_on_mac" not in call.ctl.tools and "operator_confirm" not in call.ctl.executors
    op = register(call.ctl, env={"MAC_VOICE_OPERATOR": "1"}, root=tmp_path / "operator")
    assert isinstance(op, MacOperator)
    assert call.ctl.tools["do_on_mac"].__self__ is op  # type: ignore[attr-defined]
    executor = call.ctl.executors["operator_confirm"]
    assert executor.__func__ is MacOperator.confirm_executor  # type: ignore[attr-defined]


def test_argv_helper_matches_spec_order(tmp_path: Path) -> None:
    argv = operator_argv(tmp_path / "broker.json", claude=Path("/x/claude"), schema="{}")
    assert argv[0] == "/x/claude" and argv[-1] == "--no-session-persistence"
    assert argv[argv.index("--json-schema") + 1] == "{}"
    assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--setting-sources") + 1] == ""
