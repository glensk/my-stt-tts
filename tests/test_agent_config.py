"""scripts/eleven_agent_config.py against a fake ElevenLabs API (no network, no real agent)."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import stat
import sys
from pathlib import Path
from typing import Any

from my_stt_tts.attention import make_acknowledge_tool
from my_stt_tts.bridge import BridgeController
from my_stt_tts.claude_sessions import SessionTools
from my_stt_tts.mac_control import TOOL_NAMES, MacControl
from my_stt_tts.mac_operator import MacOperator

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eleven_agent_config.py"
_spec = importlib.util.spec_from_file_location("eleven_agent_config_under_test", _SCRIPT)
assert _spec and _spec.loader
cfg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cfg  # dataclasses resolve their module while the script loads
_spec.loader.exec_module(cfg)

AGENT = "agent_test"
SECRET = "sk-" + "A" * 24
ORIGINAL_PROMPT = "You are Albert's voice assistant.\nKeep answers short."


def _agent() -> dict[str, Any]:
    return {
        "agent_id": AGENT,
        "name": "Mac voice",
        "version_id": "v1",
        "branch_id": "main",
        "main_branch_id": "main",
        "metadata": {"created_at_unix_secs": 1, "updated_at_unix_secs": 100},
        "conversation_config": {
            "tts": {"voice_id": "voice1"},
            "conversation": {"client_events": ["audio", "user_transcript"], "text_only": False},
            "agent": {
                "first_message": "",
                "prompt": {
                    "prompt": ORIGINAL_PROMPT,
                    "llm": "claude-sonnet-5-5",
                    "tool_ids": ["tool_foreign"],
                    "built_in_tools": {"end_call": {"type": "system", "name": "end_call"}},
                    "tools": [{"type": "system", "name": "end_call"}],
                },
            },
        },
        "platform_settings": {
            "overrides": {
                "conversation_config_override": {
                    "agent": {"first_message": False, "language": True}
                }
            },
            "auth": {"enable_auth": True, "shareable_token": "tok_secret_value"},
            "workspace_overrides": {"webhooks": {"url": "https://hook", "secret": "whsec_x"}},
        },
        "access_info": {"creator_email": "albert@example.com"},
    }


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key == "prompt" and isinstance(value, dict):
            dst[key] = copy.deepcopy(value)  # the prompt object is replaced as a whole
        elif isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


class FakeApi:  # pylint: disable=too-many-instance-attributes  # knobs for each test
    """An in-memory agent + workspace tools; records every call."""

    def __init__(self) -> None:
        self.agent = _agent()
        self.tools: dict[str, dict[str, Any]] = {
            "tool_foreign": {
                "type": "client",
                "name": "weather",
                "description": "Albert's own tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "expects_response": True,
                "response_timeout_secs": 5,
            }
        }
        self.calls: list[tuple[str, str]] = []
        self.next_id = 0
        self.fail: tuple[str, str] | None = None  # (method, path prefix) that raises
        self.bump_on_get: int | None = None  # the n-th agent GET sees a concurrent change
        self.ignore_events = False  # a PATCH that silently drops client_events
        self._agent_gets = 0

    @property
    def writes(self) -> list[tuple[str, str]]:
        return [c for c in self.calls if c[0] != "GET"]

    def request(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path))
        if self.fail and method == self.fail[0] and path.startswith(self.fail[1]):
            raise cfg.ApiError(f"{method} {path} → 500: boom")
        if path == f"/v1/convai/agents/{AGENT}":
            return self._agent_call(method, body)
        if path == "/v1/convai/tools" and method == "POST":
            self.next_id += 1
            tid = f"tool_new{self.next_id}"
            self.tools[tid] = copy.deepcopy(body["tool_config"])
            return {"id": tid, "tool_config": self.tools[tid]}
        tid = path.rsplit("/", 1)[1]
        if method == "GET":
            return {"id": tid, "tool_config": copy.deepcopy(self.tools[tid])}
        if method == "PATCH":
            self.tools[tid] = copy.deepcopy(body["tool_config"])
            return {}
        if method == "DELETE":
            del self.tools[tid]
            return {}
        raise AssertionError(f"unexpected {method} {path}")

    def _agent_call(self, method: str, body: Any) -> Any:
        if method == "GET":
            self._agent_gets += 1
            if self.bump_on_get == self._agent_gets:
                self._bump()
            return copy.deepcopy(self.agent)
        assert method == "PATCH"
        body = copy.deepcopy(body)
        assert "tools" not in body["conversation_config"]["agent"]["prompt"]
        if self.ignore_events:
            body["conversation_config"].pop("conversation", None)
        _merge(self.agent, body)
        self._bump()
        return {}

    def _bump(self) -> None:
        meta = self.agent["metadata"]
        meta["updated_at_unix_secs"] += 1
        self.agent["version_id"] = f"v{meta['updated_at_unix_secs']}"


def _run(api: FakeApi, tmp_path: Path, **kwargs: Any) -> tuple[int, list[str]]:
    out: list[str] = []
    code = cfg.run(api, AGENT, backup_dir=tmp_path / "backups", say=out.append, **kwargs)
    return code, out


def _prompt(api: FakeApi) -> str:
    return api.agent["conversation_config"]["agent"]["prompt"]["prompt"]


def _ours(api: FakeApi) -> set[str]:
    names = {spec.name for spec in cfg.TOOL_SPECS}
    return {tid for tid, c in api.tools.items() if c["name"] in names}


def test_dry_run_writes_a_backup_prints_the_diff_and_writes_nothing(tmp_path: Path) -> None:
    api = FakeApi()
    code, out = _run(api, tmp_path)
    assert code == 0
    assert not api.writes
    backups = list((tmp_path / "backups").glob(f"{AGENT}-*.json"))
    assert len(backups) == 1
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "backups").stat().st_mode) == 0o700
    text = "\n".join(out)
    assert "+ client tool do_on_mac (timeout 15 s)" in text
    assert "client_tool_call" in text and "first_message override allowed: False → True" in text
    assert cfg.MARK_START in text and "dry run" in text


def test_apply_patches_once_and_verifies(tmp_path: Path) -> None:
    api = FakeApi()
    code, out = _run(api, tmp_path, apply=True)
    assert code == 0, out
    agent_patches = [c for c in api.writes if c == ("PATCH", f"/v1/convai/agents/{AGENT}")]
    assert len(agent_patches) == 1
    assert len(_ours(api)) == len(cfg.TOOL_SPECS)
    prompt = api.agent["conversation_config"]["agent"]["prompt"]
    assert set(prompt["tool_ids"]) == {"tool_foreign", *_ours(api)}
    assert prompt["llm"] == "claude-sonnet-5-5"
    assert "client_tool_call" in api.agent["conversation_config"]["conversation"]["client_events"]
    ovr = api.agent["platform_settings"]["overrides"]["conversation_config_override"]["agent"]
    assert ovr == {"first_message": True, "language": True}
    assert _prompt(api).startswith(ORIGINAL_PROMPT)
    by_name = {c["name"]: c for c in api.tools.values()}
    assert by_name["send_message"]["response_timeout_secs"] == 30
    assert by_name["confirm_action"]["response_timeout_secs"] == 30
    assert by_name["do_on_mac"]["response_timeout_secs"] == 15
    assert by_name["open_url"]["response_timeout_secs"] == 10
    assert all(c["expects_response"] for n, c in by_name.items() if n != "weather")
    assert out[-1] == "✅ done"


def test_second_apply_is_a_no_op(tmp_path: Path) -> None:
    api = FakeApi()
    assert _run(api, tmp_path, apply=True)[0] == 0
    before = len(api.writes)
    code, out = _run(api, tmp_path, apply=True)
    assert code == 0
    assert len(api.writes) == before
    assert any("no changes" in line for line in out)


def test_existing_tool_with_the_same_name_is_updated_not_duplicated(tmp_path: Path) -> None:
    api = FakeApi()
    api.tools["tool_old"] = {"type": "client", "name": "open_url", "response_timeout_secs": 3}
    api.agent["conversation_config"]["agent"]["prompt"]["tool_ids"].append("tool_old")
    assert _run(api, tmp_path, apply=True)[0] == 0
    assert ("PATCH", "/v1/convai/tools/tool_old") in api.writes
    assert [c["name"] for c in api.tools.values()].count("open_url") == 1
    assert api.tools["tool_old"]["response_timeout_secs"] == 10


def test_inline_only_client_tool_gets_a_workspace_tool() -> None:
    api = FakeApi()
    api.agent["conversation_config"]["agent"]["prompt"]["tools"].append(
        {"type": "client", "name": "list_sessions"}
    )
    target = cfg.desired_target(cfg.fetch_state(api, AGENT))
    assert "list_sessions" in target.creates


def test_concurrent_change_aborts_before_the_patch(tmp_path: Path) -> None:
    api = FakeApi()
    api.bump_on_get = 2  # someone edits the agent between our read and our PATCH
    code, out = _run(api, tmp_path, apply=True)
    assert code == 1
    assert ("PATCH", f"/v1/convai/agents/{AGENT}") not in api.writes
    assert not _ours(api)  # tools created by this run removed again
    assert "tool_foreign" in api.tools
    assert any("changed since it was read" in line for line in out)
    assert _prompt(api) == ORIGINAL_PROMPT


def test_verify_failure_rolls_back(tmp_path: Path) -> None:
    api = FakeApi()
    api.ignore_events = True  # the PATCH "succeeds" but the change is not live
    code, out = _run(api, tmp_path, apply=True)
    assert code == 1
    patches = [c for c in api.writes if c == ("PATCH", f"/v1/convai/agents/{AGENT}")]
    assert len(patches) == 2  # apply + rollback
    assert _prompt(api) == ORIGINAL_PROMPT
    assert api.agent["conversation_config"]["agent"]["prompt"]["tool_ids"] == ["tool_foreign"]
    assert "tool_foreign" in api.tools and not _ours(api)
    assert "✅ rollback verified" in out


def test_failed_patch_rolls_back_updated_tools(tmp_path: Path) -> None:
    api = FakeApi()
    api.tools["tool_old"] = {"type": "client", "name": "open_url", "response_timeout_secs": 3}
    api.agent["conversation_config"]["agent"]["prompt"]["tool_ids"].append("tool_old")
    api.fail = ("PATCH", f"/v1/convai/agents/{AGENT}")
    code, _out = _run(api, tmp_path, apply=True)
    assert code == 3  # the rollback PATCH fails too (same fake failure) → not verified
    assert api.tools["tool_old"]["response_timeout_secs"] == 3  # tool restored anyway


def test_restore_brings_the_backup_back(tmp_path: Path) -> None:
    api = FakeApi()
    _run(api, tmp_path)
    backup = next((tmp_path / "backups").glob("*.json"))
    assert _run(api, tmp_path, apply=True)[0] == 0
    assert cfg.MARK_START in _prompt(api)
    code, _out = _run(api, tmp_path, apply=True, restore=backup)
    assert code == 0
    assert _prompt(api) == ORIGINAL_PROMPT
    assert api.agent["conversation_config"]["agent"]["prompt"]["tool_ids"] == ["tool_foreign"]
    ovr = api.agent["platform_settings"]["overrides"]["conversation_config_override"]["agent"]
    assert ovr["first_message"] is False


def test_foreign_tools_are_never_deleted(tmp_path: Path) -> None:
    for fail in (None, ("PATCH", f"/v1/convai/agents/{AGENT}")):
        api = FakeApi()
        api.fail = fail
        _run(api, tmp_path, apply=True)
        assert ("DELETE", "/v1/convai/tools/tool_foreign") not in api.calls
        assert "tool_foreign" in api.tools


def test_prompt_section_is_replaced_between_the_markers_only() -> None:
    old = f"Intro.\n\n{cfg.MARK_START}\nold bridge text\n{cfg.MARK_END}\n\nOutro stays."
    new = cfg.upsert_section(old)
    assert new.startswith("Intro.\n\n" + cfg.MARK_START)
    assert new.endswith(cfg.MARK_END + "\n\nOutro stays.")
    assert "old bridge text" not in new
    assert cfg.upsert_section(new) == new
    assert cfg.upsert_section("Base.") == "Base.\n\n" + cfg.SECTION
    assert cfg.upsert_section("") == cfg.SECTION


def test_secrets_never_reach_the_backup_or_the_output(tmp_path: Path) -> None:
    api = FakeApi()
    api.agent["conversation_config"]["agent"]["prompt"]["prompt"] = f"Key {SECRET} here."
    _code, out = _run(api, tmp_path)
    backup = next((tmp_path / "backups").glob("*.json")).read_text()
    for needle in (SECRET, "tok_secret_value", "whsec_x", "albert@example.com"):
        assert needle not in backup
        assert all(needle not in line for line in out)
    doc = json.loads(backup)
    assert doc["agent"]["conversation_config"]["agent"]["prompt"]["llm"] == "claude-sonnet-5-5"
    assert "tools" in doc


def test_tool_specs_match_the_controller_tools() -> None:
    funcs: dict[str, Any] = {"confirm_action": BridgeController.confirm_action}
    funcs |= {n: getattr(SessionTools, n) for n in ("list_sessions", "read_session")}
    funcs |= {n: getattr(SessionTools, n) for n in ("brief_decision", "send_message")}
    funcs["answer_decision"] = SessionTools.answer_decision
    funcs |= {n: getattr(MacControl, n) for n in TOOL_NAMES}
    funcs["do_on_mac"] = MacOperator.do_on_mac
    funcs["acknowledge_problem"] = make_acknowledge_tool(None, None)  # type: ignore[arg-type]
    assert {s.name for s in cfg.TOOL_SPECS} == set(funcs)
    for spec in cfg.TOOL_SPECS:
        params = {
            name: p
            for name, p in inspect.signature(funcs[spec.name]).parameters.items()
            if name != "self"
        }
        schema = spec.parameters
        assert set(schema["properties"]) == set(params), spec.name
        mandatory = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
        assert set(schema["required"]) == mandatory, spec.name
        assert all(prop.get("description") for prop in schema["properties"].values())
