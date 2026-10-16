"""Tests for the Claude LLM backend of mac-voice (no network: a fake ElevenLabs API)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from my_stt_tts import claude_llm, eleven_costs, eleven_voice
from my_stt_tts.claude_llm import (
    COMPAT_URL,
    ApiError,
    ClaudeBackendError,
    load_state,
    matches,
    prepare,
    resolve_backend,
    setup,
    twin_body,
    twin_problems,
)

MAIN: dict[str, Any] = {
    "agent_id": "agent_main",
    "name": "Mac voice",
    "conversation_config": {
        "agent": {
            "prompt": {
                "prompt": "be brief",
                "llm": "claude-sonnet-5-5",
                "reasoning_effort": "low",
                "tool_ids": ["tool_a", "tool_b"],
                "tools": [{"name": "legacy"}],
                "enable_parallel_tool_calls": True,
            },
            "first_message": "hi",
        },
        "tts": {"model_id": "eleven_v4_turbo"},
        "asr": {"provider": "scribe_realtime"},
    },
    "platform_settings": {
        "overrides": {"conversation_config_override": {"agent": {"first_message": True}}},
        "workspace_overrides": {"x": 1},
        "archived": False,
    },
}


class FakeApi:
    """In-memory ElevenLabs: agents + workspace secrets; PATCH merges like the server."""

    def __init__(self) -> None:
        self.agents: dict[str, dict[str, Any]] = {"agent_main": copy.deepcopy(MAIN)}
        self.secrets: dict[str, dict[str, str]] = {}
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, body: Any = None) -> Any:
        self.calls.append((method, path))
        parts = path.strip("/").split("/")
        if parts[:3] == ["v1", "convai", "secrets"]:
            return self._secrets(method, parts[3:], body)
        if path == "/v1/convai/agents/create":
            new_id = f"agent_twin{len(self.agents)}"
            self.agents[new_id] = {"agent_id": new_id, **copy.deepcopy(body)}
            return {"agent_id": new_id}
        agent_id = parts[3]
        if agent_id not in self.agents:
            raise ApiError(404, f"{method} {path} → 404")
        if method == "GET":
            return copy.deepcopy(self.agents[agent_id])
        if method == "PATCH":
            _merge(self.agents[agent_id], copy.deepcopy(body))
            return {}
        raise AssertionError(path)

    def _secrets(self, method: str, rest: list[str], body: Any) -> Any:
        if method == "GET":
            return {
                "secrets": [{"secret_id": k, "name": v["name"]} for k, v in self.secrets.items()]
            }
        if method == "POST":
            sid = f"sec{len(self.secrets) + 1}"
            self.secrets[sid] = {"name": body["name"], "value": body["value"]}
            return {"type": "stored", "secret_id": sid, "name": body["name"]}
        if method == "PATCH":
            self.secrets[rest[0]]["value"] = body["value"]
            return {}
        raise AssertionError(method)


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (claude_llm.BACKEND_ENV, claude_llm.MODEL_ENV, claude_llm.TWIN_ENV):
        monkeypatch.delenv(name, raising=False)


def test_twin_body_swaps_only_the_llm() -> None:
    body = twin_body(MAIN, "claude-haiku-5-5", "sec1")
    prompt = body["conversation_config"]["agent"]["prompt"]
    assert prompt["llm"] == "custom-llm"
    assert prompt["custom_llm"] == {
        "url": COMPAT_URL,
        "model_id": "claude-haiku-5-5",
        "api_key": {"secret_id": "sec1"},
    }
    assert "tools" not in prompt  # legacy inline mirror never sent
    assert prompt["tool_ids"] == ["tool_a", "tool_b"]  # same workspace tools
    assert prompt["prompt"] == "be brief"
    assert body["conversation_config"]["tts"] == MAIN["conversation_config"]["tts"]
    assert "workspace_overrides" not in body["platform_settings"]
    assert body["platform_settings"]["overrides"] == MAIN["platform_settings"]["overrides"]
    assert MAIN["conversation_config"]["agent"]["prompt"]["llm"] == "claude-sonnet-5-5"  # copy


def test_matches_is_a_subset_check() -> None:
    assert matches({"a": 1}, {"a": 1, "b": 2})
    assert not matches({"a": 1}, {"a": 2})
    assert not matches({"a": [1, 2]}, {"a": [1]})


def test_twin_problems_name_what_is_wrong() -> None:
    assert twin_problems(MAIN, "claude-haiku-5-5", "sec1")[0].startswith("its LLM is")
    good = {
        "conversation_config": twin_body(MAIN, "claude-haiku-5-5", "sec1")["conversation_config"]
    }
    assert not twin_problems(good, "claude-haiku-5-5", "sec1")
    assert twin_problems(good, "claude-sonnet-5-5", "sec1")


def test_backend_flag_beats_env_and_env_beats_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_backend(None) == "elevenlabs"
    monkeypatch.setenv(claude_llm.BACKEND_ENV, "Claude")
    assert resolve_backend(None) == "claude"
    assert resolve_backend("elevenlabs") == "elevenlabs"
    monkeypatch.setenv(claude_llm.BACKEND_ENV, "openai")
    with pytest.raises(ClaudeBackendError):
        resolve_backend(None)


def test_setup_creates_twin_records_state_and_prepare_is_idle(tmp_path: Path) -> None:
    api, state = FakeApi(), tmp_path / "claude-agent.json"
    twin = setup(api, "agent_main", "k-test", model="claude-haiku-5-5", state_path=state, say=str)
    assert api.secrets["sec1"] == {"name": claude_llm.SECRET_NAME, "value": "k-test"}
    assert load_state(state) == {
        "agent_id": twin,
        "secret_id": "sec1",
        "model": "claude-haiku-5-5",
        "main_agent_id": "agent_main",
    }
    assert oct(state.stat().st_mode & 0o777) == "0o600"
    assert api.agents[twin]["name"] == "Mac voice (Claude API)"
    api.calls.clear()
    assert prepare(api, "agent_main", state_path=state) == twin
    assert ("PATCH", f"/v1/convai/agents/{twin}") not in api.calls  # in sync: no write


def test_setup_twice_updates_the_secret_and_reuses_the_twin(tmp_path: Path) -> None:
    api, state = FakeApi(), tmp_path / "s.json"
    first = setup(api, "agent_main", "k1", model="claude-haiku-5-5", state_path=state, say=str)
    second = setup(api, "agent_main", "k2", model="claude-haiku-5-5", state_path=state, say=str)
    assert first == second
    assert list(api.secrets) == ["sec1"] and api.secrets["sec1"]["value"] == "k2"


def test_prepare_resyncs_after_the_main_agent_changed(tmp_path: Path) -> None:
    api, state = FakeApi(), tmp_path / "s.json"
    twin = setup(api, "agent_main", "k", model="claude-haiku-5-5", state_path=state, say=str)
    api.agents["agent_main"]["conversation_config"]["agent"]["prompt"]["prompt"] = "new prompt"
    assert prepare(api, "agent_main", state_path=state) == twin
    twin_prompt = api.agents[twin]["conversation_config"]["agent"]["prompt"]
    assert twin_prompt["prompt"] == "new prompt" and twin_prompt["llm"] == "custom-llm"


def test_model_env_switches_the_twin_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api, state = FakeApi(), tmp_path / "s.json"
    twin = setup(api, "agent_main", "k", model="claude-haiku-5-5", state_path=state, say=str)
    monkeypatch.setenv(claude_llm.MODEL_ENV, "claude-sonnet-5-5")
    prepare(api, "agent_main", state_path=state)
    custom = api.agents[twin]["conversation_config"]["agent"]["prompt"]["custom_llm"]
    assert custom["model_id"] == "claude-sonnet-5-5"


def test_prepare_without_setup_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(ClaudeBackendError, match="mac-voice -C"):
        prepare(FakeApi(), "agent_main", state_path=tmp_path / "none.json")


def test_prepare_with_a_deleted_twin_fails_loud(tmp_path: Path) -> None:
    api, state = FakeApi(), tmp_path / "s.json"
    twin = setup(api, "agent_main", "k", model="claude-haiku-5-5", state_path=state, say=str)
    del api.agents[twin]
    with pytest.raises(ClaudeBackendError, match="not found"):
        prepare(api, "agent_main", state_path=state)


def test_missing_vault_item_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(claude_llm.KEY_ENV, raising=False)
    monkeypatch.delenv("_MAC_VOICE_UNDER_SECRET_RUN", raising=False)
    monkeypatch.setattr(claude_llm, "vault_item_exists", lambda item=claude_llm.VAULT_ITEM: False)
    with pytest.raises(ClaudeBackendError, match="tp#1098"):
        claude_llm.reexec_with_key(["-C"])


def test_reexec_runs_the_module_under_secret_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(claude_llm.KEY_ENV, raising=False)
    monkeypatch.delenv("_MAC_VOICE_UNDER_SECRET_RUN", raising=False)
    monkeypatch.setattr(claude_llm, "vault_item_exists", lambda item=claude_llm.VAULT_ITEM: True)
    seen: dict[str, Any] = {}

    class _Done:
        returncode = 0

    def fake_run(cmd: list[str], env: dict[str, str], **_kw: Any) -> _Done:
        seen["cmd"], seen["env"] = cmd, env
        return _Done()

    monkeypatch.setattr(claude_llm.subprocess, "run", fake_run)
    assert claude_llm.reexec_with_key(["-C"]) == 0
    cmd = seen["cmd"]
    assert cmd[:4] == [
        "secret-run",
        "-e",
        f"{claude_llm.KEY_ENV}=anthropic-api-private-elevenlabs",
        "--",
    ]
    assert cmd[-3:] == ["-m", "my_stt_tts.eleven_voice", "-C"]
    assert seen["env"]["_MAC_VOICE_UNDER_SECRET_RUN"] == "1"


def test_key_already_in_env_skips_the_reexec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(claude_llm.KEY_ENV, "k")
    assert claude_llm.reexec_with_key(["-C"]) is None


def test_main_with_claude_backend_but_no_setup_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(eleven_voice, "_load_env", lambda: None)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_main")

    def no_setup(_api: Any, _main: str) -> str:
        raise ClaudeBackendError("the Claude backend is not set up — run once: mac-voice -C")

    monkeypatch.setattr(eleven_voice, "prepare", no_setup)
    assert eleven_voice.main(["-B", "claude"]) == 2
    assert "mac-voice -C" in capsys.readouterr().out


def test_main_passes_the_twin_id_to_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eleven_voice, "_load_env", lambda: None)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    monkeypatch.setenv("ELEVENLABS_AGENT_ID", "agent_main")
    monkeypatch.setenv(claude_llm.BACKEND_ENV, "claude")
    monkeypatch.setattr(eleven_voice, "prepare", lambda _api, _main: "agent_twin")
    seen: list[str] = []

    def fake_run(agent: str, *_args: Any) -> int:
        seen.append(agent)
        return 0

    monkeypatch.setattr(eleven_voice, "run", fake_run)
    assert eleven_voice.main([]) == 0
    assert seen == ["agent_twin"]
    assert eleven_voice.main(["-B", "elevenlabs"]) == 0
    assert seen[-1] == "agent_main"


def test_models_line_names_the_claude_model(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = {"conversation_config": twin_body(MAIN, "claude-haiku-5-5", "s")["conversation_config"]}
    monkeypatch.setattr(eleven_costs, "_get", lambda _path, _key: agent)
    line = eleven_costs.agent_models("k", "agent_twin")
    assert "claude-haiku-5-5 via Claude API credits" in line
