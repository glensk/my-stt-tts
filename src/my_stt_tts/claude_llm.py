"""Run the LLM part of mac-voice on Albert's Claude API credits instead of ElevenLabs.

An ElevenLabs agent does speech-to-text, the LLM, text-to-speech and turn-taking in
their cloud. The Anthropic API has no speech-to-text or text-to-speech, so only the LLM
can move: ElevenLabs' "custom LLM" mode calls an OpenAI-compatible chat-completions
endpoint from their servers, and Anthropic serves one at ``https://api.anthropic.com/v1``.
The LLM tokens are then billed to the private Console org's API key (its monthly
credits) and ElevenLabs charges only the per-minute voice part.

ElevenLabs only accepts a ``custom_llm`` block on an agent whose ``llm`` is
``custom-llm`` (a per-conversation ``llm`` override cannot switch to it), so the Claude
backend uses a *twin* agent: a copy of the main agent (prompt, tools, voice, speech-to-text,
overrides) whose LLM is the Claude API. Every ``mac-voice -B claude`` start re-syncs the
twin from the main agent, so edits made to the main agent (dashboard,
``scripts/eleven_agent_config.py``) reach it.

One-time setup, ``mac-voice -C``: reads the key from the Vaultwarden item
``anthropic-api-private-elevenlabs`` through ``secret-run`` (never printed), checks it
with one tiny request, stores it as the ElevenLabs workspace secret of the same name
(ElevenLabs must hold it to call Anthropic), creates or updates the twin and records its id in
``~/.local/state/mac-voice/claude-agent.json``.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

API = "https://api.elevenlabs.io"
#: Anthropic's OpenAI-compatible endpoint (ElevenLabs appends ``/chat/completions``).
COMPAT_URL = "https://api.anthropic.com/v1"
#: Fastest and cheapest current model: voice turns need a low time-to-first-token.
DEFAULT_MODEL = "claude-haiku-5-5"
BACKENDS = ("elevenlabs", "claude")
BACKEND_ENV = "MAC_VOICE_LLM_BACKEND"
MODEL_ENV = "MAC_VOICE_CLAUDE_MODEL"
TWIN_ENV = "ELEVENLABS_CLAUDE_AGENT_ID"
#: Vaultwarden item (via ``secret-run``): the private-org Console API key used only by
#: ElevenLabs (its own key, so it can be revoked without touching other automations).
VAULT_ITEM = "anthropic-api-private-elevenlabs"
#: ElevenLabs workspace secret the twin's ``custom_llm.api_key`` points at.
SECRET_NAME = "anthropic-api-private-elevenlabs"
KEY_ENV = "MAC_VOICE_ANTHROPIC_KEY"
_UNDER_SECRET_RUN = "_MAC_VOICE_UNDER_SECRET_RUN"
STATE_FILE = Path.home() / ".local" / "state" / "mac-voice" / "claude-agent.json"
TWIN_SUFFIX = " (Claude API)"
#: Platform settings the API manages itself or refuses on write.
_PS_DROP = frozenset({"workspace_overrides", "archived"})
SETUP_HINT = (
    "run once: mac-voice -C (needs the Vaultwarden item anthropic-api-private-elevenlabs, tp#1098)"
)


class ClaudeBackendError(RuntimeError):
    """The Claude backend cannot run; the message says what to do (never carries a key)."""


class ApiLike(Protocol):
    """What this module needs from the ElevenLabs REST API (tests pass a fake)."""

    def request(self, method: str, path: str, body: Any = None) -> Any: ...


class Api:
    """Minimal ElevenLabs REST client (stdlib only); the key lives only in this object."""

    def __init__(self, key: str, base: str = API, timeout: float = 30.0) -> None:
        self._key, self._base, self._timeout = key, base, timeout

    def request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={"xi-api-key": self._key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode(errors="replace")
            raise ApiError(exc.code, f"{method} {path} → {exc.code}: {detail}") from None
        return json.loads(raw) if raw else {}


class ApiError(RuntimeError):
    """A failed ElevenLabs call with its HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


# -- configuration ------------------------------------------------------------------------
def resolve_backend(cli_value: str | None) -> str:
    """The LLM backend: the ``-B`` flag, else ``MAC_VOICE_LLM_BACKEND``, else elevenlabs."""
    value = (cli_value or os.environ.get(BACKEND_ENV) or "elevenlabs").strip().lower()
    if value not in BACKENDS:
        raise ClaudeBackendError(
            f"{BACKEND_ENV}={value!r} is not a backend (choose: {', '.join(BACKENDS)})"
        )
    return value


def resolve_model(state: Mapping[str, Any] | None = None) -> str:
    """``MAC_VOICE_CLAUDE_MODEL``, else the model the setup recorded, else the default."""
    return os.environ.get(MODEL_ENV) or (state or {}).get("model") or DEFAULT_MODEL


def load_state(path: Path = STATE_FILE) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: Mapping[str, Any], path: Path = STATE_FILE) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(state), indent=1) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


# -- the twin agent (pure) ----------------------------------------------------------------
def twin_body(main: Mapping[str, Any], model: str, secret_id: str) -> dict[str, Any]:
    """PATCH/create body: the main agent's configuration with the Claude API as its LLM."""
    cc = copy.deepcopy(main.get("conversation_config") or {})
    prompt = cc.setdefault("agent", {}).setdefault("prompt", {})
    prompt.pop("tools", None)  # legacy inline mirror of tool_ids: never sent
    prompt["llm"] = "custom-llm"
    prompt["custom_llm"] = {
        "url": COMPAT_URL,
        "model_id": model,
        "api_key": {"secret_id": secret_id},
    }
    # ElevenLabs stores False for a custom LLM whatever is sent; sending it keeps syncs idle
    prompt["enable_parallel_tool_calls"] = False
    # Claude 5 models reject `temperature` (400 "deprecated for this model"); null makes
    # ElevenLabs leave it out of the request
    prompt["temperature"] = None
    ps = {
        k: copy.deepcopy(v)
        for k, v in (main.get("platform_settings") or {}).items()
        if k not in _PS_DROP
    }
    return {"conversation_config": cc, "platform_settings": ps}


def matches(want: Any, have: Any) -> bool:
    """``want`` is contained in ``have`` (dicts compared by ``want``'s keys)."""
    if isinstance(want, Mapping):
        if not isinstance(have, Mapping):
            return False
        return all(matches(v, have.get(k)) for k, v in want.items())
    if isinstance(want, list):
        return (
            isinstance(have, list)
            and len(want) == len(have)
            and all(matches(w, h) for w, h in zip(want, have, strict=True))
        )
    return bool(want == have)


def twin_problems(twin: Mapping[str, Any], model: str, secret_id: str | None) -> list[str]:
    """Why the live twin would NOT run on the Claude API (empty list = it does)."""
    prompt = ((twin.get("conversation_config") or {}).get("agent") or {}).get("prompt") or {}
    custom = prompt.get("custom_llm") or {}
    problems = []
    if prompt.get("llm") != "custom-llm":
        problems.append(f"its LLM is {prompt.get('llm')!r}, not custom-llm")
    if custom.get("url") != COMPAT_URL:
        problems.append(f"custom LLM URL is {custom.get('url')!r}")
    if custom.get("model_id") != model:
        problems.append(f"custom LLM model is {custom.get('model_id')!r}, not {model!r}")
    if not (custom.get("api_key") or {}).get("secret_id") or (
        secret_id and custom["api_key"]["secret_id"] != secret_id
    ):
        problems.append("custom LLM has no API key secret")
    return problems


# -- ElevenLabs operations ----------------------------------------------------------------
def find_secret(api: ApiLike, name: str = SECRET_NAME) -> str | None:
    """The id of the workspace secret called ``name``, or None."""
    for item in (api.request("GET", "/v1/convai/secrets") or {}).get("secrets") or []:
        if item.get("name") == name:
            return str(item.get("secret_id"))
    return None


def upsert_secret(api: ApiLike, value: str, name: str = SECRET_NAME) -> str:
    """Create or overwrite the workspace secret; returns its id (the value is never echoed)."""
    existing = find_secret(api, name)
    if existing:
        api.request(
            "PATCH",
            f"/v1/convai/secrets/{existing}",
            {"type": "update", "name": name, "value": value},
        )
        return existing
    reply = api.request("POST", "/v1/convai/secrets", {"type": "new", "name": name, "value": value})
    return str(reply["secret_id"])


def _get_agent(api: ApiLike, agent_id: str) -> dict[str, Any] | None:
    try:
        reply = api.request("GET", f"/v1/convai/agents/{agent_id}")
    except ApiError as exc:
        if exc.status == 404:
            return None
        raise
    return reply if isinstance(reply, dict) else None


def sync_twin(
    api: ApiLike, main_id: str, twin_id: str, model: str, secret_id: str
) -> tuple[bool, dict[str, Any]]:
    """PATCH the twin when it differs from the main agent; returns (changed, live twin)."""
    main = _get_agent(api, main_id)
    if main is None:
        raise ClaudeBackendError(f"main ElevenLabs agent {main_id} not found")
    twin = _get_agent(api, twin_id)
    if twin is None:
        raise ClaudeBackendError(f"Claude twin agent {twin_id} not found — {SETUP_HINT}")
    body = twin_body(main, model, secret_id)
    if matches(body, twin):
        return False, twin
    api.request("PATCH", f"/v1/convai/agents/{twin_id}", body)
    live = _get_agent(api, twin_id) or {}
    problems = twin_problems(live, model, secret_id)
    if problems:
        raise ClaudeBackendError(
            f"twin agent {twin_id} not on the Claude API: {'; '.join(problems)}"
        )
    return True, live


def prepare(api: ApiLike, main_id: str, state_path: Path = STATE_FILE) -> str:
    """Runtime check for ``-B claude``: sync the twin, return its agent id (or raise)."""
    state = load_state(state_path)
    twin_id = os.environ.get(TWIN_ENV) or state.get("agent_id")
    secret_id = state.get("secret_id")
    if not twin_id or not secret_id:
        raise ClaudeBackendError(f"the Claude backend is not set up — {SETUP_HINT}")
    model = resolve_model(state)
    try:
        changed, _ = sync_twin(api, main_id, str(twin_id), model, str(secret_id))
    except ApiError as exc:
        raise ClaudeBackendError(f"ElevenLabs refused the twin sync: {exc}") from None
    except OSError as exc:  # network: urllib.error.URLError, timeouts
        raise ClaudeBackendError(f"cannot reach ElevenLabs to sync the twin: {exc}") from None
    if changed:
        print(f"🔁 Claude twin agent {twin_id} synced from {main_id} (model {model})", flush=True)
    return str(twin_id)


def setup(  # pylint: disable=too-many-arguments
    api: ApiLike,
    main_id: str,
    key: str,
    *,
    model: str,
    state_path: Path = STATE_FILE,
    say: Any = print,
) -> str:
    """Store the key at ElevenLabs, create/sync the twin agent, record it; returns its id."""
    secret_id = upsert_secret(api, key)
    say(f"✅ ElevenLabs workspace secret {SECRET_NAME!r} stored ({secret_id})")
    state = load_state(state_path)
    twin_id = os.environ.get(TWIN_ENV) or state.get("agent_id")
    if twin_id and _get_agent(api, str(twin_id)) is None:
        say(f"⚠️  recorded twin agent {twin_id} is gone; creating a new one")
        twin_id = None
    if not twin_id:
        main = _get_agent(api, main_id)
        if main is None:
            raise ClaudeBackendError(f"main ElevenLabs agent {main_id} not found")
        body = twin_body(main, model, secret_id)
        body["name"] = f"{main.get('name') or 'mac-voice'}{TWIN_SUFFIX}"
        twin_id = str(api.request("POST", "/v1/convai/agents/create", body)["agent_id"])
        say(f"✅ created twin agent {twin_id} ({body['name']})")
    save_state(
        {"agent_id": twin_id, "secret_id": secret_id, "model": model, "main_agent_id": main_id},
        state_path,
    )
    sync_twin(api, main_id, str(twin_id), model, secret_id)
    live = _get_agent(api, str(twin_id)) or {}
    problems = twin_problems(live, model, secret_id)
    if problems:
        raise ClaudeBackendError(
            f"twin agent {twin_id} not on the Claude API: {'; '.join(problems)}"
        )
    say(f"✅ twin agent {twin_id} runs on the Claude API ({model}); state: {state_path}")
    return str(twin_id)


# -- the key ------------------------------------------------------------------------------
def vault_item_exists(item: str = VAULT_ITEM) -> bool | None:
    """Whether ``secret-run`` can inject ``item`` (None when secret-run is unavailable)."""
    exe = shutil.which("secret-run")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-l", "-j"], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        try:
            if json.loads(line).get("name") == item:
                return True
        except ValueError:
            continue
    return False


def reexec_with_key(argv: list[str]) -> int | None:
    """Re-run this command under ``secret-run`` with the key in ``KEY_ENV``.

    Returns the child's exit code, or None when the key is already in the environment.
    """
    if os.environ.get(KEY_ENV):
        return None
    if os.environ.get(_UNDER_SECRET_RUN):
        raise ClaudeBackendError(f"secret-run gave no value for {VAULT_ITEM}")
    found = vault_item_exists()
    if found is None:
        raise ClaudeBackendError(
            f"secret-run is not on PATH — export {KEY_ENV} yourself or install secret-run"
        )
    if not found:
        raise ClaudeBackendError(
            f"no Vaultwarden item {VAULT_ITEM!r} in the agent-login collection yet — "
            "tp#1098 creates the private-org Console API key; rerun mac-voice -C after that"
        )
    env = dict(os.environ, **{_UNDER_SECRET_RUN: "1"})
    module = ["-m", "my_stt_tts.eleven_voice"]
    cmd = ["secret-run", "-e", f"{KEY_ENV}={VAULT_ITEM}", "--", sys.executable, *module, *argv]
    return subprocess.run(cmd, env=env, check=False).returncode


def check_key(key: str, model: str) -> None:
    """One tiny Messages request: the key works and may use ``model`` (raises otherwise)."""
    try:
        import anthropic  # pylint: disable=import-outside-toplevel  # optional "llm" extra
    except ImportError:
        raise ClaudeBackendError(
            "the anthropic SDK is missing; run: uv sync --inexact --extra llm"
        ) from None
    client = anthropic.Anthropic(api_key=key, max_retries=1, timeout=30.0)
    try:
        client.messages.create(
            model=model, max_tokens=64, messages=[{"role": "user", "content": "Say OK."}]
        )
    except anthropic.AuthenticationError:
        raise ClaudeBackendError(f"the key in {VAULT_ITEM} was rejected (401)") from None
    except anthropic.PermissionDeniedError:
        raise ClaudeBackendError(f"the key in {VAULT_ITEM} may not use {model} (403)") from None
    except anthropic.NotFoundError:
        raise ClaudeBackendError(f"model {model!r} not found (404)") from None
    except anthropic.APIStatusError as exc:
        raise ClaudeBackendError(f"Claude API check failed ({exc.status_code})") from None
    except anthropic.APIConnectionError:
        raise ClaudeBackendError("cannot reach api.anthropic.com") from None


def setup_main(main_id: str, eleven_key: str, argv: list[str]) -> int:
    """CLI handler of ``mac-voice -C``: key via secret-run, check, store, create the twin."""
    try:
        code = reexec_with_key(argv)
        if code is not None:
            return code
        key = os.environ[KEY_ENV]
        model = resolve_model(load_state())
        check_key(key, model)
        print(f"✅ Claude API key works with {model}", flush=True)
        setup(Api(eleven_key), main_id, key, model=model)
    except ClaudeBackendError as exc:
        print(f"❌ {exc}", flush=True)
        return 2
    except ApiError as exc:
        print(f"❌ ElevenLabs API: {exc}", flush=True)
        return 1
    except OSError as exc:  # network: urllib.error.URLError, timeouts
        print(f"❌ cannot reach ElevenLabs: {exc}", flush=True)
        return 1
    print("✅ done — start it with: mac-voice -d -B claude (or MAC_VOICE_LLM_BACKEND=claude)")
    return 0
