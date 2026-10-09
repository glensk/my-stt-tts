#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Configure the ElevenLabs agent for the mac-voice bridge (PLAN_claude-bridge.md 7.7).

Flow: GET the agent → sanitised 0600 backup in
``~/.local/state/mac-voice/agent-backups/<agent>-<ts>.json`` (directory 0700) → compute
the desired configuration → print a readable diff → with ``-a`` only: create / update
the client tools, re-GET and abort when the agent changed meanwhile (version + updated
timestamp), PATCH the agent, re-GET and verify that the live main-branch version has
every change — and roll back automatically (restore the agent, the updated tools, and
delete only the tools this run created) when any step fails.

Desired configuration:

* one workspace client tool per bridge tool, keyed by NAME (an existing tool with that
  name is updated, never duplicated; tools the script did not create are never deleted),
  each with a JSON-schema ``parameters``, ``expects_response: true`` and
  ``response_timeout_secs`` 10 (send / answer / confirm 30, do_on_mac 15). The API turns
  inline ``prompt.tools`` client entries into workspace tools listed in ``tool_ids``; the
  script manages ``tool_ids`` and never sends ``prompt.tools``;
* ``client_tool_call`` in ``conversation_config.conversation.client_events``;
* ``platform_settings.overrides.conversation_config_override.agent.first_message: true``
  (the voice-on briefing);
* a marked section in the system prompt between ``<!-- mac-voice bridge start -->`` and
  ``<!-- mac-voice bridge end -->``, replaced idempotently; the rest of the prompt is
  never touched.

Secrets: the API key comes from the repo ``.env`` (``eleven_voice._load_env``) and is never
printed or stored; the backup drops secret-like keys and redacts token-shaped values.

Exit codes: 0 done (or nothing to change), 1 failed and rolled back, 2 usage / missing
configuration, 3 failed AND the rollback did not verify (check the agent by hand).

Examples:
    scripts/eleven_agent_config.py              # dry run (default): backup + diff
    scripts/eleven_agent_config.py -n           # the same, explicitly
    scripts/eleven_agent_config.py -a           # apply, verify, roll back on failure
    scripts/eleven_agent_config.py -r ~/.local/state/mac-voice/agent-backups/agent_x-….json
    scripts/eleven_agent_config.py -A agent_xyz -B /tmp/backups -n
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

API = "https://api.elevenlabs.io"
DEFAULT_BACKUP_DIR = Path.home() / ".local" / "state" / "mac-voice" / "agent-backups"
MARK_START = "<!-- mac-voice bridge start -->"
MARK_END = "<!-- mac-voice bridge end -->"
CLIENT_EVENT = "client_tool_call"
#: Keys of a tool config the script owns (compared for idempotency, written on update).
TOOL_KEYS = (
    "type",
    "name",
    "description",
    "parameters",
    "expects_response",
    "response_timeout_secs",
)
SECRET_KEY = re.compile(
    r"secret|token|api_?key|password|passwd|authori[sz]ation|credential|private_?key|cookie|email",
    re.IGNORECASE,
)
DROP_KEYS = frozenset({"access_info", "workspace_overrides"})
TOKEN_VALUE = re.compile(
    r"-----BEGIN [^\n-]*-----|glpat-[A-Za-z0-9_\-]{8,}|(?:ghp_|github_pat_)[A-Za-z0-9_]{16,}"
    r"|xox[bp]-[A-Za-z0-9\-]{8,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_\-]{16,}"
    r"|eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"
)

SECTION = f"""{MARK_START}
# Mac and Claude Code tools
You run on Albert's Mac and act on it and on his Claude Code sessions through tools.
- Sessions: list_sessions; read_session(target, what: last_reply|last_prompt); brief_decision(target); send_message(target, text); answer_decision(target, choice). Target = a session name or its number from the last list; if unclear, ask. Background sessions are read-only.
- Decision briefing: "The session <name>, which works on <aim>, needs a decision: <question>. Options: a) … b) …. It recommends <x> because <reason>." Discuss, then answer with answer_decision (a written to-do decision: send_message).
- Mac: open_url(target), open_app(name), set_volume(level|step|mute), set_brightness(step), media(command, seconds) for the video in Safari, youtube_play_first(). Anything else on the Mac: do_on_mac(request; background=true only if Albert says to do it even if he hangs up).
- acknowledge_problem(subject) when Albert says "got it, <subject>".
- SUCCESSES ARE SILENT: after a tool succeeds say nothing, at most "done". Never read routine results back. Say refusals and failures briefly.
- Risky actions (send_message, answer_decision, some do_on_mac steps) return a proposal with a two-digit code: read the proposal text back with its code, then wait. Call confirm_action(code) only when Albert himself says "confirm <code>" in a NEW sentence. Never invent or guess a code.
- A message starting with "[system notice]" comes from the system, not Albert: say it aloud in one short sentence.
- If a tool says "Mac control disabled" or is unavailable, say so in a few words.
{MARK_END}"""


def _string(desc: str, enum: list[str] | None = None) -> dict[str, Any]:
    return {"type": "string", "description": desc, **({"enum": enum} if enum else {})}


def _params(props: Mapping[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": dict(props), "required": required}


_TARGET = _string("Session name, or its number from the last list.")


@dataclass(frozen=True)
class ToolSpec:
    """One bridge client tool as the agent sees it."""

    name: str
    description: str
    parameters: dict[str, Any]
    timeout: int = 10

    def config(self) -> dict[str, Any]:
        return {
            "type": "client",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "expects_response": True,
            "response_timeout_secs": self.timeout,
        }


#: Mirrors the controller tools (bridge.confirm_action, claude_sessions, attention,
#: mac_control, mac_operator); tests/test_agent_config.py checks names + parameters.
TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "confirm_action",
        "Run the proposal whose two-digit code Albert just said after 'confirm'.",
        _params({"code": _string("The two-digit code Albert said.")}, ["code"]),
        30,
    ),
    ToolSpec("list_sessions", "List the running Claude Code sessions (numbered).", _params({}, [])),
    ToolSpec(
        "read_session",
        "What a session last replied, or the last prompt it got.",
        _params(
            {"target": _TARGET, "what": _string("What to read.", ["last_reply", "last_prompt"])},
            ["target"],
        ),
    ),
    ToolSpec(
        "brief_decision",
        "The briefing of the decision a session needs.",
        _params({"target": _TARGET}, ["target"]),
    ),
    ToolSpec(
        "send_message",
        "Propose sending a message to a session; needs confirm_action with the code.",
        _params(
            {"target": _TARGET, "text": _string("The message, as Albert said it.")},
            ["target", "text"],
        ),
        30,
    ),
    ToolSpec(
        "answer_decision",
        "Propose answering a session's pending question; needs confirm_action with the code.",
        _params(
            {"target": _TARGET, "choice": _string("The option: letter, number or label.")},
            ["target", "choice"],
        ),
        30,
    ),
    ToolSpec(
        "acknowledge_problem",
        "Clear an open problem after Albert says 'got it, <subject>'.",
        _params({"subject": _string("The problem's subject, e.g. a session name.")}, ["subject"]),
    ),
    ToolSpec(
        "open_url",
        "Open a website in Safari (youtube and jellyfin are shortcuts).",
        _params({"target": _string("Site name, host or URL.")}, ["target"]),
    ),
    ToolSpec(
        "open_app",
        "Open an installed Mac app.",
        _params({"name": _string("The app's name.")}, ["name"]),
    ),
    ToolSpec(
        "set_volume",
        "Set the output volume: a level, a step, or mute.",
        _params(
            {
                "level": {"type": "integer", "description": "Volume 0 to 100."},
                "step": _string("up, down, or a signed number like +10."),
                "mute": {"type": "boolean", "description": "true mutes, false unmutes."},
            },
            [],
        ),
    ),
    ToolSpec(
        "set_brightness",
        "Make the built-in display brighter or darker.",
        _params({"step": _string("up, down, or a signed number of steps.")}, ["step"]),
    ),
    ToolSpec(
        "media",
        "Control the video in the front Safari window (YouTube, Jellyfin).",
        _params(
            {
                "command": _string(
                    "What to do.", ["play", "pause", "seek", "next", "previous", "fullscreen"]
                ),
                "seconds": {"type": "number", "description": "Seek seconds, negative goes back."},
            },
            ["command"],
        ),
    ),
    ToolSpec(
        "youtube_play_first",
        "Play the first video visible on the YouTube page in Safari.",
        _params({}, []),
    ),
    ToolSpec(
        "do_on_mac",
        "Carry out any other request on the Mac (risky steps ask for a code).",
        _params(
            {
                "request": _string("What to do, as Albert said it."),
                "background": {
                    "type": "boolean",
                    "description": "true only if Albert said to do it even if he hangs up.",
                },
            },
            ["request"],
        ),
        15,
    ),
)


# -- HTTP ---------------------------------------------------------------------------------
class ApiLike(Protocol):
    """What the flow needs from the REST API (tests pass a fake)."""

    def request(self, method: str, path: str, body: Any = None) -> Any: ...


class ApiError(RuntimeError):
    """A failed API call (message never carries the key)."""


class Api:
    """Minimal ElevenLabs REST client; the key lives only in this object."""

    def __init__(self, key: str, base: str = API) -> None:
        self._headers = {"xi-api-key": key}
        self._base = base

    def request(self, method: str, path: str, body: Any = None) -> Any:
        import httpx  # pylint: disable=import-outside-toplevel  # only for real runs

        res = httpx.request(method, self._base + path, headers=self._headers, json=body, timeout=60)
        if res.status_code >= 400:
            raise ApiError(f"{method} {path} → {res.status_code}: {_redact(res.text[:300])}")
        return res.json() if res.content else {}


# -- sanitising ---------------------------------------------------------------------------
def _redact(text: str) -> str:
    return TOKEN_VALUE.sub("[redacted]", text)


def sanitise(obj: Any) -> Any:
    """A copy without secret-like keys (and PII blocks); token-shaped strings redacted."""
    if isinstance(obj, Mapping):
        return {
            k: sanitise(v)
            for k, v in obj.items()
            if k not in DROP_KEYS and not SECRET_KEY.search(str(k))
        }
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    if isinstance(obj, str):
        return _redact(obj)
    return obj


# -- reading the agent ----------------------------------------------------------------------
@dataclass
class State:
    """The agent as read, plus the configs of the workspace tools it lists in tool_ids."""

    agent: dict[str, Any]
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)  # tool id → tool_config

    @property
    def prompt(self) -> dict[str, Any]:
        return self.agent.get("conversation_config", {}).get("agent", {}).get("prompt", {}) or {}

    @property
    def tool_ids(self) -> list[str]:
        return list(self.prompt.get("tool_ids") or [])

    def client_tools(self) -> dict[str, tuple[str | None, dict[str, Any]]]:
        """name → (tool id or None for an inline-only entry, config) of every client tool."""
        out: dict[str, tuple[str | None, dict[str, Any]]] = {}
        for entry in self.prompt.get("tools") or []:
            if isinstance(entry, dict) and entry.get("type") == "client" and entry.get("name"):
                out[str(entry["name"])] = (None, entry)
        for tid in self.tool_ids:
            cfg = self.tools.get(tid) or {}
            if cfg.get("type") == "client" and cfg.get("name"):
                out[str(cfg["name"])] = (tid, cfg)
        return out


def fetch_state(api: ApiLike, agent_id: str) -> State:
    agent = api.request("GET", f"/v1/convai/agents/{agent_id}")
    state = State(agent)
    for tid in state.tool_ids:
        state.tools[tid] = api.request("GET", f"/v1/convai/tools/{tid}").get("tool_config") or {}
    return state


def stamp_of(agent: Mapping[str, Any]) -> tuple[Any, Any]:
    """What must not change between our read and our PATCH (version + updated time)."""
    return agent.get("version_id"), (agent.get("metadata") or {}).get("updated_at_unix_secs")


def _subset(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {k: cfg.get(k) for k in TOOL_KEYS}


def matches(want: Any, have: Any) -> bool:
    """``have`` carries everything ``want`` says (the API may add defaults of its own)."""
    if isinstance(want, Mapping):
        return isinstance(have, Mapping) and all(matches(v, have.get(k)) for k, v in want.items())
    if isinstance(want, list):
        if not isinstance(have, list) or len(want) != len(have):
            return False
        if all(isinstance(v, str) for v in want):
            return sorted(want) == sorted(have)  # required / enum: order is not meaning
        return all(matches(w, h) for w, h in zip(want, have, strict=True))
    return bool(want == have)


def same_tool(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    if a is None or b is None:
        return a is b
    return matches(a, b) or matches(b, a)


def _events(agent: Mapping[str, Any]) -> list[str]:
    conv = agent.get("conversation_config", {}).get("conversation", {}) or {}
    return list(conv.get("client_events") or [])


def _overrides(agent: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy((agent.get("platform_settings") or {}).get("overrides") or {})


def _first_message_allowed(overrides: Mapping[str, Any]) -> Any:
    ccov = overrides.get("conversation_config_override") or {}
    return (ccov.get("agent") or {}).get("first_message")


# -- the desired configuration --------------------------------------------------------------
def upsert_section(prompt: str, section: str = SECTION) -> str:
    """``prompt`` with the marked section replaced (or appended); the rest untouched."""
    start = prompt.find(MARK_START)
    end = prompt.find(MARK_END, start) if start != -1 else -1
    if start != -1 and end != -1:
        return prompt[:start] + section + prompt[end + len(MARK_END) :]
    if not prompt:
        return section
    return prompt + ("\n" if prompt.endswith("\n") else "\n\n") + section


@dataclass
class Target:
    """What the agent should look like after the run."""

    prompt: str
    client_events: list[str]
    overrides: dict[str, Any]
    tool_ids: list[str]
    updates: dict[str, dict[str, Any]] = field(default_factory=dict)  # id → config subset
    creates: dict[str, dict[str, Any]] = field(default_factory=dict)  # name → config subset


def desired_target(state: State, specs: tuple[ToolSpec, ...] = TOOL_SPECS) -> Target:
    """The bridge configuration on top of ``state`` (idempotent)."""
    events = _events(state.agent)
    if CLIENT_EVENT not in events:
        events.append(CLIENT_EVENT)
    overrides = _overrides(state.agent)
    ccov = overrides.setdefault("conversation_config_override", {})
    ccov.setdefault("agent", {})["first_message"] = True
    target = Target(
        prompt=upsert_section(str(state.prompt.get("prompt") or "")),
        client_events=events,
        overrides=overrides,
        tool_ids=state.tool_ids,
    )
    existing = state.client_tools()
    for spec in specs:
        want = _subset(spec.config())
        tid, cfg = existing.get(spec.name, (None, None))
        if tid is None:
            target.creates[spec.name] = want
        elif not matches(want, _subset(cfg or {})):
            target.updates[tid] = want
    return target


def restore_target(state: State, backup: Mapping[str, Any]) -> Target:
    """The backup's managed fields (prompt, client events, overrides, tools) on ``state``."""
    old = State(dict(backup["agent"]), dict(backup.get("tools") or {}))
    target = Target(
        prompt=str(old.prompt.get("prompt") or ""),
        client_events=_events(old.agent),
        overrides=_overrides(old.agent),
        tool_ids=old.tool_ids,
    )
    for tid in old.tool_ids:
        want = _subset(old.tools.get(tid) or {})
        if tid in state.tools and want.get("name") and not matches(want, _subset(state.tools[tid])):
            target.updates[tid] = want
    return target


def view_of(state: State) -> dict[str, Any]:
    """The managed fields of ``state`` in a diffable form."""
    return {
        "prompt": str(state.prompt.get("prompt") or ""),
        "client_events": _events(state.agent),
        "first_message_override": _first_message_allowed(_overrides(state.agent)),
        "tools": {
            name: _subset(cfg)
            for name, (tid, cfg) in sorted(state.client_tools().items())
            if tid is not None
        },
    }


def view_of_target(state: State, target: Target) -> dict[str, Any]:
    tools: dict[str, dict[str, Any]] = {}
    for tid in target.tool_ids:
        cfg = target.updates.get(tid) or state.tools.get(tid)
        if cfg and cfg.get("type") == "client" and cfg.get("name"):
            tools[str(cfg["name"])] = _subset(cfg)
    tools.update(target.creates)
    return {
        "prompt": target.prompt,
        "client_events": list(target.client_events),
        "first_message_override": _first_message_allowed(target.overrides),
        "tools": dict(sorted(tools.items())),
    }


def render_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Readable diff lines of two views (empty when nothing changes); secrets redacted."""
    lines: list[str] = []
    if before["prompt"] != after["prompt"]:
        lines.append("~ system prompt:")
        lines += [
            "    " + _redact(line)
            for line in difflib.unified_diff(
                before["prompt"].splitlines(), after["prompt"].splitlines(), lineterm="", n=1
            )
        ][2:]
    if before["client_events"] != after["client_events"]:
        lines.append(f"~ client_events: {before['client_events']} → {after['client_events']}")
    if before["first_message_override"] != after["first_message_override"]:
        lines.append(
            f"~ first_message override allowed: {before['first_message_override']} → "
            f"{after['first_message_override']}"
        )
    old_tools, new_tools = before["tools"], after["tools"]
    for name in sorted(set(old_tools) | set(new_tools)):
        old, new = old_tools.get(name), new_tools.get(name)
        if same_tool(old, new):
            continue
        if old is None:
            lines.append(f"+ client tool {name} (timeout {new.get('response_timeout_secs')} s)")
        elif new is None:
            lines.append(f"- client tool {name} (detached, not deleted)")
        else:
            keys = [k for k in TOOL_KEYS if old.get(k) != new.get(k)]
            lines.append(f"~ client tool {name}: {', '.join(keys)}")
    return lines


# -- backup -------------------------------------------------------------------------------------
def write_backup(state: State, agent_id: str, directory: Path, now: datetime | None = None) -> Path:
    """Sanitised agent + its tool configs, 0600 in a 0700 directory; returns the path."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    ts = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{agent_id}-{ts}.json"
    doc = {
        "schema": 1,
        "agent_id": agent_id,
        "saved_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "agent": sanitise(state.agent),
        "tools": sanitise(state.tools),
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path


# -- applying -------------------------------------------------------------------------------
class ConcurrencyError(RuntimeError):
    """The agent changed between our read and our PATCH."""


class VerifyError(RuntimeError):
    """The live agent does not show the change."""


def agent_body(state: State, target: Target) -> dict[str, Any]:
    """PATCH body: the whole prompt object (minus legacy ``tools``) + events + overrides."""
    prompt = copy.deepcopy(state.prompt)
    prompt.pop("tools", None)  # inline tools are a legacy mirror of tool_ids: never sent
    prompt["prompt"] = target.prompt
    prompt["tool_ids"] = list(target.tool_ids)
    return {
        "conversation_config": {
            "agent": {"prompt": prompt},
            "conversation": {"client_events": list(target.client_events)},
        },
        "platform_settings": {"overrides": copy.deepcopy(target.overrides)},
    }


def _invariants(agent: Mapping[str, Any]) -> dict[str, Any]:
    """Fields the run must never change (checked after the PATCH)."""
    cc = agent.get("conversation_config") or {}
    prompt = (cc.get("agent") or {}).get("prompt") or {}
    built_in = prompt.get("built_in_tools") or {}
    return {
        "llm": prompt.get("llm"),
        "first_message": (cc.get("agent") or {}).get("first_message"),
        "voice_id": (cc.get("tts") or {}).get("voice_id"),
        "built_in_tools": sorted(k for k, v in built_in.items() if v),
    }


Say = Callable[[str], None]


class Run:
    """One apply / restore with rollback (``say`` prints the ✅/❌ step lines)."""

    def __init__(self, api: ApiLike, agent_id: str, say: Say = print) -> None:
        self.api = api
        self.agent_id = agent_id
        self.say = say
        self.created: list[str] = []
        self.updated: dict[str, dict[str, Any]] = {}  # tool id → config before our update
        self.patched = False

    def execute(self, state: State, target: Target) -> None:
        """Tools, concurrency check, PATCH, verify — raises on any failure (caller rolls back)."""
        tool_ids = list(target.tool_ids)
        for tid, cfg in target.updates.items():
            self.updated[tid] = _subset(state.tools.get(tid) or {})
            self.api.request("PATCH", f"/v1/convai/tools/{tid}", {"tool_config": cfg})
            self.say(f"✅ updated client tool {cfg.get('name')}")
        for name, cfg in target.creates.items():
            res = self.api.request("POST", "/v1/convai/tools", {"tool_config": cfg})
            tid = str(res.get("id") or res.get("tool_id") or "")
            if not tid:
                raise ApiError(f"created tool {name} has no id")
            self.created.append(tid)
            tool_ids.append(tid)
            self.say(f"✅ created client tool {name}")
        final = Target(target.prompt, target.client_events, target.overrides, tool_ids)
        now = self.api.request("GET", f"/v1/convai/agents/{self.agent_id}")
        if stamp_of(now) != stamp_of(state.agent):
            raise ConcurrencyError("the agent changed since it was read — run again")
        self.say("✅ agent unchanged since the backup")
        self.patched = True
        self.api.request("PATCH", f"/v1/convai/agents/{self.agent_id}", agent_body(state, final))
        self.say("✅ agent patched")
        self.verify(state, view_of_target(state, target), tool_ids)

    def verify(self, before: State, want: Mapping[str, Any], tool_ids: list[str]) -> None:
        live = fetch_state(self.api, self.agent_id)
        agent = live.agent
        main_branch = agent.get("main_branch_id")
        if main_branch and agent.get("branch_id") and agent["branch_id"] != main_branch:
            raise VerifyError("the live version is not on the main branch")
        if missing := [t for t in tool_ids if t not in live.tool_ids]:
            raise VerifyError(f"{len(missing)} tool(s) not attached")
        got = view_of(live)
        diff = render_diff(got, dict(want))
        if diff:
            raise VerifyError("live agent differs: " + "; ".join(d.strip() for d in diff[:3]))
        if _invariants(agent) != _invariants(before.agent):
            raise VerifyError("an unmanaged field changed (llm, voice, first message or tools)")
        self.say("✅ verified: the live agent has every change")

    def rollback(self, original: State) -> bool:
        """Undo: restore updated tools + the agent, delete only tools this run created."""
        ok = True
        if self.patched:
            body = agent_body(original, restore_target(original, {"agent": original.agent}))
            ok &= self._step("agent restored", "PATCH", f"/v1/convai/agents/{self.agent_id}", body)
        for tid, cfg in self.updated.items():
            label = f"client tool {cfg.get('name')} restored"
            ok &= self._step(label, "PATCH", f"/v1/convai/tools/{tid}", {"tool_config": cfg})
        for tid in self.created:  # ours, and detached again by the agent restore
            label = f"removed tool {tid} created by this run"
            ok &= self._step(label, "DELETE", f"/v1/convai/tools/{tid}")
        if ok:
            try:
                live = fetch_state(self.api, self.agent_id)
                ok = not render_diff(view_of(live), view_of(original))
            except ApiError:
                ok = False
        self.say("✅ rollback verified" if ok else "❌ rollback NOT verified — check the agent")
        return ok

    def _step(self, label: str, method: str, path: str, body: Any = None) -> bool:
        try:
            self.api.request(method, path, body)
        except ApiError as exc:
            self.say(f"❌ {label}: {exc}")
            return False
        self.say(f"✅ {label}")
        return True


def run(  # pylint: disable=too-many-arguments
    api: ApiLike,
    agent_id: str,
    *,
    apply: bool = False,
    restore: Path | None = None,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    say: Say = print,
) -> int:
    """The whole flow (see module docstring); returns the exit code."""
    try:
        state = fetch_state(api, agent_id)
    except ApiError as exc:
        say(f"❌ GET agent: {exc}")
        return 1
    say(f"✅ read agent {agent_id} (version {state.agent.get('version_id')})")
    path = write_backup(state, agent_id, backup_dir)
    say(f"✅ backup → {path} (0600)")
    if restore is not None:
        target = restore_target(state, json.loads(restore.read_text(encoding="utf-8")))
    else:
        target = desired_target(state)
    diff = render_diff(view_of(state), view_of_target(state, target))
    if not diff:
        say("✅ no changes — the agent is already configured")
        return 0
    changes = sum(1 for line in diff if not line.startswith(" "))
    say(f"ℹ️  {changes} change(s):")
    for line in diff:
        say("  " + line)
    if not apply:
        say("ℹ️  dry run — nothing changed (apply with -a)")
        return 0
    job = Run(api, agent_id, say)
    try:
        job.execute(state, target)
    except (ApiError, ConcurrencyError, VerifyError) as exc:
        say(f"❌ {exc}")
        return 1 if job.rollback(state) else 3
    say("✅ done")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    doc = __doc__ or ""
    parser = argparse.ArgumentParser(
        description=doc.split("\n\n", maxsplit=1)[0],
        epilog="Examples:" + doc.split("Examples:")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-n", "--dry-run", action="store_true", help="backup + diff only (default)")
    mode.add_argument(
        "-a", "--apply", action="store_true", help="apply, verify, roll back on failure"
    )
    parser.add_argument(
        "-r", "--restore", type=Path, metavar="BACKUP", help="restore a backup's managed fields"
    )
    parser.add_argument("-A", "--agent-id", help="agent id (default: ELEVENLABS_AGENT_ID)")
    parser.add_argument(
        "-B", "--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR, help="backup directory"
    )
    args = parser.parse_args(argv)
    if args.restore is not None and not args.restore.is_file():
        print(f"❌ no such backup: {args.restore}")
        return 2

    from my_stt_tts.eleven_voice import _load_env  # pylint: disable=import-outside-toplevel

    _load_env()
    key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = args.agent_id or os.environ.get("ELEVENLABS_AGENT_ID")
    if not key or not agent_id:
        print("❌ ELEVENLABS_API_KEY / ELEVENLABS_AGENT_ID missing (repo .env)")
        return 2
    apply = args.apply or (args.restore is not None and not args.dry_run)
    return run(Api(key), agent_id, apply=apply, restore=args.restore, backup_dir=args.backup_dir)


if __name__ == "__main__":
    sys.exit(main())
