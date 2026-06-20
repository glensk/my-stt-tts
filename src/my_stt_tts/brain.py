"""The LLM 'brain': provider-agnostic streaming chat with memory + routing.

Programs against an OpenAI-compatible interface. Anthropic/Claude is the default,
but OpenAI, Ollama, vLLM, or any OpenAI-compatible server work via ``LLM_BASE_URL``.

The ``claude-cli`` provider shells out to the Claude Code CLI (``claude -p``) and
keeps a session id for multi-turn continuity — handy when you have no API key. It
is deliberately STRIPPED and ISOLATED from your general Claude use
(``--system-prompt`` replaces the agentic prompt with prompts/system_prompt.md,
``--setting-sources ""`` skips ~/.claude & ~/.llm-shared & hooks, ``--tools ""``
disables tools, runs in a non-git scratch dir) — ~8x faster, ~280x cheaper, and
unable to touch this repo.

The ``codex-cli`` provider is the OpenAI equivalent: it shells out to the OpenAI
``codex`` CLI in non-interactive ``codex exec`` mode (uses your logged-in codex
auth, so no API key). It is likewise ISOLATED — a ``read-only`` sandbox,
``--skip-git-repo-check`` + a scratch cwd so it touches nothing, and
``--ignore-user-config`` so your personal ``$CODEX_HOME/config.toml`` is skipped.
The command is overridable via ``CODEX_CLI_CMD`` (default ``codex exec``); the
system prompt is prepended to the user text since ``codex exec`` takes one prompt
argument. ``codex exec`` is stateless per call, so unlike claude-cli it keeps no
resume session id.

When the utterance starts with the agent trigger ("agent, <task>"), the request
is instead delegated to a FULL, MCP-capable Claude agent (see :mod:`.agent`).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

from .agent import AgentError, dispatch_to_agent
from .config import Config, current_time_line, locale_prompt_line
from .memory import ContextAggregator, make_memory_store
from .music import music_state_line
from .tools import ToolCall, ToolRegistry, default_tools
from .util import RateLimiter

log = logging.getLogger("my_stt_tts.brain")


class LLMError(RuntimeError):
    """Raised when the LLM backend fails or the request rate is exceeded."""


def should_use_deep(cfg: Config, text: str) -> bool:
    """True if the utterance asks for the slower, stronger 'deep' model."""
    return cfg.deep_trigger.lower() in text.lower()


class Brain:
    """Streaming chat with short conversation memory, fast/deep + agent routing."""

    def __init__(
        self,
        cfg: Config,
        *,
        tools: ToolRegistry | None = None,
        context: ContextAggregator | None = None,
    ) -> None:
        self.cfg = cfg
        # Per-speaker persistent memory + provider-agnostic context assembly (G7).
        # ``history`` is aliased to the aggregator's LIVE session (back-compat: the
        # backends and commit_spoken still read/repair ``self.history``); the
        # aggregator additionally folds in cross-session per-speaker recall when a
        # ``memory_store`` is configured. Injectable for tests.
        if context is None:
            context = ContextAggregator(
                store=make_memory_store(cfg), max_turns=getattr(cfg, "memory_max_turns", 20)
            )
        self.context = context
        self.history: list[dict[str, str]] = self.context.live
        self.speaker: str | None = None  # set per turn from speaker_id (G7)
        self._client: object | None = None
        self._rate = RateLimiter(cfg.requests_per_minute)
        self._session_id: str | None = None  # claude-cli chat session
        self._agent_session_id: str | None = None  # delegated-agent session
        # Index of the assistant turn just appended by stream(); commit_spoken()
        # uses it to repair history after a barge-in (G5).
        self._pending_assistant_index: int | None = None
        self._pending_user_text: str = ""  # user turn awaiting persistence (G7)
        self._persisted_assistant: bool = False  # whether stream() persisted a reply (G7)
        # In-conversation tool calling (R2-7). The registry exposes get_time, a
        # calculator, and home_control (routed to the existing agent / HA dispatch).
        # Injectable so tests can supply fakes; built from config when omitted.
        if tools is None and cfg.tools_enabled:
            tools = ToolRegistry(
                default_tools(
                    home_dispatch=self._home_dispatch,
                    location=cfg.location,
                    units=cfg.units,
                    music_enabled=cfg.music_enabled,
                    music_player=cfg.music_player,
                    music_volume=cfg.music_volume,
                )
            )
        self.tools = tools

    def set_speaker(self, name: str | None) -> None:
        """Set the current speaker (from speaker_id) so memory + recall are per-person (G7)."""
        self.speaker = name

    def _assembled(self) -> list[dict[str, str]]:
        """Provider-agnostic message list: per-speaker recall + the live session (G7)."""
        return self.context.assemble(self.speaker)

    def _system_prompt(self) -> str:
        """System prompt for the backend: base + locale + current time + system state.

        Keeps ``prompts/system_prompt.md`` (``cfg.system_prompt``) as the editable
        base and appends (a) a locale line so weather/distance/temperature answers
        use the configured place and measurement system, (b) a fresh
        'Current local time: …' line on EVERY turn so the assistant can tell the
        time, and (c) a 'System state: …' line reflecting live music playback so the
        assistant can answer "what's playing?" correctly even on LLM-routed turns
        (the music intent router handles play/stop locally, but a *question* about
        playback reaches the model — which must see the live state, not just the
        chat history). All three are in the prompt *text* (not behind a tool)
        precisely so they work for ``claude-cli`` too, which has no tool access in
        this loop. The timezone is derived from ``cfg.location`` (Lausanne →
        Europe/Zurich) via stdlib ``zoneinfo``, falling back to the system local tz.
        """
        base = locale_prompt_line(self.cfg.system_prompt, self.cfg.location, self.cfg.units)
        return f"{base.rstrip()}\n\n{current_time_line(self.cfg.location)}\n{music_state_line()}"

    def _home_dispatch(self, command: str) -> str:
        """Route a home_control tool call to the agent / HA dispatch (reuses agent.py).

        Mirrors the legacy "agent, ..." path: a capable agent only runs when an
        ``agent_workspace`` is configured, so without one the tool reports that
        cleanly instead of acting in an arbitrary directory.
        """
        if not command:
            return "error: no command given"
        if not self.cfg.agent_workspace:
            return "Home control is not configured (set AGENT_WORKSPACE)."
        try:
            result = dispatch_to_agent(
                command,
                workspace=self.cfg.agent_workspace,
                model=self.cfg.agent_model,
                session_id=self._agent_session_id,
            )
        except AgentError as exc:
            return f"error: {exc}"
        self._agent_session_id = result.session_id
        return result.text or "done"

    def reset(self) -> None:
        """Forget the LIVE conversation (e.g. after an idle timeout).

        Persistent per-speaker memory (G7) is NOT cleared — cross-session recall is
        the point. Use ``self.context.store.forget(speaker)`` to erase a person.
        """
        self.context.reset_live()  # clears in place -> the self.history alias stays valid
        self._session_id = None
        self._agent_session_id = None

    def _trim(self) -> None:
        # Trim the live session IN PLACE so the ``self.history`` alias to
        # ``self.context.live`` stays valid (G7).
        max_msgs = self.cfg.max_history_turns * 2
        if len(self.history) > max_msgs:
            del self.history[:-max_msgs]

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        if self.cfg.llm_provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self.cfg.anthropic_api_key, base_url=self.cfg.llm_base_url or None
            )
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.cfg.openai_api_key or "not-needed",
                base_url=self.cfg.llm_base_url,
            )
        return self._client

    def stream(self, user_text: str, *, deep: bool | None = None) -> Generator[str, None, None]:
        """Yield reply text deltas, recording the reply into history.

        History is appended in the ``finally`` block so a generation aborted by a
        barge-in still stores what was produced. **The caller is responsible for
        repairing the assistant turn to only what was actually *voiced*** — call
        :meth:`commit_spoken` after consuming the stream so the model's memory
        matches reality (G5). Without that call, the full generated reply stands.
        """
        if not self._rate.acquire():
            raise LLMError("request rate exceeded — slow down")
        use_deep = should_use_deep(self.cfg, user_text) if deep is None else deep
        model = self.cfg.llm_model_deep if use_deep else self.cfg.llm_model
        self.history.append({"role": "user", "content": user_text})
        self._trim()
        reply: list[str] = []
        self._pending_assistant_index = None
        self._pending_user_text = user_text  # for per-speaker persistence (G7)
        try:
            agent_task = self._agent_task(user_text)
            use_tools = self.tools is not None and len(self.tools) > 0
            if agent_task is not None:
                deltas = self._dispatch_agent(agent_task)
            elif self.cfg.llm_provider == "claude-cli":
                deltas = self._stream_claude_cli(model, user_text)
            elif self.cfg.llm_provider == "codex-cli":
                deltas = self._stream_codex_cli(model, user_text)
            elif self.cfg.llm_provider == "anthropic":
                deltas = (
                    self._stream_anthropic_tools(model)
                    if use_tools
                    else self._stream_anthropic(model)
                )
            else:
                deltas = (
                    self._stream_openai_tools(model) if use_tools else self._stream_openai(model)
                )
            for delta in deltas:
                reply.append(delta)
                yield delta
        except LLMError:
            raise
        except Exception as exc:  # network/SDK failure -> spoken error clip upstream
            raise LLMError(str(exc)) from exc
        finally:
            if reply:
                full = "".join(reply)
                self.history.append({"role": "assistant", "content": full})
                self._pending_assistant_index = len(self.history) - 1
                # Persist the completed exchange per-speaker (G7). commit_spoken()
                # amends the assistant turn afterwards if a barge-in truncated it.
                self.context.persist(self.speaker, self._pending_user_text, full)
                self._persisted_assistant = bool(full)

    def commit_spoken(self, spoken_text: str) -> None:
        """Repair the just-streamed assistant turn to only what was voiced (G5).

        On a barge-in, only part of the generated reply was actually spoken aloud
        before TTS was aborted. Replacing the stored assistant content with that
        spoken prefix keeps the LLM's memory honest. If nothing was voiced, the
        assistant turn is dropped entirely. A no-op when there is no pending turn
        (e.g. the reply completed normally and you keep the full text).
        """
        index = self._pending_assistant_index
        if index is None or not 0 <= index < len(self.history):
            return
        spoken = spoken_text.strip()
        if spoken:
            self.history[index]["content"] = spoken
        else:
            del self.history[index]
        # Keep persistent per-speaker memory honest too (G7): amend / drop the
        # assistant turn we already persisted in stream()'s finally.
        if getattr(self, "_persisted_assistant", False):
            self.context.amend_last_assistant(self.speaker, spoken)
            self._persisted_assistant = False
        self._pending_assistant_index = None

    # --- Agent dispatch ("agent, <task>" -> full MCP-capable Claude) ---

    def _agent_task(self, text: str) -> str | None:
        """Return the task if ``text`` starts with the agent trigger word, else None."""
        trigger = self.cfg.agent_trigger.strip().lower()
        if not trigger:
            return None
        stripped = text.strip()
        low = stripped.lower()
        if low == trigger or low.startswith(f"{trigger} ") or low.startswith(f"{trigger},"):
            return stripped[len(trigger) :].lstrip(" ,:").strip()
        return None

    def _dispatch_agent(self, task: str) -> Iterator[str]:
        if not self.cfg.agent_workspace:
            yield "Agent mode is not configured. Set an agent workspace to enable it."
            return
        if not task:
            yield "What should the agent do?"
            return
        try:
            result = dispatch_to_agent(
                task,
                workspace=self.cfg.agent_workspace,
                model=self.cfg.agent_model,
                session_id=self._agent_session_id,
            )
        except AgentError as exc:
            raise LLMError(str(exc)) from exc
        self._agent_session_id = result.session_id
        yield result.text

    # --- Chat backends ---

    @staticmethod
    def _claude_cwd() -> str:
        scratch = Path(tempfile.gettempdir()) / "my-stt-tts-claude-cwd"
        scratch.mkdir(parents=True, exist_ok=True)
        return str(scratch)

    def _stream_claude_cli(self, model: str, user_text: str) -> Iterator[str]:
        if not shutil.which("claude"):
            raise LLMError("`claude` CLI not found on PATH")
        cmd = [
            "claude",
            "-p",
            user_text,
            "--model",
            model,
            "--output-format",
            "json",
            "--system-prompt",
            self._system_prompt(),  # replace the agentic prompt (+ locale line)
            "--setting-sources",
            "",  # skip ~/.claude/CLAUDE.md, ~/.llm-shared, hooks
            "--tools",
            "",  # no tools: minimal context, ~8x faster, project-isolated
        ]
        if self._session_id is None:
            self._session_id = str(uuid.uuid4())
            cmd += ["--session-id", self._session_id]
        else:
            cmd += ["--resume", self._session_id]
        proc = subprocess.run(  # noqa: S603
            cmd, cwd=self._claude_cwd(), capture_output=True, text=True, check=False, timeout=180
        )
        data = None
        if proc.stdout.strip():
            try:
                data = json.loads(proc.stdout)
            except json.JSONDecodeError:
                data = None
        if data is None:
            detail = (proc.stderr or proc.stdout or "").strip()[:200]
            raise LLMError(f"claude CLI failed (rc={proc.returncode}): {detail}")
        if data.get("is_error"):
            raise LLMError(str(data.get("result") or "claude CLI error"))
        yield str(data.get("result", ""))

    @staticmethod
    def _codex_base_cmd() -> list[str]:
        """Resolve the codex base command (``CODEX_CLI_CMD``, default ``codex exec``).

        Overridable so a wrapper / alternate binary can be substituted in tests or by
        a user whose codex is invoked differently. Parsed with ``shlex`` so the env
        value may carry extra args (e.g. ``CODEX_CLI_CMD="codex exec --profile fast"``).
        """
        raw = os.environ.get("CODEX_CLI_CMD", "").strip()
        return shlex.split(raw) if raw else ["codex", "exec"]

    def _stream_codex_cli(self, model: str, user_text: str) -> Iterator[str]:
        """Non-interactive OpenAI codex turn (``codex exec``), isolated + key-free.

        ``codex exec`` prints only the final assistant message to stdout, so we
        capture stdout as the reply. The run is sandboxed read-only, skips the git
        check (we run in a scratch cwd), and ignores the user's codex config so it is
        decoupled from general codex use. The system prompt is prepended to the user
        text because ``codex exec`` accepts a single prompt argument.

        NOTE (assumption): the exact ``codex exec`` flags below are taken from the
        documented OpenAI Codex CLI reference (``--model``, ``--sandbox read-only``,
        ``--skip-git-repo-check``, ``--ignore-user-config``); they were not verified
        against a live binary in this environment. Override with ``CODEX_CLI_CMD`` if
        your codex build differs.
        """
        base = self._codex_base_cmd()
        if not shutil.which(base[0]):
            raise LLMError(f"`{base[0]}` CLI not found on PATH")
        prompt = f"{self._system_prompt()}\n\n{user_text}"
        cmd = [
            *base,
            "--model",
            model,
            "--sandbox",
            "read-only",  # touch nothing: read-only sandbox keeps it project-isolated
            "--skip-git-repo-check",  # we run in a scratch cwd, not a repo
            "--ignore-user-config",  # skip $CODEX_HOME/config.toml: decoupled from your codex
            prompt,
        ]
        proc = subprocess.run(  # noqa: S603
            cmd, cwd=self._codex_cwd(), capture_output=True, text=True, check=False, timeout=180
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:200]
            raise LLMError(f"codex CLI failed (rc={proc.returncode}): {detail}")
        yield proc.stdout.strip()

    @staticmethod
    def _codex_cwd() -> str:
        scratch = Path(tempfile.gettempdir()) / "my-stt-tts-codex-cwd"
        scratch.mkdir(parents=True, exist_ok=True)
        return str(scratch)

    def _stream_anthropic(self, model: str) -> Iterator[str]:
        client = self._ensure_client()
        with client.messages.stream(  # type: ignore[attr-defined]
            model=model,
            max_tokens=1024,
            system=self._system_prompt(),
            messages=self._assembled(),
        ) as stream:
            yield from stream.text_stream

    def _stream_openai(self, model: str) -> Iterator[str]:
        client = self._ensure_client()
        messages = [{"role": "system", "content": self._system_prompt()}, *self._assembled()]
        stream = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model, messages=messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # --- tool-calling round-trips (R2-7) ---------------------------------

    def _stream_anthropic_tools(self, model: str) -> Iterator[str]:
        """Anthropic tool-use loop: call -> run tools -> feed results -> stream answer.

        Runs the (non-streaming) messages API with the tool schemas. While the
        model asks for tools (``stop_reason == "tool_use"``) we execute them, append
        the assistant tool-use turn and the tool results, and iterate; any text the
        model spoke alongside the call is yielded. Once the model stops requesting
        tools the final answer is streamed token-by-token. Bounded by
        ``tools_max_iterations`` so a model can't loop forever.
        """
        assert self.tools is not None
        client = self._ensure_client()
        system = self._system_prompt()
        messages: list[dict[str, Any]] = [dict(m) for m in self._assembled()]
        tool_schemas = self.tools.anthropic_tools()
        for _ in range(self.cfg.tools_max_iterations):
            msg = client.messages.create(  # type: ignore[attr-defined]
                model=model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=tool_schemas,
            )
            calls = _anthropic_tool_calls(msg)
            if not calls:
                break  # no tool requested: stream the real answer below
            for block in _anthropic_text_blocks(msg):
                if block:
                    yield block
            messages.append({"role": "assistant", "content": _content_blocks(msg)})
            messages.append({"role": "user", "content": self._anthropic_tool_results(calls)})
        # Final pass with the (possibly tool-augmented) context — streamed.
        with client.messages.stream(  # type: ignore[attr-defined]
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tool_schemas,
        ) as stream:
            yield from stream.text_stream

    def _anthropic_tool_results(self, calls: list[ToolCall]) -> list[dict[str, Any]]:
        """Execute ``calls`` and wrap each result as an Anthropic ``tool_result`` block."""
        assert self.tools is not None
        results: list[dict[str, Any]] = []
        for call in calls:
            output = self.tools.dispatch(call.name, call.arguments)
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
        return results

    def _stream_openai_tools(self, model: str) -> Iterator[str]:
        """OpenAI tool-call loop: call -> run tools -> feed results -> stream answer.

        Mirror of the Anthropic path against the OpenAI chat-completions tool API:
        the model emits ``tool_calls``; we run each, append a ``tool`` message with
        its result, and iterate; the final answer is streamed. Bounded by
        ``tools_max_iterations``.
        """
        assert self.tools is not None
        client = self._ensure_client()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            *(dict(m) for m in self._assembled()),
        ]
        tool_schemas = self.tools.openai_tools()
        for _ in range(self.cfg.tools_max_iterations):
            completion = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model, messages=messages, tools=tool_schemas
            )
            message = completion.choices[0].message
            calls = _openai_tool_calls(message)
            if not calls:
                break
            messages.append(_openai_assistant_message(message, calls))
            for call in calls:
                output = self.tools.dispatch(call.name, call.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        stream = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model, messages=messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# --- provider-response parsing helpers (R2-7) ----------------------------------
# Defensive accessors so the round-trip works with both the real SDK objects and
# the lightweight fakes the tests inject (duck-typed: ``.type``/``.content``/etc).


def _anthropic_tool_calls(msg: Any) -> list[ToolCall]:
    """Extract ``tool_use`` blocks from an Anthropic message as :class:`ToolCall`s."""
    calls: list[ToolCall] = []
    for block in getattr(msg, "content", None) or []:
        if getattr(block, "type", None) == "tool_use":
            calls.append(
                ToolCall(
                    id=str(getattr(block, "id", "")),
                    name=str(getattr(block, "name", "")),
                    arguments=dict(getattr(block, "input", {}) or {}),
                )
            )
    return calls


def _anthropic_text_blocks(msg: Any) -> list[str]:
    """Extract the text spoken alongside any tool call (usually empty)."""
    out: list[str] = []
    for block in getattr(msg, "content", None) or []:
        if getattr(block, "type", None) == "text":
            out.append(str(getattr(block, "text", "")))
    return out


def _content_blocks(msg: Any) -> list[dict[str, Any]]:
    """Re-serialize an Anthropic message's content blocks for the next request.

    Anthropic requires the assistant ``tool_use`` turn echoed back verbatim before
    the matching ``tool_result``; rebuild the minimal block dicts (text + tool_use).
    """
    blocks: list[dict[str, Any]] = []
    for block in getattr(msg, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": str(getattr(block, "text", ""))})
        elif kind == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(getattr(block, "id", "")),
                    "name": str(getattr(block, "name", "")),
                    "input": dict(getattr(block, "input", {}) or {}),
                }
            )
    return blocks


def _openai_tool_calls(message: Any) -> list[ToolCall]:
    """Extract ``tool_calls`` from an OpenAI chat message as :class:`ToolCall`s."""
    calls: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        raw_args = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        calls.append(
            ToolCall(
                id=str(getattr(tc, "id", "")),
                name=str(getattr(fn, "name", "") if fn is not None else ""),
                arguments=args if isinstance(args, dict) else {},
            )
        )
    return calls


def _openai_assistant_message(message: Any, calls: list[ToolCall]) -> dict[str, Any]:
    """Rebuild the assistant turn (with ``tool_calls``) to echo back to OpenAI."""
    return {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ],
    }
