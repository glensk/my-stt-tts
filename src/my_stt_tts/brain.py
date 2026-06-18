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

When the utterance starts with the agent trigger ("agent, <task>"), the request
is instead delegated to a FULL, MCP-capable Claude agent (see :mod:`.agent`).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path

from .agent import AgentError, dispatch_to_agent
from .config import Config
from .util import RateLimiter

log = logging.getLogger("my_stt_tts.brain")


class LLMError(RuntimeError):
    """Raised when the LLM backend fails or the request rate is exceeded."""


def should_use_deep(cfg: Config, text: str) -> bool:
    """True if the utterance asks for the slower, stronger 'deep' model."""
    return cfg.deep_trigger.lower() in text.lower()


class Brain:
    """Streaming chat with short conversation memory, fast/deep + agent routing."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.history: list[dict[str, str]] = []
        self._client: object | None = None
        self._rate = RateLimiter(cfg.requests_per_minute)
        self._session_id: str | None = None  # claude-cli chat session
        self._agent_session_id: str | None = None  # delegated-agent session
        # Index of the assistant turn just appended by stream(); commit_spoken()
        # uses it to repair history after a barge-in (G5).
        self._pending_assistant_index: int | None = None

    def reset(self) -> None:
        """Forget the conversation (e.g. after an idle timeout)."""
        self.history.clear()
        self._session_id = None
        self._agent_session_id = None

    def _trim(self) -> None:
        max_msgs = self.cfg.max_history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

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
        try:
            agent_task = self._agent_task(user_text)
            if agent_task is not None:
                deltas = self._dispatch_agent(agent_task)
            elif self.cfg.llm_provider == "claude-cli":
                deltas = self._stream_claude_cli(model, user_text)
            elif self.cfg.llm_provider == "anthropic":
                deltas = self._stream_anthropic(model)
            else:
                deltas = self._stream_openai(model)
            for delta in deltas:
                reply.append(delta)
                yield delta
        except LLMError:
            raise
        except Exception as exc:  # network/SDK failure -> spoken error clip upstream
            raise LLMError(str(exc)) from exc
        finally:
            if reply:
                self.history.append({"role": "assistant", "content": "".join(reply)})
                self._pending_assistant_index = len(self.history) - 1

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
            self.cfg.system_prompt,  # replace the agentic prompt
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

    def _stream_anthropic(self, model: str) -> Iterator[str]:
        client = self._ensure_client()
        with client.messages.stream(  # type: ignore[attr-defined]
            model=model,
            max_tokens=1024,
            system=self.cfg.system_prompt,
            messages=self.history,
        ) as stream:
            yield from stream.text_stream

    def _stream_openai(self, model: str) -> Iterator[str]:
        client = self._ensure_client()
        messages = [{"role": "system", "content": self.cfg.system_prompt}, *self.history]
        stream = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model, messages=messages, stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
