"""The LLM 'brain': provider-agnostic streaming chat with memory + routing.

Programs against an OpenAI-compatible interface. Anthropic/Claude is the default,
but OpenAI, Ollama, vLLM, or any OpenAI-compatible server work via ``LLM_BASE_URL``.
A ``claude-cli`` provider shells out to the Claude Code CLI (``claude -p``) and
keeps a session id for multi-turn continuity — handy when you have no API key.
Heavy clients are lazy-imported so the package imports without the ``llm`` extra.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
from collections.abc import Iterator

from .config import Config
from .util import RateLimiter

log = logging.getLogger("my_stt_tts.brain")


class LLMError(RuntimeError):
    """Raised when the LLM backend fails or the request rate is exceeded."""


def should_use_deep(cfg: Config, text: str) -> bool:
    """True if the utterance asks for the slower, stronger 'deep' model."""
    return cfg.deep_trigger.lower() in text.lower()


class Brain:
    """Streaming chat with short conversation memory and fast/deep routing."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.history: list[dict[str, str]] = []
        self._client: object | None = None
        self._rate = RateLimiter(cfg.requests_per_minute)
        self._session_id: str | None = None  # claude-cli session continuity

    def reset(self) -> None:
        """Forget the conversation (e.g. after an idle timeout)."""
        self.history.clear()
        self._session_id = None

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

    def stream(self, user_text: str, *, deep: bool | None = None) -> Iterator[str]:
        """Yield reply text deltas; append the full reply to history at the end."""
        if not self._rate.acquire():
            raise LLMError("request rate exceeded — slow down")
        use_deep = should_use_deep(self.cfg, user_text) if deep is None else deep
        model = self.cfg.llm_model_deep if use_deep else self.cfg.llm_model
        self.history.append({"role": "user", "content": user_text})
        self._trim()
        reply: list[str] = []
        try:
            if self.cfg.llm_provider == "claude-cli":
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

    def _stream_claude_cli(self, model: str, user_text: str) -> Iterator[str]:
        if not shutil.which("claude"):
            raise LLMError("`claude` CLI not found on PATH")
        cmd = ["claude", "-p", user_text, "--model", model, "--output-format", "json"]
        if self._session_id is None:
            self._session_id = str(uuid.uuid4())
            cmd += ["--session-id", self._session_id]  # create + name this session
        else:
            cmd += ["--resume", self._session_id]  # continue the conversation
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=False, timeout=180
        )
        if proc.returncode != 0:
            raise LLMError(f"claude CLI failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON output: {exc}") from exc
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
