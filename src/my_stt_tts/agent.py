"""Multi-agent dispatch: hand a task to a full, MCP-capable Claude Code agent.

Unlike the stripped chat brain (fast, no tools), this invokes a *full* ``claude
-p`` — with its tools, MCP servers, and settings — inside a configured workspace
directory, so it can actually act (read/write files, call MCP, run commands).
This is the "organize other agents at home/work" primitive: the voice front-end
recognises an "agent, <task>" request and delegates the heavy lifting here.

Because a capable agent should never run in an arbitrary place, the caller must
pass an explicit ``workspace`` (the loop disables the feature until one is set).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass

from .run_ledger import (
    RunRecorder,
    claude_seat,
    emit_run,
    json_object,
    record_run,
    usage_fields,
)

log = logging.getLogger("my_stt_tts.agent")


class AgentError(RuntimeError):
    """Raised when the dispatched agent fails."""


@dataclass
class AgentResult:
    """Text result of an agent run, plus the session id for follow-ups."""

    text: str
    session_id: str | None = None


def dispatch_to_agent(
    task: str,
    *,
    workspace: str,
    model: str = "sonnet",
    session_id: str | None = None,
    timeout: float = 600.0,
    recorder: RunRecorder = record_run,
) -> AgentResult:
    """Run ``task`` on a full Claude Code agent in ``workspace`` and return its reply.

    Reuses ``session_id`` (via ``--resume``) for multi-turn agent continuity.
    """
    if not shutil.which("claude"):
        raise AgentError("`claude` CLI not found on PATH")
    cmd = ["claude", "-p", task, "--model", model, "--output-format", "json"]
    if session_id:
        cmd += ["--resume", session_id]
    log.info("dispatching to agent in %s: %s", workspace, task[:80])
    started = time.perf_counter()
    row: dict[str, object] = {
        "provider": "claude",
        "seat": claude_seat(),
        "purpose": "stt-agent",
        "outcome": "error:exec",
        "ok": False,
        "ms": 0,
        "caller": "my_stt_tts.agent",
        "requested_model": model,
        "prompt_chars": len(task),
        "cwd": workspace,
        "write": True,
    }
    try:
        try:
            proc = subprocess.run(
                cmd, cwd=workspace, capture_output=True, text=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            row.update(outcome="error:timeout", error="timeout", error_message=str(exc))
            raise AgentError(f"agent timed out after {timeout:g}s") from exc
        except OSError as exc:
            row.update(outcome="error:exec", error=type(exc).__name__, error_message=str(exc))
            raise AgentError(f"agent failed to start: {exc}") from exc

        data = json_object(proc.stdout)
        if data is None:
            detail = (proc.stderr or proc.stdout or "").strip()[:200]
            row.update(
                outcome="error:invalid-json",
                error="invalid-json",
                error_message=detail or f"exit {proc.returncode}",
            )
            raise AgentError(f"agent failed (rc={proc.returncode}): {detail}")

        row.update(usage_fields(data))
        if data.get("session_id"):
            row["session"] = str(data["session_id"])
        if data.get("is_error"):
            detail = str(data.get("result") or "agent error")
            row.update(outcome="error:claude", error="claude", error_message=detail)
            raise AgentError(detail)
        if proc.returncode != 0:
            detail = (proc.stderr or f"exit {proc.returncode}").strip()[:200]
            row.update(outcome="error:exit", error="exit", error_message=detail)
            raise AgentError(f"agent failed (rc={proc.returncode}): {detail}")

        row.update(outcome="ok", ok=True)
        return AgentResult(text=str(data.get("result", "")), session_id=data.get("session_id"))
    finally:
        row["ms"] = int((time.perf_counter() - started) * 1000)
        emit_run(recorder, row)
