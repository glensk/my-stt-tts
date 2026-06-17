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

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass

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
    proc = subprocess.run(  # noqa: S603
        cmd, cwd=workspace, capture_output=True, text=True, check=False, timeout=timeout
    )
    data = None
    if proc.stdout.strip():
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            data = None
    if data is None:
        detail = (proc.stderr or proc.stdout or "").strip()[:200]
        raise AgentError(f"agent failed (rc={proc.returncode}): {detail}")
    if data.get("is_error"):
        raise AgentError(str(data.get("result") or "agent error"))
    return AgentResult(text=str(data.get("result", "")), session_id=data.get("session_id"))
