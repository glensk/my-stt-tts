"""Shared helpers for the attended spikes S-SEND and S-PICKER (PLAN_claude-bridge.md).

Not a script: imported by ``s_send.py`` and ``s_picker.py`` (``s_ident.py`` reuses
its venv re-exec and ``claude agents`` helpers). It resolves a Claude Code session
(``claude agents --json``) to its pid, transcript and tty, validates that an iTerm
session really is that session's tab, reads only the records appended to a transcript,
and wraps ccc's ``send_text_via`` plus a raw-key sender over the iTerm2 Python API.

Everything that talks to iTerm needs the ``iterm2`` package, so the scripts re-execute
themselves under ccc's venv python (:func:`ensure_ccc_venv`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CCC_REPO = Path(
    os.environ.get("CCC_REPO", "/Users/albert/obsidian/42-Git/llms/claude-command-center")
)
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", str(Path.home() / ".local/bin/claude"))
REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests/fixtures/spikes"
ACCOUNTS: dict[str, Path | None] = {"cpriv": None, "cwork": Path.home() / ".claude-work"}

OK = "✅"
FAIL = "❌"


def step(ok: bool, msg: str) -> None:
    """Print one status line with the ✅/❌ marker."""
    print(f"{OK if ok else FAIL} {msg}", flush=True)


def now_iso() -> str:
    """Current time, ISO-8601 UTC."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def ensure_iterm2() -> None:
    """Re-exec under ccc's venv python when the ``iterm2`` package is not importable."""
    try:
        import iterm2  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        venv_py = CCC_REPO / ".venv/bin/python"
        if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
            os.execv(str(venv_py), [str(venv_py), *sys.argv])
        raise


def ensure_ccc_venv() -> None:
    """Re-exec under ccc's venv python when ``iterm2`` / ``command_center`` are missing."""
    ensure_iterm2()
    if str(CCC_REPO) not in sys.path:
        sys.path.insert(0, str(CCC_REPO))


# --------------------------------------------------------------------------- sessions


@dataclass
class ClaudeSession:
    """One interactive Claude Code session as ``claude agents --json`` reports it."""

    session_id: str
    name: str
    pid: int | None
    cwd: str
    kind: str
    status: str
    account: str


def agents_json(config_dir: Path | None, cwd: str | None = None) -> Any:
    """Parsed ``claude agents --json``; the account is pinned via *config_dir* (None = cpriv)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR")
    }
    if config_dir is not None:
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    out = subprocess.run(
        [CLAUDE_BIN, "agents", "--json"],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    return json.loads(out)


def list_sessions(account: str = "cpriv") -> list[ClaudeSession]:
    """Every session of *account* (interactive and background)."""
    rows = agents_json(ACCOUNTS[account])
    return [
        ClaudeSession(
            session_id=str(r.get("sessionId") or ""),
            name=str(r.get("name") or ""),
            pid=r.get("pid"),
            cwd=str(r.get("cwd") or ""),
            kind=str(r.get("kind") or ""),
            status=str(r.get("status") or r.get("state") or "unknown"),
            account=account,
        )
        for r in rows
        if isinstance(r, dict)
    ]


def find_session(
    *, session_id: str | None = None, name: str | None = None, account: str = "cpriv"
) -> ClaudeSession | None:
    """The interactive session with *session_id* (preferred) or exact *name*."""
    hits = [
        s
        for s in list_sessions(account)
        if s.kind == "interactive"
        and ((session_id and s.session_id == session_id) or (not session_id and s.name == name))
    ]
    return hits[0] if len(hits) == 1 else None


def session_status(session_id: str, account: str = "cpriv") -> str:
    """Raw status of *session_id* now (``busy``/``idle``/``waiting``/``blocked``/``gone``)."""
    for s in list_sessions(account):
        if s.session_id == session_id:
            return s.status
    return "gone"


def transcript_for(session_id: str, account: str = "cpriv") -> Path | None:
    """The JSONL transcript of *session_id*: owning account first, then the other one."""
    homes = [ACCOUNTS[account] or Path.home() / ".claude"]
    homes += [h or Path.home() / ".claude" for h in ACCOUNTS.values()]
    for home in homes:
        hits = sorted((home / "projects").glob(f"*/{session_id}.jsonl"))
        if hits:
            return hits[0]
    return None


def expected_transcript(session: ClaudeSession, account: str = "cpriv") -> Path:
    """Where Claude Code WILL write *session*'s transcript (a fresh session has none
    until its first prompt): ``projects/<cwd with non-alphanumerics as ->/<id>.jsonl``."""
    home = ACCOUNTS[account] or Path.home() / ".claude"
    slug = re.sub(r"[^A-Za-z0-9]", "-", session.cwd)
    return home / "projects" / slug / f"{session.session_id}.jsonl"


def pid_tty(pid: int | None) -> str:
    """``/dev/ttysNNN`` of *pid*, or ``""``."""
    if not pid:
        return ""
    out = subprocess.run(
        ["ps", "-o", "tty=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return f"/dev/{out}" if out and out not in ("??", "-") else ""


# --------------------------------------------------------------------------- iTerm


def _uuid(iterm_session_id: str) -> str:
    return (iterm_session_id or "").split(":")[-1].strip()


def _iterm_gate() -> bool:
    """ccc's bounded TCC pre-check (the iterm2 cookie request has no timeout)."""
    # command_center exists only in ccc's venv (ensure_ccc_venv re-execs into it).
    from command_center import terminal  # pylint: disable=import-outside-toplevel,import-error

    # pylint: disable=protected-access
    return terminal._iterm_api_auth_is_tcc_free() or terminal._iterm_reachable_by_apple_event()


async def _with_session(iterm_session_id: str, fn: Any) -> Any:
    import iterm2  # pylint: disable=import-outside-toplevel,import-error  # ccc venv only

    # iterm2 keeps a process-wide App singleton bound to the FIRST connection; a later
    # asyncio.run() would get that stale object and fail on the closed socket.
    iterm2.app.invalidate_app()
    conn = await asyncio.wait_for(iterm2.Connection.async_create(), timeout=8)
    app = await iterm2.async_get_app(conn)
    if app is None:
        raise RuntimeError("iTerm2 app not reachable over the Python API")
    session = app.get_session_by_id(_uuid(iterm_session_id))
    if session is None:
        raise LookupError(f"iTerm session {iterm_session_id} not found")
    return await fn(session)


def iterm_tty(iterm_session_id: str) -> str:
    """The ``tty`` variable of the iTerm session (read-only)."""
    if not _iterm_gate():
        raise RuntimeError("iTerm2 not reachable (Automation grant / API disabled)")

    async def _go(session: Any) -> str:
        return str(await session.async_get_variable("tty") or "")

    return str(asyncio.run(_with_session(iterm_session_id, _go)))


def validate_tab(iterm_session_id: str, session: ClaudeSession) -> tuple[bool, str]:
    """True when the iTerm session's tty is the tty of *session*'s pid."""
    want = pid_tty(session.pid)
    have = iterm_tty(iterm_session_id)
    return bool(want) and want == have, f"pid {session.pid} tty {want or '?'} / tab tty {have}"


def screen_lines(iterm_session_id: str) -> list[str]:
    """The mutable screen area of the iTerm session, one string per row (read-only)."""
    if not _iterm_gate():
        raise RuntimeError("iTerm2 not reachable (Automation grant / API disabled)")

    async def _go(session: Any) -> list[str]:
        contents = await session.async_get_screen_contents()
        return [contents.line(i).string for i in range(contents.number_of_lines)]

    return list(asyncio.run(_with_session(iterm_session_id, _go)))


#: Raw bytes per key name. Arrows use the normal-mode CSI form; Ink's key parser also
#: accepts the application-mode SS3 form, so either works for Claude Code's TUI.
KEYS: dict[str, str] = {
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
    "enter": "\r",
    "space": " ",
    "tab": "\t",
    "btab": "\x1b[Z",
    "esc": "\x1b",
}


def parse_keys(spec: str) -> list[tuple[str, str]]:
    """``"down,enter,text:hi\\, you,wait:0.5,2"`` → ``[(token, payload), …]``.

    Tokens: the names in :data:`KEYS`, a single digit ``0``–``9`` (typed as is),
    ``text:<str>`` (typed literally; write a comma as ``\\,``) and ``wait:<seconds>``
    (pause, payload ``""``).
    """
    tokens: list[str] = []
    buf = ""
    i = 0
    while i < len(spec):
        ch = spec[i]
        if ch == "\\" and i + 1 < len(spec) and spec[i + 1] == ",":
            buf += ","
            i += 2
            continue
        if ch == ",":
            tokens.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    tokens.append(buf)
    out: list[tuple[str, str]] = []
    for raw in tokens:
        tok = raw if raw.startswith("text:") else raw.strip().lower()
        if not tok:
            continue
        if tok in KEYS:
            out.append((tok, KEYS[tok]))
        elif len(tok) == 1 and tok.isdigit():
            out.append((tok, tok))
        elif tok.startswith("text:"):
            out.append((tok, tok[5:]))
        elif tok.startswith("wait:"):
            float(tok[5:])  # validates
            out.append((tok, ""))
        else:
            raise ValueError(f"unknown key token {raw!r}")
    return out


def send_keys(iterm_session_id: str, keys: list[tuple[str, str]], delay: float = 0.15) -> int:
    """Type *keys* into the iTerm session via the Python API ONLY; keys sent.

    No AppleScript and no focus change: ``Session.async_send_text`` writes the bytes to
    the session's pty "as though the user had typed it", whether or not the tab is
    visible. ``suppress_broadcast=True`` keeps broadcast-input groups out of it.
    """
    if not _iterm_gate():
        raise RuntimeError("iTerm2 not reachable (Automation grant / API disabled)")

    async def _go(session: Any) -> int:
        sent = 0
        for tok, payload in keys:
            if tok.startswith("wait:"):
                await asyncio.sleep(float(tok[5:]))
                continue
            await session.async_send_text(payload, suppress_broadcast=True)
            sent += 1
            await asyncio.sleep(delay)
        return sent

    return int(asyncio.run(_with_session(iterm_session_id, _go)))


def send_text(iterm_session_id: str, text: str) -> str:
    """ccc's ``send_text_via`` (bracketed paste + 0.4 s + CR); returns the channel.

    ccc's Python-API rung calls ``iterm2.async_get_app`` without invalidating the
    package's App singleton, so after any earlier connection in this process (the tab
    validation) it would hit the dead socket and silently fall back to AppleScript.
    Invalidate first so the spike measures the Python-API rung.
    """
    # Both packages exist only in ccc's venv (ensure_ccc_venv re-execs into it).
    import iterm2  # pylint: disable=import-outside-toplevel,import-error
    from command_center.terminal import (  # pylint: disable=import-outside-toplevel,import-error
        send_text_via,
    )

    iterm2.app.invalidate_app()
    return send_text_via(iterm_session_id, text)


# --------------------------------------------------------------------------- transcript


def normalise(text: str) -> str:
    """NFC, CRLF/CR → LF, trailing spaces per line dropped, outer whitespace stripped."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def text_sha(text: str) -> str:
    """sha256 hex of :func:`normalise` (*text*)."""
    return hashlib.sha256(normalise(text).encode()).hexdigest()


def payload_text(content: Any) -> str | None:
    """A message content (str or list of blocks) as plain text; None if it holds none."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(parts) if parts else None
    return None


def record_text(record: dict[str, Any]) -> tuple[str, str] | None:
    """``(kind, text)`` for the record shapes a delivered prompt can take, else None.

    ``user`` (typed at an idle prompt), ``queue-operation/enqueue`` (typed while busy:
    ``content``), ``attachment/queued_command`` (the queue draining it: ``prompt``).
    """
    rtype = record.get("type")
    if rtype == "user" and not record.get("isMeta"):
        msg = record.get("message")
        text = payload_text(msg.get("content")) if isinstance(msg, dict) else None
        return ("user", text) if text is not None else None
    if rtype == "queue-operation" and record.get("operation") == "enqueue":
        return ("queue-operation/enqueue", str(record.get("content") or ""))
    if rtype == "attachment":
        att = record.get("attachment")
        if isinstance(att, dict) and att.get("type") == "queued_command":
            text = payload_text(att.get("prompt"))
            return ("attachment/queued_command", text) if text is not None else None
    return None


def shape(record: dict[str, Any]) -> dict[str, Any]:
    """A body-free description of *record*: type, sub-type and key names only."""
    out: dict[str, Any] = {"type": record.get("type"), "keys": sorted(record)}
    if record.get("type") == "queue-operation":
        out["operation"] = record.get("operation")
    att = record.get("attachment")
    if isinstance(att, dict):
        out["attachment"] = {
            "type": att.get("type"),
            "commandMode": att.get("commandMode"),
            "keys": sorted(att),
        }
    msg = record.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            out["content_blocks"] = [b.get("type") for b in content if isinstance(b, dict)]
        else:
            out["content_blocks"] = type(content).__name__
    return out


@dataclass
class Anchor:
    """Transcript identity + size before an action; scans read only what follows."""

    path: Path
    inode: int
    size: int

    @classmethod
    def take(cls, path: Path) -> Anchor:
        if not path.exists():  # fresh session: everything written later is "appended"
            return cls(path, 0, 0)
        st = path.stat()
        return cls(path, st.st_ino, st.st_size)

    def appended(self) -> Iterator[dict[str, Any]]:
        """Complete JSON records written after the anchor (a torn last line is skipped)."""
        if not self.path.exists():
            return
        st = self.path.stat()
        if self.inode and st.st_ino != self.inode:
            raise RuntimeError("transcript was replaced (inode changed)")
        with self.path.open("rb") as handle:
            handle.seek(self.size)
            data = handle.read()
        for line in data.split(b"\n"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def all_records(path: Path) -> Iterable[dict[str, Any]]:
    """Every parseable record of *path*, file order."""
    with path.open("rb") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def wait_until(pred: Any, timeout: float, interval: float = 0.2) -> Any:
    """Poll *pred* until it returns truthy or *timeout* passes; its last value."""
    end = time.monotonic() + timeout
    while True:
        val = pred()
        if val or time.monotonic() >= end:
            return val
        time.sleep(interval)


def save_fixture(name: str, payload: dict[str, Any], append: bool = False) -> Path:
    """Write *payload* to ``tests/fixtures/spikes/<name>`` (append → JSON list of runs)."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / name
    if append:
        runs: list[Any] = []
        if path.exists():
            try:
                old = json.loads(path.read_text())
                runs = old if isinstance(old, list) else [old]
            except json.JSONDecodeError:
                runs = []
        runs.append(payload)
        path.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path
