#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/llms/claude-command-center python # pylint: disable=line-too-long
"""Spike S-IDENT (PLAN_claude-bridge.md, Phase 0): map every Claude Code session to its iTerm tab.

For every session of both accounts (``claude agents --json`` under cpriv and cwork) it
reports: session id, account, pid, the pid's tty (``ps``), ccc's stored
``iterm_session_id``, whether that iTerm session still exists, whether its tty equals the
pid's tty (= validated), and a fallback match by tty when ccc's id is wrong or missing.
Background sessions are listed but excluded from the percentage.

Read-only: it queries ``claude agents``, ``ps``, the ccc SQLite store (opened read-only)
and the iTerm2 Python API (list sessions + variables; nothing is written or focused).
It needs the ``iterm2`` package, so it re-executes itself under ccc's venv when needed.

Examples:
    scripts/spikes/s_ident.py            # table + percentage
    scripts/spikes/s_ident.py -j         # JSON (aim/prompt text never included)
    scripts/spikes/s_ident.py -j -o tests/fixtures/spikes/s_ident.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bridge_spike_lib as lib

ACCOUNTS: dict[str, Path | None] = {"private": None, "work": Path.home() / ".claude-work"}
CCC_DB = Path.home() / ".claude/command-center/state.db"


@dataclass
class ItermSession:
    """One iTerm2 session (pane) as seen through the Python API."""

    uuid: str
    tty: str
    job_pid: int | None
    job_name: str
    shell_pid: int | None


@dataclass
class Row:  # pylint: disable=too-many-instance-attributes  # one report row
    """The S-IDENT verdict for one Claude Code session."""

    session_id: str
    account: str
    kind: str
    status: str | None
    pid: int | None
    pid_tty: str
    registry_session_id: str | None
    entrypoint: str | None
    counted: bool
    ccc_row: bool
    ccc_iterm_session_id: str | None
    ccc_config_dir: str | None
    ccc_last_seen_pid: int | None
    ccc_done: bool | None
    ccc_tab_exists: bool
    ccc_tab_tty: str
    validated: bool
    foreground_is_pid: bool | None
    tty_match_uuid: str | None
    tty_match_differs_from_ccc: bool
    other_ccc_rows_on_tab: list[str] = field(default_factory=list)
    reason: str = ""


def claude_agents(account: str, config_dir: Path | None) -> list[dict[str, Any]]:
    """``claude agents --json`` for one account (env pinned explicitly)."""
    data = lib.agents_json(config_dir, cwd="/tmp")
    if not isinstance(data, list):
        raise TypeError(f"{account}: unexpected agents payload {type(data).__name__}")
    return data


def registry(config_dir: Path | None, pid: int | None) -> dict[str, Any] | None:
    """``<config>/sessions/<pid>.json`` (the live registry entry) or None."""
    if pid is None:
        return None
    base = config_dir if config_dir is not None else Path.home() / ".claude"
    path = base / "sessions" / f"{pid}.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def ps_table() -> dict[int, tuple[int, str, str]]:
    """``{pid: (ppid, tty, stat)}`` from one ``ps`` pass; tty normalised to ``/dev/...``."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,tty=,stat="], capture_output=True, text=True, check=True
    ).stdout
    table: dict[int, tuple[int, str, str]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        tty = parts[2]
        tty = "" if tty in ("??", "-") else (tty if tty.startswith("/dev/") else f"/dev/{tty}")
        table[int(parts[0])] = (int(parts[1]), tty, parts[3])
    return table


def is_descendant(pid: int, ancestor: int, table: dict[int, tuple[int, str, str]]) -> bool:
    """True when *pid* equals *ancestor* or descends from it."""
    seen: set[int] = set()
    while pid and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        row = table.get(pid)
        if row is None:
            return False
        pid = row[0]
    return False


async def _iterm_sessions_async() -> list[ItermSession]:
    import iterm2  # pylint: disable=import-outside-toplevel,import-error  # ccc venv

    sys.path.insert(0, str(lib.CCC_REPO))
    from command_center import (  # pylint: disable=import-outside-toplevel,import-error
        iterm_api,
    )

    creds = iterm_api.fetch_cookie()
    if creds is None:
        raise ConnectionError("no iTerm2 cookie (Python API disabled or iTerm not running?)")
    with iterm_api.scoped_auth_env(*creds):
        connection = await iterm2.Connection.async_create()
    app = await iterm2.async_get_app(connection)
    result: list[ItermSession] = []
    for window in app.terminal_windows:
        for tab in window.tabs:
            for session in tab.sessions:
                tty = await session.async_get_variable("tty") or ""
                job_pid = await session.async_get_variable("jobPid")
                job_name = await session.async_get_variable("jobName") or ""
                shell_pid = await session.async_get_variable("pid")
                result.append(
                    ItermSession(
                        uuid=session.session_id,
                        tty=str(tty),
                        job_pid=int(job_pid) if job_pid else None,
                        job_name=str(job_name),
                        shell_pid=int(shell_pid) if shell_pid else None,
                    )
                )
    return result


def iterm_sessions() -> list[ItermSession]:
    """Every iTerm2 session with tty / foreground job (read-only API queries)."""
    return asyncio.run(_iterm_sessions_async())


def ccc_rows(session_ids: list[str]) -> dict[str, dict[str, Any]]:
    """ccc store rows for *session_ids* (DB opened read-only)."""
    conn = sqlite3.connect(f"file:{CCC_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    marks = ",".join("?" * len(session_ids))
    rows = conn.execute(
        "SELECT session_id, iterm_session_id, config_dir, last_seen_pid, done, archived "
        f"FROM sessions WHERE session_id IN ({marks})",
        session_ids,
    ).fetchall()
    out = {r["session_id"]: dict(r) for r in rows}
    conn.close()
    return out


def ccc_rows_by_tab() -> dict[str, list[str]]:
    """``{TAB_UUID: [session_id, …]}`` over the whole store (to show tabs outliving sessions)."""
    conn = sqlite3.connect(f"file:{CCC_DB}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT session_id, iterm_session_id FROM sessions WHERE iterm_session_id != ''"
    ).fetchall()
    conn.close()
    by_tab: dict[str, list[str]] = {}
    for sid, tab in rows:
        by_tab.setdefault(tab.split(":")[-1].strip().upper(), []).append(sid)
    return by_tab


def _uuid(iterm_session_id: str | None) -> str:
    return (iterm_session_id or "").split(":")[-1].strip().upper()


def build_rows() -> list[Row]:
    """Collect everything and compute the verdict per session."""
    agents: list[tuple[str, Path | None, dict[str, Any]]] = []
    for account, cdir in ACCOUNTS.items():
        agents.extend((account, cdir, a) for a in claude_agents(account, cdir))
    table = ps_table()
    panes = iterm_sessions()
    by_uuid = {p.uuid.upper(): p for p in panes}
    by_tty: dict[str, list[ItermSession]] = {}
    for p in panes:
        if p.tty:
            by_tty.setdefault(p.tty, []).append(p)
    store = ccc_rows([a["sessionId"] for _acc, _cd, a in agents])
    tabs = ccc_rows_by_tab()
    rows: list[Row] = []
    for account, cdir, a in agents:
        sid = a["sessionId"]
        pid = a.get("pid")
        reg = registry(cdir, pid)
        pid_tty = table[pid][1] if pid in table else ""
        crow = store.get(sid)
        ccc_tab = crow["iterm_session_id"] if crow else None
        pane = by_uuid.get(_uuid(ccc_tab)) if ccc_tab else None
        tab_tty = pane.tty if pane else ""
        validated = bool(pane and pid_tty and tab_tty == pid_tty)
        tty_panes = by_tty.get(pid_tty, []) if pid_tty else []
        tty_match = tty_panes[0].uuid if len(tty_panes) == 1 else None
        fg: bool | None = None
        target = pane if validated else (tty_panes[0] if len(tty_panes) == 1 else None)
        if target is not None and pid is not None and target.job_pid is not None:
            fg = is_descendant(target.job_pid, pid, table)
        others = [s for s in tabs.get(_uuid(ccc_tab or tty_match), []) if s != sid]
        row = Row(
            session_id=sid,
            account=account,
            kind=str(a.get("kind")),
            status=a.get("status") or a.get("state"),
            pid=pid,
            pid_tty=pid_tty,
            registry_session_id=(reg or {}).get("sessionId"),
            entrypoint=(reg or {}).get("entrypoint"),
            counted=is_counted(str(a.get("kind")), (reg or {}).get("entrypoint")),
            ccc_row=crow is not None,
            ccc_iterm_session_id=ccc_tab,
            ccc_config_dir=crow["config_dir"] if crow else None,
            ccc_last_seen_pid=crow["last_seen_pid"] if crow else None,
            ccc_done=bool(crow["done"]) if crow else None,
            ccc_tab_exists=pane is not None,
            ccc_tab_tty=tab_tty,
            validated=validated,
            foreground_is_pid=fg,
            tty_match_uuid=tty_match,
            tty_match_differs_from_ccc=bool(tty_match and tty_match.upper() != _uuid(ccc_tab)),
            other_ccc_rows_on_tab=others,
        )
        row.reason = classify(row, len(tty_panes))
        rows.append(row)
    return rows


def is_counted(kind: str, entrypoint: str | None) -> bool:
    """True for a terminal session that must own a tab.

    ``claude agents`` reports headless ``claude -p`` (SDK) processes as ``kind:
    interactive`` too; only the registry's ``entrypoint`` (``cli`` vs ``sdk-cli``)
    tells them apart, so those are listed but excluded like background sessions.
    """
    return kind == "interactive" and entrypoint != "sdk-cli"


def classify(row: Row, n_tty_panes: int) -> str:  # pylint: disable=too-many-return-statements
    """One precise reason string per session."""
    if row.kind != "interactive":
        return "background session (no tab expected; excluded)"
    if not row.counted:
        return "headless claude -p (registry entrypoint sdk-cli, no tty; excluded)"
    if row.validated:
        return "ok"
    if not row.pid_tty:
        return "pid has no controlling tty (not running in a terminal / pid gone)"
    if n_tty_panes == 0:
        return "no iTerm session owns the pid's tty (not an iTerm tab, e.g. tmux/other terminal)"
    if not row.ccc_row:
        return "no ccc row for this session id (session id unknown to ccc)"
    if not row.ccc_iterm_session_id:
        return "ccc row has no iterm_session_id (missing id); tty fallback resolves it"
    if not row.ccc_tab_exists:
        return "ccc iterm_session_id points at a closed iTerm session (stale id); tty fallback"
    return (
        "ccc iterm_session_id is a live tab with a different tty (tab outlived the session /"
        " reused id); tty fallback"
    )


def summarise(rows: list[Row]) -> dict[str, Any]:
    """Percentages: validated by ccc id alone, and resolvable with the tty fallback."""
    inter = [r for r in rows if r.counted]
    ok = [r for r in inter if r.validated]
    resolvable = [r for r in inter if r.validated or r.tty_match_uuid]
    pct = round(100.0 * len(ok) / len(inter), 1) if inter else 0.0
    pct_tty = round(100.0 * len(resolvable) / len(inter), 1) if inter else 0.0
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "interactive": len(inter),
        "excluded_background_or_headless": len(rows) - len(inter),
        "validated_by_ccc_id": len(ok),
        "validated_pct": pct,
        "resolvable_with_tty_fallback": len(resolvable),
        "resolvable_pct": pct_tty,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-j", "--json", action="store_true", help="print JSON")
    parser.add_argument("-o", "--output", type=Path, help="also write the JSON to this file")
    args = parser.parse_args(argv)
    lib.ensure_iterm2()
    rows = build_rows()
    summary = summarise(rows)
    payload: dict[str, Any] = {"summary": summary, "sessions": [asdict(r) for r in rows]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for r in rows:
        mark = "✅" if r.validated else ("·" if not r.counted else "❌")
        print(
            f"{mark} {r.account:7} {r.kind:11} {r.session_id[:8]} pid={r.pid} "
            f"tty={r.pid_tty or '-'} ccc={_uuid(r.ccc_iterm_session_id)[:8] or '-'} "
            f"tty_match={(r.tty_match_uuid or '-')[:8]} — {r.reason}"
        )
    s = summary
    print(
        f"\nvalidated {s['validated_by_ccc_id']}/{s['interactive']} = {s['validated_pct']} %;"
        f" with tty fallback {s['resolvable_with_tty_fallback']}/{s['interactive']}"
        f" = {s['resolvable_pct']} %"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
