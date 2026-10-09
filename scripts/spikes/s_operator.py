#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Spike S-OPERATOR: run the exact section-7.4 operator argv against the prototype broker.

For each run it creates a fresh 0700 call dir under
``~/.local/state/mac-voice/operator/<call-id>``, writes a broker MCP config there,
and runs (env built from scratch with ``env -i``)::

    ~/.local/bin/claude -p --model sonnet --tools "" --strict-mcp-config
      --mcp-config <broker.json> --permission-mode bypassPermissions
      --setting-sources "" --max-turns 30 --output-format json
      --json-schema '<contents of schemas/operator_result.json>'

with the request on stdin. Afterwards it independently reads the target app's
display through System Events, records wall time / turns / cost / per-step broker
timing, and writes a JSON summary (no screenshots, no content, no secrets).

Wrap it in the shared screen lock when other agents drive the screen::

    screen_lock.py -- .venv/bin/python scripts/spikes/s_operator.py -r 3

Usage:
    s_operator.py [-r RUNS] [-q REQUEST] [-x EXPECT] [-a APP] [-m MODEL] [-M MAX_PX]
                  [-t MAX_TURNS] [-o OUT] [-l LABEL] [-A] [-N] [-k KEEP_DIR] [-n]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import broker_proto  # same directory: on sys.path[0] when run as a script

REPO = Path(__file__).resolve().parents[2]
BROKER = REPO / "scripts" / "spikes" / "broker_proto.py"
SCHEMA = REPO / "src" / "my_stt_tts" / "schemas" / "operator_result.json"
VENV_PY = REPO / ".venv" / "bin" / "python"
CLAUDE = Path.home() / ".local" / "bin" / "claude"
OPERATOR_ROOT = Path.home() / ".local" / "state" / "mac-voice" / "operator"
DEFAULT_OUT = REPO / "tests" / "fixtures" / "spikes" / "s_operator.json"
THRESHOLD_S = 25.0

PREAMBLE = """You operate this Mac through the `mac` broker tools only.
Do the request below with as few tool calls as possible. Apps may show leftover
state from earlier use. After your LAST action, call observe_screen and check the
image; then call finish(status, summary, evidence_obs_id) with that observation's
obs_id. Your structured result must repeat the status the broker recorded.
Keep the summary to one short sentence.

Request: """

_AS_RUNNING = "on run argv\nreturn application (item 1 of argv) is running\nend run"
_AS_QUIT = "on run argv\ntell application (item 1 of argv) to quit\nend run"
# Static texts of the app's first window (Calculator: expression, then result).
_AS_TEXTS = (
    """
on run argv
    set out to {}
    tell application "System Events"
        tell process (item 1 of argv)
            if (count of windows) = 0 then return ""
            set els to entire contents of window 1
        end tell
        repeat with e in els
            try
                if role of e is "AXStaticText" then set end of out to (value of e as text)
            end try
        end repeat
    end tell
"""
    + broker_proto.AS_RETURN_OUT_LINES
)
# Leave a distinct value (49) so a later run cannot pass on a restored "4".
_AS_POISON = """
on run argv
    tell application (item 1 of argv) to activate
    delay 0.2
    tell application "System Events"
        key code 53
        keystroke "7*7="
    end tell
end run
"""


def _osa(script: str, *args: str, timeout: float = 30) -> str:
    proc = broker_proto.run_osascript(script, *args, timeout=timeout)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def app_running(app: str) -> bool:
    return _osa(_AS_RUNNING, app) == "true"


def quit_app(app: str) -> None:
    _osa(_AS_QUIT, app)
    for _ in range(50):
        if not app_running(app):
            return
        time.sleep(0.1)


def bare_env() -> list[str]:
    """`env -i` assignments: the 7.4 set plus USER (needed for the Keychain login lookup)."""
    home = str(Path.home())
    return [
        "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
        f"HOME={home}",
        "LANG=en_US.UTF-8",
        f"USER={os.environ.get('USER') or Path.home().name}",
        f"CLAUDE_CONFIG_DIR={home}/.claude-work",
        "AI_NO_AUTOCOMMIT=1",
    ]


def make_call_dir(call_id: str) -> Path:
    OPERATOR_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    OPERATOR_ROOT.chmod(0o700)
    call_dir = OPERATOR_ROOT / call_id
    call_dir.mkdir(mode=0o700)
    return call_dir


def write_broker_config(call_dir: Path, max_px: int, keep_dir: Path | None) -> Path:
    args = [str(BROKER), "-c", str(call_dir), "-m", str(max_px)]
    if keep_dir is not None:
        args += ["-k", str(keep_dir)]
    cfg = {"mcpServers": {"mac": {"type": "stdio", "command": str(VENV_PY), "args": args}}}
    path = call_dir / "broker.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(cfg, fh)
    return path


def operator_argv(broker_json: Path, model: str, max_turns: int, no_persist: bool) -> list[str]:
    schema_inline = json.dumps(json.loads(SCHEMA.read_text()), separators=(",", ":"))
    argv = [
        str(CLAUDE),
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        str(broker_json),
        "--permission-mode",
        "bypassPermissions",
        "--setting-sources",
        "",
        "--max-turns",
        str(max_turns),
        "--output-format",
        "json",
        "--json-schema",
        schema_inline,
    ]
    if no_persist:
        argv.append("--no-session-persistence")
    return argv


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def one_run(i: int, opts: argparse.Namespace) -> dict[str, Any]:
    call_id = f"s-op-{dt.datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}-{i}"
    call_dir = make_call_dir(call_id)
    broker_json = write_broker_config(call_dir, opts.max_px, opts.keep_dir)
    argv = [
        "env",
        "-i",
        *bare_env(),
        *operator_argv(broker_json, opts.model, opts.max_turns, opts.no_persist),
    ]
    stdin = PREAMBLE + opts.request + "\n"
    t0_wall = time.time()
    t0 = time.monotonic()
    proc = subprocess.run(
        argv, input=stdin, capture_output=True, text=True, cwd=call_dir, timeout=300, check=False
    )
    wall_s = time.monotonic() - t0
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {}
    calls = read_jsonl(call_dir / "broker_calls.jsonl")
    finish = (
        json.loads((call_dir / "finish.json").read_text())
        if (call_dir / "finish.json").exists()
        else {}
    )
    texts = _osa(_AS_TEXTS, opts.app).splitlines() if app_running(opts.app) else []
    display = texts[-1] if texts else None
    steps = [
        {
            "seq": c["seq"],
            "tool": c["tool"],
            "args_sha256": c["args_sha256"][:16],
            "t_rel_s": round(c["t_wall"] - t0_wall, 2),
            "dur_ms": c["dur_ms"],
            "ok": c.get("ok"),
            **({"recorded_status": c["recorded_status"]} if "recorded_status" in c else {}),
            **({"img_kb": round(c["img_bytes"] / 1024, 1)} if "img_bytes" in c else {}),
        }
        for c in calls
    ]
    last_finish = (finish.get("finishes") or [{}])[-1]
    structured = out.get("structured_output") or {}
    passed = (
        structured.get("status") == "done"
        and last_finish.get("recorded_status") == "done"
        and display == opts.expect
        and wall_s <= THRESHOLD_S
    )
    usage = out.get("usage") or {}
    rec = {
        "run": i,
        "call_id": call_id,
        "exit_code": proc.returncode,
        "stderr_tail": proc.stderr.strip()[-300:] if proc.returncode else "",
        "wall_s": round(wall_s, 2),
        "pass": passed,
        "claude": {
            k: out.get(k)
            for k in (
                "is_error",
                "subtype",
                "terminal_reason",
                "num_turns",
                "duration_ms",
                "duration_api_ms",
                "time_to_request_ms",
                "ttft_ms",
                "total_cost_usd",
            )
        },
        "usage": {
            k: usage.get(k)
            for k in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            )
        },
        "models": sorted((out.get("modelUsage") or {}).keys()),
        "structured_status": structured.get("status"),
        "structured_evidence_matches_broker": structured.get("evidence_obs_id")
        == last_finish.get("evidence_obs_id"),
        "broker_finishes": [
            {
                k: f.get(k)
                for k in (
                    "requested_status",
                    "recorded_status",
                    "reason",
                    "evidence_seq",
                    "last_action_seq",
                )
            }
            for f in finish.get("finishes", [])
        ],
        "startup_to_first_tool_s": steps[0]["t_rel_s"] if steps else None,
        "finish_at_s": next((s["t_rel_s"] for s in reversed(steps) if s["tool"] == "finish"), None),
        "steps": steps,
        "independent_display": display,
        "independent_display_ok": display == opts.expect,
    }
    return rec


def evidence_check(out: Path) -> int:
    """Drive the broker in-process to show finish()'s evidence-ordering rule (no screen change)."""
    import asyncio  # pylint: disable=import-outside-toplevel
    import tempfile  # pylint: disable=import-outside-toplevel

    from mcp import Client  # pylint: disable=import-outside-toplevel

    def _text(res: Any) -> str:
        return next(c.text for c in res.content if getattr(c, "type", "") == "text")

    async def _go(call_dir: Path) -> list[dict[str, Any]]:
        state = broker_proto.BrokerState(call_dir, 400, None)
        cases: list[dict[str, Any]] = []
        async with Client(broker_proto.build_server(state)) as client:
            obs1 = json.loads(_text(await client.call_tool("observe_screen", {})))["obs_id"]
            r = json.loads(
                _text(
                    await client.call_tool(
                        "finish", {"status": "done", "summary": "s", "evidence_obs_id": obs1}
                    )
                )
            )
            cases.append(
                {
                    "case": "observation after last action",
                    "expect": "done",
                    "got": r["recorded_status"],
                }
            )
            # An action that fails at argument parsing still counts as an action
            # (safe, no keystroke).
            await client.call_tool("key", {"combo": "hyper+x", "app": ""})
            r = json.loads(
                _text(
                    await client.call_tool(
                        "finish", {"status": "done", "summary": "s", "evidence_obs_id": obs1}
                    )
                )
            )
            cases.append(
                {
                    "case": "observation older than last action",
                    "expect": "unknown_partial",
                    "got": r["recorded_status"],
                }
            )
            r = json.loads(
                _text(
                    await client.call_tool(
                        "finish",
                        {"status": "done", "summary": "s", "evidence_obs_id": "obs-999-dead"},
                    )
                )
            )
            cases.append(
                {"case": "unknown obs_id", "expect": "unknown_partial", "got": r["recorded_status"]}
            )
            r = json.loads(
                _text(await client.call_tool("finish", {"status": "done", "summary": "s"}))
            )
            cases.append(
                {"case": "no evidence", "expect": "unknown_partial", "got": r["recorded_status"]}
            )
            obs2 = json.loads(_text(await client.call_tool("observe_screen", {})))["obs_id"]
            r = json.loads(
                _text(
                    await client.call_tool(
                        "finish", {"status": "done", "summary": "s", "evidence_obs_id": obs2}
                    )
                )
            )
            cases.append(
                {"case": "re-observe then finish", "expect": "done", "got": r["recorded_status"]}
            )
        mode = oct((call_dir / "broker_calls.jsonl").stat().st_mode & 0o777)
        cases.append({"case": "call log mode", "expect": "0o600", "got": mode})
        return cases

    with tempfile.TemporaryDirectory() as td:
        cases = asyncio.run(_go(Path(td)))
    for c in cases:
        c["ok"] = c["expect"] == c["got"]
        print(f"{'✅' if c['ok'] else '❌'} {c['case']}: {c['got']}")
    data: dict[str, Any] = {
        "spike": "S-OPERATOR",
        "plan": "PLAN_claude-bridge.md 7.4 / tp#855",
        "batches": [],
    }
    if out.exists():
        data = json.loads(out.read_text())
    data["evidence_ordering_check"] = cases
    out.write_text(json.dumps(data, indent=2) + "\n")
    return 0 if all(c["ok"] for c in cases) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spike S-OPERATOR: exact 7.4 operator argv + prototype broker, timed.",
        epilog=(
            "Examples:\n"
            "  screen_lock.py -- .venv/bin/python scripts/spikes/s_operator.py -r 3\n"
            "  s_operator.py -r 1 -M 600 -l small-shots -A\n"
            "  s_operator.py -r 1 -k /tmp/evidence -n   # keep screenshots, leave the app open"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-r", "--runs", type=int, default=3, help="number of runs (default 3)")
    ap.add_argument(
        "-q", "--request", default="open Calculator and compute 2+2", help="operator request"
    )
    ap.add_argument("-x", "--expect", default="4", help="expected final display text (default 4)")
    ap.add_argument("-a", "--app", default="Calculator", help="app to verify and quit afterwards")
    ap.add_argument("-m", "--model", default="sonnet", help="claude --model (default sonnet)")
    ap.add_argument(
        "-M", "--max-px", type=int, default=800, help="screenshot max edge px (default 800)"
    )
    ap.add_argument(
        "-t", "--max-turns", type=int, default=30, help="claude --max-turns (default 30)"
    )
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help="results JSON path")
    ap.add_argument("-l", "--label", default="exact-7.4", help="label for this batch of runs")
    ap.add_argument(
        "-A", "--append", action="store_true", help="append the batch to an existing results file"
    )
    ap.add_argument(
        "-N", "--no-persist", action="store_true", help="variant: add --no-session-persistence"
    )
    ap.add_argument(
        "-k", "--keep-dir", type=Path, default=None, help="debug: keep screenshots in this dir"
    )
    ap.add_argument("-n", "--no-quit", action="store_true", help="leave the app running afterwards")
    ap.add_argument(
        "-e",
        "--evidence-check",
        action="store_true",
        help="only run the in-process finish() ordering check",
    )
    opts = ap.parse_args(argv)
    if opts.evidence_check:
        return evidence_check(opts.out)

    was_running = app_running(opts.app)
    if opts.keep_dir is not None:
        opts.keep_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    runs = []
    for i in range(1, opts.runs + 1):
        if not was_running and app_running(opts.app):
            quit_app(opts.app)
        rec = one_run(i, opts)
        runs.append(rec)
        mark = "✅" if rec["pass"] else "❌"
        c = rec["claude"]
        print(
            f"{mark} run {i}: wall {rec['wall_s']} s, turns {c['num_turns']}, "
            f"cost ${c['total_cost_usd']}, structured {rec['structured_status']}, "
            f"display {rec['independent_display']!r}, "
            f"broker {[f['recorded_status'] for f in rec['broker_finishes']]}"
        )
        if app_running(opts.app) and opts.app == "Calculator":
            _osa(_AS_POISON, opts.app)
    if not was_running and not opts.no_quit and app_running(opts.app):
        quit_app(opts.app)

    walls = [r["wall_s"] for r in runs]
    batch = {
        "label": opts.label,
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "claude_version": subprocess.run(
            [str(CLAUDE), "--version"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "request_sha256": hashlib.sha256(opts.request.encode()).hexdigest(),
        "request": opts.request,
        "model": opts.model,
        "max_px": opts.max_px,
        "no_session_persistence": opts.no_persist,
        "threshold_s": THRESHOLD_S,
        "app_was_running": was_running,
        "summary": {
            "passed": sum(r["pass"] for r in runs),
            "runs": len(runs),
            "wall_s_min": min(walls),
            "wall_s_max": max(walls),
            "wall_s_mean": round(sum(walls) / len(walls), 2),
            "first_run_wall_s": walls[0],
            "cost_usd_total": round(sum((r["claude"]["total_cost_usd"] or 0) for r in runs), 4),
        },
        "runs": runs,
    }
    data: dict[str, Any] = {
        "spike": "S-OPERATOR",
        "plan": "PLAN_claude-bridge.md 7.4 / tp#855",
        "batches": [],
    }
    if opts.append and opts.out.exists():
        data = json.loads(opts.out.read_text())
    data["batches"].append(batch)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"results → {opts.out}")
    return 0 if batch["summary"]["passed"] == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
