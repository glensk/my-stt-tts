#!/usr/bin/env python3
"""Spike S-PICKER (PLAN_claude-bridge.md §6 ``answer``, §9 Phase 0): can a pending
AskUserQuestion picker in a Claude Code tab be answered with raw keys sent over the iTerm2
Python API, without focusing the tab?

ATTENDED ONLY — it types into the given iTerm session. Use a scratch session from
``scratch_setup.sh``, never a real one.

Steps, in this order, each optional:

1. ``-p``: send (via ccc ``send_text_via``) a prompt that makes the session call
   AskUserQuestion (``-q single|multi|both``), then wait until the transcript holds that
   call with no tool_result yet, and print its questions.
2. ``-c``: dump the session's current screen (iTerm ``async_get_screen_contents``) — to
   watch the picker state between key batches.
3. ``-k``: send a key sequence with :func:`bridge_spike_lib.send_keys` (Python API only,
   no AppleScript, no focus change). Refused unless an AskUserQuestion is pending (``-F``
   overrides). Then wait ``-W`` s (10) for the call's tool_result and print
   ``toolUseResult.answers``; with ``-e`` compare against the expected answers.

Key tokens (comma-separated): up down left right enter space tab btab esc, a digit
0-9, ``text:<str>`` (``\\,`` for a literal comma), ``wait:<s>``.

``-e`` takes JSON mapping question text OR index ("0", "1") to a label (single-select)
or a list of labels (multi-select, order ignored). Every run is appended to
``tests/fixtures/spikes/s_picker.json`` (questions, keys, answers, verdict, screen line
counts; no screen text).

Examples:
    scripts/spikes/s_picker.py -s UUID -N scratch-idle -p -q single -c
    scripts/spikes/s_picker.py -s UUID -N scratch-idle -c                 # screen only
    scripts/spikes/s_picker.py -s UUID -N scratch-idle -k "down,enter" -e '{"0": "green"}'
    scripts/spikes/s_picker.py -s UUID -N scratch-idle -k "1,3,down,down,down,down,enter" \\
        -e '{"0": ["apple", "plum"]}' -c
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bridge_spike_lib as lib  # same directory: on sys.path[0] when run as a script

PROMPTS = {
    "single": (
        "Use the AskUserQuestion tool to ask me exactly one single-select question: which "
        "colour I prefer, with the options red, green, blue. Ask nothing else and do not "
        "call any other tool; after my answer reply with one short sentence."
    ),
    "multi": (
        "Use the AskUserQuestion tool to ask me exactly one multi-select question: which "
        "fruits I like, with the options apple, pear, plum. Ask nothing else and do not "
        "call any other tool; after my answer reply with one short sentence."
    ),
    "both": (
        "Use the AskUserQuestion tool to ask me which colour I prefer with options red, "
        "green, blue, and in the same call a multi-select question which fruits I like "
        "with options apple, pear, plum. Do not call any other tool; after my answers "
        "reply with one short sentence."
    ),
}


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s_picker.py",
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0],
        epilog="Examples:" + (__doc__ or "").rsplit("Examples:", maxsplit=1)[-1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-s", "--iterm-session", required=True, help="iTerm session id (w0t0p0:UUID)")
    p.add_argument("-S", "--session-id", help="Claude session id")
    p.add_argument("-N", "--name", default="scratch-idle", help="Claude session name")
    p.add_argument("-t", "--transcript", type=Path, help="transcript path (else resolved)")
    p.add_argument("-A", "--account", choices=tuple(lib.ACCOUNTS), default="cpriv")
    p.add_argument("-p", "--prompt", action="store_true", help="send the AskUserQuestion prompt")
    p.add_argument("-q", "--questions", choices=tuple(PROMPTS), default="both")
    p.add_argument("-P", "--prompt-text", help="custom prompt instead of -q's")
    p.add_argument("-w", "--wait", type=float, default=120.0, help="max wait for the picker (120)")
    p.add_argument("-c", "--capture", action="store_true", help="dump the screen contents")
    p.add_argument("-k", "--keys", help="key sequence, e.g. 'down,enter'")
    p.add_argument("-d", "--delay", type=float, default=0.15, help="pause between keys, s")
    p.add_argument("-W", "--result-timeout", type=float, default=10.0, help="tool_result wait")
    p.add_argument("-e", "--expect", help="expected answers as JSON")
    p.add_argument("-F", "--force", action="store_true", help="skip tab + pending checks")
    p.add_argument("-n", "--dry-run", action="store_true", help="resolve + validate, send nothing")
    return p.parse_args(argv)


def _ask_calls(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """AskUserQuestion tool_use blocks (file order) and tool_result records by id."""
    calls: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for rec in lib.all_records(path):
        msg = rec.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list) or rec.get("isSidechain"):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if (
                rec.get("type") == "assistant"
                and block.get("type") == "tool_use"
                and block.get("name") == "AskUserQuestion"
            ):
                calls.append(block)
            elif rec.get("type") == "user" and block.get("type") == "tool_result":
                results[str(block.get("tool_use_id"))] = {"record": rec, "block": block}
    return calls, results


def pending_call(path: Path) -> dict[str, Any] | None:
    """The newest AskUserQuestion call that has no tool_result yet."""
    calls, results = _ask_calls(path)
    for call in reversed(calls):
        if str(call.get("id")) not in results:
            return call
    return None


def result_for(path: Path, tool_use_id: str) -> dict[str, Any] | None:
    """``{record, block}`` of the tool_result answering *tool_use_id*, or None."""
    return _ask_calls(path)[1].get(tool_use_id)


def describe(call: dict[str, Any]) -> list[dict[str, Any]]:
    """The call's questions as ``[{index, text, header, multi_select, options}]``."""
    qs = (call.get("input") or {}).get("questions") or []
    return [
        {
            "index": i,
            "text": q.get("question"),
            "header": q.get("header"),
            "multi_select": bool(q.get("multiSelect")),
            "options": [o.get("label") for o in q.get("options") or [] if isinstance(o, dict)],
        }
        for i, q in enumerate(qs)
        if isinstance(q, dict)
    ]


def compare(
    questions: list[dict[str, Any]], answers: dict[str, Any], expect: dict[str, Any]
) -> tuple[bool, list[dict[str, Any]]]:
    """``(all_match, per-question rows)``; list expectations compare as label sets."""
    rows: list[dict[str, Any]] = []
    ok = True
    for key, want in expect.items():
        q = next((x for x in questions if str(x["index"]) == key or x["text"] == key), None)
        if q is None:
            rows.append({"question": key, "error": "no such question"})
            ok = False
            continue
        got = answers.get(q["text"])
        if isinstance(want, list):
            hit = got is not None and set(str(got).split(", ")) == {str(w) for w in want}
        else:
            hit = got == want
        ok = ok and hit
        rows.append({"question": q["text"], "want": want, "got": got, "match": hit})
    return ok, rows


def capture(iterm_session: str, run: dict[str, Any], label: str) -> None:
    """Print the screen; record only its line count in *run*."""
    lines = lib.screen_lines(iterm_session)
    while lines and not lines[-1].strip():
        lines.pop()
    print(f"----- screen ({label}, {len(lines)} lines) -----")
    for line in lines:
        print(line)
    print("----- end screen -----")
    run.setdefault("screens", []).append({"label": label, "lines": len(lines)})


@dataclass
class _Run:
    """What the steps share: the transcript, the parsed ``-k`` / ``-e`` and the run record."""

    transcript: Path
    keys: list[tuple[str, str]]
    expect: dict[str, Any] | None
    record: dict[str, Any]


def _resolve(
    args: argparse.Namespace, keys: list[tuple[str, str]], expect: dict[str, Any] | None
) -> tuple[lib.ClaudeSession, _Run] | None:
    """Session + the run context, after the tab-tty validation; None (reported) on failure."""
    session = lib.find_session(session_id=args.session_id, name=args.name, account=args.account)
    if session is None:
        lib.step(False, f"no unique interactive session {args.session_id or args.name!r}")
        return None
    lib.step(True, f"session {session.name} {session.session_id} [{session.status}]")
    transcript = args.transcript or (
        lib.transcript_for(session.session_id, args.account)
        or lib.expected_transcript(session, args.account)
    )
    ok, why = lib.validate_tab(args.iterm_session, session)
    lib.step(ok or args.force, f"tab validation: {why}")
    if not ok and not args.force:
        return None
    record: dict[str, Any] = {"spike": "S-PICKER", "at": lib.now_iso(), "keys": args.keys}
    return session, _Run(transcript=transcript, keys=keys, expect=expect, record=record)


def _send_prompt(args: argparse.Namespace, session: lib.ClaudeSession, transcript: Path) -> bool:
    """``-p``: send the AskUserQuestion prompt and wait for the new pending call."""
    if args.dry_run:
        lib.step(True, f"dry run: would send the {args.questions!r} prompt")
        return True
    if session.status != "idle":
        lib.step(False, f"prompt needs an idle session, got {session.status}")
        return False
    prev = pending_call(transcript)
    before = prev.get("id") if prev else None
    text = args.prompt_text or PROMPTS[args.questions]
    channel = lib.send_text(args.iterm_session, text)
    lib.step(bool(channel), f"prompt sent via {channel or 'nothing'}")
    if not channel:
        return False
    t0 = time.monotonic()
    call = lib.wait_until(
        lambda: (c := pending_call(transcript)) and c.get("id") != before and c,
        args.wait,
        0.5,
    )
    lib.step(bool(call), f"pending AskUserQuestion after {time.monotonic() - t0:.1f} s")
    if not call:
        return False
    time.sleep(1.0)  # let the picker render
    return True


def _show_pending(ctx: _Run) -> dict[str, Any] | None:
    """Record + print the pending AskUserQuestion call (if any); returns it."""
    call = pending_call(ctx.transcript)
    questions = describe(call) if call else []
    ctx.record["tool_use_id"] = call.get("id") if call else None
    ctx.record["questions"] = questions
    if call:
        lib.step(True, f"pending AskUserQuestion {call.get('id')}")
        for q in questions:
            kind = "multi" if q["multi_select"] else "single"
            opts = ", ".join(f"{i + 1}={o}" for i, o in enumerate(q["options"]))
            print(f"   Q{q['index']} [{kind}] {q['text']} -> {opts}")
    else:
        lib.step(not ctx.keys, "no pending AskUserQuestion")
    return call


def _record_answers(ctx: _Run, res: dict[str, Any], t0: float) -> None:
    """Store the tool_result's answers in the run record and compare them with ``-e``."""
    run = ctx.record
    tur = res["record"].get("toolUseResult")
    answers = tur.get("answers") if isinstance(tur, dict) else None
    run["latency_s"] = round(time.monotonic() - t0, 2)
    run["is_error"] = bool(res["block"].get("is_error"))
    run["answers"] = answers
    run["tool_use_result_keys"] = sorted(tur) if isinstance(tur, dict) else type(tur).__name__
    lib.step(not run["is_error"], f"tool_result after {run['latency_s']} s")
    print("   toolUseResult.answers =", json.dumps(answers, ensure_ascii=False))
    if ctx.expect is not None:
        match, rows = compare(run["questions"], answers or {}, ctx.expect)
        run["expect"], run["compare"], run["match"] = ctx.expect, rows, match
        lib.step(match, f"answers {'match' if match else 'do NOT match'} -e")


def _answer(args: argparse.Namespace, ctx: _Run, call: dict[str, Any] | None) -> int | None:
    """``-k``: send the keys and wait for the answer; an exit code ends ``main`` early."""
    if not call and not args.force:
        lib.step(False, "refusing to send keys: no picker is pending (-F overrides)")
        return 1
    if args.dry_run:
        lib.step(True, f"dry run: would send {[k for k, _ in ctx.keys]}")
        return 0
    t0 = time.monotonic()
    sent = lib.send_keys(args.iterm_session, ctx.keys, args.delay)
    lib.step(True, f"sent {sent} keys via the iTerm2 Python API (no focus change)")
    res = None
    if call:
        res = lib.wait_until(
            lambda: result_for(ctx.transcript, str(call.get("id"))), args.result_timeout, 0.2
        )
    if res:
        _record_answers(ctx, res, t0)
    else:
        ctx.record["answers"] = None
        lib.step(False, f"no tool_result within {args.result_timeout} s (picker still open?)")
        if args.capture:
            capture(args.iterm_session, ctx.record, "after keys")
    return None


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        keys = lib.parse_keys(args.keys) if args.keys else []
        expect = json.loads(args.expect) if args.expect else None
    except (ValueError, json.JSONDecodeError) as exc:
        lib.step(False, f"bad -k/-e: {exc}")
        return 2
    lib.ensure_ccc_venv()
    target = _resolve(args, keys, expect)
    if target is None:
        return 1
    session, ctx = target
    run = ctx.record

    if args.prompt and not _send_prompt(args, session, ctx.transcript):
        return 1

    call = _show_pending(ctx)

    if args.capture:
        capture(args.iterm_session, run, "before keys" if keys else "now")

    if keys:
        early = _answer(args, ctx, call)
        if early is not None:
            return early
    if not args.dry_run and (keys or args.prompt):
        path = lib.save_fixture("s_picker.json", run, append=True)
        lib.step(True, f"appended run to {path}")
    return 0 if (not keys or run.get("answers") is not None) and run.get("match", True) else 1


if __name__ == "__main__":
    sys.exit(main())
