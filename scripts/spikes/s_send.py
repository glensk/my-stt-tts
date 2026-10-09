#!/usr/bin/env python3
"""Spike S-SEND (PLAN_claude-bridge.md §6 ``send``, §9 Phase 0): does a message typed into a
Claude Code tab by ccc's ``send_text_via`` land as the expected transcript record?

ATTENDED ONLY — it types into the given iTerm session. Use the scratch sessions from
``scratch_setup.sh``, never a real one.

Flow: resolve the Claude session (``-N`` name or ``-S`` id → pid, status, transcript),
check that the iTerm tab ``-s`` really belongs to it (tab tty == pid tty), record the
transcript inode + byte size, send the test text, then scan ONLY the appended records for
``-T`` seconds (default 8) for a record whose normalised text sha256 matches:

* ``-m idle``: the session must be idle; expected record ``user``.
* ``-m busy``: first sends a long-running prompt (``sleep 40`` in Bash), waits until the
  session reports ``busy``, then sends the test text; expected record
  ``queue-operation/enqueue`` (``attachment/queued_command`` also counts).

Outcome ``accepted`` (matched record type + latency) or ``unknown``. A body-free record
of what was seen goes to ``tests/fixtures/spikes/s_send_<mode>.json`` (record types and
key names; the only text kept is the test text itself).

Examples:
    scripts/spikes/s_send.py -m idle -N scratch-idle -s w0t3p0:UUID
    scripts/spikes/s_send.py -m busy -N scratch-busy -s UUID -T 8
    scripts/spikes/s_send.py -m idle -S <session-uuid> -s UUID -n     # dry run, sends nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import bridge_spike_lib as lib  # same directory: on sys.path[0] when run as a script

BUSY_PROMPT = "Run `sleep 40` in bash, then say done."
EXPECT = {
    "idle": ("user",),
    "busy": ("queue-operation/enqueue", "attachment/queued_command"),
}


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s_send.py",
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        epilog=(__doc__ or "").rsplit("Examples:", maxsplit=1)[-1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-m", "--mode", choices=("idle", "busy"), required=True)
    p.add_argument("-s", "--iterm-session", required=True, help="iTerm session id (w0t0p0:UUID)")
    p.add_argument("-S", "--session-id", help="Claude session id (resolves pid + transcript)")
    p.add_argument("-N", "--name", help="Claude session name (default scratch-<mode>)")
    p.add_argument("-t", "--transcript", type=Path, help="transcript path (else resolved)")
    p.add_argument("-A", "--account", choices=tuple(lib.ACCOUNTS), default="cpriv")
    p.add_argument("-x", "--text", help="test text (default: a unique 's-send <mode> <ts>')")
    p.add_argument("-T", "--timeout", type=float, default=8.0, help="match window, s (8)")
    p.add_argument("-B", "--busy-wait", type=float, default=30.0, help="max wait for busy, s (30)")
    p.add_argument("-F", "--force", action="store_true", help="skip the tab-tty validation")
    p.add_argument("-n", "--dry-run", action="store_true", help="resolve + validate, send nothing")
    p.add_argument("-o", "--output", type=Path, help="fixture path (default s_send_<mode>.json)")
    return p.parse_args(argv)


def _match(
    anchor: lib.Anchor, want_sha: str, kinds: tuple[str, ...], timeout: float, t0: float
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """First appended record of *kinds* whose text sha matches; plus every shape seen."""
    end = t0 + timeout
    while True:
        seen: list[dict[str, Any]] = []
        for rec in anchor.appended():
            info = lib.shape(rec)
            got = lib.record_text(rec)
            if got is not None:
                info["prompt_kind"] = got[0]
                info["sha_match"] = lib.text_sha(got[1]) == want_sha
                if info["sha_match"] and got[0] in kinds:
                    info["latency_s"] = round(time.monotonic() - t0, 3)
                    seen.append(info)
                    return info, seen
            seen.append(info)
        if time.monotonic() >= end:
            return None, seen
        time.sleep(0.1)


def _resolve(args: argparse.Namespace) -> tuple[lib.ClaudeSession, Path] | None:
    """Session + transcript, with the tab-tty validation; None (reported) on failure."""
    name = args.name or (None if args.session_id else f"scratch-{args.mode}")
    session = lib.find_session(session_id=args.session_id, name=name, account=args.account)
    if session is None:
        lib.step(
            False, f"no unique interactive session {args.session_id or name!r} ({args.account})"
        )
        return None
    lib.step(
        True, f"session {session.name} {session.session_id} pid {session.pid} [{session.status}]"
    )
    transcript = args.transcript or (
        lib.transcript_for(session.session_id, args.account)
        or lib.expected_transcript(session, args.account)
    )
    lib.step(True, f"transcript {transcript}")
    ok, why = lib.validate_tab(args.iterm_session, session)
    lib.step(
        ok or args.force, f"tab validation: {why}" + (" (forced)" if args.force and not ok else "")
    )
    if not ok and not args.force:
        return None
    return session, transcript


def _make_busy(args: argparse.Namespace, session: lib.ClaudeSession, run: dict[str, Any]) -> bool:
    """Busy mode: send the long-running prompt and wait until the session is busy."""
    if session.status != "idle":
        lib.step(False, f"busy mode starts from idle, got {session.status}")
        return False
    channel = lib.send_text(args.iterm_session, BUSY_PROMPT)
    lib.step(bool(channel), f"long-running prompt sent via {channel or 'nothing'}")
    if not channel:
        return False
    t_busy = time.monotonic()
    status = lib.wait_until(
        lambda: lib.session_status(session.session_id, args.account) == "busy",
        args.busy_wait,
        0.5,
    )
    run["busy_after_s"] = round(time.monotonic() - t_busy, 2)
    lib.step(bool(status), f"session busy after {run['busy_after_s']} s")
    if not status:
        return False
    time.sleep(2.0)  # let the Bash tool call start, so the turn is mid-flight
    return True


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    lib.ensure_ccc_venv()
    target = _resolve(args)
    if target is None:
        return 1
    session, transcript = target
    text = args.text or f"s-send {args.mode} {lib.now_iso()} - reply with just: ok"
    if args.mode == "idle" and session.status != "idle":
        lib.step(False, f"idle mode needs status idle, got {session.status}")
        return 1
    if args.dry_run:
        lib.step(True, f"dry run: would send {text!r} (mode {args.mode}); nothing sent")
        return 0

    run: dict[str, Any] = {"spike": "S-SEND", "mode": args.mode, "at": lib.now_iso()}
    if args.mode == "busy" and not _make_busy(args, session, run):
        return 1
    return _send_and_match(args, session, transcript, text, run)


def _send_and_match(
    args: argparse.Namespace,
    session: lib.ClaudeSession,
    transcript: Path,
    text: str,
    run: dict[str, Any],
) -> int:
    """Anchor the transcript, send the test text, scan the appended records, save."""
    status_before = lib.session_status(session.session_id, args.account)
    anchor = lib.Anchor.take(transcript)
    run.update(
        status_before=status_before,
        anchor={"inode": anchor.inode, "size": anchor.size},
        test_text=text,
        expect=list(EXPECT[args.mode]),
    )
    t0 = time.monotonic()
    channel = lib.send_text(args.iterm_session, text)
    run["send_s"] = round(time.monotonic() - t0, 3)
    run["channel"] = channel
    lib.step(bool(channel), f"test text sent via {channel or 'nothing'} in {run['send_s']} s")
    if not channel:
        run["outcome"] = "failed"
        lib.save_fixture(f"s_send_{args.mode}.json", run)
        return 1
    hit, seen = _match(anchor, lib.text_sha(text), EXPECT[args.mode], args.timeout, t0)
    run["appended"] = seen
    run["outcome"] = "accepted" if hit else "unknown"
    if hit:
        run["matched"] = {"prompt_kind": hit["prompt_kind"], "latency_s": hit["latency_s"]}
        lib.step(True, f"accepted: {hit['prompt_kind']} after {hit['latency_s']} s")
    else:
        kinds = sorted({str(s.get("prompt_kind") or s.get("type")) for s in seen})
        lib.step(False, f"unknown: no matching record within {args.timeout} s (saw {kinds})")
    out = args.output
    path = lib.save_fixture(f"s_send_{args.mode}.json", run) if out is None else _write(out, run)
    lib.step(True, f"saved {path}")
    return 0 if hit else 1


def _write(path: Path, run: dict[str, Any]) -> Path:
    import json  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    sys.exit(main())
