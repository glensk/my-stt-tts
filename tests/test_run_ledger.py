"""Tests for the best-effort ccc LLM run-ledger boundary."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from my_stt_tts.run_ledger import claude_seat, record_run, usage_fields


def test_record_run_pipes_one_json_object_to_ccc():
    row = {
        "provider": "anthropic",
        "seat": "default",
        "purpose": "stt-brain",
        "outcome": "ok",
        "ok": True,
        "ms": 12,
    }
    completed = MagicMock(returncode=0, stdout="", stderr="")
    with (
        patch("my_stt_tts.run_ledger.shutil.which", return_value="/usr/bin/ccc"),
        patch("my_stt_tts.run_ledger.subprocess.run", return_value=completed) as run,
    ):
        record_run(row)

    assert run.call_count == 1
    assert run.call_args.args[0] == ["/usr/bin/ccc", "record-run", "-q", "-"]
    assert json.loads(run.call_args.kwargs["input"]) == row
    assert run.call_args.kwargs["timeout"] == 10
    assert run.call_args.kwargs["check"] is False


def test_claude_usage_and_seat_are_normalized():
    envelope = {
        "usage": {
            "input_tokens": 13,
            "output_tokens": 4,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 6,
        },
        "modelUsage": {"claude-test": {}},
    }
    assert usage_fields(envelope) == {
        "model": "claude-test",
        "tokens_in": 13,
        "tokens_out": 4,
        "tokens_cache_read": 6,
        "tokens_cache_create": 2,
    }
    assert claude_seat({}) == "private"
    assert claude_seat({"CLAUDE_CONFIG_DIR": "/tmp/.claude-work"}) == "work"
    assert claude_seat({"CLAUDE_CONFIG_DIR": "/tmp/custom"}) == "default"
