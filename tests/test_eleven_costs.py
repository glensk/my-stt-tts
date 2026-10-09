"""Tests for the ElevenLabs cost report formatter (no network)."""

from __future__ import annotations

from typing import Any

from my_stt_tts.eleven_costs import format_report

CONV: dict[str, Any] = {
    "conversation_id": "conv_x",
    "metadata": {
        "start_time_unix_secs": 0,
        "call_duration_secs": 87,
        "cost": 579,
        "cost_fiat": 0.0575,
        "charging": {
            "call_charge": 474,
            "llm_charge": 105,
            "platform_price": 0.047,
            "llm_price": 0.0105,
            "tts_usage": {"primary_tts_model": "eleven_v4_turbo", "total_audio_output_seconds": 39},
            "asr_usage": {"asr_model": "scribe_realtime"},
        },
    },
    "transcript": [
        {"role": "user", "message": "Hi", "time_in_call_secs": 1},
        {
            "role": "agent",
            "message": "Hello there.",
            "time_in_call_secs": 2,
            "llm_usage": {
                "model_usage": {
                    "claude-sonnet-5-5": {
                        "input": {"tokens": 10, "price": 0.0003},
                        "output_total": {"tokens": 5, "price": 0.0002},
                    }
                }
            },
        },
        {"role": "agent", "message": "No LLM turn", "time_in_call_secs": 3},
    ],
}


def test_report_prices_each_llm_sentence_and_totals() -> None:
    conv = CONV | {
        "metadata": CONV["metadata"] | {"termination_reason": "Client disconnected: 1006"}
    }
    lines = format_report(conv, {"character_count": 3347, "character_limit": 10000, "tier": "free"})
    sentence = [line for line in lines if "Hello there" in line]
    assert len(sentence) == 1 and "$0.0005  Hello there." in sentence[0]
    assert not any("No LLM turn" in line for line in lines)  # nothing to price
    summary = lines[-1]
    assert summary.startswith("🧾 call 87 s  $0.0575 (voice $0.0470 + LLM $0.0105 = 579 cr)")
    assert "ended: connection dropped (1006)" in summary
    assert "month so far: 3347/10000 credits (free plan)" in summary
    assert len(lines) == 2  # one line per priced sentence + ONE summary line


def test_report_without_subscription_has_no_period_line() -> None:
    assert not any("month so far" in line for line in format_report(CONV, None))


def test_tool_only_turn_names_the_tool() -> None:
    conv: dict[str, Any] = {
        "metadata": {},
        "transcript": [
            {
                "role": "agent",
                "message": "",
                "tool_calls": [{"tool_name": "end_call"}],
                "llm_usage": {"model_usage": {"m": {"output_total": {"price": 0.001}}}},
            }
        ],
    }
    assert any("$0.0010  [end_call]" in line for line in format_report(conv))
