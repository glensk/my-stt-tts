"""Tests for the ElevenLabs cost report formatter (no network)."""

from __future__ import annotations

from my_stt_tts.eleven_costs import format_report

CONV = {
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
    lines = format_report(CONV, {"character_count": 3347, "character_limit": 10000, "tier": "free"})
    sentence = [line for line in lines if "Hello there" in line]
    assert len(sentence) == 1 and "claude-sonnet-5-5 $0.00050" in sentence[0]
    assert not any("No LLM turn" in line for line in lines)  # nothing to price
    total = next(line for line in lines if line.strip().startswith("total"))
    assert "579 credits = $0.0575" in total and "voice 474 cr" in total and "LLM 105 cr" in total
    assert "eleven_v4_turbo" in total and "scribe_realtime" in total
    period = lines[-1]
    assert "3347 / 10000 credits used (free tier)" in period and "≈ $0.33" in period


def test_report_without_subscription_has_no_period_line() -> None:
    assert not any("billing period" in line for line in format_report(CONV, None))


def test_tool_only_turn_names_the_tool() -> None:
    conv = {
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
    assert any("[end_call] — m $0.00100" in line for line in format_report(conv))
