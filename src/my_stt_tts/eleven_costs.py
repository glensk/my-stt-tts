"""Models and costs of ElevenLabs agent conversations, for the mac-voice console log.

The realtime websocket carries no billing data, so costs are read from the REST API
after a conversation ends (ElevenLabs needs a few seconds to finalise it):

* per agent sentence: the exact LLM and its $ price (``transcript[].llm_usage``);
* per conversation: credits + $ split into voice/platform (billed per minute — not
  attributable to single sentences) and LLM, plus the TTS / ASR models used;
* per billing period: credits used so far (``/v1/user/subscription``) and their $ value
  at this conversation's credit price.

:func:`format_report` is pure (dicts in, lines out); :func:`fetch_report` does the HTTP.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

API = "https://api.elevenlabs.io"


def _get(path: str, api_key: str, timeout: float = 10.0) -> dict[str, Any]:
    req = urllib.request.Request(API + path, headers={"xi-api-key": api_key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    return data if isinstance(data, dict) else {}


def agent_models(api_key: str, agent_id: str) -> str:
    """One line naming the agent's LLM (+ reasoning effort), voice model and ASR."""
    cfg = _get(f"/v1/convai/agents/{agent_id}", api_key).get("conversation_config", {})
    prompt = cfg.get("agent", {}).get("prompt", {})
    effort = prompt.get("reasoning_effort")
    llm = prompt.get("llm", "?") + (f" (reasoning {effort})" if effort else "")
    tts = cfg.get("tts", {}).get("model_id", "?")
    asr = cfg.get("asr", {}).get("provider", "?")
    return f"🧠 models: LLM {llm} · voice {tts} · speech-to-text {asr}"


def _turn_price(turn: dict[str, Any]) -> tuple[str, float] | None:
    usage = (turn.get("llm_usage") or {}).get("model_usage") or {}
    if not usage:
        return None
    model = next(iter(usage))
    price = sum(
        float(part.get("price") or 0.0)
        for parts in usage.values()
        for part in parts.values()
        if isinstance(part, dict)
    )
    return model, price


def _clock(start_unix: float | None, offset: float | None) -> str:
    if start_unix is None:
        return "--:--:--"
    return time.strftime("%H:%M:%S", time.localtime(start_unix + float(offset or 0)))


def _ended(reason: str | None) -> str:
    """Short, human wording for ElevenLabs' ``termination_reason``."""
    text = (reason or "").strip()
    low = text.lower()
    if "1006" in low:
        return "connection dropped (1006)"
    if "end_call" in low or "end call" in low:
        return "agent hung up"
    if "1000" in low or "client disconnected" in low:
        return "closed by mac-voice"
    return text[:40] or "?"


def format_report(conv: dict[str, Any], subscription: dict[str, Any] | None = None) -> list[str]:
    """Console lines for one finished conversation (see module docstring).

    One aligned line per priced agent sentence (time · $ · text), then ONE summary line.
    """
    meta = conv.get("metadata", {})
    charging = meta.get("charging") or {}
    start = meta.get("start_time_unix_secs")
    lines: list[str] = []
    for turn in conv.get("transcript", []):
        priced = _turn_price(turn) if turn.get("role") == "agent" else None
        if priced is None:
            continue
        _model, price = priced
        text = (turn.get("message") or "").strip().replace("\n", " ")
        if not text:  # a tool-only turn, e.g. the agent hanging up
            tools = [c.get("tool_name", "?") for c in turn.get("tool_calls") or []]
            text = f"[{', '.join(tools) or 'no speech'}]"
        text = text if len(text) <= 70 else text[:69] + "…"
        lines.append(f"💶 {_clock(start, turn.get('time_in_call_secs'))}  ${price:.4f}  {text}")
    conv_credits = meta.get("cost")
    usd = float(meta.get("cost_fiat") or 0.0)
    summary = (
        f"🧾 call {meta.get('call_duration_secs', '?')} s  ${usd:.4f}"
        f" (voice ${float(charging.get('platform_price') or 0):.4f}"
        f" + LLM ${float(charging.get('llm_price') or 0):.4f} = {conv_credits} cr)"
        f" · ended: {_ended(meta.get('termination_reason'))}"
    )
    if subscription:
        used, limit = subscription.get("character_count"), subscription.get("character_limit")
        per_credit = usd / conv_credits if conv_credits else 0.0
        worth = f" ≈ ${used * per_credit:.2f}" if per_credit and used else ""
        summary += f" · period {used}/{limit} cr{worth}"
    lines.append(summary)
    return lines


def fetch_report(api_key: str, conversation_id: str, *, wait_s: float = 45.0) -> list[str]:
    """Wait until ElevenLabs has finalised the conversation, then format its costs."""
    deadline = time.monotonic() + wait_s
    conv: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            conv = _get(f"/v1/convai/conversations/{conversation_id}", api_key)
        except (OSError, urllib.error.URLError, ValueError):
            conv = {}
        if conv.get("status") in ("done", "failed") and conv.get("metadata", {}).get("cost"):
            break
        time.sleep(3.0)
    if not conv.get("metadata", {}).get("cost"):
        return [f"⚠️  costs for {conversation_id} not available yet (ElevenLabs still processing)"]
    try:
        subscription = _get("/v1/user/subscription", api_key)
    except (OSError, urllib.error.URLError, ValueError):
        subscription = {}
    return format_report(conv, subscription)
