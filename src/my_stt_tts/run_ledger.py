"""Best-effort recording of physical LLM attempts in the ccc run ledger.

The voice assistant must never fail because observability is unavailable.  This
module therefore keeps the recorder deliberately small: callers construct one
contract-compliant row per physical attempt, and :func:`record_run` pipes it to
``ccc record-run -q -`` with a short timeout.  Missing/refusing ``ccc`` produces
one warning per process and does not alter the LLM call's result.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("my_stt_tts.run_ledger")

RunRow = dict[str, object]
RunRecorder = Callable[[RunRow], None]

_warning_lock = threading.Lock()
_warning_emitted = threading.Event()


def _warn_once(message: str) -> None:
    """Emit at most one recorder warning, even across concurrent voice turns."""
    with _warning_lock:
        if _warning_emitted.is_set():
            return
        _warning_emitted.set()
    log.warning("LLM run ledger unavailable: %s", message)


def record_run(row: RunRow) -> None:
    """Write ``row`` to ``ccc`` without ever raising into the calling job."""
    try:
        ccc = shutil.which("ccc")
        if not ccc:
            _warn_once("`ccc` binary not found on PATH")
            return
        proc = subprocess.run(
            [ccc, "record-run", "-q", "-"],
            input=f"{json.dumps(row, ensure_ascii=False)}\n",
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:300]
            _warn_once(detail)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        _warn_once(str(exc))


def emit_run(recorder: RunRecorder, row: RunRow) -> None:
    """Invoke an injected recorder defensively so instrumentation stays harmless."""
    try:
        recorder(row)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        _warn_once(str(exc))


def json_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object, returning ``None`` for empty, malformed, or non-object data."""
    if not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def claude_seat(env: Mapping[str, str] | None = None) -> str:
    """Map Claude's selected config directory to its ledger seat label."""
    values = os.environ if env is None else env
    config_dir = values.get("CLAUDE_CONFIG_DIR", "").strip()
    if not config_dir:
        return "private"
    name = Path(config_dir).name.lower()
    if name in {".claude", "claude"}:
        return "private"
    if "work" in name:
        return "work"
    return "default"


def endpoint_seat(base_url: str | None, *, default: str = "default") -> str:
    """Return a short, stable seat label for an API endpoint or local CLI config."""
    if not base_url:
        return default
    parsed = urlparse(base_url if "://" in base_url else f"//{base_url}")
    label = parsed.hostname or parsed.path.split("/", 1)[0]
    return _short_label(label) or default


def codex_seat(env: Mapping[str, str] | None = None) -> str:
    """Derive a short label for the Codex CLI account/config in use."""
    values = os.environ if env is None else env
    home = values.get("CODEX_HOME", "").strip()
    return _short_label(Path(home).name) if home else "default"


def _short_label(value: str) -> str:
    clean = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return clean[:48]


def usage_fields(response: Any) -> RunRow:
    """Extract ledger token/model fields from Claude, Anthropic, or OpenAI shapes."""
    try:
        return _usage_fields(response)
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        _warn_once(f"could not read provider usage: {exc}")
        return {}


def _usage_fields(response: Any) -> RunRow:
    if response is None:
        return {}
    model = _value(response, "model")
    usage = _value(response, "usage")
    result: RunRow = {}
    if isinstance(model, str) and model:
        result["model"] = model

    model_usage = _value(response, "modelUsage", "model_usage")
    if isinstance(model_usage, Mapping) and model_usage:
        models = [str(name) for name in model_usage if name]
        if models:
            result["model"] = models[0] if len(models) == 1 else ",".join(models)
        if usage is None:
            usage = _sum_model_usage(model_usage.values())

    if usage is None:
        return result
    token_names = {
        "tokens_in": ("input_tokens", "inputTokens", "prompt_tokens"),
        "tokens_out": ("output_tokens", "outputTokens", "completion_tokens"),
        "tokens_cache_read": (
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cached_tokens",
        ),
        "tokens_cache_create": (
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
        ),
    }
    for ledger_name, source_names in token_names.items():
        value = _integer(usage, *source_names)
        if value is not None:
            result[ledger_name] = value

    # OpenAI nests cached prompt tokens under prompt_tokens_details.
    if "tokens_cache_read" not in result:
        prompt_details = _value(usage, "prompt_tokens_details", "input_token_details")
        cached = _integer(prompt_details, "cached_tokens")
        if cached is not None:
            result["tokens_cache_read"] = cached
    return result


def _sum_model_usage(usages: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for name in (
            "inputTokens",
            "outputTokens",
            "cacheReadInputTokens",
            "cacheCreationInputTokens",
        ):
            value = _integer(usage, name)
            if value is not None:
                totals[name] = totals.get(name, 0) + value
    return totals


def _value(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _integer(obj: Any, *names: str) -> int | None:
    value = _value(obj, *names)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    return None
