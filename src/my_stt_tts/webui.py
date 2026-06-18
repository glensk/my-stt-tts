"""Local web UI for ``--browser``: live state, transcript, settings, controls.

Dependency-light: stdlib ``http.server`` + Server-Sent Events (no extra packages).
Serves a single page (``webui.html`` next to this module), streams live events
from :data:`events.bus`, and exposes ``/api/settings``, ``/api/turn``,
``/api/action``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import (
    AEC_MODES,
    BARGE_IN_MODES,
    BRAIN_PRESETS,
    PROVIDERS,
    TURN_ANALYZERS,
    Config,
)
from .events import bus
from .tts import _VOICE_NOTES, VOICE_PRESETS

log = logging.getLogger("my_stt_tts.webui")

_HTML_PATH = Path(__file__).with_name("webui.html")
_FALLBACK_HTML = (
    "<!doctype html><meta charset=utf-8><title>my-stt-tts</title>"
    '<body style="font:16px system-ui;background:#0f1115;color:#e6e9ef;padding:2rem">'
    "<h1>my-stt-tts</h1><p>UI asset missing — rebuild. Live events at "
    "<code>/events</code>; API under <code>/api/</code>.</p></body>"
)


def settings_dict(cfg: Config) -> dict[str, Any]:
    """The settable subset of the config plus the choice lists, for the UI."""
    return {
        "provider": cfg.llm_provider,
        "model": cfg.llm_model,
        "model_deep": cfg.llm_model_deep,
        "voice_en": cfg.tts_voices.get("en"),
        "length_scale": cfg.tts_length_scale,
        "wake_phrase": cfg.wake_phrase,
        "agent_workspace": cfg.agent_workspace or "",
        "agent_model": cfg.agent_model,
        "system_prompt": cfg.system_prompt,
        "barge_in": cfg.barge_in,
        "aec_mode": cfg.aec_mode,
        "turn_analyzer": cfg.turn_analyzer,
        "stt_streaming": cfg.stt_streaming,
        "stt_window_s": cfg.stt_window_s,
        "interrupt_min_words": cfg.interrupt_min_words,
        "interrupt_predict": cfg.interrupt_predict,
        "brain_presets": sorted(BRAIN_PRESETS),
        "providers": list(PROVIDERS),
        "barge_in_modes": list(BARGE_IN_MODES),
        "aec_modes": list(AEC_MODES),
        "turn_analyzers": list(TURN_ANALYZERS),
        "voices": [
            {"name": name, "id": VOICE_PRESETS[name], "note": _VOICE_NOTES.get(name, "")}
            for name in VOICE_PRESETS
        ],
    }


def apply_settings(cfg: Config, data: dict[str, Any]) -> None:
    """Apply changed fields from the UI onto ``cfg`` (live)."""
    if data.get("brain_preset"):
        cfg.apply_brain_preset(str(data["brain_preset"]))
    if "provider" in data:
        cfg.llm_provider = str(data["provider"])
    if "model" in data:
        cfg.llm_model = str(data["model"])
    if "model_deep" in data:
        cfg.llm_model_deep = str(data["model_deep"])
    if "voice_en" in data:
        cfg.tts_voices["en"] = str(data["voice_en"])
    if "length_scale" in data:
        cfg.tts_length_scale = float(data["length_scale"])
    if "wake_phrase" in data:
        cfg.wake_phrase = str(data["wake_phrase"])
    if "agent_workspace" in data:
        cfg.agent_workspace = str(data["agent_workspace"]) or None
    if "agent_model" in data:
        cfg.agent_model = str(data["agent_model"])
    if "system_prompt" in data:
        cfg.system_prompt = str(data["system_prompt"])
    if "barge_in" in data:
        cfg.barge_in = str(data["barge_in"])
    if "aec_mode" in data:
        cfg.aec_mode = str(data["aec_mode"])
    if "turn_analyzer" in data:
        cfg.turn_analyzer = str(data["turn_analyzer"])
    if "stt_streaming" in data:
        cfg.stt_streaming = bool(data["stt_streaming"])
    if "stt_window_s" in data:
        cfg.stt_window_s = float(data["stt_window_s"])
    if "interrupt_min_words" in data:
        cfg.interrupt_min_words = int(data["interrupt_min_words"])
    if "interrupt_predict" in data:
        cfg.interrupt_predict = bool(data["interrupt_predict"])


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:  # keep the console quiet
        pass

    @property
    def _ui(self) -> WebUI:
        return self.server.ui  # type: ignore[attr-defined]

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, "application/json", json.dumps(obj).encode("utf-8"))

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", self._ui.html.encode("utf-8"))
        elif self.path == "/api/settings":
            self._json(200, settings_dict(self._ui.cfg))
        elif self.path == "/events":
            self._sse()
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        if self.path == "/api/settings":
            try:
                apply_settings(self._ui.cfg, body)
            except Exception as exc:  # bad value from the UI shouldn't 500-crash
                self._json(400, {"error": str(exc)})
                return
            self._json(200, settings_dict(self._ui.cfg))
        elif self.path == "/api/turn":
            text = str(body.get("text", "")).strip()
            if text:
                self._ui.run_turn_async(text)
            self._json(200, {"ok": True})
        elif self.path == "/api/action":
            self._ui.action(str(body.get("action", "")), body)
            self._json(200, {"ok": True})
        else:
            self._send(404, "text/plain", b"not found")

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        sub = bus.subscribe()
        try:
            while True:
                try:
                    data = sub.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # keep-alive comment
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ValueError):
            pass
        finally:
            bus.unsubscribe(sub)


class WebUI:
    """Serves the page + API; ``on_turn`` runs a typed turn, ``on_action`` controls."""

    def __init__(
        self,
        cfg: Config,
        on_turn: Callable[[str], None],
        on_action: Callable[[str, dict[str, Any]], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.cfg = cfg
        self._on_turn = on_turn
        self._on_action = on_action
        self.host = host
        self.port = port
        self.html = (
            _HTML_PATH.read_text(encoding="utf-8") if _HTML_PATH.exists() else _FALLBACK_HTML
        )
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.ui = self  # type: ignore[attr-defined]
        self._busy = threading.Lock()

    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def run_turn_async(self, text: str) -> None:
        """Run a turn in a worker thread (ignores overlapping requests)."""

        def _work() -> None:
            if not self._busy.acquire(blocking=False):
                bus.log("busy — finish the current turn first", "error")
                return
            try:
                self._on_turn(text)
            finally:
                self._busy.release()

        threading.Thread(target=_work, daemon=True).start()

    def action(self, name: str, data: dict[str, Any]) -> None:
        if self._on_action is not None:
            self._on_action(name, data)

    def serve_forever(self) -> None:
        self._server.serve_forever()
