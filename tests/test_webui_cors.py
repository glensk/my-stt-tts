"""CORS / cross-origin behaviour of the WebUI HTTP server.

The hosted GitHub Pages ``gui.html`` page can be pointed at a user-run server with
``?backend=https://<host>``. For the browser to permit those cross-origin calls the
server must answer ``/api/*`` and ``/events`` with permissive CORS headers and reply
to the ``OPTIONS`` preflight. These tests start a real :class:`WebUI` on an ephemeral
port and assert exactly that — no network beyond loopback, no devices, no API keys.
"""

from __future__ import annotations

import http.client
import threading
from collections.abc import Iterator

import pytest

from my_stt_tts.config import Config
from my_stt_tts.webui import WebUI


@pytest.fixture()
def served_ui() -> Iterator[tuple[WebUI, int]]:
    """A running WebUI on 127.0.0.1:<ephemeral>, torn down after the test."""
    cfg = Config(sample_rate=16000)
    ui = WebUI(cfg, on_turn=lambda _t: None, on_action=lambda _n, _d: None, port=0)
    port = ui._server.server_address[1]  # type: ignore[attr-defined]  # noqa: SLF001
    thread = threading.Thread(target=ui.serve_forever, daemon=True)
    thread.start()
    try:
        yield ui, port
    finally:
        ui._server.shutdown()  # type: ignore[attr-defined]  # noqa: SLF001
        ui._server.server_close()  # type: ignore[attr-defined]  # noqa: SLF001


def test_api_settings_get_emits_cors_headers(served_ui: tuple[WebUI, int]) -> None:
    """A GET on /api/settings carries Access-Control-Allow-Origin: * for cross-origin."""
    _ui, port = served_ui
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/api/settings", headers={"Origin": "https://glensk.github.io"})
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        assert "application/json" in (resp.getheader("Content-Type") or "")
        assert b"brain_presets" in body  # the settings payload actually came back
    finally:
        conn.close()


def test_options_preflight_returns_204_with_cors(served_ui: tuple[WebUI, int]) -> None:
    """An OPTIONS preflight on /api/turn returns 204 + the CORS allow headers."""
    _ui, port = served_ui
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request(
            "OPTIONS",
            "/api/turn",
            headers={
                "Origin": "https://glensk.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 204
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        allow_methods = (resp.getheader("Access-Control-Allow-Methods") or "").upper()
        assert "POST" in allow_methods
        assert "OPTIONS" in allow_methods
        assert "content-type" in (resp.getheader("Access-Control-Allow-Headers") or "").lower()
    finally:
        conn.close()


def test_events_stream_carries_cors_and_tolerates_token_query(
    served_ui: tuple[WebUI, int],
) -> None:
    """/events sends CORS headers, and a client-supplied ?token=… still routes (no 404)."""
    _ui, port = served_ui
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        # A ?token=… is appended by the page when the user passes &token=; the bundled
        # server ignores it but must still match the route rather than 404.
        conn.request("GET", "/events?token=secret", headers={"Origin": "https://glensk.github.io"})
        resp = conn.getresponse()
        assert resp.status == 200
        assert "text/event-stream" in (resp.getheader("Content-Type") or "")
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
    finally:
        # The SSE stream is open-ended; close the socket without reading the body.
        conn.close()
