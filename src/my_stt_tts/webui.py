"""Local web UI for ``--browser``: live state, transcript, settings, controls.

Dependency-light: stdlib ``http.server`` + Server-Sent Events (no extra packages).
Serves a single page (``webui.html`` next to this module), streams live events
from :data:`events.bus`, and exposes ``/api/settings``, ``/api/turn``,
``/api/action``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import ws_frame
from .config import (
    AEC_MODES,
    BARGE_IN_MODES,
    BRAIN_PRESETS,
    DENOISER_MODES,
    MUSIC_PLAYBACK_MODES,
    PROVIDERS,
    TRANSPORT_MODES,
    TURN_ANALYZERS,
    UNITS,
    Config,
    available_wake_words,
    model_label,
    wake_word_info,
)
from .events import bus
from .tts import _VOICE_NOTES, VOICE_PRESETS
from .ws_transport import WebSocketTransport

log = logging.getLogger("my_stt_tts.webui")

_HTML_PATH = Path(__file__).with_name("webui.html")
_FALLBACK_HTML = (
    "<!doctype html><meta charset=utf-8><title>my-stt-tts</title>"
    '<body style="font:16px system-ui;background:#0f1115;color:#e6e9ef;padding:2rem">'
    "<h1>my-stt-tts</h1><p>UI asset missing — rebuild. Live events at "
    "<code>/events</code>; API under <code>/api/</code>.</p></body>"
)


def settings_dict(
    cfg: Config,
    *,
    audio_enabled: bool = False,
    voice_available: bool = False,
    voice_hint: str = "",
) -> dict[str, Any]:
    """The settable subset of the config plus the choice lists, for the UI.

    ``audio_enabled`` tells the page whether the backend can carry real mic/TTS
    audio over the WebSocket channel (R2-5); when False the page stays state /
    transcript only and uses the demo fallback offline.

    ``voice_available`` tells the page whether the *server-side* wake / push-to-talk
    controls can actually do anything (an STT engine is loaded AND the wake model
    exists AND a mic is usable). When False the page disables those buttons and shows
    ``voice_hint`` (a short reason) instead of letting them POST and silently error.
    """
    from .audio import mic_permission_status
    from .config import is_official_wake_word

    # Is the sherpa KeywordSpotter usable at all (enabled + importable + model fetchable)?
    # Per-word detector annotation: official words are openWakeWord-ONLY ("oww"); a custom
    # word is ALSO served by KWS (zero-train open-vocab) when KWS is available, and by the
    # few-shot enrolled detector when that word has saved references — so the detector string
    # OR-joins the available branches ("oww", "oww+kws", "oww+fewshot", "oww+kws+fewshot").
    from .enrolled_wake import load_references
    from .kws import kws_available
    from .platform import host_app_name

    kws_ok = kws_available(cfg)
    fewshot_on = getattr(cfg, "fewshot_wake_enabled", True)
    word_info = wake_word_info()
    for word, info in word_info.items():
        if is_official_wake_word(word):
            info["detector"] = "oww"
            continue
        branches = ["oww"]
        if kws_ok:
            branches.append("kws")
        if fewshot_on and load_references(word) is not None:
            branches.append("fewshot")
        info["detector"] = "+".join(branches)

    return {
        "audio_enabled": audio_enabled,
        "voice_available": voice_available,
        "voice_hint": voice_hint,
        # macOS mic authorization (authorized/denied/notDetermined/restricted/unavailable) so
        # the page can show whether System Settings › Microphone is enabled for this app.
        "mic_permission": mic_permission_status(),
        # The friendly name of the app the SERVER runs in (from TERM_PROGRAM) — the
        # app whose macOS mic permission governs the server capture. The GUI labels
        # the server mic "uses <App>'s microphone permission".
        "host_app": host_app_name(),
        "provider": cfg.llm_provider,
        "model": cfg.llm_model,
        "model_deep": cfg.llm_model_deep,
        # The EXACT model + effort/size-tier label the GUI shows for the active brain
        # (e.g. ``claude-cli / opus-4.8 xlarge``) — the SAME string carried on the
        # bus.response(model=…) event, so the settings panel and the ASSISTANT
        # transcript bubble agree. Built via config.model_label (shared contract).
        "model_label": model_label(cfg.llm_provider, cfg.llm_model),
        "voice_en": cfg.tts_voices.get("en"),
        "length_scale": cfg.tts_length_scale,
        "wake_phrase": cfg.wake_phrase,
        # Wake sensitivity (openWakeWord score, 0..1) a frame must clear to fire.
        # Lower = triggers more easily / more false-positives; higher = stricter.
        "wake_threshold": cfg.wake_threshold,
        # The pre-shipped wake words present on disk, so the UI offers the real
        # choices as a dropdown (empty list -> the page falls back to free text).
        "wake_words": available_wake_words(),
        # Per-wake-word reliability metadata so the page can render a reliability BAR
        # (width = reliability) + tier colour + note, steering users to reliable
        # models. Shape (GUI contract): {"<word>": {"tier": "green"|"orange"|"red",
        # "note": str, "reliability": float 0-1, "tested": int, "measured": bool}}.
        # ``reliability`` is data-driven from THIS user's wake tests when they exist
        # (debug/wake_stats.json, server-biased), else a static prior (official 0.9,
        # self-trained 0.3); ``tier`` is derived from it. Each entry ALSO carries
        # ``detector`` ("oww" | "kws" | "oww+kws") naming which detector(s) serve the
        # word: official = "oww" only; a custom word = "oww+kws" when the sherpa
        # KeywordSpotter is available (the OR'd open-vocabulary path), else "oww".
        "wake_word_info": word_info,
        # Whether the sherpa KeywordSpotter (KWS) — the SECOND, OR'd wake detector for
        # CUSTOM / self-trained words — is usable at all (enabled + sherpa importable +
        # the GigaSpeech model present or auto-downloadable). Official words never use it.
        "kws_available": kws_ok,
        "kws_enabled": cfg.kws_enabled,
        # The few-shot ENROLLED wake detector (third, OR'd path for CUSTOM words): whether it
        # is enabled, plus its operating knobs. A custom word gains the "fewshot" detector tag
        # (in wake_word_info[word]["detector"]) once it has saved references under
        # models/wake_embeddings/. Official words never use it.
        "fewshot_wake_enabled": getattr(cfg, "fewshot_wake_enabled", True),
        "fewshot_threshold": getattr(cfg, "fewshot_threshold", 0.96),
        "fewshot_patience": getattr(cfg, "fewshot_patience", 2),
        # Software input gain applied to SERVER mic captures (mic_check / wake_test),
        # clip-protected to ±1.0. Reported back to the page as processing.gain.
        "mic_gain": cfg.mic_gain,
        # Software gain applied to each frame in the LIVE wake loop BEFORE openWakeWord
        # scores it (clip-protected). THE fix knob for a dead wake word: a quiet mic
        # starves the (un-normalized) model, collapsing the score; lifting this restores
        # it. Default 1.0 = no change. The wake gain sweep finds the right value.
        "wake_gain": cfg.wake_gain,
        "agent_workspace": cfg.agent_workspace or "",
        "agent_model": cfg.agent_model,
        "system_prompt": cfg.system_prompt,
        "location": cfg.location,
        "units": cfg.units,
        "barge_in": cfg.barge_in,
        "aec_mode": cfg.aec_mode,
        "turn_analyzer": cfg.turn_analyzer,
        "stt_streaming": cfg.stt_streaming,
        "stt_window_s": cfg.stt_window_s,
        "interrupt_min_words": cfg.interrupt_min_words,
        "interrupt_predict": cfg.interrupt_predict,
        "transport": cfg.transport,
        "stt_backend": cfg.stt_backend,
        "tts_backend": cfg.tts_backend,
        "tts_streaming": cfg.tts_streaming,
        "denoiser": cfg.denoiser,
        "tools_enabled": cfg.tools_enabled,
        # Where the GUI shows music (audio always plays server-side via mpv): the
        # page uses this to decide whether to ALSO embed the muted YouTube video when
        # it is local to the server. "server" = audio-only; "hybrid" = may embed video.
        "music_playback": cfg.music_playback,
        # Within-turn speaker diarization (G7+): split one turn that holds several
        # voices + TV into per-speaker segments, each named via the ECAPA path. Opt-in
        # and active only when speaker ID is usable AND sherpa-onnx + the models exist.
        "speaker_diarize_enabled": cfg.speaker_diarize_enabled,
        "diarize_num_speakers": cfg.diarize_num_speakers,
        "brain_mode": cfg.brain_mode,
        "realtime_keyed": bool(cfg.realtime_api_key),
        "telephony": cfg.telephony,
        "telemetry": cfg.telemetry,
        "brain_presets": sorted(BRAIN_PRESETS),
        "providers": list(PROVIDERS),
        "barge_in_modes": list(BARGE_IN_MODES),
        "aec_modes": list(AEC_MODES),
        "turn_analyzers": list(TURN_ANALYZERS),
        "transport_modes": list(TRANSPORT_MODES),
        "denoiser_modes": list(DENOISER_MODES),
        "music_playback_modes": list(MUSIC_PLAYBACK_MODES),
        "units_modes": list(UNITS),
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
        # Picking a wake word in the UI re-derives the model path
        # (wakewords/<phrase>.onnx) so the selection takes effect live.
        cfg.select_wake_word(str(data["wake_phrase"]))
    if "wake_threshold" in data:
        # Clamp to [0, 1]: the slider can't produce out-of-range values, but a
        # hand-crafted POST shouldn't push the wake detector past validate()'s bound.
        cfg.wake_threshold = max(0.0, min(1.0, float(data["wake_threshold"])))
    if "kws_enabled" in data:
        # Toggle the OR'd sherpa-KWS path for custom words (official words ignore it).
        cfg.kws_enabled = bool(data["kws_enabled"])
    if "fewshot_wake_enabled" in data:
        # Toggle the OR'd few-shot enrolled path for custom words (official words ignore it).
        cfg.fewshot_wake_enabled = bool(data["fewshot_wake_enabled"])
    if "fewshot_threshold" in data:
        # Clamp to [0, 1] like wake_threshold: a hand-crafted POST shouldn't push it past
        # validate()'s bound.
        cfg.fewshot_threshold = max(0.0, min(1.0, float(data["fewshot_threshold"])))
    if "fewshot_patience" in data:
        # Floor at 1 (validate() requires >= 1); a single window is the most-sensitive setting.
        cfg.fewshot_patience = max(1, int(data["fewshot_patience"]))
    if "mic_gain" in data:
        # Clamp to (0, 10]: a hand-crafted POST shouldn't push it past validate()'s
        # bound (0 would zero out the capture; >10 risks pure clipping).
        cfg.mic_gain = max(0.01, min(10.0, float(data["mic_gain"])))
    if "wake_gain" in data:
        # Clamp to (0, 10] like mic_gain — the live-loop wake gain knob (the dead-wake
        # fix); a hand-crafted POST shouldn't push it past validate()'s bound.
        cfg.wake_gain = max(0.01, min(10.0, float(data["wake_gain"])))
    if "agent_workspace" in data:
        cfg.agent_workspace = str(data["agent_workspace"]) or None
    if "agent_model" in data:
        cfg.agent_model = str(data["agent_model"])
    if "system_prompt" in data:
        cfg.system_prompt = str(data["system_prompt"])
    if "location" in data:
        cfg.location = str(data["location"])
    if "units" in data:
        cfg.units = str(data["units"])
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
    if "tts_streaming" in data:
        cfg.tts_streaming = bool(data["tts_streaming"])
    if "denoiser" in data:
        cfg.denoiser = str(data["denoiser"])
    if "music_playback" in data:
        # Audio still always plays server-side; this only steers whether the GUI
        # also embeds the muted YouTube video. Ignore an unrecognised value so a
        # hand-crafted POST can't push the config past validate()'s allowed set.
        choice = str(data["music_playback"])
        if choice in MUSIC_PLAYBACK_MODES:
            cfg.music_playback = choice


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:  # keep the console quiet
        pass

    @property
    def _ui(self) -> WebUI:
        return self.server.ui  # type: ignore[attr-defined]

    def _cors(self) -> None:
        """Emit permissive CORS headers.

        Lets a cross-origin page (e.g. the hosted GitHub Pages ``gui.html`` opened
        with ``?backend=https://<your-server>``) call this user-run server's
        ``/api/*`` and ``/events`` endpoints. The server holds no secrets and is
        started by the user, so a wildcard origin is acceptable and matches the
        "point the demo at your own box" workflow.
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
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

    def _route(self) -> str:
        """The request path without any query string (so a client-supplied
        ``?token=…`` on ``/events``/``/ws/audio``/``/api/settings`` still matches).
        """
        return self.path.split("?", 1)[0]

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        route = self._route()
        if route in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", self._ui.html.encode("utf-8"))
        elif route == "/api/settings":
            self._json(200, self._ui.settings_payload())
        elif route == "/events":
            self._sse()
        elif route == "/ws/audio":
            self._ws_audio()
        elif route.startswith("/recordings/"):
            self._serve_recording(route[len("/recordings/") :])
        else:
            self._send(404, "text/plain", b"not found")

    def _serve_recording(self, name: str) -> None:
        """Serve a saved mic/wake WAV from ``debug/recordings/`` (path-traversal-safe).

        The GUI links each diagnostic clip via its ``wav_url`` — a flat
        ``/recordings/<file>`` for a mic check or a per-word
        ``/recordings/wake/<word>/<file>`` for a wake clip. Either way only the
        **basename** of the requested name is honoured: :func:`audio.resolve_recording`
        strips any ``..`` / path segments and searches the recordings dir and its
        subfolders for that ``.wav``, re-checking every candidate stays under the
        recordings dir — so a crafted ``/recordings/../../etc/passwd`` can never escape.
        """
        from .audio import resolve_recording

        path = resolve_recording(os.path.basename(name))
        if path is None:
            self._send(404, "text/plain", b"not found")
            return
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send(404, "text/plain", b"not found")
            return
        self._send(200, "audio/wav", body)

    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server API)
        """Answer a CORS preflight: 204 + the CORS headers, no body.

        A cross-origin page POSTing JSON (``Content-Type: application/json``) to
        ``/api/turn``/``/api/action``/``/api/settings`` triggers a preflight; reply
        for any path so the subsequent real request is allowed.
        """
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _ws_audio(self) -> None:
        """Upgrade to a WebSocket and bridge browser PCM ⇄ the pipeline (R2-5).

        Performs the RFC 6455 handshake on the same-origin connection (so the page
        CSP ``connect-src 'self'`` permits it), then runs a
        :class:`~my_stt_tts.ws_transport.WebSocketTransport` session: inbound binary
        frames are decoded mic PCM fed to the loop, and TTS PCM the loop produces is
        framed back to the browser for playback. Audio carriage is only enabled when
        the WebUI was built with an ``on_audio_session`` callback.
        """
        if self._ui.on_audio_session is None:
            self._send(404, "text/plain", b"audio transport disabled")
            return
        key = self.headers.get("Sec-WebSocket-Key")
        if not key or self.headers.get("Upgrade", "").lower() != "websocket":
            self._send(400, "text/plain", b"expected a websocket upgrade")
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", ws_frame.accept_key(key))
        self.end_headers()
        self._ui.run_audio_session(self.connection)

    def _webrtc_offer(self, body: dict[str, Any]) -> None:
        """Answer a browser WebRTC SDP offer (R3-1): negotiate Opus + run the session.

        The browser opens a real ``RTCPeerConnection`` with
        ``getUserMedia({audio:{echoCancellation:true}})`` and POSTs its SDP offer
        here; the server builds the aiortc peer, wires the transport into the turn
        loop (``on_audio_session``), and returns the SDP answer. Only available when
        the WebUI carries audio AND the ``webrtc`` extra is installed; otherwise
        returns a clear error so the page falls back to the WS PCM path.
        """
        if self._ui.on_audio_session is None:
            self._json(404, {"error": "audio transport disabled"})
            return
        sdp = str(body.get("sdp", ""))
        if not sdp:
            self._json(400, {"error": "missing sdp offer"})
            return
        try:
            answer = self._ui.run_webrtc_session(
                {"sdp": sdp, "type": str(body.get("type", "offer"))}
            )
        except RuntimeError as exc:  # webrtc extra missing
            self._json(501, {"error": str(exc)})
            return
        self._json(200, answer)

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        route = self._route()
        if route == "/api/settings":
            try:
                apply_settings(self._ui.cfg, body)
            except Exception as exc:  # bad value from the UI shouldn't 500-crash
                self._json(400, {"error": str(exc)})
                return
            self._json(200, self._ui.settings_payload())
        elif route == "/api/turn":
            text = str(body.get("text", "")).strip()
            if text:
                self._ui.run_turn_async(text)
            self._json(200, {"ok": True})
        elif route == "/api/webrtc/offer":
            self._webrtc_offer(body)
        elif route == "/api/action":
            self._ui.action(str(body.get("action", "")), body)
            self._json(200, {"ok": True})
        else:
            self._send(404, "text/plain", b"not found")

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()  # allow a cross-origin (?backend=…) page to open this stream
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
        on_audio_session: Callable[[WebSocketTransport], None] | None = None,
        voice_available: bool = False,
        voice_hint: str = "",
    ) -> None:
        self.cfg = cfg
        self._on_turn = on_turn
        self._on_action = on_action
        # When set, the GUI can carry REAL audio over a same-origin WebSocket
        # (browser mic PCM in, TTS PCM out). When None, the page stays state/transcript
        # only (and falls back to demo mode offline) — the prior behaviour (R2-5).
        self.on_audio_session = on_audio_session
        # Whether the server-side wake / push-to-talk controls can actually run
        # (STT + wake model + mic). Surfaced via /api/settings so the page disables
        # those buttons (and shows ``voice_hint``) when voice is off.
        self.voice_available = voice_available
        self.voice_hint = voice_hint
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

    def settings_payload(self) -> dict[str, Any]:
        """The full ``/api/settings`` payload (config + capability flags) for the UI."""
        return settings_dict(
            self.cfg,
            audio_enabled=self.on_audio_session is not None,
            voice_available=self.voice_available,
            voice_hint=self.voice_hint,
        )

    def run_audio_session(self, sock: Any) -> None:
        """Bridge an upgraded WebSocket ``sock`` to the pipeline (R2-5, browser audio).

        Decodes inbound (masked) client frames into mic PCM fed to a
        :class:`WebSocketTransport`, runs ``on_audio_session`` (the turn loop) in a
        worker thread, and frames the transport's outbound TTS PCM back to the
        browser. Runs until the socket closes; all socket/frame errors are swallowed
        so a dropped tab can't crash the server.
        """
        assert self.on_audio_session is not None
        transport = WebSocketTransport(sample_rate=self.cfg.sample_rate)
        worker = threading.Thread(target=self.on_audio_session, args=(transport,), daemon=True)
        worker.start()
        sender = threading.Thread(target=self._ws_send_loop, args=(sock, transport), daemon=True)
        sender.start()
        try:
            self._ws_recv_loop(sock, transport)
        finally:
            transport.end_mic()
            transport.close()
            with contextlib.suppress(Exception):
                sock.sendall(ws_frame.close_frame())

    def run_webrtc_session(self, offer: dict[str, str]) -> dict[str, str]:
        """Negotiate a WebRTC peer for ``offer`` and run the turn loop on it (R3-1).

        Builds a :class:`~my_stt_tts.webrtc_transport.WebRtcTransport`, runs the real
        aiortc negotiation in a private asyncio loop on a worker thread (so the
        stdlib ``http.server`` request thread isn't blocked), starts the synchronous
        turn loop (``on_audio_session``) on the transport, and returns the SDP
        answer. Raises ``RuntimeError`` if the ``webrtc`` extra is missing.
        """
        import asyncio

        from .webrtc_transport import WebRtcTransport, run_webrtc_offer

        assert self.on_audio_session is not None
        transport = WebRtcTransport(sample_rate=self.cfg.sample_rate)
        result: dict[str, dict[str, str] | Exception] = {}
        done = threading.Event()

        def _negotiate() -> None:
            loop = asyncio.new_event_loop()
            try:
                answer = loop.run_until_complete(run_webrtc_offer(transport, offer))
                result["answer"] = answer
            except Exception as exc:  # surface to the request thread
                result["error"] = exc
            finally:
                done.set()
                # keep the loop alive so the peer connection's media pumps run
                with contextlib.suppress(Exception):
                    loop.run_forever()

        threading.Thread(target=_negotiate, daemon=True).start()
        threading.Thread(target=self.on_audio_session, args=(transport,), daemon=True).start()
        done.wait(timeout=15)
        if isinstance(result.get("error"), Exception):
            raise RuntimeError(str(result["error"]))
        answer = result.get("answer")
        if not isinstance(answer, dict):
            raise RuntimeError("WebRTC negotiation timed out")
        return answer

    @staticmethod
    def _ws_recv_loop(sock: Any, transport: WebSocketTransport) -> None:
        """Read masked client frames; feed binary PCM to ``transport`` until close."""
        buf = b""
        while not transport.closed:
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while True:
                try:
                    decoded = ws_frame.decode_frame(buf)
                except ValueError:
                    return  # protocol violation -> drop the connection
                if decoded is None:
                    break  # need more bytes
                frame, consumed = decoded
                buf = buf[consumed:]
                if frame.opcode == ws_frame.OP_CLOSE:
                    return
                if frame.opcode == ws_frame.OP_BINARY:
                    transport.feed_mic(frame.payload)

    @staticmethod
    def _ws_send_loop(sock: Any, transport: WebSocketTransport) -> None:
        """Frame the transport's outbound TTS PCM back to the browser as binary."""
        while not transport.closed:
            data = transport.iter_outbound(timeout=0.1)
            if not data:
                continue
            try:
                sock.sendall(ws_frame.encode_frame(data, opcode=ws_frame.OP_BINARY))
            except OSError:
                transport.close()
                return

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
