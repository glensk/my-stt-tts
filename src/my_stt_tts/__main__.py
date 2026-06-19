"""CLI entrypoint: wake-word / push-to-talk / typed / browser voice loop.

Modes:
  (default)     push-to-talk: press Enter to start/stop recording each turn
  --wake        always-listening: say the wake phrase ("maziko"), then speak
  --type        interactive typed input (no mic/STT)
  --text "..."  run one typed turn, then exit
  --browser     serve a live web GUI (state, transcript, settings, controls)

Brain presets (``--brain``) switch provider+model in one word, e.g. ``haiku-sub``
(subscription via the Claude CLI, no API key) or ``opus-api``. Say "agent, <task>"
to delegate to a full MCP-capable Claude agent (set AGENT_WORKSPACE to enable).
``--voice`` picks a Piper voice; ``--list-voices``/``--settings``/``-h`` print info.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import audio, chimes
from .aec import make_echo_canceller
from .brain import Brain, LLMError
from .config import (
    AEC_MODES,
    BARGE_IN_MODES,
    BRAIN_PRESETS,
    DENOISER_MODES,
    TRANSPORT_MODES,
    TURN_ANALYZERS,
    Config,
    ConfigError,
)
from .events import bus
from .interrupt import InterruptGate, make_interrupt_predictor
from .metrics import TurnMetrics
from .text import SentenceChunker, strip_non_spoken
from .tts import VOICE_PRESETS, TTSRouter, list_voice_presets

log = logging.getLogger("my_stt_tts")
_CHIME_SR = chimes.DEFAULT_SR
_BLUE = "\033[34m"
_RESET = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def settings_text(cfg: Config, *, color: bool | None = None) -> str:
    """A human-readable summary of the resolved settings (brain, voice, etc.)."""
    use = _use_color() if color is None else color
    blue = _BLUE if use else ""
    reset = _RESET if use else ""
    prompt_head = (cfg.system_prompt.strip().splitlines() or [""])[0][:68]
    agent_ws = cfg.agent_workspace or "(disabled — set AGENT_WORKSPACE)"
    rows = [
        "current settings (override via .env or flags):",
        f"  brain      {blue}{cfg.llm_provider} / {cfg.llm_model}{reset}  (deep: {cfg.llm_model_deep})",
        f"  voice      en={blue}{cfg.tts_voices.get('en')}{reset}"
        f"  de={cfg.tts_voices.get('de')}  fr={cfg.tts_voices.get('fr')}"
        f"  length-scale {cfg.tts_length_scale}",
        f"  stt        {blue}{cfg.stt_model}{reset}  streaming {cfg.stt_streaming}"
        f"  window {cfg.stt_window_s}s  backend {cfg.stt_backend}",
        f"  transport  {blue}{cfg.transport}{reset}"
        f"  {cfg.transport_host}:{cfg.transport_port}"
        f"  tools {cfg.tools_enabled}  tts-backend {cfg.tts_backend}"
        f"  tts-stream {cfg.tts_streaming}",
        f"  turn       barge-in {blue}{cfg.barge_in}{reset}  aec {cfg.aec_mode}"
        f"  hw-capture {cfg.aec_hw_capture}  denoiser {cfg.denoiser}"
        f"  analyzer {cfg.turn_analyzer}  min-words {cfg.interrupt_min_words}"
        f"  predict {cfg.interrupt_predict}",
        f"  wake       phrase {blue}{cfg.wake_phrase}{reset}  model {cfg.wake_model_path}",
        f"  agent      trigger '{cfg.agent_trigger}'  workspace {blue}{agent_ws}{reset}"
        f"  model {cfg.agent_model}",
        f"  prompt     {blue}{prompt_head}…{reset}  (edit prompts/system_prompt.md)",
    ]
    return "\n".join(rows)


def _epilog() -> str:
    try:
        return settings_text(Config.from_env())
    except Exception:  # never let a bad .env break --help
        return "current settings: (unavailable — check your .env)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="my-stt-tts",
        description="Local voice assistant: wake/typed -> STT -> LLM -> TTS (Anthropic by default).",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wake", action="store_true", help="Always-listen for the wake phrase.")
    parser.add_argument(
        "--type",
        dest="type_mode",
        action="store_true",
        help="Interactive typed input (no mic/STT).",
    )
    parser.add_argument("--text", help="Run one typed turn with this text, then exit.")
    parser.add_argument("--once", action="store_true", help="Run a single mic turn, then exit.")
    parser.add_argument("--browser", action="store_true", help="Serve the live web GUI.")
    parser.add_argument(
        "--browser-audio",
        action="store_true",
        help="Carry REAL audio in the GUI: browser mic → WS → pipeline → TTS back "
        "(loads on-device STT). Without it the GUI is state/transcript only.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Web GUI port (default 8765).")
    parser.add_argument("--debug", action="store_true", help="Speak stage cues + verbose logs.")
    parser.add_argument(
        "--brain",
        choices=sorted(BRAIN_PRESETS),
        help="One-word provider+model preset (e.g. haiku-sub, opus-api). Overrides --provider/--model.",
    )
    parser.add_argument(
        "--provider", help="Override LLM_PROVIDER (anthropic/openai/ollama/claude-cli/...)."
    )
    parser.add_argument("--model", help="Override LLM_MODEL.")
    parser.add_argument(
        "--voice", choices=sorted(VOICE_PRESETS), help="English TTS voice (see --list-voices)."
    )
    parser.add_argument(
        "--barge-in",
        choices=BARGE_IN_MODES,
        help="Interrupt playback on user speech: off | headphones | always "
        "(needs headphones without AEC; default off).",
    )
    parser.add_argument(
        "--turn-analyzer",
        choices=TURN_ANALYZERS,
        help="End-of-turn detector: smart (Smart Turn v3, default; auto-downloads, "
        "falls back to silence) | silence (fixed timer).",
    )
    parser.add_argument(
        "--aec",
        dest="aec_mode",
        choices=AEC_MODES,
        help="Acoustic echo cancellation so barge-in works on open speakers: "
        "off | nlms (software) | voiceprocessing (macOS HW) | auto.",
    )
    parser.add_argument(
        "--stt-streaming",
        action="store_true",
        help="Emit partial transcripts during the turn (incremental STT).",
    )
    parser.add_argument(
        "--stt-window",
        dest="stt_window_s",
        type=float,
        help="Seconds of trailing audio re-decoded per streaming partial (default 7).",
    )
    parser.add_argument(
        "--no-interrupt-predict",
        dest="interrupt_predict",
        action="store_false",
        default=None,
        help="Disable the acoustic interruption predictor (3rd barge-in guard).",
    )
    parser.add_argument(
        "--transport",
        choices=TRANSPORT_MODES,
        help="Audio transport: local (sound card, default) | websocket (network "
        "server for remote satellites / the browser) | webrtc (real RTCPeerConnection: "
        "Opus + jitter buffer + NAT traversal; needs the webrtc extra).",
    )
    parser.add_argument(
        "--denoiser",
        choices=DENOISER_MODES,
        help="Pre-VAD noise suppression on mic frames: off | spectral (pure-numpy, "
        "default-safe) | rnnoise (needs the denoiser extra; falls back to spectral).",
    )
    parser.add_argument(
        "--no-tts-streaming",
        dest="tts_streaming",
        action="store_false",
        default=None,
        help="Disable streamed clause-by-clause TTS playout (render whole sentences first).",
    )
    parser.add_argument(
        "--transport-port", type=int, help="WebSocket transport port (default 8770)."
    )
    parser.add_argument(
        "--transport-token", help="Shared token a transport client must present (optional auth)."
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List the English voice presets and exit."
    )
    parser.add_argument(
        "--settings", action="store_true", help="Print the resolved settings and exit."
    )
    return parser.parse_args(argv)


# CLI flag -> Config field for the plain "override when present" overrides, applied
# table-driven in _build_config (keeps its branch count low as flags grow). A None
# arg leaves the .env/default value untouched; the special cases (brain preset,
# voice-name mapping, store_true toggles) are handled explicitly below.
_CONFIG_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("provider", "llm_provider"),
    ("model", "llm_model"),
    ("barge_in", "barge_in"),
    ("aec_mode", "aec_mode"),
    ("turn_analyzer", "turn_analyzer"),
    ("stt_window_s", "stt_window_s"),
    ("interrupt_predict", "interrupt_predict"),
    ("transport", "transport"),
    ("transport_port", "transport_port"),
    ("transport_token", "transport_token"),
    ("denoiser", "denoiser"),
)


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if args.brain:
        cfg.apply_brain_preset(args.brain)
    if args.voice:
        cfg.tts_voices["en"] = VOICE_PRESETS[args.voice]
    for arg_name, field_name in _CONFIG_OVERRIDES:
        value = getattr(args, arg_name)
        if value is not None and value != "":
            setattr(cfg, field_name, value)
    if args.stt_streaming:
        cfg.stt_streaming = True
    if args.tts_streaming is not None:
        cfg.tts_streaming = args.tts_streaming
    if args.debug:
        cfg.debug = True
    return cfg


def _play(samples: object) -> None:
    audio.play(samples, _CHIME_SR)  # type: ignore[arg-type]


@dataclass
class RespondResult:
    """Outcome of one assistant response (G1).

    ``interrupted`` is True if the user barged in; ``captured`` holds the audio
    recorded from the barge-in onward (to seed the next turn without clipping).
    """

    interrupted: bool = False
    captured: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


@dataclass
class _BargeInCtx:
    """Barge-in machinery shared across one reply's sentences (R2-1/3/4/6).

    Bundles the VAD, the false-interrupt gate, the AEC front-end and the acoustic
    interruption predictor so :func:`_voice_sentence` keeps a small signature and
    the adaptive state (echo filter, predictor score) persists across sentences.
    Half-duplex when ``vad`` / ``igate`` are None.
    """

    vad: object | None = None
    igate: InterruptGate | None = None
    aec: object | None = None
    predictor: object | None = None
    source: object | None = None  # R3-4: HW VoiceProcessingIO capture, if active
    denoiser: object | None = None  # R3-6: pre-VAD noise suppression, if enabled

    @property
    def live(self) -> bool:
        """True when the mic stays live during playback (barge-in is armed)."""
        return self.vad is not None and self.igate is not None


def _speak(tts: TTSRouter, gate: audio.MicGate, sentence: str) -> None:
    """Half-duplex speak one sentence (mic gated during playback). Legacy path."""
    spoken = strip_non_spoken(sentence)
    if not spoken:
        return
    gate.gate()
    tts.speak(spoken)  # lang auto-detected from the answer text
    gate.release()


def _respond(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    text: str,
    *,
    vad: object | None = None,
    source: object | None = None,
    denoiser: object | None = None,
) -> RespondResult:
    """Stream the LLM for ``text``, speak it, and publish live state to the bus.

    When ``cfg.barge_in`` is enabled and a ``vad`` is supplied, the mic stays live
    during playback; confirmed user speech (gated by :class:`InterruptGate`)
    aborts TTS + the LLM stream and the spoken-prefix is committed to history
    (G1/G4/G5). Otherwise playback is half-duplex (mic gated)."""
    metrics = TurnMetrics()
    metrics.note(transcript=text)
    bus.transcript(text)
    chunker = SentenceChunker()
    barge_in = cfg.barge_in != "off" and vad is not None
    voiced_chars = 0
    result = RespondResult()
    frame_ms = 512 / cfg.sample_rate * 1000.0
    igate = (
        InterruptGate(
            min_speech_ms=cfg.interrupt_min_speech_ms,
            min_words=cfg.interrupt_min_words,
            frame_ms=frame_ms,
        )
        if barge_in
        else None
    )
    # R2-1 AEC front-end + R2-3 acoustic interruption predictor, built once per
    # reply and shared across its sentences so adaptation/scoring persists.
    # R3-4: with HW capture active the OS already cancelled the echo, so the
    # software AEC front-end is a no-op (avoid double-processing).
    sw_aec = make_echo_canceller(cfg) if (barge_in and source is None) else None
    ctx = _BargeInCtx(
        vad=vad if barge_in else None,
        igate=igate,
        aec=sw_aec,
        predictor=make_interrupt_predictor(cfg, frame_ms) if barge_in else None,
        source=source if barge_in else None,
        denoiser=denoiser if barge_in else None,
    )
    stream = brain.stream(text)
    try:
        with metrics.stage("llm_tts"):
            bus.state("llm_request", f"{cfg.llm_provider} / {cfg.llm_model}")
            bus.state("llm_wait")
            first = True
            for delta in stream:
                if first:
                    bus.state("llm_response")
                    first = False
                bus.response(delta, final=False)
                for sentence in chunker.feed(delta):
                    spoken_n, res = _voice_sentence(cfg, tts, gate, ctx, sentence)
                    voiced_chars += spoken_n
                    if res is not None and res.interrupted:
                        result = res
                        break
                if result.interrupted:
                    break
            else:
                tail = chunker.flush()
                if tail:
                    spoken_n, res = _voice_sentence(cfg, tts, gate, ctx, tail)
                    voiced_chars += spoken_n
                    if res is not None and res.interrupted:
                        result = res
        bus.response("", final=True)
    except LLMError as exc:
        log.error("LLM error: %s", exc)
        bus.log(str(exc), "error")
        _play(chimes.chime_error())
        tts.speak("Sorry, I had a problem.")
    finally:
        # Stop consuming the LLM stream (best-effort cancel of in-flight tokens).
        with contextlib.suppress(Exception):
            stream.close()
        if result.interrupted:
            # G5: the model should only remember what was actually voiced.
            brain.commit_spoken(_spoken_prefix(brain, voiced_chars))
            bus.interrupted(voiced_chars)
    bus.state("idle")
    metrics.log()
    return result


def _spoken_prefix(brain: Brain, voiced_chars: int) -> str:
    """The first ``voiced_chars`` characters of the just-streamed assistant reply."""
    index = brain._pending_assistant_index  # noqa: SLF001 — same package, intentional
    if index is None or not 0 <= index < len(brain.history):
        return ""
    return brain.history[index]["content"][:voiced_chars]


def _voice_sentence(
    cfg: Config,
    tts: TTSRouter,
    gate: audio.MicGate,
    ctx: _BargeInCtx,
    sentence: str,
) -> tuple[int, RespondResult | None]:
    """Speak one sentence; return (chars actually voiced, barge-in result or None).

    Half-duplex unless barge-in is armed (``ctx.live``). When armed, the mic stays
    live with the AEC front-end + acoustic predictor (R2-1/R2-3) and interruption
    is published as bus events (R2-6) so all stages flush consistently."""
    spoken = strip_non_spoken(sentence)
    if not spoken:
        return 0, None
    bus.state("speaking")
    if not ctx.live:
        gate.gate()
        tts.speak(spoken)
        gate.release()
        return len(sentence), None
    # R3-3: stream the clause-chunked PCM so first audio plays within a few hundred
    # ms; the StreamingPlayback handle keeps the same cancel surface for barge-in.
    playback = (
        tts.start_speaking_stream(spoken) if cfg.tts_streaming else tts.start_speaking(spoken)
    )
    res = audio.monitor_during_playback(
        playback,
        cfg.sample_rate,
        ctx.vad,
        ctx.igate,
        energy_floor=cfg.barge_in_energy,
        aec=ctx.aec,
        predictor=ctx.predictor,
        source=ctx.source,
        denoiser=ctx.denoiser,
    )
    bus.bot_stopped_speaking()  # R2-6: playback ended (cancelled or finished)
    if res.interrupted:
        bus.state("interrupted")
        bus.interrupt_start()  # R2-6: abort/flush downstream stages
        # Only the audio that played counts as voiced; approximate as the whole
        # sentence (it had started) — refined truncation is at sentence boundary.
        return len(sentence), RespondResult(interrupted=True, captured=res.captured)
    return len(sentence), None


def _maybe_streamer(cfg: Config, stt: object) -> Any:
    """A :class:`StreamingTranscriber` when streaming STT is on, else None (G6)."""
    if not cfg.stt_streaming:
        return None
    from .stt import StreamingTranscriber

    return StreamingTranscriber(
        stt,  # type: ignore[arg-type]
        cfg.sample_rate,
        partial_interval_ms=cfg.stt_partial_interval_ms,
        window_s=cfg.stt_window_s,
    )


def _capture_ptt(cfg: Config, stt: object, gate: audio.MicGate) -> str:
    gate.gate()
    _play(chimes.chime_listening())
    gate.release()
    bus.state("recording")
    clip = audio.record_push_to_talk(cfg.sample_rate, cfg.max_record_seconds)
    gate.gate()
    _play(chimes.chime_done())
    gate.release()
    if clip.size == 0:
        bus.state("idle")
        return ""
    bus.state("stt")
    return str(stt.transcribe(clip, cfg.sample_rate).text)  # type: ignore[attr-defined]


def _transcribe(cfg: Config, stt: object, clip: np.ndarray) -> str:
    """Final transcription of a recorded clip (used for barge-in re-capture)."""
    if clip.size == 0:
        return ""
    bus.state("stt")
    return str(stt.transcribe(clip, cfg.sample_rate).text).strip()  # type: ignore[attr-defined]


def _transcribe_barge_in(cfg: Config, stt: object, clip: np.ndarray) -> str:
    """Transcribe a captured barge-in clip as the next turn's text (R2-6).

    The audio captured *while the bot was speaking* is handed straight to the
    streaming transcriber via ``feed_clip`` instead of being re-transcribed from
    scratch, so the next turn does not pay an extra round-trip. Emits
    ``interrupt_stop`` once the hand-off is done so all stages know it is safe to
    resume from a clean state. Falls back to a one-shot transcribe when streaming
    STT is off."""
    if clip.size == 0:
        bus.interrupt_stop()
        return ""
    bus.state("stt")
    if cfg.stt_streaming:
        from .stt import StreamingTranscriber

        streamer = StreamingTranscriber(
            stt,  # type: ignore[arg-type]
            cfg.sample_rate,
            partial_interval_ms=cfg.stt_partial_interval_ms,
            window_s=cfg.stt_window_s,
        )
        streamer.feed_clip(clip)
        text = str(streamer.final().text).strip()
    else:
        text = _transcribe(cfg, stt, clip)
    bus.interrupt_stop()  # R2-6: hand-off complete; downstream may resume
    return text


def run_turn(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    *,
    stt: object | None = None,
    typed_text: str | None = None,
    vad: object | None = None,
) -> None:
    """Run one push-to-talk or typed turn (chaining barge-in follow-ups)."""
    text = typed_text.strip() if typed_text is not None else _capture_ptt(cfg, stt, gate)
    if not text:
        return
    if cfg.debug and typed_text is None:
        tts.speak("recorded")
    active_vad = vad if typed_text is None else None
    while text:
        result = _respond(cfg, brain, tts, gate, text, vad=active_vad)
        if not result.interrupted:
            return
        # User barged in: their captured audio is handed straight to the streaming
        # transcriber (no from-scratch re-transcribe) as the next turn (R2-6).
        text = _transcribe_barge_in(cfg, stt, result.captured) if stt is not None else ""


def run_wake_loop(
    cfg: Config, brain: Brain, tts: TTSRouter, gate: audio.MicGate, stt: object
) -> int:
    """Always-listen: on the wake phrase, capture (VAD), respond, then follow up."""
    from .aec import make_voiceprocessing_capture
    from .denoise import make_denoiser
    from .turn import make_turn_analyzer
    from .vad import SileroVad
    from .wake import WakeWord

    wake = WakeWord.from_config(cfg)
    if not wake.available():
        msg = (
            f'Wake model not found at {cfg.wake_model_path}. Train "{cfg.wake_phrase}" first '
            "— see wakewords/WAKEWORD.md."
        )
        print(msg, file=sys.stderr)
        bus.log(msg, "error")
        return 2
    vad = SileroVad(cfg.sample_rate)
    frame_seconds = 512 / cfg.sample_rate
    barge_vad = vad if cfg.barge_in != "off" else None
    # R3-4: capture through macOS VoiceProcessingIO (hardware AEC) when requested
    # and available; None means fall back to sounddevice. R3-6: pre-VAD denoiser.
    hw_source = make_voiceprocessing_capture(cfg)
    denoiser = make_denoiser(cfg)
    print(f'Listening for "{cfg.wake_phrase}". Ctrl-C to quit.')
    while True:
        bus.state("listening", cfg.wake_phrase)
        audio.listen_for_wake(wake, cfg.sample_rate)
        bus.wake()
        gate.gate()
        _play(chimes.chime_listening())
        gate.release()
        follow_up = False
        while True:
            bus.state("recording")
            analyzer = make_turn_analyzer(cfg, frame_seconds)
            streamer = _maybe_streamer(cfg, stt)
            max_s = cfg.follow_up_seconds if follow_up else cfg.max_record_seconds
            clip = audio.record_turn(
                cfg.sample_rate,
                vad,
                analyzer,
                max_seconds=max_s,
                streamer=streamer,
                on_partial=lambda t: bus.transcript(t, partial=True),
                source=hw_source,
                denoiser=denoiser,
            )
            if clip.size == 0:
                break  # silence -> back to listening for the wake word
            text = (
                str(streamer.final().text).strip()  # type: ignore[union-attr]
                if streamer is not None
                else _transcribe(cfg, stt, clip)
            )
            if not text:
                break
            result = _respond(
                cfg, brain, tts, gate, text, vad=barge_vad, source=hw_source, denoiser=denoiser
            )
            while result.interrupted:
                # Hand the captured barge-in audio straight to the transcriber as
                # the next turn (no from-scratch re-transcribe, R2-6), and keep
                # chaining if the new reply is itself interrupted.
                text = _transcribe_barge_in(cfg, stt, result.captured)
                if not text:
                    break
                result = _respond(
                    cfg, brain, tts, gate, text, vad=barge_vad, source=hw_source, denoiser=denoiser
                )
            follow_up = True  # subsequent turns are short follow-ups (no re-wake)


def _run_browser(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    stt: object | None,
    *,
    wake: bool,
    port: int = 8765,
) -> int:
    """Serve the web GUI; turns are driven from the page (and optional wake loop)."""
    import webbrowser

    from .webui import WebUI

    def on_turn(text: str) -> None:
        _respond(cfg, brain, tts, gate, text)

    def on_action(name: str, _data: dict) -> None:
        if name == "reset":
            brain.reset()
            bus.log("conversation reset")
        elif name == "list_voices":
            bus.log("voices: " + ", ".join(VOICE_PRESETS))
        elif name in {"wake_start", "wake_stop", "ptt"}:
            bus.log(
                f"'{name}' runs from the terminal in this build (use --wake / push-to-talk)",
                "error",
            )
        else:
            bus.log(f"unknown action '{name}'", "error")

    # Real browser audio (R2-5): when an STT engine is available, the page can
    # stream mic PCM over a same-origin WebSocket and play TTS PCM back, driving the
    # full pipeline (not just state). Disabled (state/transcript only) without STT.
    def _audio_session(transport: object) -> None:
        from .net_loop import run_transport_session

        run_transport_session(transport, cfg, brain, tts, stt)  # type: ignore[arg-type]

    on_audio_session = _audio_session if stt is not None else None
    ui = WebUI(cfg, on_turn, on_action, port=port, on_audio_session=on_audio_session)
    bus.state("idle")
    if wake and stt is not None:
        threading.Thread(
            target=run_wake_loop, args=(cfg, brain, tts, gate, stt), daemon=True
        ).start()
    print(f"my-stt-tts web UI → {ui.url()}  (Ctrl-C to quit)")
    with contextlib.suppress(Exception):  # headless / no browser is fine
        webbrowser.open(ui.url())
    try:
        ui.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def _run_websocket_server(cfg: Config, brain: Brain, tts: TTSRouter, stt: object | None) -> int:
    """Serve the WebSocket audio transport (R2-5): bridge remote clients to the pipeline.

    Each accepted client (satellite or browser) gets a :class:`WebSocketTransport`
    that becomes the mic source + audio sink for a full :func:`run_transport_session`
    turn loop — the same STT/LLM/TTS stages, just with the device boundary on the
    wire. Needs the ``transport`` extra; prints a clear message if it is missing.
    """
    from .net_loop import run_transport_session
    from .ws_transport import serve_websocket

    if stt is None:
        from .stt import make_transcriber

        stt = make_transcriber(cfg)

    def on_session(transport: object) -> None:
        run_transport_session(transport, cfg, brain, tts, stt)  # type: ignore[arg-type]

    print(
        f"my-stt-tts WebSocket transport → ws://{cfg.transport_host}:{cfg.transport_port}"
        "  (Ctrl-C to quit)"
    )
    try:
        serve_websocket(
            on_session,
            host=cfg.transport_host,
            port=cfg.transport_port,
            token=cfg.transport_token,
            sample_rate=cfg.sample_rate,
        )
    except RuntimeError as exc:  # missing 'transport' extra
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def _run_webrtc_server(
    cfg: Config, brain: Brain, tts: TTSRouter, stt: object | None, *, port: int = 8765
) -> int:
    """Serve the WebRTC signaling + GUI (R3-1): browser connects via RTCPeerConnection.

    WebRTC is browser-first: the page opens a real ``RTCPeerConnection`` (Opus +
    jitter buffer + ICE NAT traversal) and POSTs its SDP offer to
    ``/api/webrtc/offer``, which the WebUI answers and bridges into the same
    pipeline as the WebSocket path. Reuses the GUI server so the WS PCM channel
    stays available as a fallback. Needs the ``webrtc`` extra (aiortc).
    """
    from .webrtc_transport import webrtc_available

    if not webrtc_available():
        print("WebRTC transport needs the 'webrtc' extra: uv sync --extra webrtc", file=sys.stderr)
        return 2
    if stt is None:
        from .stt import make_transcriber

        stt = make_transcriber(cfg)
    gate = audio.MicGate(cfg.mic_gate_tail_seconds)
    print(f"my-stt-tts WebRTC server (browser RTCPeerConnection) → port {port}  (Ctrl-C to quit)")
    return _run_browser(cfg, brain, tts, gate, stt, wake=False, port=port)


def main(argv: list[str] | None = None) -> int:
    """Validate config, build components, run the chosen loop. Returns exit code."""
    args = _parse_args(argv)
    if args.list_voices:
        print("English voice presets (use --voice <name>):")
        print(list_voice_presets())
        return 0
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    try:
        cfg = _build_config(args)
        if args.settings:
            print(settings_text(cfg))
            return 0
        cfg.validate()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    brain = Brain(cfg)
    tts = TTSRouter(cfg)
    gate = audio.MicGate(cfg.mic_gate_tail_seconds)

    needs_stt = (
        cfg.transport in ("websocket", "webrtc")
        or (args.browser and args.browser_audio)
        or (not args.type_mode and args.text is None and (not args.browser or args.wake))
    )
    stt: object | None = None
    if needs_stt:
        from .stt import make_transcriber  # heavy (MLX); only needed for mic modes

        stt = make_transcriber(cfg)

    if not args.browser:
        print(settings_text(cfg))
    try:
        if cfg.transport == "websocket":
            return _run_websocket_server(cfg, brain, tts, stt)
        if cfg.transport == "webrtc":
            return _run_webrtc_server(cfg, brain, tts, stt, port=args.port)
        if args.browser:
            return _run_browser(cfg, brain, tts, gate, stt, wake=args.wake, port=args.port)
        if args.wake:
            return run_wake_loop(cfg, brain, tts, gate, stt)
        _run_terminal_modes(args, cfg, brain, tts, gate, stt)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    return 0


def _run_terminal_modes(
    args: argparse.Namespace,
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    stt: object | None,
) -> None:
    """One typed turn, the interactive typed loop, or the push-to-talk mic loop."""
    if args.text is not None:
        run_turn(cfg, brain, tts, gate, typed_text=args.text)
        return
    if args.type_mode:
        print("Type a message ('quit' or blank to exit).")
        while True:
            line = input("you> ").strip()
            if line in {"", "quit", "exit"}:
                return
            run_turn(cfg, brain, tts, gate, typed_text=line)
    ptt_vad = None
    if cfg.barge_in != "off":
        from .vad import SileroVad

        ptt_vad = SileroVad(cfg.sample_rate)
    print("Push-to-talk: Enter to start/stop each turn. Ctrl-C to quit.")
    while True:
        run_turn(cfg, brain, tts, gate, stt=stt, vad=ptt_vad)
        if args.once:
            return


if __name__ == "__main__":
    raise SystemExit(main())
