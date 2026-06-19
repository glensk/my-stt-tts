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
    STT_BACKENDS,
    TRANSPORT_MODES,
    TTS_BACKENDS,
    TURN_ANALYZERS,
    UNITS,
    Config,
    ConfigError,
    available_wake_words,
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

# R3-7: one telemetry sink per process (JSON-lines log + aggregator + OTel span),
# built lazily from config the first time a turn emits metrics. ``None`` when
# telemetry is disabled — the headline log line + bus event still fire regardless.
_SESSION_SINK: object | None = None
_SESSION_SINK_BUILT = False


def _session_sink(cfg: Config) -> Any:
    """Return the process-wide telemetry sink (built once from config), or None."""
    global _SESSION_SINK, _SESSION_SINK_BUILT  # noqa: PLW0603 — one-time process singleton
    if not _SESSION_SINK_BUILT:
        from .metrics import make_sink

        _SESSION_SINK = make_sink(cfg)
        _SESSION_SINK_BUILT = True
    return _SESSION_SINK


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
        f"  brain      {blue}{cfg.llm_provider} / {cfg.llm_model}{reset}  (deep: {cfg.llm_model_deep})"
        f"  memory {blue}{cfg.memory_store or 'in-memory'}{reset}",
        f"  voice      en={blue}{cfg.tts_voices.get('en')}{reset}"
        f"  de={cfg.tts_voices.get('de')}  fr={cfg.tts_voices.get('fr')}"
        f"  length-scale {cfg.tts_length_scale}",
        f"  stt        {blue}{cfg.stt_model}{reset}  streaming {cfg.stt_streaming}"
        f"  window {cfg.stt_window_s}s  backend {blue}{cfg.stt_backend}{reset}"
        f"  platform {cfg.platform}  playback {cfg.playback_backend}",
        f"  transport  {blue}{cfg.transport}{reset}"
        f"  {cfg.transport_host}:{cfg.transport_port}"
        f"  tools {cfg.tools_enabled}  tts-backend {cfg.tts_backend}"
        f"  tts-stream {cfg.tts_streaming}",
        f"  turn       barge-in {blue}{cfg.barge_in}{reset}  aec {cfg.aec_mode}"
        f"  hw-capture {cfg.aec_hw_capture}  denoiser {cfg.denoiser}"
        f"  analyzer {cfg.turn_analyzer}  min-words {cfg.interrupt_min_words}"
        f"  predict {cfg.interrupt_predict}",
        f"  brain-mode {blue}{cfg.brain_mode}{reset}"
        f"  (realtime: {cfg.realtime_model} keyed={bool(cfg.realtime_api_key)})"
        f"  telephony {cfg.telephony}  telemetry {cfg.telemetry}",
        f"  wake       phrase {blue}{cfg.wake_phrase}{reset}  model {cfg.wake_model_path}"
        f"  exists {os.path.isfile(cfg.wake_model_path)}"
        f"  available [{', '.join(available_wake_words()) or 'none — see wakewords/WAKEWORD.md'}]",
        f"  speaker-id {blue}{cfg.speaker_id_enabled}{reset}  enroll {cfg.enroll_dir}"
        f"  threshold {cfg.speaker_threshold}  margin {cfg.speaker_margin}",
        f"  agent      trigger '{cfg.agent_trigger}'  workspace {blue}{agent_ws}{reset}"
        f"  model {cfg.agent_model}",
        f"  prompt     {blue}{prompt_head}…{reset}  (edit prompts/system_prompt.md)",
        f"  locale     location {blue}{cfg.location}{reset}  units {blue}{cfg.units}{reset}",
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
        "--location",
        help="User location for weather + units-aware answers (default 'Lausanne, Switzerland').",
    )
    parser.add_argument(
        "--units",
        choices=UNITS,
        help="Measurement system for answers + the weather tool: metric (default) | imperial.",
    )
    parser.add_argument(
        "--voice", choices=sorted(VOICE_PRESETS), help="English TTS voice (see --list-voices)."
    )
    parser.add_argument(
        "--wake-word",
        dest="wake_word",
        help="Pick a pre-shipped wake word by NAME (see --settings for the choices). "
        "Sets the wake phrase and auto-derives the model path wakewords/<name>.onnx.",
    )
    parser.add_argument(
        "--wake-model-path",
        dest="wake_model_path",
        help="Explicit path to a custom-trained wake-word .onnx (overrides --wake-word "
        "/ WAKE_PHRASE derivation; same as WAKE_MODEL_PATH).",
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
        "--stt-backend",
        dest="stt_backend",
        choices=STT_BACKENDS,
        help="STT backend (G1/G8): local (parakeet-mlx) | whispercpp | faster-whisper "
        "(cross-platform) | cloud/openai | deepgram. Cloud is key-gated (falls back to local).",
    )
    parser.add_argument(
        "--tts-backend",
        dest="tts_backend",
        choices=TTS_BACKENDS,
        help="TTS backend (G1): local (Piper/say) | cloud/openai | elevenlabs | cartesia. "
        "Cloud is key-gated (falls back to local Piper / say).",
    )
    parser.add_argument(
        "--memory-store",
        dest="memory_store",
        help="Per-speaker persistent memory store path (G7): a .json file (JSON backend) "
        "or any other path (SQLite). Cross-session recall keyed by enrolled speaker. "
        "Unset = in-memory only.",
    )
    parser.add_argument(
        "--platform",
        dest="platform",
        choices=("auto", "macos", "linux"),
        help="Host platform for the brain (G8): auto-detect (default) | macos | linux. "
        "Linux selects native playback + WebRTC-APM AEC so the brain runs off-Mac.",
    )
    parser.add_argument(
        "--playback",
        dest="playback_backend",
        choices=("auto", "sounddevice", "aplay", "afplay"),
        help="Audio playback sink (G8): auto | sounddevice | aplay (Linux) | afplay (macOS).",
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
        "--realtime",
        dest="brain_mode_realtime",
        action="store_true",
        help="Speech-to-speech: stream mic audio to a realtime LLM (OpenAI Realtime) "
        "and play its audio back, bypassing the STT->LLM->TTS cascade. Key-gated "
        "(REALTIME_API_KEY/OPENAI_API_KEY); falls back to the cascade without a key.",
    )
    parser.add_argument(
        "--telephony",
        action="store_true",
        help="Answer phone calls via Twilio Media Streams (μ-law 8 kHz) over a "
        "WebSocket; the same pipeline takes the call. Needs the 'transport' extra.",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Record per-stage latency telemetry per turn to events.bus + a "
        "JSON-lines log (set TELEMETRY_LOG_FILE) keyed by a speech_id.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Fetch + checksum the Smart-Turn model and Piper voices ahead of time, "
        "report what's ready, then exit (verified first-run bootstrap).",
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
    ("location", "location"),
    ("units", "units"),
    ("barge_in", "barge_in"),
    ("aec_mode", "aec_mode"),
    ("turn_analyzer", "turn_analyzer"),
    ("stt_window_s", "stt_window_s"),
    ("interrupt_predict", "interrupt_predict"),
    ("transport", "transport"),
    ("transport_port", "transport_port"),
    ("transport_token", "transport_token"),
    ("denoiser", "denoiser"),
    ("stt_backend", "stt_backend"),
    ("tts_backend", "tts_backend"),
    ("platform", "platform"),
    ("playback_backend", "playback_backend"),
    ("memory_store", "memory_store"),
)


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if args.brain:
        cfg.apply_brain_preset(args.brain)
    if args.voice:
        cfg.tts_voices["en"] = VOICE_PRESETS[args.voice]
    # Wake-word selection: --wake-word NAME sets the phrase + derives the path; an
    # explicit --wake-model-path then wins (custom model anywhere on disk).
    if args.wake_word:
        cfg.select_wake_word(args.wake_word)
    if args.wake_model_path:
        cfg.wake_model_path = args.wake_model_path
    for arg_name, field_name in _CONFIG_OVERRIDES:
        value = getattr(args, arg_name)
        if value is not None and value != "":
            setattr(cfg, field_name, value)
    if args.stt_streaming:
        cfg.stt_streaming = True
    if args.tts_streaming is not None:
        cfg.tts_streaming = args.tts_streaming
    if args.brain_mode_realtime:
        cfg.brain_mode = "realtime"
    if args.telephony:
        cfg.telephony = True
    if args.telemetry:
        cfg.telemetry = True
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


def _set_speaker(brain: Brain, speaker_id: object | None, clip: np.ndarray | None) -> None:
    """Resolve a clip to an enrolled speaker and set it on the brain (G7).

    Runs sequentially just before ``brain.stream`` so per-speaker memory keys to the
    right person. Gated: with no pipeline (disabled / no enrollment / no speechbrain)
    or a typed turn (``clip is None``) the speaker is ``None`` (shared guest bucket)
    and there is no embed cost. The pipeline's :meth:`identify` is itself defensive,
    so a model/clip failure simply yields ``None`` — never a crashed turn. The
    identified name is published to the bus so the web UI can show who is talking.
    """
    name = speaker_id.identify(clip) if speaker_id is not None and clip is not None else None  # type: ignore[attr-defined]
    brain.set_speaker(name)
    bus.speaker(name)


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
    speaker_id: object | None = None,
    clip: np.ndarray | None = None,
) -> RespondResult:
    """Stream the LLM for ``text``, speak it, and publish live state to the bus.

    When ``cfg.barge_in`` is enabled and a ``vad`` is supplied, the mic stays live
    during playback; confirmed user speech (gated by :class:`InterruptGate`)
    aborts TTS + the LLM stream and the spoken-prefix is committed to history
    (G1/G4/G5). Otherwise playback is half-duplex (mic gated).

    ``speaker_id`` + ``clip`` (G7): when both are supplied the clip is embedded and
    resolved to an enrolled name before streaming, so memory is per-person. The
    barge-in chaining loops call this again with the freshly *captured* clip, so a
    different person interrupting is re-identified for the follow-up turn."""
    _set_speaker(brain, speaker_id, clip)
    metrics = TurnMetrics()
    metrics.note(transcript=text)
    bus.transcript(text)
    chunker = SentenceChunker()
    voiced_any = False  # first-audio (R3-7): mark the instant the first audio plays
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
                    # R3-7: latency from turn start to the model's first token.
                    metrics.mark("llm_first_token")
                    bus.state("llm_response")
                    first = False
                bus.response(delta, final=False)
                for sentence in chunker.feed(delta):
                    spoken_n, res = _voice_sentence(cfg, tts, gate, ctx, sentence)
                    if spoken_n and not voiced_any:
                        metrics.mark("first_audio")  # R3-7: time-to-first-audio
                        voiced_any = True
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
                    if spoken_n and not voiced_any:
                        metrics.mark("first_audio")
                        voiced_any = True
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
    # R3-7: log + publish to the bus + (when telemetry is on) the JSON-lines file,
    # the session aggregator, and the optional OpenTelemetry span.
    metrics.emit(_session_sink(cfg))
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


@dataclass
class _Captured:
    """A push-to-talk capture: the transcript plus the raw clip it came from (G7).

    The clip is kept (not discarded after STT) so the speaker-ID pipeline can embed
    the same audio and key conversation memory per-person."""

    text: str = ""
    clip: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


def _capture_ptt(cfg: Config, stt: object, gate: audio.MicGate) -> _Captured:
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
        return _Captured()
    bus.state("stt")
    text = str(stt.transcribe(clip, cfg.sample_rate).text)  # type: ignore[attr-defined]
    return _Captured(text=text, clip=clip)


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
    speaker_id: object | None = None,
) -> None:
    """Run one push-to-talk or typed turn (chaining barge-in follow-ups).

    For spoken turns the recorded clip is also speaker-identified (G7) so per-person
    memory is keyed correctly; typed turns set the speaker to ``None`` (guest)."""
    if typed_text is not None:
        text = typed_text.strip()
        clip: np.ndarray | None = None
    else:
        captured = _capture_ptt(cfg, stt, gate)
        text, clip = captured.text, captured.clip
    if not text:
        return
    if cfg.debug and typed_text is None:
        tts.speak("recorded")
    active_vad = vad if typed_text is None else None
    while text:
        result = _respond(
            cfg, brain, tts, gate, text, vad=active_vad, speaker_id=speaker_id, clip=clip
        )
        if not result.interrupted:
            return
        # User barged in: their captured audio is handed straight to the streaming
        # transcriber (no from-scratch re-transcribe) as the next turn (R2-6). The
        # interrupter may be a different person, so the captured clip is re-identified
        # on the next _respond (G7).
        clip = result.captured
        text = _transcribe_barge_in(cfg, stt, result.captured) if stt is not None else ""


def run_wake_loop(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    stt: object,
    *,
    speaker_id: object | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Always-listen: on the wake phrase, capture (VAD), respond, then follow up.

    The VAD-recorded utterance clip is speaker-identified (G7) before responding so
    memory is per-person; each barge-in follow-up re-identifies its captured clip.

    ``stop`` (optional) makes the loop cleanly tearable-down: it is checked between
    iterations and passed to :func:`audio.listen_for_wake` so a GUI-driven loop
    exits promptly while it sits idle waiting for the wake phrase. ``None`` keeps the
    classic run-forever terminal behaviour (Ctrl-C to quit)."""
    from .aec import make_voiceprocessing_capture
    from .denoise import make_denoiser
    from .turn import make_turn_analyzer
    from .vad import SileroVad
    from .wake import WakeUnavailable, WakeWord

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
        if stop is not None and stop.is_set():
            bus.state("idle")
            return 0
        bus.state("listening", cfg.wake_phrase)
        try:
            fired = audio.listen_for_wake(wake, cfg.sample_rate, stop=stop)
        except WakeUnavailable as exc:
            # Construction/predict failed (e.g. an openwakeword API/version
            # mismatch). Log ONCE with a clear hint and stop — never spin the
            # same error forever.
            msg = f"wake word disabled: {exc}"
            print(msg, file=sys.stderr)
            bus.log(msg, "error")
            bus.state("idle")
            return 2
        if not fired:
            bus.state("idle")
            return 0  # stop requested while idle-listening for the wake word
        bus.wake()
        gate.gate()
        _play(chimes.chime_listening())
        gate.release()
        follow_up = False
        while True:
            if stop is not None and stop.is_set():
                bus.state("idle")
                return 0
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
                cfg,
                brain,
                tts,
                gate,
                text,
                vad=barge_vad,
                source=hw_source,
                denoiser=denoiser,
                speaker_id=speaker_id,
                clip=clip,
            )
            while result.interrupted:
                # Hand the captured barge-in audio straight to the transcriber as
                # the next turn (no from-scratch re-transcribe, R2-6), and keep
                # chaining if the new reply is itself interrupted. The captured clip
                # is re-identified (a different person may have barged in) (G7).
                barge_clip = result.captured
                text = _transcribe_barge_in(cfg, stt, result.captured)
                if not text:
                    break
                result = _respond(
                    cfg,
                    brain,
                    tts,
                    gate,
                    text,
                    vad=barge_vad,
                    source=hw_source,
                    denoiser=denoiser,
                    speaker_id=speaker_id,
                    clip=barge_clip,
                )
            follow_up = True  # subsequent turns are short follow-ups (no re-wake)


def _announce_browser_url(url: str) -> None:
    """Show the GUI URL prominently AND auto-open it in the default browser.

    So ``--browser`` both opens the page and prints a clickable link (handy when
    the auto-open is suppressed, e.g. on a headless host or over SSH).
    """
    import webbrowser

    print(f"\n▶ Open in your browser:  {url}\n  (auto-opening… Ctrl-C to quit)")
    with contextlib.suppress(Exception):  # headless / no browser is fine
        webbrowser.open(url)


class _WakeController:
    """Start/stop the server-side wake loop + run one-shot push-to-talk for the GUI.

    All three GUI voice actions (``wake_start`` / ``wake_stop`` / ``ptt``) are driven
    through here so they share one lock and can never double-start the loop or run a
    push-to-talk turn while the always-listen loop is active. The wake loop runs in a
    daemon thread with a :class:`threading.Event` stop flag; a one-shot push-to-talk
    runs in its own short-lived worker. Thread-safe: every transition takes ``_lock``.
    """

    def __init__(
        self,
        cfg: Config,
        brain: Brain,
        tts: TTSRouter,
        gate: audio.MicGate,
        stt: object,
        *,
        speaker_id: object | None = None,
    ) -> None:
        self._cfg = cfg
        self._brain = brain
        self._tts = tts
        self._gate = gate
        self._stt = stt
        self._speaker_id = speaker_id
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._ptt_busy = False

    def start_wake(self) -> None:
        """Start the always-listen wake loop in a daemon thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                bus.log("wake loop already listening")
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._wake_target, args=(self._stop,), daemon=True
            )
            self._thread.start()
        bus.log(f'wake loop started — say "{self._cfg.wake_phrase}"')

    def stop_wake(self) -> None:
        """Signal the wake loop to exit between iterations and return to idle."""
        with self._lock:
            stop, thread = self._stop, self._thread
            self._stop, self._thread = None, None
        if stop is None or thread is None or not thread.is_alive():
            bus.log("wake loop not running")
            bus.state("idle")
            return
        stop.set()
        bus.log("wake loop stopping")
        bus.state("idle")

    def push_to_talk(self) -> None:
        """Run one server-side push-to-talk capture + respond in a worker thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                bus.log("push-to-talk unavailable while the wake loop is listening", "error")
                return
            if self._ptt_busy:
                bus.log("push-to-talk already capturing", "error")
                return
            self._ptt_busy = True
        threading.Thread(target=self._ptt_target, daemon=True).start()

    def mic_test(self) -> None:
        """Run a short server-side mic capture + report a verdict, in a worker thread.

        Runs regardless of wake state: if the wake loop is holding the input device
        it is stopped first (so the test owns the mic), then restarted afterwards so
        the user lands back where they were. Never blocks the HTTP handler."""
        threading.Thread(target=self._mic_test_target, daemon=True).start()

    def _mic_test_target(self) -> None:
        # Free the device if the wake loop currently holds it; remember to restore.
        with self._lock:
            was_listening = self._thread is not None and self._thread.is_alive()
        thread = self._thread
        if was_listening:
            self.stop_wake()
            if thread is not None:
                thread.join(timeout=3.0)
        _run_mic_test(self._cfg)  # capture + publish verdict (single source of truth)
        if was_listening:
            self.start_wake()  # restore the always-listen loop we paused for the test

    def _wake_target(self, stop: threading.Event) -> None:
        try:
            run_wake_loop(
                self._cfg,
                self._brain,
                self._tts,
                self._gate,
                self._stt,
                speaker_id=self._speaker_id,
                stop=stop,
            )
        except Exception as exc:  # never let a mic/model failure kill the thread silently
            log.error("wake loop error: %s", exc)
            bus.log(f"wake loop error: {exc}", "error")
        finally:
            bus.state("idle")

    def _ptt_target(self) -> None:
        try:
            captured = _capture_ptt(self._cfg, self._stt, self._gate)
            if not captured.text:
                # macOS: a blank capture is most often an ungranted mic permission.
                bus.log(
                    "no audio captured — check the microphone permission "
                    "(macOS prompts the Terminal/app on first capture)",
                    "error",
                )
                return
            _respond(
                self._cfg,
                self._brain,
                self._tts,
                self._gate,
                captured.text,
                speaker_id=self._speaker_id,
                clip=captured.clip,
            )
        except Exception as exc:
            log.error("push-to-talk error: %s", exc)
            bus.log(f"push-to-talk error: {exc}", "error")
        finally:
            with self._lock:
                self._ptt_busy = False
            bus.state("idle")


def _run_mic_test(cfg: Config) -> None:
    """Capture a short mic sample, publish the verdict, and log it (standalone).

    Used by the GUI ``mic_test`` action when no wake controller exists (voice off)
    — so the user can still diagnose the mic. :func:`audio.mic_test` is defensive,
    but the outer guard keeps a surprise from killing the worker thread silently.
    """
    bus.log("testing microphone (1.5s)…")
    try:
        result = audio.mic_test(cfg.sample_rate)
    except Exception as exc:  # mic_test is defensive; belt-and-braces
        log.error("mic test error: %s", exc)
        result = audio.MicTestResult(ok=False, verdict="error", message=f"microphone error: {exc}")
    bus.mic_result(ok=result.ok, verdict=result.verdict, message=result.message, level=result.level)
    bus.log(("✓ " if result.ok else "✗ ") + result.message, "info" if result.ok else "error")
    bus.state("idle")


def _voice_status(cfg: Config, stt: object | None) -> tuple[bool, str]:
    """Whether the server can actually do GUI wake / push-to-talk, and why not.

    Voice is available only when (1) an STT engine is loaded, (2) the trained wake
    model file exists, and (3) a microphone is usable. Returns ``(available, hint)``
    where ``hint`` is a short human reason shown in the GUI when unavailable (empty
    when available)."""
    from .wake import WakeWord

    if stt is None:
        return False, "Voice off — relaunch with `./mstt --browser --wake` and grant mic access"
    if not WakeWord.from_config(cfg).available():
        return (
            False,
            f'Wake model missing ({cfg.wake_model_path}) — train "{cfg.wake_phrase}" first',
        )
    if not audio.mic_available():
        return False, "No microphone detected — connect one and grant mic access, then relaunch"
    return True, ""


def _run_browser(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    stt: object | None,
    *,
    wake: bool,
    port: int = 8765,
    speaker_id: object | None = None,
) -> int:
    """Serve the web GUI; turns are driven from the page (and optional wake loop).

    Typed turns from the page are guest (no audio); the browser-audio session, the
    push-to-talk one-shots, and the wake loop are speaker-identified (G7) when
    ``speaker_id`` is set. When the voice pipeline is available (STT loaded + wake
    model present + a usable mic) the GUI's Start-Wake / Push-to-Talk buttons drive
    the server-side loop; otherwise they are reported as disabled (honest UI)."""
    from .webui import WebUI

    voice_available, voice_hint = _voice_status(cfg, stt)
    controller = (
        _WakeController(cfg, brain, tts, gate, stt, speaker_id=speaker_id)
        if voice_available and stt is not None
        else None
    )

    def on_turn(text: str) -> None:
        # Typed-only from the page in this build -> guest bucket (no clip to embed).
        _respond(cfg, brain, tts, gate, text)

    def on_action(name: str, _data: dict) -> None:
        if name == "reset":
            brain.reset()
            bus.log("conversation reset")
        elif name == "list_voices":
            bus.log("voices: " + ", ".join(VOICE_PRESETS))
        elif name == "mic_test":
            # Diagnostic: capture ~1.5s and report a clear verdict. Runs regardless
            # of voice/wake state — it's *most* useful when voice is off (to find
            # out why). With a controller it pauses/restores the wake loop; without
            # one it runs a standalone capture in a worker (never block the handler).
            if controller is not None:
                controller.mic_test()
            else:
                threading.Thread(target=lambda: _run_mic_test(cfg), daemon=True).start()
        elif name in {"wake_start", "wake_stop", "ptt"}:
            if controller is None:
                bus.log(f"'{name}' unavailable: {voice_hint}", "error")
            elif name == "wake_start":
                controller.start_wake()
            elif name == "wake_stop":
                controller.stop_wake()
            else:  # ptt
                controller.push_to_talk()
        else:
            bus.log(f"unknown action '{name}'", "error")

    # Real browser audio (R2-5): when an STT engine is available, the page can
    # stream mic PCM over a same-origin WebSocket and play TTS PCM back, driving the
    # full pipeline (not just state). Disabled (state/transcript only) without STT.
    def _audio_session(transport: object) -> None:
        from .net_loop import run_transport_session

        run_transport_session(
            transport,  # type: ignore[arg-type]
            cfg,
            brain,
            tts,
            stt,  # type: ignore[arg-type]
            speaker_id=speaker_id,
        )

    on_audio_session = _audio_session if stt is not None else None
    ui = WebUI(
        cfg,
        on_turn,
        on_action,
        port=port,
        on_audio_session=on_audio_session,
        voice_available=voice_available,
        voice_hint=voice_hint,
    )
    bus.state("idle")
    # ``--wake`` on launch means "start listening immediately" — but the GUI can
    # still stop/restart it via the Start-Wake button (same controller).
    if wake and controller is not None:
        controller.start_wake()
    elif wake and controller is None:
        bus.log(f"--wake requested but voice is off: {voice_hint}", "error")
    _announce_browser_url(ui.url())
    try:
        ui.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def _make_session_runner(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    stt: object,
    realtime_brain: object | None,
    speaker_id: object | None = None,
) -> Any:
    """Return the ``on_session(transport)`` callback for a network/telephony client.

    Picks the brain at the transport boundary (R3-5): when a keyed realtime brain
    is present, the client's audio is bridged straight to the speech-to-speech
    endpoint (cascade bypassed); otherwise it runs the full STT->LLM->TTS
    :func:`run_transport_session` loop (with speaker ID, G7). Same callable shape.
    """
    if realtime_brain is not None:
        from .realtime import run_realtime_session

        def _realtime_session(transport: object) -> None:
            run_realtime_session(transport, cfg)  # type: ignore[arg-type]

        return _realtime_session

    from .net_loop import run_transport_session

    def _cascade_session(transport: object) -> None:
        run_transport_session(
            transport,  # type: ignore[arg-type]
            cfg,
            brain,
            tts,
            stt,  # type: ignore[arg-type]
            speaker_id=speaker_id,
        )

    return _cascade_session


def _run_websocket_server(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    stt: object | None,
    realtime_brain: object | None = None,
    speaker_id: object | None = None,
) -> int:
    """Serve the WebSocket audio transport (R2-5): bridge remote clients to the pipeline.

    Each accepted client (satellite or browser) gets a :class:`WebSocketTransport`
    that becomes the mic source + audio sink for a full :func:`run_transport_session`
    turn loop — the same STT/LLM/TTS stages, just with the device boundary on the
    wire — or, with a keyed realtime brain, a speech-to-speech session (R3-5). Needs
    the ``transport`` extra; prints a clear message if it is missing.
    """
    from .ws_transport import serve_websocket

    if stt is None:
        from .stt import make_transcriber

        stt = make_transcriber(cfg)

    on_session = _make_session_runner(cfg, brain, tts, stt, realtime_brain, speaker_id)
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


def _run_telephony_server(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    stt: object | None,
    realtime_brain: object | None = None,
) -> int:
    """Serve the Twilio Media Streams endpoint (R3-9): answer phone calls.

    Point a Twilio ``<Stream>`` at ``ws://TELEPHONY_HOST:TELEPHONY_PORT/``; each
    call gets a :class:`~my_stt_tts.telephony.TwilioTransport` (μ-law 8 kHz <->
    PCM, 8k<->16k resample) bridged into the same pipeline as the WebSocket path —
    cascade by default, speech-to-speech with a keyed realtime brain. Needs the
    ``transport`` extra (websockets); prints a clear message if it is missing.
    """
    from .telephony import serve_twilio

    if stt is None and realtime_brain is None:
        from .stt import make_transcriber

        stt = make_transcriber(cfg)

    on_session = _make_session_runner(cfg, brain, tts, stt, realtime_brain)
    print(
        f"my-stt-tts Twilio telephony → ws://{cfg.telephony_host}:{cfg.telephony_port}/"
        "  (point a Twilio <Stream> here; Ctrl-C to quit)"
    )
    try:
        serve_twilio(
            on_session,
            host=cfg.telephony_host,
            port=cfg.telephony_port,
            sample_rate=cfg.sample_rate,
        )
    except RuntimeError as exc:  # missing 'transport' extra
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def _run_webrtc_server(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    stt: object | None,
    *,
    port: int = 8765,
    speaker_id: object | None = None,
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
    return _run_browser(cfg, brain, tts, gate, stt, wake=False, port=port, speaker_id=speaker_id)


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
        if args.preflight:
            from .preflight import preflight_main

            return preflight_main(cfg)
        cfg.validate()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    brain = Brain(cfg)
    tts = TTSRouter(cfg)
    gate = audio.MicGate(cfg.mic_gate_tail_seconds)

    # The realtime brain (R3-5) does its own ASR, so it needs no local STT engine.
    realtime_selected = cfg.brain_mode == "realtime" and bool(cfg.realtime_api_key)
    needs_stt = not realtime_selected and (
        cfg.transport in ("websocket", "webrtc")
        or cfg.telephony
        or (args.browser and args.browser_audio)
        or (not args.type_mode and args.text is None and (not args.browser or args.wake))
    )
    stt: object | None = None
    if needs_stt:
        from .stt import make_transcriber  # heavy (MLX); only needed for mic modes

        stt = make_transcriber(cfg)

    # R3-5: a realtime speech-to-speech brain (key-gated). When selected AND keyed,
    # it drives the loop at the audio level (bypassing this cascade Brain); None
    # falls back to the cascade with a clear log line.
    from .realtime import make_realtime_brain

    realtime_brain = make_realtime_brain(cfg)

    # G7: build the speaker-ID pipeline ONCE (or None). Gated + defensive — it loads
    # nothing and adds no latency unless speaker ID is enabled AND voices are
    # enrolled AND speechbrain is installed. Threaded into the live loops so the
    # recorded clip resolves to a person and memory is per-speaker.
    from .speaker_pipeline import SpeakerPipeline

    speaker_id = SpeakerPipeline.from_config(cfg)

    if not args.browser:
        print(settings_text(cfg))
    try:
        if cfg.telephony:  # R3-9: answer Twilio phone calls over the WS transport
            return _run_telephony_server(cfg, brain, tts, stt, realtime_brain)
        if cfg.transport == "websocket":
            return _run_websocket_server(cfg, brain, tts, stt, realtime_brain, speaker_id)
        if cfg.transport == "webrtc":
            return _run_webrtc_server(cfg, brain, tts, stt, port=args.port, speaker_id=speaker_id)
        if args.browser:
            return _run_browser(
                cfg, brain, tts, gate, stt, wake=args.wake, port=args.port, speaker_id=speaker_id
            )
        if args.wake:
            return run_wake_loop(cfg, brain, tts, gate, stt, speaker_id=speaker_id)
        _run_terminal_modes(args, cfg, brain, tts, gate, stt, speaker_id=speaker_id)
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
    *,
    speaker_id: object | None = None,
) -> None:
    """One typed turn, the interactive typed loop, or the push-to-talk mic loop.

    Push-to-talk turns are speaker-identified (G7); typed turns are guest (None)."""
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
        run_turn(cfg, brain, tts, gate, stt=stt, vad=ptt_vad, speaker_id=speaker_id)
        if args.once:
            return


if __name__ == "__main__":
    raise SystemExit(main())
