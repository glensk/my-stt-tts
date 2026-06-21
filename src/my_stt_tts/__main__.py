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
import shutil
import sys
import threading
import time
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
    model_label,
)
from .events import bus, install_log_bridge
from .interrupt import InterruptGate, make_interrupt_predictor
from .metrics import TurnMetrics
from .speaker_pipeline import resolve_turn_speaker
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

# Turn-source tags threaded onto bus.transcript(...) so the GUI can label the user
# bubble with how the turn was entered (e.g. "YOU · push-to-talk").
SOURCE_TYPED = "typed"
SOURCE_PTT = "push_to_talk"
SOURCE_WAKE = "wake"
SOURCE_LIVE_AUDIO = "live_audio"


def _session_sink(cfg: Config) -> Any:
    """Return the process-wide telemetry sink (built once from config), or None."""
    global _SESSION_SINK, _SESSION_SINK_BUILT  # noqa: PLW0603 — one-time process singleton
    if not _SESSION_SINK_BUILT:
        from .metrics import make_sink

        _SESSION_SINK = make_sink(cfg)
        _SESSION_SINK_BUILT = True
    return _SESSION_SINK


def _signal_mic_confirmed(clip: np.ndarray, sample_rate: int) -> None:
    """Emit a "mic working" signal once a real (non-silent) capture is confirmed.

    A successful wake / push-to-talk capture proves the microphone path works, so we
    publish a ``mic_result(ok=True)`` — the same event the GUI's "Test mic" uses — so
    the page can hide the macOS permission hint the instant audio is confirmed,
    without the user having to run a separate mic test. Best-effort + no-op on an
    empty/silent clip (that would not prove anything)."""
    arr = np.asarray(clip, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if arr.size == 0 or peak < audio._SILENCE_PEAK:  # noqa: SLF001 — shared silence floor
        return
    level = int(round(min(1.0, peak) * 100))
    duration = round(arr.size / sample_rate, 1) if sample_rate else 0.0
    with contextlib.suppress(Exception):
        bus.mic_result(
            ok=True,
            verdict="ok",
            message=f"Microphone confirmed working — captured {duration}s at level {level}%",
            level=level,
            permission=audio.mic_permission_status(),
        )


def debug_audio_enabled(cfg: Config, *, browser: bool = False) -> bool:
    """Whether the heavy audio debug instrument is on for this run.

    ``cfg.debug_audio`` is tri-state: an explicit ``DEBUG_AUDIO`` env (``True``/
    ``False``) always wins; ``None`` (the default) means "auto" — ON under
    ``--browser`` so the GUI EVENT LOG shows the trace out of the box, OFF otherwise.
    """
    if cfg.debug_audio is not None:
        return cfg.debug_audio
    return browser


def wake_debug_capture_enabled(cfg: Config, *, browser: bool = False) -> bool:
    """Whether the wake-debug recorder dumps the first seconds of the wake stream.

    ``cfg.wake_debug_capture`` is tri-state: an explicit ``WAKE_DEBUG_CAPTURE`` env
    always wins; ``None`` (the default) means "auto" — ON whenever the audio debug
    instrument is on (so under ``--browser`` it just works and the saved WAV path is
    logged), OFF otherwise. So a user diagnosing a never-firing wake word gets the
    recording without flipping an extra switch.
    """
    if cfg.wake_debug_capture is not None:
        return cfg.wake_debug_capture
    return debug_audio_enabled(cfg, browser=browser)


class _AudioDebug:
    """The capture/VAD/wake/STT debug instrument (the GUI "debugger").

    When enabled (:func:`debug_audio_enabled`) every event is logged to **stderr**
    AND published as a ``bus.debug`` event so it appears in the GUI EVENT LOG —
    making WHERE audio is lost (sample rate, #samples, rms/peak, VAD/endpoint
    decisions, wake max-score, STT input length + transcript) obvious. A no-op when
    disabled, so the loops can call it unconditionally. VAD-frame traces are
    rate-limited to one line per ~25 frames (≈0.8 s) so the log isn't flooded.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._frame_n = 0

    def __call__(self, stage: str, **fields: Any) -> None:
        if not self.enabled:
            return
        if stage in ("vad_frame", "wake_score"):
            self._frame_n += 1
            # These fire every ~80 ms audio frame — heartbeat 1-in-50 so the log isn't a
            # wall of lines, but ALWAYS surface a wake score climbing toward detection
            # (>= 0.3) so a near-miss stays visible.
            notable = stage == "wake_score" and float(fields.get("score") or 0.0) >= 0.3
            if not notable and self._frame_n % 50 != 1:
                return
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        msg = f"[audio:{stage}] {kv}".rstrip()
        print(msg, file=sys.stderr)
        with contextlib.suppress(Exception):
            bus.debug(msg, stage=stage, **fields)

    # Maps a raw GUI action name to the EXACT button label the user clicked, so the
    # EVENT LOG reads "clicked PUSH-TO-TALK" instead of the opaque "action:ptt". The
    # structured stage="action:<name>" field is still carried for machine use.
    _ACTION_LABELS = {
        "ptt": "clicked PUSH-TO-TALK",
        "wake_start": "clicked START WAKE",
        "wake_stop": "clicked STOP WAKE",
        "mic_test": "clicked TEST SERVER MIC",
        "mic_record_replay": "clicked RECORD & PLAY · SERVER",
        "mic_check": "clicked MIC CHECK",
        "play_recording": "clicked PLAY RECORDING",
        "live_audio": "clicked LIVE AUDIO",
        "reset": "clicked RESET",
        "wake_test": "clicked WAKE TEST",
        "wake_gain_sweep": "clicked WAKE GAIN SWEEP",
        "score_clip": "clicked SCORE CLIP",
        "score_histogram": "clicked SCORE HISTOGRAM",
        "fa_eval": "clicked FA EVAL",
        "train_verifier": "clicked TRAIN VERIFIER",
        "spectrogram": "clicked SPECTROGRAM",
        "turn": "submitted a turn",
    }

    @classmethod
    def action_label(cls, name: str) -> str:
        """The human EVENT-LOG line for a GUI action (unknown -> "clicked <NAME>")."""
        return cls._ACTION_LABELS.get(name, "clicked " + name.upper())

    def action(self, name: str, **fields: Any) -> None:
        """Log a GUI action with its friendly button label (EVENT LOG) + structured stage.

        The human message uses the EXACT GUI button label (e.g. "clicked
        PUSH-TO-TALK") so the EVENT LOG is readable, while the event still carries the
        machine-stable ``stage="action:<name>"`` field. A huge ``pcm`` payload (the
        browser wake-test clip) is dropped from the logged fields so the log isn't
        flooded with thousands of floats. A no-op when audio debugging is off.
        """
        if not self.enabled:
            return
        fields = {k: v for k, v in fields.items() if k != "pcm"}
        stage = f"action:{name}"
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        msg = f"{self.action_label(name)} {kv}".rstrip()
        print(f"[audio:{stage}] {msg}", file=sys.stderr)
        with contextlib.suppress(Exception):
            bus.debug(msg, stage=stage, **fields)


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
        f"  brain      {blue}{model_label(cfg.llm_provider, cfg.llm_model)}{reset}"
        f"  (deep: {cfg.llm_model_deep})"
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
        f"  wake       phrase {blue}{cfg.wake_phrase}{reset}  threshold {blue}{cfg.wake_threshold}{reset}"
        f"  sensitivity {blue}{cfg.sensitivity_for(cfg.wake_phrase)}{reset}"
        f"{' (master)' if cfg.wake_sensitivity_set else ' (threshold is master)'}"
        f"  phases {blue}{cfg.wake_phases}{reset}"
        f"  window {blue}{cfg.wake_window}{reset}  refractory {blue}{cfg.wake_refractory}{reset}"
        f"  calibration {blue}{cfg.wake_calibration}{reset}"
        f"  mic-gain {blue}{cfg.mic_gain}{reset}"
        f"  wake-gain {blue}{cfg.wake_gain}{reset}"
        f"  noise-corpus {cfg.noise_corpus_dir}"
        f"  model {cfg.wake_model_path}  exists {os.path.isfile(cfg.wake_model_path)}"
        f"  available [{', '.join(available_wake_words()) or 'none — see wakewords/WAKEWORD.md'}]",
        f"  kws        enabled {blue}{cfg.kws_enabled}{reset}  (OR'd detector for CUSTOM words;"
        f" official words stay openWakeWord-only)  boost {cfg.kws_boost}  threshold {cfg.kws_threshold}",
        f"  fewshot    enabled {blue}{cfg.fewshot_wake_enabled}{reset}  (few-shot ENROLLED OR'd"
        f" detector for CUSTOM words; enroll via scripts/enroll_wakeword.py)"
        f"  threshold {blue}{cfg.fewshot_threshold}{reset}  patience {blue}{cfg.fewshot_patience}{reset}",
        f"  speaker-id {blue}{cfg.speaker_id_enabled}{reset}  enroll {cfg.enroll_dir}"
        f"  threshold {cfg.speaker_threshold}  margin {cfg.speaker_margin}"
        f"  diarize {blue}{cfg.speaker_diarize_enabled}{reset}"
        f"  speakers {cfg.diarize_num_speakers if cfg.diarize_num_speakers > 0 else 'auto'}",
        f"  agent      trigger '{cfg.agent_trigger}'  workspace {blue}{agent_ws}{reset}"
        f"  model {cfg.agent_model}",
        f"  music      enabled {blue}{cfg.music_enabled}{reset}  player {blue}{cfg.music_player}{reset}"
        f"  volume {cfg.music_volume if cfg.music_volume is not None else 'default'}"
        f"  playback {blue}{cfg.music_playback}{reset}",
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
        "--skip-audio-preflight",
        dest="skip_audio_preflight",
        action="store_true",
        default=None,
        help="Bypass the startup audio preflight HARD STOP for mic modes (power users): "
        "run even when capture can't deliver 16 kHz or the mic queue overflows.",
    )
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
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the noise×SNR wake benchmark for the selected word (saved positives + "
        "the negative corpus mixed with the noise corpus at each SNR), print a per-SNR "
        "miss@target-FA table, write JSON artifacts under debug/benchmark/, then exit.",
    )
    parser.add_argument(
        "--benchmark-snr",
        help="Comma-separated SNRs (dB) for --benchmark, e.g. 'clean,10,5' "
        "(default clean,10,5). 'clean' (or 'none') = the noise-free condition.",
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
    if args.skip_audio_preflight:
        cfg.skip_audio_preflight = True
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


def maybe_handle_music(cfg: Config, tts: TTSRouter, gate: audio.MicGate, text: str) -> bool:
    """Handle ``text`` locally as a music command, returning True if it was one.

    This is the cross-brain LOCAL INTENT ROUTER: it runs in the turn path BEFORE
    the LLM so "play We Will Rock You by Queen" actually plays the song for every
    brain (the default ``claude-cli`` does not do our tool-calling, so a tool alone
    would not work). On a match it resolves + plays / stops / pauses / resumes via
    :mod:`my_stt_tts.music`, speaks a short confirmation through the normal TTS
    path, publishes a bus state/log so the GUI shows "▶ Playing: <title>", and the
    caller then SKIPS the LLM. Disabled (always returns False) when
    ``cfg.music_enabled`` is off. Never raises — a failure degrades to a spoken
    reason and a handled turn so the music word never reaches the LLM as "I can't
    play music"."""
    if not cfg.music_enabled:
        return False
    from . import music

    intent = music.match_music_intent(text)
    if intent is None:
        return False
    action = intent["action"]
    player = music.get_player(player=cfg.music_player, volume=cfg.music_volume)
    bus.transcript(text, source=SOURCE_TYPED)
    if action == "play":
        query = intent.get("query") or "popular music playlist"
        bus.state("music_search", query)
        result = player.play(query)
        if result.ok:
            bus.state("music_playing", result.title)
            bus.log(f"▶ Playing: {result.title}")
            bus.music("playing", title=result.title, video_id=result.video_id, url=result.url)
            _music_respond(
                cfg, tts, gate, f"▶ Playing: {result.title}.", f"Playing {result.title}."
            )
        else:
            bus.log(result.reason, "error")
            _music_respond(cfg, tts, gate, result.reason, result.reason)
    elif action == "stop":
        was = player.stop()
        bus.log("⏹ Stopped the music" if was else "no music was playing")
        if was:
            bus.music("stopped")
            _music_respond(cfg, tts, gate, "⏹ Stopped the music.", "Stopped the music.")
        else:
            _music_respond(
                cfg, tts, gate, "Nothing is playing right now.", "Nothing is playing right now."
            )
    elif action == "pause":
        ok = player.pause()
        bus.log("⏸ Paused the music" if ok else "could not pause")
        if ok:
            snap = player.status()
            bus.music("paused", title=snap["title"], video_id=snap["video_id"], url=snap["url"])
            _music_respond(cfg, tts, gate, "⏸ Paused.", "Paused.")
        elif player.is_playing():
            _music_respond(cfg, tts, gate, "I can't pause this track.", "I can't pause this track.")
        else:
            _music_respond(
                cfg, tts, gate, "Nothing is playing right now.", "Nothing is playing right now."
            )
    elif action == "resume":
        ok = player.resume()
        bus.log("▶ Resumed the music" if ok else "could not resume")
        if ok:
            snap = player.status()
            bus.music("resumed", title=snap["title"], video_id=snap["video_id"], url=snap["url"])
            _music_respond(cfg, tts, gate, "▶ Resumed.", "Resumed.")
        else:
            _music_respond(
                cfg, tts, gate, "There's nothing to resume.", "There's nothing to resume."
            )
    bus.state("idle")
    return True


def _music_respond(
    cfg: Config, tts: TTSRouter, gate: audio.MicGate, display: str, spoken: str
) -> None:
    """Render an assistant bubble for a locally-handled music turn AND speak it.

    The music intent router answers WITHOUT the LLM, so it must emit the same bus
    surface a normal reply does or the transcript shows no assistant bubble (the
    original bug: music turns produced only a log line). Emits a brief
    ``llm_response`` state + a final :meth:`EventBus.response` carrying ``display``
    (with the ▶/⏹/⏸ glyph the page renders) and the active model label (so the page
    draws an "ASSISTANT · <model>" bubble), then speaks the glyph-free ``spoken``
    text through the normal half-duplex TTS path (the symbols must not be read
    aloud). The trailing ``idle`` state is set by the caller."""
    model = _model_label(cfg)
    bus.state("llm_response", model)
    bus.response(display, final=True, model=model)
    _speak(tts, gate, spoken)


def _music_action(name: str, *, player: str = "auto", volume: int | None = None) -> None:
    """Drive the shared player from a GUI button (``music_stop``/``pause``/``resume``).

    Mirrors the intent-router control path so a GUI button and a spoken "stop" act
    on the SAME process-wide player and publish the SAME structured ``music`` event
    + log line. Server-side mpv playback is unchanged — this only adds the control
    surface + the event the page listens for. Never raises (a missing player simply
    no-ops with a log)."""
    from . import music

    plr = music.get_player(player=player, volume=volume)
    if name == "music_stop":
        was = plr.stop()
        bus.log("⏹ Stopped the music" if was else "no music was playing")
        if was:
            bus.music("stopped")
    elif name == "music_pause":
        if plr.pause():
            snap = plr.status()
            bus.music("paused", title=snap["title"], video_id=snap["video_id"], url=snap["url"])
            bus.log("⏸ Paused the music")
        else:
            bus.log("could not pause")
    elif name == "music_resume":
        if plr.resume():
            snap = plr.status()
            bus.music("resumed", title=snap["title"], video_id=snap["video_id"], url=snap["url"])
            bus.log("▶ Resumed the music")
        else:
            bus.log("could not resume")


def _speak(tts: TTSRouter, gate: audio.MicGate, sentence: str) -> None:
    """Half-duplex speak one sentence (mic gated during playback). Legacy path."""
    spoken = strip_non_spoken(sentence)
    if not spoken:
        return
    gate.gate()
    tts.speak(spoken)  # lang auto-detected from the answer text
    gate.release()


def _model_label(cfg: Config) -> str:
    """Exact model + effort/size-tier label for the active brain (e.g.
    ``claude-cli / opus-4.8 xlarge``) for the UI.

    Carried on the ``llm_request`` state detail and the final ``response`` event so
    the page can show an "ASSISTANT · <model>" label. Delegates to
    :func:`config.model_label` so the bare CLI alias / pinned API id resolves to the
    precise marketing version and the claude-cli effort/size tier is appended."""
    return model_label(cfg.llm_provider, cfg.llm_model)


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
    turn_source: str = "",
) -> RespondResult:
    """Stream the LLM for ``text``, speak it, and publish live state to the bus.

    ``turn_source`` tags where the turn's text came from (``"typed"`` /
    ``"push_to_talk"`` / ``"wake"`` / ``"live_audio"``) so the GUI can label the
    user bubble (e.g. "YOU · push-to-talk"); it rides the published transcript.

    When ``cfg.barge_in`` is enabled and a ``vad`` is supplied, the mic stays live
    during playback; confirmed user speech (gated by :class:`InterruptGate`)
    aborts TTS + the LLM stream and the spoken-prefix is committed to history
    (G1/G4/G5). Otherwise playback is half-duplex (mic gated).

    ``speaker_id`` + ``clip`` (G7): when both are supplied the clip is embedded and
    resolved to an enrolled name before streaming, so memory is per-person. The
    barge-in chaining loops call this again with the freshly *captured* clip, so a
    different person interrupting is re-identified for the follow-up turn."""
    # G7+: resolve who is speaking FIRST (before the music router + LLM) so within-turn
    # diarization can DROP a turn that is only unknown voices / background TV — that
    # chatter must never drive a command. Emits per-segment speaker events itself when
    # diarization split the turn.
    turn_speakers = resolve_turn_speaker(speaker_id, clip, bus=bus)  # type: ignore[arg-type]
    if turn_speakers.should_drop:
        bus.log("dropped a turn with no enrolled speaker (only unknown / background voices)")
        bus.state("idle")
        return RespondResult()
    brain.set_speaker(turn_speakers.routed)
    # LOCAL INTENT ROUTER (cross-brain): handle a music command ("play <song>",
    # "stop", …) HERE, before the LLM, so it works for every brain. On a match the
    # song is played + a short confirmation is spoken, and the turn ends without
    # ever asking the model (which would otherwise answer "I can't play music").
    if maybe_handle_music(cfg, tts, gate, text):
        return RespondResult()
    metrics = TurnMetrics()
    metrics.note(transcript=text)
    # When diarization split the turn, emit a LABELLED transcript per known segment;
    # otherwise the single plain transcript (today's behaviour).
    if not turn_speakers.emit_transcripts(text, bus=bus, source=turn_source):
        bus.transcript(text, source=turn_source)
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
    model_label = _model_label(cfg)
    stream = brain.stream(text)
    try:
        with metrics.stage("llm_tts"):
            bus.state("llm_request", model_label)
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
        # Carry the active model on the end-of-turn response so the page can label
        # the bubble "ASSISTANT · <model>" (the llm_request state detail has it too).
        bus.response("", final=True, model=model_label)
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


def _capture_ptt(
    cfg: Config,
    stt: object,
    gate: audio.MicGate,
    *,
    stop: threading.Event | None = None,
    debug: Any = None,
) -> _Captured:
    """Capture one push-to-talk turn, hands-free (VAD-driven, NO stdin).

    Used by both the terminal PTT loop and the GUI's server-side PTT button. It
    records until a natural pause via Silero VAD + the silence endpointer rather
    than waiting on an Enter key — the old ``record_push_to_talk`` blocked on
    ``input()``, which in the browser server worker has no interactive stdin and
    returned an empty clip (the "push-to-talk records nothing" bug). The recorder
    also pins the audio to 16 kHz mono (resampling a 48 kHz device) so STT gets a
    valid clip. ``debug`` (the audio instrument) surfaces capture/VAD/STT telemetry.
    """
    from .vad import SilenceEndpointer, SileroVad

    dbg = debug if debug is not None else (lambda *_a, **_k: None)
    gate.gate()
    _play(chimes.chime_listening())
    gate.release()
    bus.state("recording")
    vad = SileroVad(cfg.sample_rate, cfg.vad_threshold)
    endpointer = SilenceEndpointer(cfg.vad_silence_seconds, frame_seconds=512 / cfg.sample_rate)
    clip = audio.record_until_silence(
        cfg.sample_rate,
        vad,
        endpointer,
        max_seconds=cfg.max_record_seconds,
        stop=stop,
        on_debug=dbg,
    )
    gate.gate()
    _play(chimes.chime_done())
    gate.release()
    if clip.size == 0:
        bus.state("idle")
        return _Captured()
    _signal_mic_confirmed(clip, cfg.sample_rate)  # audio confirmed -> GUI can hide perm hint
    bus.state("stt")
    dbg("stt_input", **audio.capture_stats(clip, cfg.sample_rate))
    text = str(stt.transcribe(clip, cfg.sample_rate).text)  # type: ignore[attr-defined]
    dbg("stt_output", chars=len(text), transcript=text[:120])
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
        turn_source = SOURCE_TYPED
    else:
        captured = _capture_ptt(cfg, stt, gate)
        text, clip = captured.text, captured.clip
        turn_source = SOURCE_PTT
    if not text:
        return
    if cfg.debug and typed_text is None:
        tts.speak("recorded")
    active_vad = vad if typed_text is None else None
    while text:
        result = _respond(
            cfg,
            brain,
            tts,
            gate,
            text,
            vad=active_vad,
            speaker_id=speaker_id,
            clip=clip,
            turn_source=turn_source,
        )
        if not result.interrupted:
            return
        # User barged in: their captured audio is handed straight to the streaming
        # transcriber (no from-scratch re-transcribe) as the next turn (R2-6). The
        # interrupter may be a different person, so the captured clip is re-identified
        # on the next _respond (G7). Barge-in audio came from the live mic.
        clip = result.captured
        turn_source = SOURCE_LIVE_AUDIO
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
    debug: Any = None,
    fire_buffer: audio.WakeFireBuffer | None = None,
) -> int:
    """Always-listen: on the wake phrase, capture (VAD), respond, then follow up.

    The VAD-recorded utterance clip is speaker-identified (G7) before responding so
    memory is per-person; each barge-in follow-up re-identifies its captured clip.

    ``stop`` (optional) makes the loop cleanly tearable-down: it is checked between
    iterations and passed to :func:`audio.listen_for_wake` so a GUI-driven loop
    exits promptly while it sits idle waiting for the wake phrase. ``None`` keeps the
    classic run-forever terminal behaviour (Ctrl-C to quit). ``debug`` (the audio
    instrument) surfaces wake max-score + capture/VAD/STT telemetry."""
    from .aec import make_voiceprocessing_capture
    from .denoise import make_denoiser
    from .turn import make_turn_analyzer
    from .vad import SileroVad
    from .wake import WakeUnavailable, make_wake_detector

    dbg = debug if debug is not None else _AudioDebug(False)
    # The detector is openWakeWord for an official word, or openWakeWord-OR-sherpa-KWS for
    # a custom word when KWS is enabled + available (fires if EITHER fires). Both expose
    # the identical detect/reset/last_score surface listen_for_wake drives.
    wake = make_wake_detector(cfg)
    if not wake.available():
        msg = (
            f'Wake model not found at {cfg.wake_model_path}. Train "{cfg.wake_phrase}" first '
            "— see wakewords/WAKEWORD.md."
        )
        print(msg, file=sys.stderr)
        bus.log(msg, "error")
        return 2
    vad = SileroVad(cfg.sample_rate, cfg.vad_threshold)
    frame_seconds = 512 / cfg.sample_rate
    barge_vad = vad if cfg.barge_in != "off" else None
    # R3-4: capture through macOS VoiceProcessingIO (hardware AEC) when requested
    # and available; None means fall back to sounddevice. R3-6: pre-VAD denoiser.
    hw_source = make_voiceprocessing_capture(cfg)
    denoiser = make_denoiser(cfg)
    # Wake-debug recorder: dump the first ~N s of the EXACT 16 kHz frames the wake
    # model receives (so a never-firing wake word is diagnosable from one WAV). Built
    # once so it captures the first window after Start; goes inert once flushed.
    recorder: audio.WakeDebugRecorder | None = None
    # Arm the recorder whenever the audio debug instrument is actually live (the GUI
    # / --browser path turns it on), honoring an explicit WAKE_DEBUG_CAPTURE override.
    # Keyed off ``dbg.enabled`` — the real source of truth — NOT a re-derivation via
    # ``wake_debug_capture_enabled(cfg)`` that defaulted browser=False and so stayed
    # off in exactly the GUI session where a user is trying to diagnose a dead wake
    # word (the WAV was silently never written).
    arm_recorder = cfg.wake_debug_capture if cfg.wake_debug_capture is not None else dbg.enabled
    if arm_recorder:
        recorder = audio.WakeDebugRecorder(
            cfg.wake_debug_path, cfg.sample_rate, cfg.wake_debug_seconds, on_debug=dbg
        )
        bus.log(f"wake-debug recorder armed — WAV will be saved to {recorder.path}")
    print(f'Listening for "{cfg.wake_phrase}". Ctrl-C to quit.')
    while True:
        if stop is not None and stop.is_set():
            bus.state("idle")
            return 0
        bus.state("listening", cfg.wake_phrase)
        try:
            fired = audio.listen_for_wake(
                wake,
                cfg.sample_rate,
                gain=cfg.wake_gain,
                stop=stop,
                on_debug=dbg,
                recorder=recorder,
                fire_buffer=fire_buffer,
            )
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
        # Wake phrase fired: flash the GUI cue (bus.wake) AND play a distinct
        # detection chime so the user gets an unmistakable acknowledgement. This
        # runs once per detection (outside the follow-up loop), so it never beeps
        # repeatedly while a conversation is in progress. ``detector`` names which
        # detector fired ("oww"|"kws") so the GUI/log can show the OR'd path's winner.
        bus.wake(detector=getattr(wake, "last_detector", "oww") or "oww")
        gate.gate()
        _play(chimes.chime_wake())
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
            partial_src = SOURCE_LIVE_AUDIO if follow_up else SOURCE_WAKE
            clip = audio.record_turn(
                cfg.sample_rate,
                vad,
                analyzer,
                max_seconds=max_s,
                streamer=streamer,
                on_partial=lambda t, s=partial_src: bus.transcript(t, partial=True, source=s),
                source=hw_source,
                denoiser=denoiser,
            )
            if clip.size == 0:
                dbg("no_speech", **audio.capture_stats(clip, cfg.sample_rate))
                break  # silence -> back to listening for the wake word
            _signal_mic_confirmed(clip, cfg.sample_rate)  # audio confirmed -> hide perm hint
            dbg("stt_input", **audio.capture_stats(clip, cfg.sample_rate))
            text = (
                str(streamer.final().text).strip()  # type: ignore[union-attr]
                if streamer is not None
                else _transcribe(cfg, stt, clip)
            )
            dbg("stt_output", chars=len(text), transcript=text[:120])
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
                # The first utterance after the phrase fires is tagged "wake";
                # subsequent follow-ups (no re-wake) are live-mic audio.
                turn_source=SOURCE_WAKE if not follow_up else SOURCE_LIVE_AUDIO,
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
                    turn_source=SOURCE_LIVE_AUDIO,  # barge-in audio is from the live mic
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
        debug: Any = None,
    ) -> None:
        self._cfg = cfg
        self._brain = brain
        self._tts = tts
        self._gate = gate
        self._stt = stt
        self._speaker_id = speaker_id
        self._debug = debug if debug is not None else _AudioDebug(False)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._ptt_busy = False
        # Active-learning ring buffer (repo #5): the wake loop feeds every scored frame and
        # snapshots the window on each fire, so capture_last_fire can save the triggering
        # audio as a negative. Owned here (survives loop restarts) and read from the action
        # thread.
        self._fire_buffer = audio.WakeFireBuffer(cfg.sample_rate)

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
        """Run one server-side push-to-talk capture + respond in a worker thread.

        Works whether or not the always-listen wake loop is running: if the loop
        holds the mic it is PAUSED for the duration of the capture+respond and
        RESTORED afterwards (the same pattern :meth:`mic_record_replay` uses), so
        the user lands back listening. Re-entrancy is guarded — a second push while
        one is in flight is refused, so two captures never fight over the device."""
        with self._lock:
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

    def mic_record_replay(self) -> None:
        """Record ~3s from the mic and play it back, in a worker thread.

        Like :meth:`mic_test`, it owns the input device for the duration: if the wake
        loop holds the mic it is paused first and restored afterwards. Never blocks
        the HTTP handler."""
        threading.Thread(target=self._mic_record_replay_target, daemon=True).start()

    def mic_check(self) -> None:
        """Run the unified 2.0 s mic check (gain + level graph + saved WAV), in a worker.

        Like :meth:`mic_test`, it owns the input device for the duration: if the wake
        loop holds the mic it is paused first and restored afterwards. Never blocks
        the HTTP handler."""
        threading.Thread(target=self._mic_check_target, daemon=True).start()

    def _mic_test_target(self) -> None:
        self._with_paused_wake(lambda: _run_mic_test(self._cfg))

    def _mic_record_replay_target(self) -> None:
        self._with_paused_wake(lambda: _run_mic_record_replay(self._cfg))

    def _mic_check_target(self) -> None:
        self._with_paused_wake(lambda: _run_mic_check_server(self._cfg))

    def _with_paused_wake(self, fn: Any) -> None:
        """Run ``fn`` with exclusive use of the mic: pause the wake loop if it holds
        the device, run ``fn``, then restore the loop — so a mic diagnostic always
        owns the input device and the user lands back where they were.

        The hand-off is the subtle part. The wake loop's ``sounddevice`` InputStream
        is a context manager inside :func:`audio.listen_for_wake`; setting the stop
        event makes it return, which exits the ``with`` block and STOPS+CLOSES the
        stream. We must wait for the wake THREAD to actually join before opening the
        diagnostic's own stream — otherwise both hold the input device at once. Even
        after the close, macOS (CoreAudio/AUHAL) frees the device asynchronously, so
        opening immediately races and the diagnostic records near-silence
        (PaMacCore err=-50 / paramErr). A short ``mic_check_settle_s`` gap after the
        join lets the OS release the device before ``fn`` opens its capture stream;
        ``fn`` (via ``record_fixed``) also retries on a contention error as a backstop.
        """
        with self._lock:
            was_listening = self._thread is not None and self._thread.is_alive()
            thread = self._thread  # snapshot under the lock (stop_wake clears self._thread)
        if was_listening:
            self.stop_wake()  # signal the loop; its InputStream context-exit stops+closes it
            if thread is not None:
                thread.join(timeout=3.0)  # wait for the stream to be fully released
            # CoreAudio frees the input device asynchronously after close() returns;
            # settle before the diagnostic opens its own stream so they don't collide.
            settle = max(0.0, float(getattr(self._cfg, "mic_check_settle_s", 0.0)))
            if settle > 0:
                time.sleep(settle)
        fn()
        if was_listening:
            self.start_wake()  # restore the always-listen loop we paused for the diagnostic

    def last_fire_audio(self) -> np.ndarray | None:
        """The most-recent live wake fire's audio window (for capture_last_fire), or None."""
        return self._fire_buffer.last_fire

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
                debug=self._debug,
                fire_buffer=self._fire_buffer,
            )
        except Exception as exc:  # never let a mic/model failure kill the thread silently
            log.error("wake loop error: %s", exc)
            bus.log(f"wake loop error: {exc}", "error")
        finally:
            bus.state("idle")

    def _ptt_target(self) -> None:
        try:
            # Own the mic for the whole capture+respond: if the wake loop is
            # listening it is paused first and restored after, so push-to-talk works
            # even under --browser --wake (which auto-starts the loop).
            self._with_paused_wake(self._capture_and_respond)
        except Exception as exc:
            log.error("push-to-talk error: %s", exc)
            bus.log(f"push-to-talk error: {exc}", "error")
        finally:
            with self._lock:
                self._ptt_busy = False
            bus.state("idle")

    def _capture_and_respond(self) -> None:
        """One push-to-talk capture + respond (called with the mic already owned)."""
        captured = _capture_ptt(self._cfg, self._stt, self._gate, debug=self._debug)
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
            turn_source=SOURCE_PTT,
        )


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
    bus.mic_result(
        ok=result.ok,
        verdict=result.verdict,
        message=result.message,
        level=result.level,
        permission=result.permission,
    )
    bus.log(("✓ " if result.ok else "✗ ") + result.message, "info" if result.ok else "error")
    bus.state("idle")


def _run_mic_record_replay(cfg: Config, *, seconds: float = 3.0) -> None:
    """Record ~``seconds`` from the mic, play it back, and report capture stats.

    The GUI "record & replay" diagnostic: the user hears their OWN microphone played
    back through the speaker, which makes a working capture path unmistakable. Emits
    the measured level / duration / sample-rate via the bus (a ``mic_result`` event
    plus a human ``bus.log`` line) and, on a successful (non-silent) capture, emits
    ``mic_result(ok=True)`` so the GUI can hide the macOS permission hint (the audio
    is confirmed working). Never raises — a missing device / PortAudio / permission
    is turned into a clear failing verdict so the worker thread stays alive.
    """
    permission = audio.mic_permission_status()
    bus.log(f"recording {seconds:.0f}s from the mic for replay…")
    bus.state("recording")
    try:
        clip, device_rate = audio.record_fixed(
            cfg.sample_rate,
            seconds=seconds,
            retries=cfg.mic_check_retries,
            settle_s=cfg.mic_check_settle_s,
        )
    except Exception as exc:  # noqa: BLE001 — no device / PortAudio / capture error
        contention = audio.is_device_contention_error(exc)
        log.error(
            "mic record-replay error%s: %s", " (device contention)" if contention else "", exc
        )
        # A device-contention failure (the wake loop still held the input device) is
        # NOT a permission problem — say so rather than sending the user to System
        # Settings. A genuine device/PortAudio error keeps the plain message.
        msg = (
            "Microphone was busy (device contention) — the wake loop hadn't released "
            "the input device yet. Try record & replay again."
            if contention
            else f"microphone error: {exc}"
        )
        bus.log(f"✗ record & replay failed: {msg}", "error")  # SYSTEM log first…
        bus.mic_result(  # …then the verdict, so it isn't flushed by the error log
            ok=False,
            verdict="busy" if contention else "error",
            message=msg,
            permission=permission,
        )
        bus.state("idle")
        return
    # The clip is RAW at the device rate (record_fixed does not resample for the
    # human replay), so duration + playback both use device_rate — playing it at
    # any other rate (e.g. the 24 kHz chime rate) is what made the replay sped-up
    # and high-pitched. Same rate in, same rate out -> faithful pitch + duration.
    stats = audio.capture_stats(clip, device_rate)
    peak = float(stats["peak"])
    level = int(round(min(1.0, max(0.0, peak)) * 100))
    ok = clip.size > 0 and peak >= audio._SILENCE_PEAK  # noqa: SLF001 — shared silence floor
    if ok:
        bus.log(
            f"playing back your recording ({stats['duration_s']}s @ {device_rate} Hz, "
            f"level {level}%)…"
        )
        try:
            audio.play(clip, device_rate)  # play at the capture rate (faithful pitch/speed)
        except Exception as exc:  # noqa: BLE001 — playback backend missing/failed
            log.error("mic replay playback error: %s", exc)
            bus.log(f"recorded OK but playback failed: {exc}", "error")
    message = (
        f"Microphone OK — recorded & replayed {stats['duration_s']}s, level {level}%"
        if ok
        else "No audio captured to replay — check the microphone permission and input device."
    )
    # Log first, then the authoritative mic_result LAST: a failing verdict logs at
    # "error" level, which is a SYSTEM frame that flushes the subscriber's data queue
    # — emitting mic_result after it keeps the verdict from being flushed away.
    bus.log(("✓ " if ok else "✗ ") + message, "info" if ok else "error")
    bus.mic_result(
        ok=ok,
        verdict="ok" if ok else "silent",
        message=message,
        level=level,
        permission=permission,
    )
    bus.state("idle")


def _level_from_peak(peak: float) -> int:
    """The 0..100 UI level meter value for a 0..1 peak amplitude (clamped)."""
    return int(round(min(1.0, max(0.0, float(peak))) * 100))


def _mic_check_stats(clip16k: np.ndarray) -> tuple[float, float, int, float, list[float]]:
    """Peak / rms / level / duration / per-window levels for a 16 kHz mic-check clip."""
    arr = np.asarray(clip16k, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    duration_s = round(arr.size / 16000, 3) if arr.size else 0.0
    return peak, rms, _level_from_peak(peak), duration_s, audio.compute_levels(arr)


def _run_mic_check_server(cfg: Config, *, seconds: float = 2.0) -> None:
    """Capture ~``seconds`` from the SERVER mic, apply gain, save + emit a mic_check_result.

    The unified GUI ``mic_check`` / ``source="server"`` action: record a fixed window,
    apply the software ``cfg.mic_gain`` (clip-protected), compute peak/level/rms + the
    ~48-window level-over-time array, save the 16 kHz clip under ``debug/recordings/``,
    and publish a ``mic_check_result`` carrying ``processing={agc,ns,ec,gain}`` (server
    processing is gain-only) + the saved-WAV ``hash``/``wav_url``. Never raises — a
    missing device / permission becomes a clear failing result so the worker survives.
    """
    bus.log(f"mic check: recording {seconds:.1f}s from the server mic…")
    bus.state("recording")
    try:
        clip, device_rate = audio.record_fixed(
            cfg.sample_rate,
            seconds=seconds,
            retries=cfg.mic_check_retries,
            settle_s=cfg.mic_check_settle_s,
        )
    except Exception as exc:  # noqa: BLE001 — no device / PortAudio / capture error
        contention = audio.is_device_contention_error(exc)
        log.error(
            "mic check capture error%s: %s", " (device contention)" if contention else "", exc
        )
        # Distinguish a device-contention failure (wake loop still held the input
        # device — permission is fine) from a genuine permission/no-device problem, so
        # we don't send the user chasing a non-issue.
        msg = (
            "Microphone was busy (device contention) — the wake loop hadn't released "
            "the input device yet. Try the mic check again."
            if contention
            else f"microphone error: {exc}"
        )
        bus.log(f"✗ mic check failed: {msg}", "error")
        bus.mic_check_result(
            source="server",
            peak=0.0,
            level=0,
            rms=0.0,
            duration_s=0.0,
            sample_rate=16000,
            levels=audio.compute_levels(np.zeros(0, dtype=np.float32)),
            processing={"agc": False, "ns": False, "ec": False, "gain": cfg.mic_gain},
            hash="",
            wav_url="",
            message=msg,
        )
        bus.state("idle")
        return
    clip16k = audio.resample_to(clip, device_rate, 16000)
    clip16k = audio.apply_gain(clip16k, cfg.mic_gain)  # software input gain (clip-protected)
    path, hash8, wav_url = audio.save_recording(clip16k, 16000, kind="mic", source="server")
    peak, rms, level, duration_s, levels = _mic_check_stats(clip16k)
    ok = peak >= audio._SILENCE_PEAK  # noqa: SLF001 — shared silence floor
    message = (
        f"Microphone OK — level {level}% (gain {cfg.mic_gain:g}×)"
        if ok
        else "No audio captured — check the microphone permission and input device."
    )
    bus.log(("✓ " if ok else "✗ ") + message, "info" if ok else "error")
    bus.mic_check_result(
        source="server",
        peak=peak,
        level=level,
        rms=rms,
        duration_s=duration_s,
        sample_rate=16000,
        levels=levels,
        processing={"agc": False, "ns": False, "ec": False, "gain": cfg.mic_gain},
        hash=hash8,
        wav_url=wav_url,
        message=message,
        stats=audio.capture_stats(clip16k, 16000),
    )
    bus.state("idle")
    _ = path  # path is informational; the GUI addresses the clip via wav_url/hash


def _run_mic_check_browser(
    cfg: Config,
    pcm: list[float],
    sample_rate: int,
    processing: dict[str, Any] | None = None,
) -> None:
    """Save + analyse a BROWSER-supplied mic clip, emit a mic_check_result.

    The unified GUI ``mic_check`` / ``source="browser"`` action: the page records the
    clip locally (with its own AGC/NS/EC flags) and POSTs raw float PCM + its sample
    rate + the ``processing`` flags. The server resamples to 16 kHz, saves it under
    ``debug/recordings/``, computes the same peak/level/rms + level-over-time graph, and
    emits the result. No server gain is applied to a browser clip (the browser owns its
    own processing); ``processing.gain`` is reported as 1.0. Never raises.
    """
    _ = cfg  # signature parity with the server variant; browser clips carry their rate
    proc = dict(processing or {})
    flags = {
        "agc": bool(proc["agc"]) if "agc" in proc else None,
        "ns": bool(proc["ns"]) if "ns" in proc else None,
        "ec": bool(proc["ec"]) if "ec" in proc else None,
        "gain": float(proc.get("gain", 1.0)),
    }
    clip = np.asarray(pcm, dtype=np.float32).ravel()
    rate = int(sample_rate) or 16000
    clip16k = audio.resample_to(clip, rate, 16000)
    path, hash8, wav_url = audio.save_recording(clip16k, 16000, kind="mic", source="browser")
    peak, rms, level, duration_s, levels = _mic_check_stats(clip16k)
    ok = peak >= audio._SILENCE_PEAK  # noqa: SLF001 — shared silence floor
    message = (
        f"Microphone OK — level {level}%"
        if ok
        else "No audio in the recording — check the browser mic permission."
    )
    bus.log(("✓ " if ok else "✗ ") + message, "info" if ok else "error")
    bus.mic_check_result(
        source="browser",
        peak=peak,
        level=level,
        rms=rms,
        duration_s=duration_s,
        sample_rate=16000,
        levels=levels,
        processing=flags,
        hash=hash8,
        wav_url=wav_url,
        message=message,
        stats=audio.capture_stats(clip16k, 16000),
    )
    bus.state("idle")
    _ = path


def _play_recording(hash8: str) -> None:
    """Play a saved mic/wake WAV addressed by its 8-hex content ``hash`` (best-effort).

    The GUI ``play_recording`` action: find ``debug/recordings/*-<hash8>.wav`` and play
    it back server-side through the normal playback path (:func:`audio.play`) so the
    user hears exactly what was captured. Never raises — a missing hash / file / player
    degrades to a clear log line so the worker thread stays alive.
    """
    import wave

    hash8 = str(hash8 or "").strip()
    if not hash8:
        bus.log("play_recording: no hash given", "error")
        return
    # Search flat (mic) AND per-word wake subfolders for the hash-named clip.
    matches = audio.find_recordings(f"*-{hash8}.wav")
    if not matches:
        bus.log(f"play_recording: no saved recording for {hash8}", "error")
        return
    target = matches[0]
    try:
        with wave.open(target, "rb") as wf:
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        clip = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    except (OSError, wave.Error) as exc:  # unreadable WAV must not crash the worker
        log.error("play_recording read error: %s", exc)
        bus.log(f"play_recording: could not read {os.path.basename(target)}: {exc}", "error")
        return
    bus.log(f"▶ playing recording {os.path.basename(target)}")
    try:
        audio.play(clip, rate)
    except Exception as exc:  # noqa: BLE001 — playback backend missing/failed
        log.error("play_recording playback error: %s", exc)
        bus.log(f"play_recording: playback failed: {exc}", "error")
    finally:
        bus.state("idle")


def _wake_test_wav_path(word: str, source: str) -> str:
    """Where a wake-test clip is saved (kept for later debugging).

    ``~/.cache/my-stt-tts/wake-test-<word>-<source>.wav`` — one stable file per
    (word, source) so a re-test overwrites the last clip of the same kind."""
    return os.path.expanduser(f"~/.cache/my-stt-tts/wake-test-{word}-{source}.wav")


def _save_wake_test_wav(clip16k: np.ndarray, word: str, source: str) -> str:
    """Write the scored 16 kHz clip as a mono WAV; return the path ("" on failure)."""
    from .util import wav_bytes_from_float

    path = _wake_test_wav_path(word, source)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(wav_bytes_from_float(np.asarray(clip16k, dtype=np.float32).ravel(), 16000))
        return path
    except OSError as exc:  # disk/permission issue must not kill the worker
        log.warning("wake-test WAV write failed (%s): %s", path, exc)
        return ""


def _classify_wake_outcome(
    word: str,
    *,
    fired: bool,
    confidence: float,
    int16_peak: int,
    threshold: float,
    available: bool = True,
) -> tuple[str, str]:
    """Classify a wake score into the contract's ``(reason, detail)`` (shared by the
    wake-test + score-clip paths).

    ``reason`` is one of ``"fired"`` / ``"level_too_low"`` / ``"not_detected"`` /
    ``"unavailable"``; ``detail`` is a human one-liner naming the word + why. The
    level floor is :data:`audio._INT16_LOW_PEAK` — the same int16 peak under which
    :func:`audio._expected_level` reads "low" and the (un-normalized) wake model is
    starved. So a quiet capture is reported as ``level_too_low`` (the dominant
    failure: "move closer") rather than blamed on the word's recall, while a loud
    capture that still doesn't clear ``threshold`` is a genuine ``not_detected``
    recall miss (level OK)."""
    if not available:
        return (
            "unavailable",
            f"{word}: wake model unavailable — train it (see wakewords/WAKEWORD.md)",
        )
    if fired:
        return "fired", f"{word}: fired (confidence {confidence:.2f})"
    if int16_peak < audio._INT16_LOW_PEAK:  # noqa: SLF001 — shared wake-usable floor
        return (
            "level_too_low",
            f"{word}: level too low (pk {int16_peak}) — move closer or raise the mic gain",
        )
    return (
        "not_detected",
        f"{word}: wake word not detected (level OK, confidence {confidence:.2f} < {threshold:.2f})",
    )


def _emit_wake_test(
    word: str,
    source: str,
    confidence: float,
    fired: bool,
    wav_path: str,
    *,
    available: bool = True,
    stats: dict[str, Any] | None = None,
    detector: str = "oww",
    threshold: float = 0.4,
) -> None:
    """Score + format + publish a ``wake_test_result`` (and a human EVENT-LOG line).

    Order matters: an "error"-level ``bus.log`` is a SYSTEM frame that FLUSHES the
    subscriber's data queue, so the human log + idle state go FIRST and the
    authoritative ``wake_test_result`` (DATA) is published LAST — otherwise the
    unavailable-model result would be flushed away before the GUI ever sees it.

    ``stats`` (optional) carries the level-meter extras the GUI shares with a
    mic-check: ``peak``/``level``/``rms``/``duration_s``/``sample_rate``/``levels``/
    ``processing``/``hash``/``wav_url`` + a nested ``stats`` (:func:`audio.capture_stats`).
    Absent (unavailable model) -> the event still publishes with the level fields at
    their zero defaults.

    ALWAYS classifies the outcome into the contract's ``reason``/``detail`` (see
    :func:`_classify_wake_outcome`): ``level_too_low`` when the captured int16 peak is
    below the wake-usable floor, ``not_detected`` when the level is OK but confidence
    stayed sub-threshold, ``fired`` otherwise — so the GUI always explains WHY a clip
    did or didn't fire, naming the word.
    """
    # The int16 peak the wake model actually saw (from the nested capture_stats), used
    # to tell a too-quiet capture apart from a genuine recall miss.
    inner = (stats or {}).get("stats") or {}
    int16_peak = int(inner.get("int16_peak", 0))
    reason, detail = _classify_wake_outcome(
        word,
        fired=fired,
        confidence=confidence,
        int16_peak=int16_peak,
        threshold=threshold,
        available=available,
    )
    if not available:
        message = detail
    else:
        verdict = "detected" if fired else "not detected"
        # Name the OR'd detector when a custom word fired via the sherpa-KWS path.
        via = f" via {detector}" if fired and detector and detector != "oww" else ""
        message = f"{word}: confidence {confidence:.2f} — {verdict}{via}"
    bus.log(("✓ " if fired else "✗ ") + message, "info" if available else "error")
    bus.state("idle")
    bus.wake_test_result(
        word=word,
        source=source,
        confidence=round(float(confidence), 3),
        fired=bool(fired),
        message=message,
        wav_path=wav_path,
        detector=detector,
        reason=reason,
        detail=detail,
        **(stats or {}),
    )


def _wake_test_stats(
    clip16k: np.ndarray,
    *,
    processing: dict[str, Any],
    hash8: str,
    wav_url: str,
    score_trace: list[float] | None = None,
) -> dict[str, Any]:
    """The level-meter extras (peak/level/levels/processing/hash/wav_url) for a wake test.

    Also carries ``stats`` (the full :func:`audio.capture_stats` dict — int16 magnitudes
    + quality metrics) and the per-frame ``score_trace``, both surfaced on the
    ``wake_test_result`` event so the GUI can diagnose a too-quiet capture / a localized
    sub-threshold spike.
    """
    peak, rms, level, duration_s, levels = _mic_check_stats(clip16k)
    return {
        "peak": peak,
        "level": level,
        "rms": rms,
        "duration_s": duration_s,
        "sample_rate": 16000,
        "levels": levels,
        "processing": processing,
        "hash": hash8,
        "wav_url": wav_url,
        "stats": audio.capture_stats(clip16k, 16000),
        "score_trace": list(score_trace or []),
    }


def _run_wake_test_server(cfg: Config, word: str) -> None:
    """Record ~2 s from the SERVER mic, score it against ``word``, save + emit.

    The GUI ``wake_test`` / ``source="server"`` diagnostic: capture a 2.0 s clip from
    the server microphone, apply the software ``cfg.mic_gain`` (clip-protected), run it
    through the REAL phase-diverse wake path via :func:`wake.score_wake_clip` (so it
    matches the always-listening loop), save the gained 16 kHz clip under
    ``debug/recordings/`` (and the legacy cache path), and publish a
    ``wake_test_result`` carrying the same peak/level/levels/processing as a mic-check.
    Never raises — a missing model or a capture error becomes a clear failing message.
    """
    from .wake import score_wake_clip_combined, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        _emit_wake_test(word, "server", 0.0, False, "", available=False)
        return
    bus.log(f"wake test: recording 2.0 s to score against '{word}'…")
    bus.state("recording")
    try:
        clip, device_rate = audio.record_fixed(cfg.sample_rate, seconds=2.0)
    except Exception as exc:  # noqa: BLE001 — no device / PortAudio / capture error
        log.error("wake test capture error: %s", exc)
        bus.error(f"wake test failed: {exc}")
        bus.wake_test_result(
            word=word,
            source="server",
            confidence=0.0,
            fired=False,
            message=f"{word}: microphone error — {exc}",
            wav_path="",
        )
        bus.state("idle")
        return
    clip16k = audio.resample_to(clip, device_rate, 16000)
    clip16k = audio.apply_gain(clip16k, cfg.mic_gain)  # software input gain (clip-protected)
    # OR-combined: openWakeWord, then (custom word only) the open-vocabulary sherpa-KWS.
    # ``detector`` names which fired ("oww"|"kws"); official words stay oWW-only.
    confidence, fired, detector, score_trace = score_wake_clip_combined(clip16k, 16000, word, cfg)
    wav_path = _save_wake_test_wav(clip16k, word, "server")
    saved_path, hash8, wav_url = audio.save_recording(
        clip16k, 16000, kind="wake", source="server", word=word
    )
    # Persist the outcome so per-word reliability is data-driven from REAL tests, LINKED to the
    # saved clip (hash/path) so a logged miss/false-fire is actionable by the relabel loop.
    from .config import record_wake_outcome

    record_wake_outcome(
        word,
        confidence=confidence,
        fired=fired,
        source="server",
        clip_hash=hash8,
        clip_path=saved_path,
    )
    stats = _wake_test_stats(
        clip16k,
        processing={"agc": False, "ns": False, "ec": False, "gain": cfg.mic_gain},
        hash8=hash8,
        wav_url=wav_url,
        score_trace=score_trace,
    )
    _emit_wake_test(
        word,
        "server",
        confidence,
        fired,
        wav_path,
        stats=stats,
        detector=detector,
        threshold=cfg.wake_threshold,
    )


def _run_wake_test_browser(
    cfg: Config,
    word: str,
    pcm: list[float],
    sample_rate: int,
    processing: dict[str, Any] | None = None,
) -> None:
    """Score a BROWSER-supplied 2.0 s clip against ``word``, save + emit.

    The GUI ``wake_test`` / ``source="browser"`` diagnostic: the page records the
    clip locally and POSTs raw float PCM (+ its sample rate + its AGC/NS/EC flags); the
    server resamples to 16 kHz, scores it through the same phase-diverse wake path,
    saves the clip under ``debug/recordings/`` (and the legacy cache path), and emits a
    ``wake_test_result`` carrying the same peak/level/levels/processing as a mic-check.
    No server gain is applied to a browser clip. Never raises — bad/empty PCM scores 0.0.
    """
    from .wake import score_wake_clip_combined, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        _emit_wake_test(word, "browser", 0.0, False, "", available=False)
        return
    proc = dict(processing or {})
    flags = {
        "agc": bool(proc["agc"]) if "agc" in proc else None,
        "ns": bool(proc["ns"]) if "ns" in proc else None,
        "ec": bool(proc["ec"]) if "ec" in proc else None,
        "gain": float(proc.get("gain", 1.0)),
    }
    clip = np.asarray(pcm, dtype=np.float32).ravel()
    rate = int(sample_rate) or cfg.sample_rate
    # OR-combined: openWakeWord, then (custom word only) the open-vocabulary sherpa-KWS.
    confidence, fired, detector, score_trace = score_wake_clip_combined(clip, rate, word, cfg)
    clip16k = audio.resample_to(clip, rate, 16000)
    wav_path = _save_wake_test_wav(clip16k, word, "browser")
    saved_path, hash8, wav_url = audio.save_recording(
        clip16k, 16000, kind="wake", source="browser", word=word
    )
    # Persist the outcome so per-word reliability is data-driven from REAL tests, LINKED to the
    # saved clip (hash/path) so a logged miss/false-fire is actionable by the relabel loop.
    from .config import record_wake_outcome

    record_wake_outcome(
        word,
        confidence=confidence,
        fired=fired,
        source="browser",
        clip_hash=hash8,
        clip_path=saved_path,
    )
    stats = _wake_test_stats(
        clip16k, processing=flags, hash8=hash8, wav_url=wav_url, score_trace=score_trace
    )
    _emit_wake_test(
        word,
        "browser",
        confidence,
        fired,
        wav_path,
        stats=stats,
        detector=detector,
        threshold=cfg.wake_threshold,
    )


def _load_saved_wake_clip(word: str, hash8: str | None) -> tuple[np.ndarray | None, str, str]:
    """Resolve a SAVED wake clip to ``(clip16k, hash, basename)`` for re-scoring.

    Addresses a clip in ``debug/recordings/`` either by its 8-hex content ``hash``
    (``*-<hash8>.wav``, searched flat + per-word subfolders) or, when ``hash8`` is
    falsy, by the NEWEST wake recording for ``word`` (the per-word training folder
    ``wake/<word>/*.wav`` by mtime). The WAV is already 16 kHz mono (see
    :func:`audio.save_recording`). Returns ``(None, "", "")`` when nothing matches or
    the file is unreadable — the caller turns that into a clear "no clip" message. The
    returned ``hash`` is the resolved 8-hex id parsed from the filename.
    """
    import glob

    rec_dir = audio.recordings_dir()
    h = str(hash8 or "").strip()
    if h:
        matches = audio.find_recordings(f"*-{h}.wav")
    else:
        # Newest wake clip for this word: it lives in the per-word folder
        # debug/recordings/wake/<word>/<ts>-<source>-<hash>.wav.
        from .audio import _sanitize_word

        word_dir = os.path.join(rec_dir, "wake", _sanitize_word(word))
        matches = sorted(
            glob.glob(os.path.join(word_dir, "*.wav")),
            key=lambda p: os.path.getmtime(p) if os.path.isfile(p) else 0.0,
            reverse=True,
        )
    if not matches:
        return None, "", ""
    target = matches[0]
    try:
        clip16k, _rate = audio.read_wav_float(target, target_rate=16000)
    except Exception as exc:  # noqa: BLE001 — unreadable WAV must not crash the worker
        log.error("load saved wake clip read error: %s", exc)
        return None, "", ""
    base = os.path.basename(target)
    resolved_hash = base.rsplit("-", 1)[-1].removesuffix(".wav") if "-" in base else (h or "")
    return clip16k, resolved_hash, base


def _load_positive_clips(word: str) -> list[np.ndarray]:
    """Every SAVED positive clip for ``word`` as 16 kHz float32 (the eval positives).

    Reads all WAVs in the per-word training folder ``debug/recordings/wake/<word>/``
    via :func:`audio.read_wav_float`. An unreadable file is skipped (logged), so a
    single corrupt clip never sinks the whole histogram / FA-eval. Returns ``[]`` when
    the folder is missing or empty.
    """
    from .audio import _sanitize_word

    word_dir = os.path.join(audio.recordings_dir(), "wake", _sanitize_word(word))
    clips: list[np.ndarray] = []
    for path in audio.list_wavs(word_dir):
        try:
            clip, _rate = audio.read_wav_float(path, target_rate=16000)
            clips.append(clip)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not sink the set
            log.warning("skipping unreadable positive clip %s: %s", path, exc)
    return clips


def _load_negative_clips(cfg: Config, word: str | None = None) -> tuple[list[np.ndarray], str]:
    """Every WAV in the negative corpus as 16 kHz float32 + the resolved dir.

    Reads :data:`Config.negative_corpus_dir` (env ``WAKE_NEG_CORPUS``) flat via
    :func:`audio.list_wavs` / :func:`audio.read_wav_float`. When ``word`` is given, the
    per-word active-learning NEGATIVES dir ``debug/recordings/wake_neg/<word>/`` (clips the
    user marked "wasn't me" / captured live false fires, Mycroft Precise's closed-loop idea —
    repo #5) is UNIONed in via :func:`my_stt_tts.active_learning.load_negative_clips_union`, so
    the eval toolkit + the rebuild gate see the user's own negatives. Returns ``(clips, dir)``;
    an empty list signals the caller to emit the "drop WAVs into <dir>" message. ``dir`` is the
    primary ``negative_corpus_dir`` (the drop-in path the message names).
    """
    if word is not None:
        from . import active_learning

        clips, _dirs = active_learning.load_negative_clips_union(word, cfg)
        return clips, cfg.negative_corpus_dir
    neg_dir = cfg.negative_corpus_dir
    clips = []
    for path in audio.list_wavs(neg_dir):
        try:
            clip, _rate = audio.read_wav_float(path, target_rate=16000)
            clips.append(clip)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not sink the set
            log.warning("skipping unreadable negative clip %s: %s", path, exc)
    return clips, neg_dir


def _load_noise_clips(cfg: Config) -> tuple[list[np.ndarray], str]:
    """Every WAV in the NOISE corpus as 16 kHz float32 + the resolved dir.

    Reads :data:`Config.noise_corpus_dir` (env ``WAKE_NOISE_CORPUS``) flat via
    :func:`audio.list_wavs` / :func:`audio.read_wav_float` — the wake-word-FREE noise the
    noise×SNR benchmark mixes into the positives + negatives. Returns ``(clips, dir)``;
    an empty list signals the caller to run clean-only + emit the "drop noise WAVs into
    <dir>" note.
    """
    noise_dir = cfg.noise_corpus_dir
    clips: list[np.ndarray] = []
    for path in audio.list_wavs(noise_dir):
        try:
            clip, _rate = audio.read_wav_float(path, target_rate=16000)
            clips.append(clip)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not sink the set
            log.warning("skipping unreadable noise clip %s: %s", path, exc)
    return clips, noise_dir


def _run_wake_gain_sweep(cfg: Config, word: str, hash8: str | None, gains: list[float]) -> None:
    """Re-score a SAVED wake clip at each gain (the smoking-gun diagnostic).

    The GUI ``wake_gain_sweep`` action: load the clip by ``hash8`` (or the newest wake
    recording for ``word``), run :func:`wake.score_wake_clip` at every gain in
    ``gains``, and emit a ``wake_gain_sweep_result`` with each ``{gain, confidence,
    fired}``. If confidence CLIMBS with gain, a too-quiet capture (low mel energy ->
    collapsed openWakeWord score) is the cause — and ``wake_gain`` is the fix. Never
    raises — a missing model / clip becomes a clear message with no points.
    """
    from .wake import score_wake_clip, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        bus.wake_gain_sweep_result(
            word=word,
            hash=str(hash8 or ""),
            points=[],
            message=f"{word}: wake model unavailable — train it (see wakewords/WAKEWORD.md)",
        )
        bus.state("idle")
        return
    clip16k, resolved_hash, base = _load_saved_wake_clip(word, hash8)
    if clip16k is None:
        which = f"hash {hash8}" if hash8 else f"newest '{word}' recording"
        bus.log(f"wake gain sweep: no saved clip ({which})", "error")
        bus.wake_gain_sweep_result(
            word=word, hash=str(hash8 or ""), points=[], message=f"no saved clip ({which})"
        )
        bus.state("idle")
        return
    # The sweep is a DIAGNOSTIC — it deliberately probes gains beyond the persistent
    # wake_gain bound (the contract's default sweep ends at 16x) to FIND where a quiet
    # capture starts firing. Only require each gain to be a positive, finite number;
    # cap at 64x to reject an abusive payload while keeping the useful range.
    clean_gains = [g for g in (float(x) for x in gains) if 0.0 < g <= 64.0] or [1.0]
    points: list[dict[str, Any]] = []
    for gain in clean_gains:
        confidence, fired = score_wake_clip(
            clip16k, 16000, word, threshold=cfg.wake_threshold, phases=cfg.wake_phases, gain=gain
        )
        points.append({"gain": gain, "confidence": confidence, "fired": fired})
    climbed = len(points) >= 2 and points[-1]["confidence"] > points[0]["confidence"] + 1e-3
    hint = " — confidence rises with gain (a quiet capture is the cause)" if climbed else ""
    message = f"{word}: swept {len(points)} gains on {base}{hint}"
    bus.log("✓ " + message)
    bus.wake_gain_sweep_result(word=word, hash=resolved_hash, points=points, message=message)
    bus.state("idle")


def _run_score_clip(cfg: Config, word: str, hash8: str | None) -> None:
    """Score the EXACT saved WAV through the model (capture-vs-model isolation).

    The GUI ``score_clip`` action: load the saved clip by ``hash8`` and run it through
    :func:`wake.score_wake_clip` with NO extra gain. If a clip that SHOULD fire also
    scores ~0 here, the model/word is the problem, not capture. ALWAYS computes the
    score and classifies the outcome into the contract's ``reason``/``detail`` (see
    :func:`_classify_wake_outcome`): ``level_too_low`` when the clip's ``int16_peak`` is
    below the wake-usable floor, ``not_detected`` when the level is OK but confidence
    stayed sub-threshold, ``fired`` otherwise. Never raises — a missing clip/model is a
    clear message.
    """
    from .wake import score_wake_clip, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        _, detail = _classify_wake_outcome(
            word,
            fired=False,
            confidence=0.0,
            int16_peak=0,
            threshold=cfg.wake_threshold,
            available=False,
        )
        bus.score_clip_result(
            word=word,
            hash=str(hash8 or ""),
            confidence=0.0,
            fired=False,
            int16_peak=0,
            message=detail,
            reason="unavailable",
            detail=detail,
        )
        bus.state("idle")
        return
    clip16k, resolved_hash, base = _load_saved_wake_clip(word, hash8)
    if clip16k is None:
        which = f"hash {hash8}" if hash8 else f"newest '{word}' recording"
        bus.log(f"score_clip: no saved clip ({which})", "error")
        bus.score_clip_result(
            word=word,
            hash=str(hash8 or ""),
            confidence=0.0,
            fired=False,
            int16_peak=0,
            message=f"no saved clip ({which})",
            reason="no_clip",
            detail=f"{word}: no saved clip ({which}) to score",
        )
        bus.state("idle")
        return
    confidence, fired = score_wake_clip(
        clip16k,
        16000,
        word,
        threshold=cfg.wake_threshold,
        phases=cfg.wake_phases,
        window=cfg.wake_window,
        refractory=cfg.wake_refractory,
    )
    int16_peak = int(audio.capture_stats(clip16k, 16000)["int16_peak"])
    reason, detail = _classify_wake_outcome(
        word,
        fired=fired,
        confidence=confidence,
        int16_peak=int16_peak,
        threshold=cfg.wake_threshold,
    )
    verdict = "detected" if fired else "not detected"
    message = f"{word}: confidence {confidence:.2f} on {base} (int16 peak {int16_peak}) — {verdict}"
    bus.log(("✓ " if fired else "✗ ") + message)
    bus.score_clip_result(
        word=word,
        hash=resolved_hash,
        confidence=confidence,
        fired=fired,
        int16_peak=int16_peak,
        message=message,
        reason=reason,
        detail=detail,
    )
    bus.state("idle")


# --------------------------------------------------------------------------- #
# Wake EVALUATION toolkit actions: histogram / FA-eval / verifier / spectrogram #
# --------------------------------------------------------------------------- #
def _run_score_histogram(cfg: Config, word: str) -> None:
    """Score the SAVED positives + the negative corpus for ``word`` (the eval histogram).

    The GUI ``score_histogram`` action (Task 1): load every saved positive clip for
    ``word`` AND the negative corpus, score each through the REAL phase-diverse path
    (:func:`wake.score_clip_set`), and emit ``score_histogram_result`` with both
    max-score arrays + the :func:`wake.separation` scalar. This is the recall-vs-level
    proof the positives-only diagnostics never gave — pos and neg distributions side by
    side. Never raises: a missing model / no positives / empty corpus each become a clear
    message with empty arrays.
    """
    from .wake import score_clip_set, separation, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        bus.score_histogram_result(
            word=word,
            pos_scores=[],
            neg_scores=[],
            threshold=cfg.wake_threshold,
            separation=0.0,
            message=f"{word}: wake model unavailable — train it (see wakewords/WAKEWORD.md)",
        )
        bus.state("idle")
        return
    pos_clips = _load_positive_clips(word)
    neg_clips, neg_dir = _load_negative_clips(cfg, word)
    if not pos_clips:
        bus.score_histogram_result(
            word=word,
            pos_scores=[],
            neg_scores=[],
            threshold=cfg.wake_threshold,
            separation=0.0,
            message=f"{word}: no saved positive clips yet — run WAKE TEST a few times first",
        )
        bus.state("idle")
        return
    from .calibration import calibrator_for

    calibrator = calibrator_for(word, enabled=getattr(cfg, "wake_calibration", False))
    pos_scores, _ = score_clip_set(
        pos_clips, word, threshold=cfg.wake_threshold, phases=cfg.wake_phases, calibrator=calibrator
    )
    neg_scores, _ = (
        score_clip_set(
            neg_clips,
            word,
            threshold=cfg.wake_threshold,
            phases=cfg.wake_phases,
            calibrator=calibrator,
        )
        if neg_clips
        else ([], [])
    )
    sep = separation(pos_scores, neg_scores)
    neg_note = (
        f"{len(neg_scores)} negatives" if neg_scores else f"NO negatives (drop WAVs into {neg_dir})"
    )
    message = f"{word}: {len(pos_scores)} positives vs {neg_note} — separation {sep}"
    bus.log("✓ " + message)
    bus.score_histogram_result(
        word=word,
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        threshold=cfg.wake_threshold,
        separation=sep,
        message=message,
    )
    bus.state("idle")


def _run_fa_eval(cfg: Config, word: str) -> None:
    """Sweep the threshold over positives + the timed negative corpus (FA/hour + ROC/DET).

    The GUI ``fa_eval`` action (Task 2): score the saved positives and the negative
    corpus to per-frame traces, then :func:`wake.fa_eval` sweeps the threshold counting
    false-accept EVENTS (not frames) over the corpus → ``(threshold, fa_per_hour,
    true_accept)`` points + the miss-rate at a target FA/hour. Emits ``fa_eval_result``.
    An empty corpus emits a clear "drop WAVs into <dir>" message and DOES NOT crash.
    """
    from .wake import fa_eval, score_clip_set, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        bus.fa_eval_result(
            word=word,
            points=[],
            miss_at_target_fa=1.0,
            target_fa=0.5,
            message=f"{word}: wake model unavailable — train it (see wakewords/WAKEWORD.md)",
        )
        bus.state("idle")
        return
    neg_clips, neg_dir = _load_negative_clips(cfg, word)
    if not neg_clips:
        bus.fa_eval_result(
            word=word,
            points=[],
            miss_at_target_fa=1.0,
            target_fa=0.5,
            message=(
                f"{word}: negative corpus empty — drop wake-word-free WAVs into {neg_dir} "
                "(set WAKE_NEG_CORPUS to change the dir), then re-run"
            ),
        )
        bus.state("idle")
        return
    from .calibration import calibrator_for

    calibrator = calibrator_for(word, enabled=getattr(cfg, "wake_calibration", False))
    pos_clips = _load_positive_clips(word)
    _, pos_traces = (
        score_clip_set(
            pos_clips,
            word,
            threshold=cfg.wake_threshold,
            phases=cfg.wake_phases,
            calibrator=calibrator,
        )
        if pos_clips
        else ([], [])
    )
    _, neg_traces = score_clip_set(
        neg_clips, word, threshold=cfg.wake_threshold, phases=cfg.wake_phases, calibrator=calibrator
    )
    result = fa_eval(
        pos_traces,
        neg_traces,
        window=cfg.wake_window,
        refractory=cfg.wake_refractory,
        target_fa=0.5,
    )
    # NOISE×SNR axis (Porcupine's idea, repo #4): when a noise corpus is present, mix the
    # positives + negatives at each SNR and re-measure miss@target-FA per condition.
    # Empty noise dir -> clean-only + a clear "drop noise WAVs into <dir>" note.
    from .wake import fa_eval_snr

    noise_clips, noise_dir = _load_noise_clips(cfg)
    snr_result = fa_eval_snr(
        pos_clips,
        neg_clips,
        noise_clips,
        word,
        threshold=cfg.wake_threshold,
        phases=cfg.wake_phases,
        window=cfg.wake_window,
        refractory=cfg.wake_refractory,
        target_fa=0.5,
    )
    noise_note = (
        f" · {len(noise_clips)} noise clips ×{len(snr_result['snr_list'])} SNR"
        if noise_clips
        else f" · clean only (drop noise WAVs into {noise_dir})"
    )
    message = (
        f"{word}: {len(pos_traces)} positives vs {result['neg_seconds']}s of negatives — "
        f"miss {result['miss_at_target_fa'] * 100:.0f}% at {result['target_fa']} FA/h{noise_note}"
    )
    bus.log("✓ " + message)
    bus.fa_eval_result(
        word=word,
        points=result["points"],
        miss_at_target_fa=result["miss_at_target_fa"],
        target_fa=result["target_fa"],
        neg_seconds=result["neg_seconds"],
        per_snr=snr_result["per_snr"] if noise_clips else None,
        snr_list=snr_result["snr_list"] if noise_clips else None,
        message=message,
    )
    bus.state("idle")


def _parse_benchmark_snrs(raw: str | None) -> list[float | None]:
    """Parse ``--benchmark-snr`` (``clean,10,5``) -> ``[None, 10.0, 5.0]``.

    ``clean`` / ``none`` (any case) -> ``None`` (the noise-free condition); everything
    else is a float SNR in dB. A blank value -> the default ladder
    (:data:`my_stt_tts.wake.DEFAULT_SNR_LIST`). Malformed tokens are skipped. Never raises.
    """
    from .wake import DEFAULT_SNR_LIST

    if not raw or not raw.strip():
        return list(DEFAULT_SNR_LIST)
    out: list[float | None] = []
    for token in raw.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok in ("clean", "none"):
            out.append(None)
            continue
        try:
            out.append(float(tok))
        except ValueError:
            log.warning("ignoring unparseable --benchmark-snr token %r", token)
    return out or list(DEFAULT_SNR_LIST)


def _run_benchmark(cfg: Config, snr_raw: str | None) -> int:
    """Run the noise×SNR wake benchmark for the selected word, print a table, write artifacts.

    The reproducible CLI harness (Porcupine's idea, repo #4): score the saved positives
    + the negative corpus, mixed with the noise corpus at each SNR, through the REAL
    phase-diverse detector → miss@target-FA per condition (:func:`my_stt_tts.wake.fa_eval_snr`).
    Prints a per-SNR table to stdout and writes ``debug/benchmark/<word>-<ts>.json`` (the
    full per-SNR points). Returns a process exit code (0 always when it ran; 1 only if the
    wake model is missing). An empty noise corpus runs clean-only with a clear note.
    """
    import datetime as _dt
    import json as _json

    from .wake import fa_eval_snr, wake_model_for

    word = cfg.wake_phrase
    if not os.path.isfile(wake_model_for(word)):
        print(
            f"benchmark: wake model for {word!r} unavailable — train it "
            "(see wakewords/WAKEWORD.md) or pick another --wake-word",
            file=sys.stderr,
        )
        return 1
    pos_clips = _load_positive_clips(word)
    neg_clips, neg_dir = _load_negative_clips(cfg)
    noise_clips, noise_dir = _load_noise_clips(cfg)
    snr_list = _parse_benchmark_snrs(snr_raw)
    print(
        f"wake benchmark · word={word} · {len(pos_clips)} positives · {len(neg_clips)} "
        f"negatives ({neg_dir}) · {len(noise_clips)} noise clips ({noise_dir})"
    )
    if not pos_clips:
        print(
            "  (no saved positive clips — run WAKE TEST a few times; recall column will be empty)"
        )
    if not noise_clips:
        print(
            f"  (noise corpus empty — clean-only; drop noise WAVs into {noise_dir} "
            "for the SNR axis)"
        )
    result = fa_eval_snr(
        pos_clips,
        neg_clips,
        noise_clips,
        word,
        snr_list=snr_list,
        threshold=cfg.wake_threshold,
        phases=cfg.wake_phases,
        window=cfg.wake_window,
        refractory=cfg.wake_refractory,
        target_fa=0.5,
    )
    print(f"  target {0.5} FA/h · threshold {cfg.wake_threshold} · phases {cfg.wake_phases}")
    print("  SNR (dB)   miss@target-FA   operating points")
    for entry in result["per_snr"]:
        snr_db = entry["snr_db"]
        label = "clean" if snr_db is None else f"{snr_db:g}"
        miss = entry["miss_at_target_fa"]
        print(f"  {label:>8}   {miss * 100:>13.0f}%   {len(entry['points'])}")
    # Write the full artifact (gitignored debug/ tree).
    out_dir = os.path.join(audio.recordings_dir(), "..", "benchmark")
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(out_dir, f"{word}-{stamp}.json")
    artifact = {
        "word": word,
        "threshold": cfg.wake_threshold,
        "phases": cfg.wake_phases,
        "target_fa": 0.5,
        "n_positives": len(pos_clips),
        "n_negatives": len(neg_clips),
        "n_noise": len(noise_clips),
        "snr_list": result["snr_list"],
        "per_snr": result["per_snr"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        _json.dump(artifact, fh, indent=2)
    print(f"  artifact: {out_path}")
    return 0


def _run_train_verifier(cfg: Config, word: str) -> None:
    """Train an openWakeWord-style custom verifier for ``word`` (saved positives + negatives).

    The GUI ``train_verifier`` action (Task 3): fit a logistic-regression head on the
    shared oWW embedding from the saved positive clips (>=3) + the negative corpus and
    persist it to a git-ignored path. Emits ``verifier_result``. scikit-learn lives in
    the optional ``debug`` extra — when absent, the result is ``trained=False`` with a
    clear install hint (never raises, core stays clean).
    """
    from .wake_verifier import train_verifier

    pos_clips = _load_positive_clips(word)
    neg_clips, _neg_dir = _load_negative_clips(cfg, word)
    result = train_verifier(pos_clips, neg_clips, word)
    bus.log(("✓ " if result["trained"] else "✗ ") + result["message"])
    bus.verifier_result(
        word=word,
        trained=bool(result["trained"]),
        path=str(result["path"]),
        n_pos=int(result["n_pos"]),
        n_neg=int(result["n_neg"]),
        message=str(result["message"]),
    )
    bus.state("idle")


def _run_relabel(cfg: Config, word: str, hash8: str, action: str) -> None:
    """Relabel a saved clip (mark_false_fire / mark_miss) + eval-gated rebuild → relabel_result.

    The active-learning closed loop (Mycroft Precise's idea, repo #5): move the clip to the
    per-word NEGATIVES (``mark_false_fire``) or POSITIVES (``mark_miss``) dir, then rebuild the
    cheap CPU detector (few-shot refs + verifier + calibration) and KEEP it only if the eval
    gate (d-prime separation + miss@target-FA) does not regress, else ROLL BACK. Emits a
    ``relabel_result`` the GUI renders as an accepted / rolled-back card. Never raises.
    """
    from . import active_learning

    result = active_learning.relabel_clip(word, hash8, action, cfg)
    bus.log(("✓ " if result.accepted else "✗ ") + result.message)
    active_learning.emit_relabel_result(result)
    bus.state("idle")


def _run_capture_last_fire(cfg: Config, word: str, controller: Any) -> None:
    """Capture the last live wake fire's audio as a NEGATIVE + eval-gated rebuild → relabel_result.

    The ``capture_last_fire`` action (repo #5): pull the most-recent fire's audio window from
    the wake loop's ring buffer (owned by ``controller``), save it under
    ``debug/recordings/wake_neg/<word>/``, and run the SAME eval-gated rebuild as
    ``mark_false_fire``. When no fire has happened yet (or no controller) it reports
    ``rebuilt=False`` with a clear message. Never raises.
    """
    from . import active_learning

    clip = controller.last_fire_audio() if controller is not None else None
    if clip is None:
        result = active_learning.RelabelResult(
            word=word,
            action="capture_last_fire",
            rebuilt=False,
            accepted=False,
            sep_before=0.0,
            sep_after=0.0,
            fa_before=1.0,
            fa_after=1.0,
            message=f"{word}: no recent live wake fire to capture (start the wake loop first)",
        )
    else:
        result = active_learning.capture_last_fire(word, clip, cfg.sample_rate, cfg)
    bus.log(("✓ " if result.accepted else "✗ ") + result.message)
    active_learning.emit_relabel_result(result)
    bus.state("idle")


def _run_spectrogram(cfg: Config, word: str, hash8: str | None) -> None:
    """Compute a log-mel spectrogram + wake score-trace of a saved clip (the visual inspector).

    The GUI ``spectrogram`` action (Task 4): load the saved clip by ``hash`` (or the
    newest for ``word``), compute a downsampled log-mel grid (:func:`wake.log_mel_spectrogram`)
    AND the per-frame wake ``score_trace`` (:func:`wake.score_wake_clip`), and emit
    ``spectrogram_result``. Never raises — a missing clip is a clear message.
    """
    from .wake import log_mel_spectrogram, score_wake_clip, wake_model_for

    clip16k, resolved_hash, base = _load_saved_wake_clip(word, hash8)
    if clip16k is None:
        which = f"hash {hash8}" if hash8 else f"newest '{word}' recording"
        bus.log(f"spectrogram: no saved clip ({which})", "error")
        bus.spectrogram_result(
            hash=str(hash8 or ""),
            mels=0,
            frames=0,
            grid=[],
            score_trace=[],
            message=f"no saved clip ({which})",
        )
        bus.state("idle")
        return
    spec = log_mel_spectrogram(clip16k, 16000)
    score_trace: list[float] = []
    if os.path.isfile(wake_model_for(word)):
        _, _, score_trace = score_wake_clip(
            clip16k,
            16000,
            word,
            threshold=cfg.wake_threshold,
            phases=cfg.wake_phases,
            with_trace=True,
        )
    message = f"{word}: log-mel {spec['mels']}x{spec['frames']} on {base}"
    bus.log("✓ " + message)
    bus.spectrogram_result(
        hash=resolved_hash,
        mels=spec["mels"],
        frames=spec["frames"],
        grid=spec["grid"],
        score_trace=score_trace,
        freqs=spec.get("freqs"),
        times=spec.get("times"),
        message=message,
    )
    bus.state("idle")


def _voice_preset_name(voice_id: str) -> str:
    """Map a Piper voice id (``en_US-lessac-medium``) back to its friendly preset name."""
    for name, vid in VOICE_PRESETS.items():
        if vid == voice_id:
            return name
    return voice_id or "default"


def _voice_test(cfg: Config, tts: TTSRouter, data: dict[str, Any]) -> None:
    """Speak a short fixed sample line in the currently-selected voice (GUI play button).

    Synthesizes "This is the <voice> voice." with the active English voice and plays
    it server-side, so the GUI's voice-preview button produces audio. Honours a
    per-request ``voice_en`` (the page can preview a voice before saving it) by
    applying it to ``cfg`` first. Runs in a worker thread (the caller spawns it), so
    it never blocks the HTTP handler; defensive — a TTS failure is logged, not raised.
    """
    if data.get("voice_en"):
        cfg.tts_voices["en"] = str(data["voice_en"])
    voice = _voice_preset_name(cfg.tts_voices.get("en", ""))
    line = f"This is the {voice} voice."
    bus.log(f"playing voice sample: {voice}")
    try:
        tts.speak(line, "en")
    except Exception as exc:  # never let a TTS hiccup kill the worker
        log.error("voice test error: %s", exc)
        bus.log(f"voice test error: {exc}", "error")
    finally:
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

    # The audio debug instrument defaults ON under --browser (so the GUI EVENT LOG
    # shows the trace out of the box); an explicit DEBUG_AUDIO env overrides.
    dbg = _AudioDebug(debug_audio_enabled(cfg, browser=True))
    voice_available, voice_hint = _voice_status(cfg, stt)
    controller = (
        _WakeController(cfg, brain, tts, gate, stt, speaker_id=speaker_id, debug=dbg)
        if voice_available and stt is not None
        else None
    )

    def on_turn(text: str) -> None:
        # Typed-only from the page in this build -> guest bucket (no clip to embed).
        dbg.action("turn", chars=len(text))
        _respond(cfg, brain, tts, gate, text, turn_source=SOURCE_TYPED)

    def on_action(name: str, data: dict) -> None:
        dbg.action(name, **{k: v for k, v in data.items() if k != "action"})
        if name == "reset":
            brain.reset()
            bus.log("conversation reset")
        elif name == "list_voices":
            bus.log("voices: " + ", ".join(VOICE_PRESETS))
        elif name == "voice_test":
            # Play a short fixed sample line in the currently-selected voice so the
            # GUI's "play voice" button works. Server-side TTS in a worker thread so
            # the HTTP handler never blocks; honours a per-request "voice_en" override.
            threading.Thread(target=lambda: _voice_test(cfg, tts, data), daemon=True).start()
        elif name == "mic_test":
            # Diagnostic: capture ~1.5s and report a clear verdict. Runs regardless
            # of voice/wake state — it's *most* useful when voice is off (to find
            # out why). With a controller it pauses/restores the wake loop; without
            # one it runs a standalone capture in a worker (never block the handler).
            if controller is not None:
                controller.mic_test()
            else:
                threading.Thread(target=lambda: _run_mic_test(cfg), daemon=True).start()
        elif name == "mic_record_replay":
            # Record ~3s and play it back so the user hears their own mic — the most
            # unambiguous "is the mic working" check. Owns the device (pauses the wake
            # loop if needed); a non-silent capture emits mic_result(ok=True) so the
            # GUI can hide the macOS permission hint. Always runs in a worker thread.
            if controller is not None:
                controller.mic_record_replay()
            else:
                threading.Thread(target=lambda: _run_mic_record_replay(cfg), daemon=True).start()
        elif name == "mic_check":
            # Unified 2.0 s mic diagnostic (the new backend): server captures (with the
            # software mic_gain applied + clip protection) or a browser-recorded clip,
            # both saved under debug/recordings/ and reported via mic_check_result with
            # the level meter + ~48-window level-over-time graph + the saved-WAV link.
            source = str(data.get("source") or "server")
            if source == "browser":
                pcm = list(data.get("pcm") or [])
                rate = int(data.get("sample_rate") or cfg.sample_rate)
                proc = data.get("processing") if isinstance(data.get("processing"), dict) else None
                threading.Thread(
                    target=lambda: _run_mic_check_browser(cfg, pcm, rate, proc),
                    daemon=True,
                ).start()
            elif controller is not None:
                controller.mic_check()  # owns the device: pauses/restores the wake loop
            else:
                threading.Thread(target=lambda: _run_mic_check_server(cfg), daemon=True).start()
        elif name == "play_recording":
            # Play a saved mic/wake WAV (addressed by its 8-hex content hash) back
            # through the server speaker — best-effort, in a worker thread.
            threading.Thread(
                target=lambda: _play_recording(str(data.get("hash") or "")), daemon=True
            ).start()
        elif name in {"wake_start", "wake_stop", "ptt"}:
            if controller is None:
                bus.log(f"'{name}' unavailable: {voice_hint}", "error")
            elif name == "wake_start":
                controller.start_wake()
            elif name == "wake_stop":
                controller.stop_wake()
            else:  # ptt
                controller.push_to_talk()
        elif name in {"music_stop", "music_pause", "music_resume"}:
            _music_action(name, player=cfg.music_player, volume=cfg.music_volume)
        elif name == "wake_test":
            # Score a ~2 s clip against the model for the requested word (NOT
            # necessarily the configured wake word) and report whether it would
            # fire. source=server records from the server mic; source=browser scores
            # a clip the page recorded and POSTed as raw float PCM. Worker thread so
            # the HTTP handler never blocks; defensive against missing models / mics.
            word = str(data.get("word") or cfg.wake_phrase)
            source = str(data.get("source") or "server")
            if source == "browser":
                pcm = list(data.get("pcm") or [])
                rate = int(data.get("sample_rate") or cfg.sample_rate)
                proc = data.get("processing") if isinstance(data.get("processing"), dict) else None
                threading.Thread(
                    target=lambda: _run_wake_test_browser(cfg, word, pcm, rate, proc),
                    daemon=True,
                ).start()
            else:
                threading.Thread(
                    target=lambda: _run_wake_test_server(cfg, word), daemon=True
                ).start()
        elif name == "wake_gain_sweep":
            # Re-score a SAVED clip (by hash, or the newest wake recording for the
            # word) at each gain. THE smoking-gun test: if confidence climbs with
            # gain, a too-quiet capture was the dead-wake cause. Worker thread.
            word = str(data.get("word") or cfg.wake_phrase)
            hash8 = data.get("hash")
            hash8 = str(hash8) if hash8 else None
            raw_gains = data.get("gains")
            gains = (
                [float(g) for g in raw_gains]
                if isinstance(raw_gains, list) and raw_gains
                else [1.0, 2.0, 4.0, 8.0, 16.0]
            )
            threading.Thread(
                target=lambda: _run_wake_gain_sweep(cfg, word, hash8, gains), daemon=True
            ).start()
        elif name == "score_clip":
            # Score the EXACT saved WAV through the model (capture-vs-model fork): if a
            # known clip also scores ~0, the model/word is the problem, not capture.
            word = str(data.get("word") or cfg.wake_phrase)
            hash8 = data.get("hash")
            hash8 = str(hash8) if hash8 else None
            threading.Thread(target=lambda: _run_score_clip(cfg, word, hash8), daemon=True).start()
        elif name == "score_histogram":
            # Eval toolkit (Task 1): score the saved POSITIVES + the negative corpus and
            # emit both max-score arrays + a separation scalar — the recall-vs-level proof.
            word = str(data.get("word") or cfg.wake_phrase)
            threading.Thread(target=lambda: _run_score_histogram(cfg, word), daemon=True).start()
        elif name == "fa_eval":
            # Eval toolkit (Task 2): sweep the threshold over positives + the TIMED negative
            # corpus, counting FA EVENTS (not frames) -> FA/hour + ROC/DET points + miss@target.
            word = str(data.get("word") or cfg.wake_phrase)
            threading.Thread(target=lambda: _run_fa_eval(cfg, word), daemon=True).start()
        elif name == "train_verifier":
            # Eval toolkit (Task 3): train an openWakeWord-style logistic-regression verifier
            # from the saved positives (+ negatives). scikit-learn is the optional `debug` extra.
            word = str(data.get("word") or cfg.wake_phrase)
            threading.Thread(target=lambda: _run_train_verifier(cfg, word), daemon=True).start()
        elif name in {"mark_false_fire", "mark_miss"}:
            # Active-learning closed loop (Mycroft Precise's idea, repo #5): relabel a saved
            # clip as a NEGATIVE (mark_false_fire) or POSITIVE (mark_miss), then eval-gated
            # rebuild the cheap CPU detector (kept only if d-prime/FA don't regress, else
            # rolled back). Worker thread; emits relabel_result.
            word = str(data.get("word") or cfg.wake_phrase)
            relabel_hash = str(data.get("hash") or "")
            action_name = str(name)

            def _relabel(a: str = action_name, w: str = word, h: str = relabel_hash) -> None:
                _run_relabel(cfg, w, h, a)

            threading.Thread(target=_relabel, daemon=True).start()
        elif name == "capture_last_fire":
            # Active-learning (repo #5): save the last LIVE wake fire's audio (from the wake
            # loop's ring buffer) as a NEGATIVE, then behave like mark_false_fire (eval-gated
            # rebuild). Needs the wake loop to have fired at least once. Worker thread.
            word = str(data.get("word") or cfg.wake_phrase)
            threading.Thread(
                target=lambda: _run_capture_last_fire(cfg, word, controller), daemon=True
            ).start()
        elif name == "spectrogram":
            # Eval toolkit (Task 4): log-mel spectrogram + wake score-trace of a saved clip
            # (by hash, or the newest for the word) — the visual capture inspector.
            word = str(data.get("word") or cfg.wake_phrase)
            hash8 = data.get("hash")
            hash8 = str(hash8) if hash8 else None
            threading.Thread(target=lambda: _run_spectrogram(cfg, word, hash8), daemon=True).start()
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


def _uses_local_mic(args: argparse.Namespace, cfg: Config) -> bool:
    """Whether THIS run opens the local microphone (so the preflight gate applies).

    The mic-using modes are: ``--wake`` (terminal or GUI), ``--browser --browser-audio``
    (server-side capture for the browser-audio session), and the default terminal
    push-to-talk loop. The mic-LESS modes are skipped: ``--type`` / ``--text`` (typed,
    no STT), ``--browser`` without ``--wake``/``--browser-audio`` (state/transcript
    only), and the network/telephony servers whose mic lives on a remote client (not
    this host's sound card). Pure over the parsed args + resolved config.
    """
    if cfg.telephony or cfg.transport in ("websocket", "webrtc"):
        return False  # the device boundary is on the wire, not this host's mic
    if args.text is not None or args.type_mode:
        return False  # typed: no capture at all
    if args.browser:
        return bool(args.wake or args.browser_audio)  # GUI is mic-less unless these
    return True  # default terminal push-to-talk loop opens the local mic


def _audio_preflight_gate(cfg: Config, args: argparse.Namespace) -> int | None:
    """HARD STOP: run the startup audio preflight for mic modes; gate the launch.

    Returns ``None`` when it is safe to proceed (mic-less mode, the preflight passed,
    or it was explicitly skipped) and a NON-ZERO exit code when the preflight failed —
    so :func:`main` returns it WITHOUT opening the GUI or starting any capture, rather
    than presenting a control room that silently records nothing. A passing-but-marginal
    preflight still wires its device-rate / drop-ratio / reason into the debug
    instrument so the numbers are visible.
    """
    if not _uses_local_mic(args, cfg):
        return None  # mic-less mode: nothing to capture, nothing to gate
    if cfg.skip_audio_preflight:
        bus.log("audio preflight skipped (--skip-audio-preflight / SKIP_AUDIO_PREFLIGHT)", "info")
        return None
    result = audio.audio_preflight(cfg.sample_rate)
    if not result.ok:
        print(result.message, file=sys.stderr)
        bus.log(result.message, "error")
        return 3
    # Passed (possibly marginal): surface the numbers via the audio debug instrument.
    _AudioDebug(debug_audio_enabled(cfg, browser=args.browser)).action(
        "preflight",
        reason=result.reason,
        device_rate=result.device_rate,
        drop_ratio=result.drop_ratio,
        permission=result.permission,
    )
    return None


def _mpv_preflight_gate(cfg: Config) -> int | None:
    """HALT EARLY when launching the GUI without mpv (mirrors quickstart.sh).

    Music plays SERVER-side through mpv, and pause/resume need its IPC socket; without
    mpv "play <song>" can't work. ``quickstart.sh`` already checks before opening the
    browser, but a DIRECT ``./mstt --browser`` launch bypasses it — this covers that
    path. Returns a NON-ZERO exit code (so :func:`main` returns it BEFORE opening the
    browser) when mpv is missing, ``None`` when it is safe to proceed. No-op (returns
    ``None``) when music is disabled or the check is opted out with
    ``MSTT_SKIP_MPV_CHECK=1``. Unlike quickstart it does not try to install mpv — it
    only gates and points at the install command (the browser launch is non-recoverable
    here, so a clear stop beats a half-working control room)."""
    if os.environ.get("MSTT_SKIP_MPV_CHECK"):
        return None
    if not cfg.music_enabled:
        return None  # no music -> mpv is irrelevant
    if shutil.which("mpv") is not None:
        return None
    # Only HARD-HALT on an INTERACTIVE launch (a real terminal). In CI / tests /
    # headless runs (no TTY) mpv legitimately may be absent, and exiting non-zero
    # there would break the suite and any headless deployment — so warn, don't halt.
    if not sys.stderr.isatty():
        warn = "mpv not installed — music playback disabled (set MSTT_SKIP_MPV_CHECK=1 to silence)."
        bus.log(warn, "warning")
        return None
    msg = (
        "mpv is not installed (needed for music playback + pause/resume) — halting "
        "before opening the browser.\n"
        "  Install it (macOS: brew install mpv | Linux: apt install mpv), then re-run.\n"
        "  Or skip this check: MSTT_SKIP_MPV_CHECK=1 ./mstt --browser"
    )
    print(msg, file=sys.stderr)
    bus.log(msg, "error")
    return 4


def main(argv: list[str] | None = None) -> int:
    """Validate config, build components, run the chosen loop. Returns exit code."""
    args = _parse_args(argv)
    if args.list_voices:
        print("English voice presets (use --voice <name>):")
        print(list_voice_presets())
        return 0
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    # Sink B: bridge the Python logging library (+ captured warnings) onto the event
    # bus so the GUI EVENT LOG / file sink become a superset of library + app logs
    # (onnxruntime / Hugging Face warnings, httpx requests, our own logging). Installed
    # ONCE here at startup for every run mode (browser, wake, terminal, transport);
    # library/test imports never wire it. Sink A (bus -> stderr console) is auto-on
    # whenever MSTT_EVENT_LOG is set (quickstart sets it) — no wiring needed here.
    install_log_bridge()
    try:
        cfg = _build_config(args)
        if args.settings:
            print(settings_text(cfg))
            return 0
        if args.preflight:
            from .preflight import preflight_main

            return preflight_main(cfg)
        if args.benchmark:
            # The noise×SNR benchmark reads SAVED clips only — no mic, no heavy STT.
            # Validate config (so a bad wake word fails clearly) then run + exit.
            cfg.validate()
            return _run_benchmark(cfg, args.benchmark_snr)
        cfg.validate()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    # HARD STOP (broken-audio gate): for the mic-using modes, confirm a usable 16 kHz
    # capture path BEFORE loading the heavy STT model, opening the GUI, or starting any
    # capture — so a broken-audio host refuses to run with a clear error instead of a
    # control room that silently records nothing. Mic-less modes pass through.
    gate_rc = _audio_preflight_gate(cfg, args)
    if gate_rc is not None:
        return gate_rc

    # HALT EARLY (mpv preflight, mirroring quickstart.sh): a DIRECT ./mstt --browser
    # launch bypasses quickstart's mpv check, so cover it here — refuse to open the
    # control room with no mpv (music + pause/resume would silently fail) unless the
    # user opts out with MSTT_SKIP_MPV_CHECK=1. Only the GUI launches a long-lived
    # server where this matters; terminal/transport modes pass through.
    if args.browser:
        mpv_rc = _mpv_preflight_gate(cfg)
        if mpv_rc is not None:
            return mpv_rc

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

        ptt_vad = SileroVad(cfg.sample_rate, cfg.vad_threshold)
    print("Push-to-talk: start speaking; it stops on a natural pause. Ctrl-C to quit.")
    while True:
        run_turn(cfg, brain, tts, gate, stt=stt, vad=ptt_vad, speaker_id=speaker_id)
        if args.once:
            return


if __name__ == "__main__":
    raise SystemExit(main())
