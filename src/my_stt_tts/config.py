"""Central configuration: load from environment / .env, validate fail-fast."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROVIDERS = ("anthropic", "openai", "openai-compatible", "ollama", "claude-cli", "codex-cli")
LANGUAGES = ("de", "fr", "en")

# Where pre-shipped wake-word ONNX models live. The repo ships several trained
# models here (e.g. maziko.onnx, jarvis.onnx, computer.onnx); a user PICKS one by
# name via ``wake_phrase`` and the model path auto-derives (see wake_model_for /
# Config.from_env). The actual .onnx files are committed by the orchestrator.
WAKEWORDS_DIR = "wakewords"


def wake_model_for(phrase: str, wakewords_dir: str = WAKEWORDS_DIR) -> str:
    """The conventional wake-model path for a wake-word ``phrase``.

    Selecting a pre-shipped wake word is just setting ``wake_phrase``: the model is
    found at ``<wakewords_dir>/<phrase>.onnx``. Keeps the path convention in one
    place so the CLI, env, and web UI all derive it identically.
    """
    return f"{wakewords_dir.rstrip('/')}/{phrase}.onnx"


def available_wake_words(wakewords_dir: str | os.PathLike[str] = WAKEWORDS_DIR) -> list[str]:
    """Names of the pre-shipped wake-word models present in ``wakewords_dir``.

    Discovers whatever ``*.onnx`` models are committed (the stem of each file),
    sorted, so the UI / CLI / ``--settings`` can offer the real choices without
    hard-coding filenames. Returns ``[]`` when the directory is missing or empty
    — callers must not assume any specific model exists.
    """
    directory = Path(wakewords_dir)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.onnx") if p.is_file())


# Measurement systems the assistant can answer in (see Config.units). Injected into
# the system prompt so temperatures/distances/etc. come back in the right system,
# and consumed by the get_weather tool to pick °C/km·h vs °F/mph.
UNITS = ("metric", "imperial")

# Brain mode (R3-5). The cascade is STT -> LLM -> TTS (everything to date); the
# realtime mode streams mic audio to a speech-to-speech endpoint (OpenAI Realtime
# over WebSocket) and plays the returned audio back, bypassing the cascade's
# irreducible turn latency. ``realtime`` is key-gated and falls back to ``cascade``
# when no key/endpoint is configured (see realtime.make_realtime_brain).
BRAIN_MODES = ("cascade", "realtime")

# Audio transport modes (R2-5/R3-1; see transport.py). Where the loop sources mic
# audio and sinks TTS audio:
#   local      — the local sound card via sounddevice (default, today's behaviour)
#   websocket  — a network link: a WS server feeds remote satellites / the browser
#   webrtc     — a real WebRTC peer (Opus + jitter buffer + ICE/NAT, the `webrtc`
#                extra: aiortc); the browser uses a real RTCPeerConnection
TRANSPORT_MODES = ("local", "websocket", "webrtc")

# Pre-VAD noise-suppression backends (R3-6; see denoise.py). Applied to mic frames
# AFTER echo cancellation and BEFORE VAD/STT so STT accuracy rises and false
# barge-ins fall in noisy rooms:
#   off       — no denoising (default)
#   spectral  — pure-numpy spectral-gate noise reduction (always available)
#   rnnoise   — RNNoise via an optional wheel; falls back to spectral if missing
DENOISER_MODES = ("off", "spectral", "rnnoise")

# STT backend selection (R2-7 / G1). Local-first; cloud backends are opt-in and
# key-gated (graceful fallback to local). Names are resolved by the backend
# registry (registry.py); validation cross-checks against the registered set.
#   local          — on-device parakeet-mlx (default, Apple Silicon)
#   whispercpp     — whisper.cpp via pywhispercpp (cross-platform; G8 off-Mac brain)
#   faster-whisper — faster-whisper / CTranslate2 (Linux CPU/GPU; G8 off-Mac brain)
#   cloud / openai — an OpenAI-compatible transcription endpoint
#   deepgram       — Deepgram streaming STT (real adapter; key-gated)
STT_BACKENDS = ("local", "whispercpp", "faster-whisper", "cloud", "openai", "deepgram")

# TTS backend selection (R2-7 / G1). Local-first; cloud backends are opt-in and
# key-gated (e.g. a high-quality cloud German voice, the local weak spot).
#   local            — Piper / macOS say (default)
#   cloud / openai   — an OpenAI-compatible speech endpoint (e.g. OpenAI TTS)
#   elevenlabs       — ElevenLabs neural TTS (real adapter; key-gated)
#   cartesia         — Cartesia Sonic neural TTS (real adapter; key-gated)
TTS_BACKENDS = ("local", "cloud", "openai", "elevenlabs", "cartesia")

# Barge-in safety modes (see Config.barge_in). Without acoustic echo cancellation
# (AEC) an open speaker bleeds into the mic, so interruption is opt-in:
#   off         — half-duplex: mic is gated shut during playback (legacy behaviour)
#   headphones  — barge-in ON; safe because headphones don't leak into the mic
#   always      — barge-in ON even on open speakers (relies on the energy gate;
#                 may self-trigger without AEC — documented caveat)
BARGE_IN_MODES = ("off", "headphones", "always")

# End-of-turn analyzer choices (see turn.py).
TURN_ANALYZERS = ("silence", "smart")

# Acoustic echo cancellation modes (see aec.py). Removes the assistant's own TTS
# from the mic so barge-in works on open speakers, not just headphones:
#   off             — no AEC (legacy; barge-in reliable only with headphones)
#   nlms            — pure-numpy adaptive filter referencing the played signal
#   voiceprocessing — macOS hardware AEC (AVAudioEngine VoiceProcessingIO), NLMS fallback
#   webrtc          — Linux WebRTC Audio Processing Module (APM) AEC, NLMS fallback (G8)
#   auto            — macOS HW AEC if available, else WebRTC-APM on Linux, else NLMS
AEC_MODES = ("off", "nlms", "voiceprocessing", "webrtc", "auto")

# Smart Turn v3 ONNX model: auto-downloaded on first run (like Piper voices) so
# smart endpointing works out of the box. Pinned to the upstream pipecat release.
SMART_TURN_MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx"
)
# Pinned SHA-256 of the upstream smart-turn-v3.0.onnx (verified 2026-06-19). The
# preflight (R3-8) and the runtime download checksum the file against this so a
# silently-truncated or tampered download is rejected rather than used.
SMART_TURN_MODEL_SHA256 = "07a133aba31e2d0b523f17f8c2e4e65efe6d8f685efd12ca4fe21ebf4e798991"

# Friendly one-word brain presets -> (provider, model).
# "-sub" uses your Claude subscription via the Claude Code CLI (no API key); the
# bare aliases haiku/sonnet/opus resolve to the latest version automatically.
# "-api" uses the Anthropic API (needs ANTHROPIC_API_KEY) pinned to latest ids.
# "codex" uses your logged-in OpenAI codex CLI (no API key, `codex exec`).
BRAIN_PRESETS: dict[str, tuple[str, str]] = {
    "haiku-sub": ("claude-cli", "haiku"),
    "sonnet-sub": ("claude-cli", "sonnet"),
    "opus-sub": ("claude-cli", "opus"),
    "haiku-api": ("anthropic", "claude-haiku-4-5"),
    "sonnet-api": ("anthropic", "claude-sonnet-4-6"),
    "opus-api": ("anthropic", "claude-opus-4-8"),
    "ollama": ("ollama", "llama3.1"),  # also set LLM_BASE_URL=http://localhost:11434/v1
    "codex": ("codex-cli", "gpt-5-codex"),  # OpenAI codex CLI (no API key, `codex exec`)
}

# Fallback if the editable repo's prompts/system_prompt.md can't be found.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a calm, concise voice assistant. Your reply is spoken aloud and the "
    "user never sees text: no markdown, lists, code, emoji, or URLs. Speak in one "
    "to three short sentences, spell numbers and dates as words, reply in the "
    "language the user spoke, and use metric units and ISO-8601 dates."
)


class ConfigError(ValueError):
    """Raised when the resolved configuration is invalid."""


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _repo_prompt_file() -> Path:
    """Path to the editable in-repo system prompt (resolved from this package)."""
    return Path(__file__).resolve().parents[2] / "prompts" / "system_prompt.md"


def load_system_prompt(override: str | os.PathLike[str] | None = None) -> str:
    """Read the system prompt from a file (override or repo default), or fall back."""
    path = Path(override) if override else _repo_prompt_file()
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return _DEFAULT_SYSTEM_PROMPT
    return text or _DEFAULT_SYSTEM_PROMPT


def locale_prompt_line(base_prompt: str, location: str, units: str) -> str:
    """Append a location + units awareness line to ``base_prompt``.

    Keeps the editable base prompt (``prompts/system_prompt.md``) intact and adds a
    single sentence so the assistant is generally location- and units-aware (weather,
    distances, temperatures). Returns the base unchanged when location is blank.
    """
    place = location.strip()
    if not place:
        return base_prompt
    line = (
        f"The user is in {place} and uses {units} units; "
        "answer measurements, distances, and temperatures accordingly."
    )
    return f"{base_prompt.rstrip()}\n\n{line}"


# Map a free-text location to an IANA timezone. Lausanne and the rest of
# Switzerland live in Europe/Zurich; extend as new default locations are added.
# Anything unmatched falls back to the system local timezone.
_LOCATION_TZ: dict[str, str] = {
    "lausanne": "Europe/Zurich",
    "geneva": "Europe/Zurich",
    "zurich": "Europe/Zurich",
    "bern": "Europe/Zurich",
    "switzerland": "Europe/Zurich",
}


def timezone_for_location(location: str) -> Any:  # noqa: ANN401 — tzinfo | None
    """Resolve a free-text ``location`` to a :class:`zoneinfo.ZoneInfo`, else None.

    Matches a known place/country substring (Lausanne → ``Europe/Zurich``) using
    stdlib ``zoneinfo``; returns ``None`` (caller uses the system local tz) when the
    location is unknown or the tz database is unavailable. Pure + dependency-free.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    place = location.strip().lower()
    for key, tz in _LOCATION_TZ.items():
        if key in place:
            try:
                return ZoneInfo(tz)
            except ZoneInfoNotFoundError:
                return None
    return None


def current_time_line(location: str, *, now: Any = None) -> str:
    """A single 'current local time' sentence for the assembled system prompt.

    Injected EVERY turn so the assistant can tell the time regardless of brain —
    including ``claude-cli``, which has no tool access here, so the time must be in
    the prompt text, not behind a tool. The timezone is derived from ``location``
    (Lausanne → ``Europe/Zurich``) via stdlib ``zoneinfo``, falling back to the
    system local tz. ``now`` is injectable for deterministic tests.
    """
    import datetime as _dt

    tz = timezone_for_location(location)
    moment = now if now is not None else _dt.datetime.now(tz=tz)
    if moment.tzinfo is None:
        moment = moment.astimezone(tz) if tz is not None else moment.astimezone()
    elif tz is not None:
        moment = moment.astimezone(tz)
    label = str(moment.tzinfo) if moment.tzinfo is not None else "local time"
    return f"Current local time: {moment.strftime('%Y-%m-%d %H:%M')} ({label})."


@dataclass(slots=True)
class Config:
    """All tunables for the voice loop, with sane defaults.

    Build via :meth:`from_env` (reads ``.env`` + environment), then call
    :meth:`validate` before starting the pipeline.
    """

    # --- LLM (provider-agnostic; Anthropic is the default) ---
    llm_provider: str = "anthropic"
    llm_model: str = "claude-haiku-4-5"
    llm_model_deep: str = "claude-opus-4-8"
    llm_base_url: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    deep_trigger: str = "think hard"
    max_history_turns: int = 10
    requests_per_minute: int = 30
    # Spoken-output system prompt; edit prompts/system_prompt.md to change it.
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT

    # --- Locale (location + measurement units) ---
    # Injected into the assembled system prompt so the assistant is generally
    # location- and units-aware (weather, "how far is…", temperatures), and used as
    # the default for the get_weather tool. ``units`` is one of :data:`UNITS`.
    location: str = "Lausanne, Switzerland"
    units: str = "metric"

    # --- Agent dispatch (Phase 6): "agent, <task>" hands off to a full,
    # MCP-capable Claude Code agent in `agent_workspace`. Disabled until a
    # workspace is set (a capable agent should not run in an arbitrary dir). ---
    agent_trigger: str = "agent"
    agent_workspace: str | None = None
    agent_model: str = "sonnet"

    # --- Wake / capture ---
    wake_phrase: str = "maziko"
    wake_model_path: str = "wakewords/maziko.onnx"
    # openWakeWord score (0..1) a frame must clear to fire the wake word. Lower =
    # triggers more easily (more false-positives); higher = stricter (may miss a
    # quiet "maziko"). 0.4 is a touch more sensitive than openWakeWord's 0.5
    # default so a soft wake word still fires. Env: WAKE_THRESHOLD.
    wake_threshold: float = 0.4
    # How many sub-frame PHASE-OFFSET detectors evaluate each wake-word pass.
    # openWakeWord scores once per 1280-sample (80 ms) frame, locked to ONE phase
    # relative to the spoken word — and the maziko score swings ~25x (0.03..0.85)
    # purely with where that frame boundary lands. In an always-listening loop the
    # frame grid is fixed by capture timing, so a single utterance gets ONE phase:
    # an unlucky alignment scores ~0.03 and never fires even though the SAME audio
    # at a better offset scores ~0.7 (the "fires offline, never live" bug). Running
    # K detectors offset by 1280/K samples and firing on the max covers the phase
    # space — measured to lift recall from 2/8 to 5/8 synthesized voices with no
    # extra false-positives and a 0.22 real-time factor at K=8. 1 disables it.
    # Env: WAKE_PHASES.
    wake_phases: int = 8
    follow_up_seconds: float = 8.0
    sample_rate: int = 16000
    preroll_seconds: float = 0.3
    max_record_seconds: float = 30.0
    vad_silence_seconds: float = 0.7
    # Silero-VAD speech probability above which a frame counts as speech. Kept low
    # so a quiet-but-present voice (a ~10% mic level) is still captured — a high
    # threshold was treating soft speech as silence and ending the turn empty.
    vad_threshold: float = 0.3
    mic_gate_tail_seconds: float = 0.2

    # --- Barge-in / interruption (Phase 7) ---
    # Keep the mic live during playback and abort TTS + the in-flight LLM stream
    # on confirmed user speech. `barge_in` is the master switch (see BARGE_IN_MODES).
    barge_in: str = "off"
    # Ignore playback bleed: a frame only counts as a barge-in candidate when its
    # RMS energy clears this floor (0..1, float32 mono). Tuned to reject the
    # low-level speaker leak heard by an open mic without AEC.
    barge_in_energy: float = 0.02
    # False-interrupt suppression (pipecat MinWords equivalent): an interruption
    # is only honoured once the user has spoken for at least this long AND/OR
    # produced at least this many words. Backchannels / coughs / TV stay ignored.
    interrupt_min_speech_ms: float = 350.0
    interrupt_min_words: int = 2

    # --- Acoustic echo cancellation (R2-1) ---
    # Remove the assistant's own TTS from the mic so an OPEN speaker doesn't
    # self-trigger barge-in. "auto" uses macOS hardware AEC when available and the
    # software NLMS filter otherwise (see AEC_MODES / aec.py). When AEC is active
    # the barge-in energy floor is relaxed (echo is already gone).
    aec_mode: str = "off"
    aec_nlms_taps: int = 256  # FIR length of the software adaptive filter
    aec_nlms_mu: float = 0.3  # NLMS step size in (0, 2]; higher = faster but less stable
    # R3-4: when aec_mode=voiceprocessing, capture mic audio THROUGH the macOS
    # AVAudioEngine VoiceProcessingIO input node (PyObjC) so HARDWARE-cancelled PCM
    # reaches Python — instead of capturing via plain sounddevice and cancelling in
    # software. Falls back to sounddevice + NLMS if the PyObjC bridge is unavailable.
    aec_hw_capture: bool = True

    # --- Pre-VAD noise suppression (R3-6) ---
    # Clean mic frames AFTER echo cancellation and BEFORE VAD/STT to raise STT
    # accuracy and cut false barge-ins in noisy rooms. See DENOISER_MODES / denoise.py.
    denoiser: str = "off"
    denoiser_strength: float = 1.0  # spectral-gate over-subtraction factor (>=0)

    # --- Acoustic interruption prediction (R2-3) ---
    # A 3rd, purely-acoustic barge-in guard: score sustained voiced energy + pitch/
    # spectral-flux to detect *intent to take the floor* before words transcribe.
    # Composes with the duration + word guards. Set 0 to require accumulated score.
    interrupt_predict: bool = True
    interrupt_predict_threshold: float = 0.6  # score in [0,1] above which it fires
    interrupt_predict_min_ms: float = 240.0  # sustained voiced time before it can fire

    # --- End-of-turn analysis (Phase 4/7, R2-4) ---
    # "smart": Smart Turn v3 prosodic model (auto-downloaded on first run, like
    # Piper voices) — the DEFAULT so a natural pause is not cut off. Falls back to
    # the fixed silence timer only when the model/runtime is genuinely unavailable.
    # "silence": always-available fixed silence timer (explicit opt-out).
    turn_analyzer: str = "smart"
    smart_turn_model_path: str = "models/smart-turn-v3.0.onnx"
    smart_turn_model_url: str = SMART_TURN_MODEL_URL
    smart_turn_sha256: str = SMART_TURN_MODEL_SHA256  # integrity pin (R3-8); "" disables
    smart_turn_auto_download: bool = True
    smart_turn_threshold: float = 0.5

    # --- STT ---
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    # Incremental STT: emit partial transcripts by re-transcribing a BOUNDED sliding
    # window of recent audio (R2-2) so partial latency / CPU don't grow with the
    # utterance length; final on end-of-turn. ``stt_window_s`` caps the re-decode.
    stt_streaming: bool = False
    stt_partial_interval_ms: float = 600.0
    stt_window_s: float = 7.0

    # --- TTS (per-language voice maps; Piper voice ids and macOS `say` voices) ---
    default_language: str = "en"
    tts_voices: dict[str, str] = field(
        default_factory=lambda: {
            "de": "de_DE-thorsten-high",
            "fr": "fr_FR-tom-medium",
            "en": "en_US-lessac-medium",
        }
    )
    say_voices: dict[str, str] = field(
        default_factory=lambda: {"de": "Anna", "fr": "Thomas", "en": "Ava"}
    )
    piper_data_dir: str = "voices"
    tts_length_scale: float = 1.1  # Piper duration multiplier; >1 = slower/calmer
    # --- Streamed, low-latency TTS playout (R3-3) ---
    # Render + play the reply CLAUSE-by-clause through a sounddevice OutputStream
    # (local) / the transport sink (network), so first audio plays within a few
    # hundred ms instead of after the whole sentence is synthesized. Keeps cancel
    # semantics for barge-in. `tts_streaming` is the master switch.
    tts_streaming: bool = True
    tts_stream_min_chars: int = 12  # min speakable chars before a sub-sentence clause fires
    tts_stream_frame: int = 1024  # OutputStream write block (samples) for streamed playout

    # --- Speaker ID (G7): per-utterance ECAPA embedding -> enrolled name so
    # conversation memory keys per-person. Opt-in: only active when enabled AND
    # ``enroll_dir`` holds centroids AND speechbrain is installed; otherwise the
    # loop runs with speaker=None (shared guest bucket) at zero added latency. ---
    speaker_id_enabled: bool = False
    speaker_threshold: float = 0.45
    speaker_margin: float = 0.06
    enroll_dir: Path = field(default_factory=lambda: Path("enroll"))

    # --- Audio transport (R2-5): move mic/TTS audio over the wire for satellites /
    # the browser. `local` (sounddevice) is the default; `websocket` runs a server
    # that bridges remote clients into this same pipeline. `transport_token`, when
    # set, is a shared secret the client must present in its handshake. ---
    transport: str = "local"
    transport_host: str = "0.0.0.0"  # noqa: S104 — bind LAN-wide so satellites can reach it
    transport_port: int = 8770
    transport_token: str | None = None

    # --- Tool / function calling (R2-7): let the model call tools mid-conversation
    # (get_time, calculator, home_control routing to the agent/HA dispatch). Works
    # with the anthropic + openai providers' native tool-use round-trip. The legacy
    # "agent, ..." trigger still works; this is the inline upgrade. ---
    tools_enabled: bool = True
    tools_max_iterations: int = 4  # cap tool-use loops so a model can't spin forever

    # --- Cloud STT/TTS backends (R2-7): optional, behind the existing seams.
    # Local-first defaults; cloud is selected explicitly and degrades gracefully
    # when no API key is set. Both speak an OpenAI-compatible API by default. ---
    stt_backend: str = "local"
    stt_cloud_model: str = "whisper-1"
    stt_cloud_base_url: str | None = None  # defaults to the OpenAI endpoint
    stt_cloud_api_key: str | None = None
    tts_backend: str = "local"
    tts_cloud_model: str = "gpt-4o-mini-tts"
    tts_cloud_voice: str = "alloy"
    tts_cloud_base_url: str | None = None
    tts_cloud_api_key: str | None = None

    # --- Real provider adapters (G1): Deepgram STT, ElevenLabs/Cartesia TTS.
    # Each is key-gated and selected via stt_backend/tts_backend; a missing key
    # degrades gracefully to the local backend (registry.py). ---
    deepgram_model: str = "nova-3"
    deepgram_api_key: str | None = None
    deepgram_language: str | None = None  # None => Deepgram auto-detects
    elevenlabs_model: str = "eleven_multilingual_v2"  # DE/FR/EN
    elevenlabs_voice_id: str = "Rachel"
    elevenlabs_api_key: str | None = None
    cartesia_model: str = "sonic-2"
    cartesia_voice_id: str = ""  # required for Cartesia (available() is false without it)
    cartesia_api_key: str | None = None

    # --- Cross-platform / off-Mac brain (G8): a Linux box can be the central brain
    # with Mac/ESP32 satellites. ``platform`` auto-detects the OS; ``playback`` and
    # ``aec`` seams pick a native path. macOS path is unchanged when auto-detected. ---
    platform: str = "auto"  # auto | macos | linux
    playback_backend: str = "auto"  # auto | sounddevice | aplay (Linux) | afplay (macOS)
    # Non-MLX STT model id for the whispercpp / faster-whisper cross-platform backends.
    whispercpp_model: str = "large-v3-turbo"
    faster_whisper_compute: str = "int8"  # int8 | int8_float16 | float16 | float32

    # --- Per-speaker persistent memory (G7): cross-session recall keyed by the
    # enrolled speaker. Disabled (in-memory only) until a store path is set. ---
    memory_store: str | None = None  # path to the SQLite/JSON store; None => off
    memory_max_turns: int = 40  # per-speaker history cap loaded into context

    # --- Speech-to-speech / realtime LLM (R3-5): bypass the STT->LLM->TTS cascade.
    # ``brain=realtime`` streams mic audio to a realtime speech-to-speech endpoint
    # (OpenAI Realtime over WebSocket) and plays the returned audio back. Key-gated:
    # without a key/endpoint the loop falls back to the cascade. ---
    brain_mode: str = "cascade"
    realtime_model: str = "gpt-4o-realtime-preview"
    realtime_url: str = "wss://api.openai.com/v1/realtime"
    realtime_api_key: str | None = None  # falls back to OPENAI_API_KEY
    realtime_voice: str = "alloy"
    # The endpoint's audio format. ``pcm16`` (24 kHz mono int16) is the default;
    # ``g711_ulaw`` (8 kHz μ-law) is handy when bridging straight to telephony.
    realtime_audio_format: str = "pcm16"

    # --- Per-stage latency telemetry (R3-7): record STT / LLM-first-token / TTS /
    # first-audio latencies per turn, keyed by a speech_id, to events.bus + a
    # JSON-lines log, with an optional OpenTelemetry span. Off by default. ---
    telemetry: bool = False
    telemetry_log_file: str | None = None  # JSON-lines path; None = no file
    telemetry_otel: bool = False  # emit an OpenTelemetry span per turn (lazy import)

    # --- Telephony reach (R3-9): answer a phone call via Twilio Media Streams over
    # the existing WebSocket transport (base64 μ-law 8 kHz <-> int16 PCM, 8k<->16k
    # resample). Behind a toggle; uses the same 'transport' extra (websockets). ---
    telephony: bool = False
    telephony_host: str = "0.0.0.0"  # noqa: S104 — bind LAN-wide so Twilio can reach it
    telephony_port: int = 8771

    debug: bool = False
    # Heavy audio-pipeline tracing (the GUI "debugger"): logs every GUI action and,
    # for every capture, the sample_rate / #samples / duration / rms / peak, the
    # VAD + endpoint decisions, the per-evaluation wake max-score, and the STT input
    # length + transcript. Surfaced to stderr AND the event bus (``bus.debug``) so it
    # shows in the GUI EVENT LOG, making *where* audio is lost obvious. ``None`` =
    # auto (ON under ``--browser``); an explicit ``DEBUG_AUDIO`` env var overrides.
    debug_audio: bool | None = None

    # Wake-debug recorder: when the wake loop starts, capture the first ~5 s of the
    # EXACT post-resample 16 kHz int16 frames fed to the wake model, save them as a
    # mono WAV (``wake_debug_path``) and log rate / #samples / duration / RMS / peak /
    # max+mean wake score — so a never-firing wake word is diagnosable as capture
    # (wrong rate / silent / clipped) vs model recall (good audio, low score) from a
    # single file. ``None`` = auto (ON when the audio debug instrument is on, e.g.
    # under ``--browser``); an explicit ``WAKE_DEBUG_CAPTURE`` env var overrides.
    wake_debug_capture: bool | None = None
    # Where the wake-debug recorder writes its WAV. Default lands under the user
    # cache dir so it's easy to find and attach. Env: WAKE_DEBUG_PATH.
    wake_debug_path: str = "~/.cache/my-stt-tts/wake-debug.wav"
    # Seconds of the wake stream captured by the recorder (the first N s after Start).
    wake_debug_seconds: float = 5.0

    # Skip the startup audio preflight HARD STOP (the broken-audio gate that refuses
    # to open the GUI / start a mic loop when capture can't deliver 16 kHz or the mic
    # queue persistently overflows). Power-user escape hatch — also ``--skip-audio-
    # preflight`` and the ``SKIP_AUDIO_PREFLIGHT`` env var.
    skip_audio_preflight: bool = False

    @classmethod
    def from_env(cls, dotenv_path: str | os.PathLike[str] | None = None) -> Config:
        """Build a Config from environment variables (loading ``.env`` first)."""
        load_dotenv(dotenv_path, override=False)
        env = os.environ
        cfg = cls(
            llm_provider=env.get("LLM_PROVIDER", "anthropic"),
            llm_model=env.get("LLM_MODEL", "claude-haiku-4-5"),
            llm_model_deep=env.get("LLM_MODEL_DEEP", "claude-opus-4-8"),
            llm_base_url=env.get("LLM_BASE_URL") or None,
            anthropic_api_key=env.get("ANTHROPIC_API_KEY") or None,
            openai_api_key=env.get("OPENAI_API_KEY") or None,
            system_prompt=load_system_prompt(env.get("SYSTEM_PROMPT_FILE")),
            location=env.get("LOCATION", "Lausanne, Switzerland"),
            units=env.get("UNITS", "metric"),
            agent_trigger=env.get("AGENT_TRIGGER", "agent"),
            agent_workspace=env.get("AGENT_WORKSPACE") or None,
            agent_model=env.get("AGENT_MODEL", "sonnet"),
            # Selecting a pre-shipped wake word = setting WAKE_PHRASE: the model path
            # auto-derives as wakewords/<phrase>.onnx. WAKE_MODEL_PATH is an explicit
            # override (a custom-trained model elsewhere) and wins when set.
            wake_phrase=env.get("WAKE_PHRASE", "maziko"),
            wake_model_path=env.get("WAKE_MODEL_PATH")
            or wake_model_for(env.get("WAKE_PHRASE", "maziko")),
            stt_model=env.get("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3"),
            piper_data_dir=env.get("PIPER_DATA_DIR", "voices"),
            barge_in=env.get("BARGE_IN", "off"),
            aec_mode=env.get("AEC_MODE", "off"),
            aec_hw_capture=_env_bool("AEC_HW_CAPTURE", default=True),
            denoiser=env.get("DENOISER", "off"),
            tts_streaming=_env_bool("TTS_STREAMING", default=True),
            turn_analyzer=env.get("TURN_ANALYZER", "smart"),
            smart_turn_model_path=env.get("SMART_TURN_MODEL_PATH", "models/smart-turn-v3.0.onnx"),
            smart_turn_model_url=env.get("SMART_TURN_MODEL_URL", SMART_TURN_MODEL_URL),
            smart_turn_sha256=env.get("SMART_TURN_SHA256", SMART_TURN_MODEL_SHA256),
            smart_turn_auto_download=_env_bool("SMART_TURN_AUTO_DOWNLOAD", default=True),
            stt_streaming=_env_bool("STT_STREAMING", default=False),
            interrupt_predict=_env_bool("INTERRUPT_PREDICT", default=True),
            transport=env.get("TRANSPORT", "local"),
            transport_host=env.get("TRANSPORT_HOST", "0.0.0.0"),  # noqa: S104 — LAN bind
            transport_token=env.get("TRANSPORT_TOKEN") or None,
            tools_enabled=_env_bool("TOOLS_ENABLED", default=True),
            speaker_id_enabled=_env_bool("SPEAKER_ID", default=False),
            stt_backend=env.get("STT_BACKEND", "local"),
            stt_cloud_model=env.get("STT_CLOUD_MODEL", "whisper-1"),
            stt_cloud_base_url=env.get("STT_CLOUD_BASE_URL") or None,
            stt_cloud_api_key=env.get("STT_CLOUD_API_KEY") or env.get("OPENAI_API_KEY") or None,
            tts_backend=env.get("TTS_BACKEND", "local"),
            tts_cloud_model=env.get("TTS_CLOUD_MODEL", "gpt-4o-mini-tts"),
            tts_cloud_voice=env.get("TTS_CLOUD_VOICE", "alloy"),
            tts_cloud_base_url=env.get("TTS_CLOUD_BASE_URL") or None,
            tts_cloud_api_key=env.get("TTS_CLOUD_API_KEY") or env.get("OPENAI_API_KEY") or None,
            deepgram_model=env.get("DEEPGRAM_MODEL", "nova-3"),
            deepgram_api_key=env.get("DEEPGRAM_API_KEY") or None,
            deepgram_language=env.get("DEEPGRAM_LANGUAGE") or None,
            elevenlabs_model=env.get("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            elevenlabs_voice_id=env.get("ELEVENLABS_VOICE_ID", "Rachel"),
            elevenlabs_api_key=env.get("ELEVENLABS_API_KEY") or None,
            cartesia_model=env.get("CARTESIA_MODEL", "sonic-2"),
            cartesia_voice_id=env.get("CARTESIA_VOICE_ID", ""),
            cartesia_api_key=env.get("CARTESIA_API_KEY") or None,
            platform=env.get("PLATFORM", "auto"),
            playback_backend=env.get("PLAYBACK_BACKEND", "auto"),
            whispercpp_model=env.get("WHISPERCPP_MODEL", "large-v3-turbo"),
            faster_whisper_compute=env.get("FASTER_WHISPER_COMPUTE", "int8"),
            memory_store=env.get("MEMORY_STORE") or None,
            brain_mode=env.get("BRAIN_MODE", "cascade"),
            realtime_model=env.get("REALTIME_MODEL", "gpt-4o-realtime-preview"),
            realtime_url=env.get("REALTIME_URL", "wss://api.openai.com/v1/realtime"),
            realtime_api_key=env.get("REALTIME_API_KEY") or env.get("OPENAI_API_KEY") or None,
            realtime_voice=env.get("REALTIME_VOICE", "alloy"),
            realtime_audio_format=env.get("REALTIME_AUDIO_FORMAT", "pcm16"),
            telemetry=_env_bool("TELEMETRY", default=False),
            telemetry_log_file=env.get("TELEMETRY_LOG_FILE") or None,
            telemetry_otel=_env_bool("TELEMETRY_OTEL", default=False),
            telephony=_env_bool("TELEPHONY", default=False),
            telephony_host=env.get("TELEPHONY_HOST", "0.0.0.0"),  # noqa: S104 — LAN bind
            debug=_env_bool("DEBUG", default=False),
            # None => auto (the browser loop turns it on); an explicit env wins.
            debug_audio=(_env_bool("DEBUG_AUDIO", default=False) if "DEBUG_AUDIO" in env else None),
            wake_debug_capture=(
                _env_bool("WAKE_DEBUG_CAPTURE", default=False)
                if "WAKE_DEBUG_CAPTURE" in env
                else None
            ),
            wake_debug_path=env.get("WAKE_DEBUG_PATH", "~/.cache/my-stt-tts/wake-debug.wav"),
            skip_audio_preflight=_env_bool("SKIP_AUDIO_PREFLIGHT", default=False),
        )
        if env.get("TELEPHONY_PORT"):
            cfg.telephony_port = int(env["TELEPHONY_PORT"])
        if env.get("TRANSPORT_PORT"):
            cfg.transport_port = int(env["TRANSPORT_PORT"])
        if env.get("TTS_VOICE_EN"):
            cfg.tts_voices["en"] = env["TTS_VOICE_EN"]
        if env.get("TTS_LENGTH_SCALE"):
            cfg.tts_length_scale = float(env["TTS_LENGTH_SCALE"])
        if env.get("BARGE_IN_ENERGY"):
            cfg.barge_in_energy = float(env["BARGE_IN_ENERGY"])
        if env.get("INTERRUPT_MIN_WORDS"):
            cfg.interrupt_min_words = int(env["INTERRUPT_MIN_WORDS"])
        if env.get("INTERRUPT_MIN_SPEECH_MS"):
            cfg.interrupt_min_speech_ms = float(env["INTERRUPT_MIN_SPEECH_MS"])
        if env.get("SMART_TURN_THRESHOLD"):
            cfg.smart_turn_threshold = float(env["SMART_TURN_THRESHOLD"])
        if env.get("WAKE_THRESHOLD"):
            cfg.wake_threshold = float(env["WAKE_THRESHOLD"])
        if env.get("WAKE_PHASES"):
            cfg.wake_phases = int(env["WAKE_PHASES"])
        if env.get("WAKE_DEBUG_SECONDS"):
            cfg.wake_debug_seconds = float(env["WAKE_DEBUG_SECONDS"])
        if env.get("VAD_THRESHOLD"):
            cfg.vad_threshold = float(env["VAD_THRESHOLD"])
        if env.get("VAD_SILENCE_SECONDS"):
            cfg.vad_silence_seconds = float(env["VAD_SILENCE_SECONDS"])
        if env.get("AEC_NLMS_TAPS"):
            cfg.aec_nlms_taps = int(env["AEC_NLMS_TAPS"])
        if env.get("AEC_NLMS_MU"):
            cfg.aec_nlms_mu = float(env["AEC_NLMS_MU"])
        if env.get("STT_WINDOW_S"):
            cfg.stt_window_s = float(env["STT_WINDOW_S"])
        if env.get("INTERRUPT_PREDICT_THRESHOLD"):
            cfg.interrupt_predict_threshold = float(env["INTERRUPT_PREDICT_THRESHOLD"])
        if env.get("INTERRUPT_PREDICT_MIN_MS"):
            cfg.interrupt_predict_min_ms = float(env["INTERRUPT_PREDICT_MIN_MS"])
        if env.get("TTS_STREAM_MIN_CHARS"):
            cfg.tts_stream_min_chars = int(env["TTS_STREAM_MIN_CHARS"])
        if env.get("TTS_STREAM_FRAME"):
            cfg.tts_stream_frame = int(env["TTS_STREAM_FRAME"])
        if env.get("DENOISER_STRENGTH"):
            cfg.denoiser_strength = float(env["DENOISER_STRENGTH"])
        if env.get("MEMORY_MAX_TURNS"):
            cfg.memory_max_turns = int(env["MEMORY_MAX_TURNS"])
        if env.get("ENROLL_DIR"):
            cfg.enroll_dir = Path(env["ENROLL_DIR"])
        if env.get("SPEAKER_THRESHOLD"):
            cfg.speaker_threshold = float(env["SPEAKER_THRESHOLD"])
        if env.get("SPEAKER_MARGIN"):
            cfg.speaker_margin = float(env["SPEAKER_MARGIN"])
        return cfg

    def apply_brain_preset(self, name: str) -> None:
        """Set provider + model from a :data:`BRAIN_PRESETS` key."""
        if name not in BRAIN_PRESETS:
            raise ConfigError(f"unknown brain preset {name!r}; choose from {tuple(BRAIN_PRESETS)}")
        self.llm_provider, self.llm_model = BRAIN_PRESETS[name]

    def select_wake_word(self, phrase: str) -> None:
        """Pick a pre-shipped wake word by name: set the phrase + derive the path.

        This is the one-call "choose a wake word" used by ``--wake-word`` and the
        web UI dropdown — it both sets ``wake_phrase`` and re-derives
        ``wake_model_path`` to ``wakewords/<phrase>.onnx`` so the selection takes
        effect without touching paths.
        """
        self.wake_phrase = phrase
        self.wake_model_path = wake_model_for(phrase)

    def validate(self) -> None:
        """Raise :class:`ConfigError` listing every problem (fail-fast)."""
        errors: list[str] = []
        if self.llm_provider not in PROVIDERS:
            errors.append(f"LLM_PROVIDER must be one of {PROVIDERS}; got {self.llm_provider!r}")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append(
                "ANTHROPIC_API_KEY is required for provider 'anthropic'. No API key? "
                "Run ./quickstart.sh to auto-pick a key-free brain, or use "
                "--brain haiku-sub (Claude CLI), or set LLM_PROVIDER=ollama "
                "(with a local model), or --brain codex (OpenAI codex CLI)."
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for provider 'openai'")
        if self.llm_provider in {"openai-compatible", "ollama"} and not self.llm_base_url:
            errors.append(f"LLM_BASE_URL is required for provider {self.llm_provider!r}")
        if self.llm_provider == "claude-cli" and not shutil.which("claude"):
            errors.append("provider 'claude-cli' needs the `claude` CLI on PATH")
        if self.llm_provider == "codex-cli" and not shutil.which("codex"):
            errors.append("provider 'codex-cli' needs the `codex` CLI on PATH")
        if self.units not in UNITS:
            errors.append(f"units must be one of {UNITS}; got {self.units!r}")
        if not self.location.strip():
            errors.append("location must not be empty")
        if self.sample_rate <= 0:
            errors.append(f"sample_rate must be > 0; got {self.sample_rate}")
        if not 0.0 < self.speaker_threshold < 1.0:
            errors.append(f"speaker_threshold must be in (0, 1); got {self.speaker_threshold}")
        if self.requests_per_minute <= 0:
            errors.append(f"requests_per_minute must be > 0; got {self.requests_per_minute}")
        if self.barge_in not in BARGE_IN_MODES:
            errors.append(f"barge_in must be one of {BARGE_IN_MODES}; got {self.barge_in!r}")
        if self.aec_mode not in AEC_MODES:
            errors.append(f"aec_mode must be one of {AEC_MODES}; got {self.aec_mode!r}")
        if self.aec_nlms_taps <= 0:
            errors.append(f"aec_nlms_taps must be > 0; got {self.aec_nlms_taps}")
        if not 0.0 < self.aec_nlms_mu <= 2.0:
            errors.append(f"aec_nlms_mu must be in (0, 2]; got {self.aec_nlms_mu}")
        if self.stt_window_s <= 0:
            errors.append(f"stt_window_s must be > 0; got {self.stt_window_s}")
        if not 0.0 <= self.interrupt_predict_threshold <= 1.0:
            errors.append(
                "interrupt_predict_threshold must be in [0, 1]; "
                f"got {self.interrupt_predict_threshold}"
            )
        if self.interrupt_predict_min_ms < 0:
            errors.append(
                f"interrupt_predict_min_ms must be >= 0; got {self.interrupt_predict_min_ms}"
            )
        if self.turn_analyzer not in TURN_ANALYZERS:
            errors.append(
                f"turn_analyzer must be one of {TURN_ANALYZERS}; got {self.turn_analyzer!r}"
            )
        if not 0.0 <= self.smart_turn_threshold <= 1.0:
            errors.append(
                f"smart_turn_threshold must be in [0, 1]; got {self.smart_turn_threshold}"
            )
        if not 0.0 <= self.wake_threshold <= 1.0:
            errors.append(f"wake_threshold must be in [0, 1]; got {self.wake_threshold}")
        if not 1 <= self.wake_phases <= 16:
            errors.append(f"wake_phases must be in [1, 16]; got {self.wake_phases}")
        if not 0.0 <= self.vad_threshold <= 1.0:
            errors.append(f"vad_threshold must be in [0, 1]; got {self.vad_threshold}")
        if self.interrupt_min_words < 0:
            errors.append(f"interrupt_min_words must be >= 0; got {self.interrupt_min_words}")
        if self.interrupt_min_speech_ms < 0:
            errors.append(
                f"interrupt_min_speech_ms must be >= 0; got {self.interrupt_min_speech_ms}"
            )
        if self.transport not in TRANSPORT_MODES:
            errors.append(f"transport must be one of {TRANSPORT_MODES}; got {self.transport!r}")
        if not 0 < self.transport_port < 65536:
            errors.append(f"transport_port must be in (0, 65535]; got {self.transport_port}")
        if self.tools_max_iterations <= 0:
            errors.append(f"tools_max_iterations must be > 0; got {self.tools_max_iterations}")
        self._validate_backends(errors)
        if self.platform not in ("auto", "macos", "linux"):
            errors.append(f"platform must be auto|macos|linux; got {self.platform!r}")
        if self.playback_backend not in ("auto", "sounddevice", "aplay", "afplay"):
            errors.append(
                "playback_backend must be auto|sounddevice|aplay|afplay; "
                f"got {self.playback_backend!r}"
            )
        if self.memory_max_turns <= 0:
            errors.append(f"memory_max_turns must be > 0; got {self.memory_max_turns}")
        if self.denoiser not in DENOISER_MODES:
            errors.append(f"denoiser must be one of {DENOISER_MODES}; got {self.denoiser!r}")
        if self.denoiser_strength < 0:
            errors.append(f"denoiser_strength must be >= 0; got {self.denoiser_strength}")
        if self.tts_stream_min_chars <= 0:
            errors.append(f"tts_stream_min_chars must be > 0; got {self.tts_stream_min_chars}")
        if self.tts_stream_frame <= 0:
            errors.append(f"tts_stream_frame must be > 0; got {self.tts_stream_frame}")
        if self.brain_mode not in BRAIN_MODES:
            errors.append(f"brain_mode must be one of {BRAIN_MODES}; got {self.brain_mode!r}")
        if self.realtime_audio_format not in ("pcm16", "g711_ulaw"):
            errors.append(
                "realtime_audio_format must be 'pcm16' or 'g711_ulaw'; "
                f"got {self.realtime_audio_format!r}"
            )
        if not 0 < self.telephony_port < 65536:
            errors.append(f"telephony_port must be in (0, 65535]; got {self.telephony_port}")
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    def _validate_backends(self, errors: list[str]) -> None:
        """Cross-check stt/tts backend names against the registered set (G1).

        Prefers the live registry's registered names (so a newly registered backend
        validates without a config edit); falls back to the static tuples if the
        registry can't be imported (keeps validate() dependency-free in isolation).
        """
        try:
            from .registry import globals_reg

            reg = globals_reg()
            stt_names: tuple[str, ...] = reg.names("stt")
            tts_names: tuple[str, ...] = (*reg.names("tts"), "local")
        except Exception:  # registry import problem -> use the documented tuples
            stt_names, tts_names = STT_BACKENDS, TTS_BACKENDS
        if self.stt_backend not in stt_names:
            errors.append(f"stt_backend must be one of {stt_names}; got {self.stt_backend!r}")
        if self.tts_backend not in tts_names:
            errors.append(f"tts_backend must be one of {tts_names}; got {self.tts_backend!r}")
