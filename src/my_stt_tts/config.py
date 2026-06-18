"""Central configuration: load from environment / .env, validate fail-fast."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROVIDERS = ("anthropic", "openai", "openai-compatible", "ollama", "claude-cli")
LANGUAGES = ("de", "fr", "en")

# Audio transport modes (R2-5; see transport.py). Where the loop sources mic audio
# and sinks TTS audio:
#   local      — the local sound card via sounddevice (default, today's behaviour)
#   websocket  — a network link: a WS server feeds remote satellites / the browser
TRANSPORT_MODES = ("local", "websocket")

# STT backend selection (R2-7). Local-first; cloud is opt-in and needs an API key.
#   local — on-device parakeet-mlx (default)
#   cloud — an OpenAI-compatible transcription endpoint (OpenAI / Deepgram-compat)
STT_BACKENDS = ("local", "cloud")

# TTS backend selection (R2-7). Local-first; cloud is opt-in (e.g. a high-quality
# cloud German voice, since local German TTS is the weak spot).
#   local — Piper / macOS say (default)
#   cloud — an OpenAI-compatible speech endpoint (e.g. OpenAI TTS)
TTS_BACKENDS = ("local", "cloud")

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
#   auto            — hardware AEC if available, else NLMS (recommended default)
AEC_MODES = ("off", "nlms", "voiceprocessing", "auto")

# Smart Turn v3 ONNX model: auto-downloaded on first run (like Piper voices) so
# smart endpointing works out of the box. Pinned to the upstream pipecat release.
SMART_TURN_MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.0.onnx"
)

# Friendly one-word brain presets -> (provider, model).
# "-sub" uses your Claude subscription via the Claude Code CLI (no API key); the
# bare aliases haiku/sonnet/opus resolve to the latest version automatically.
# "-api" uses the Anthropic API (needs ANTHROPIC_API_KEY) pinned to latest ids.
BRAIN_PRESETS: dict[str, tuple[str, str]] = {
    "haiku-sub": ("claude-cli", "haiku"),
    "sonnet-sub": ("claude-cli", "sonnet"),
    "opus-sub": ("claude-cli", "opus"),
    "haiku-api": ("anthropic", "claude-haiku-4-5"),
    "sonnet-api": ("anthropic", "claude-sonnet-4-6"),
    "opus-api": ("anthropic", "claude-opus-4-8"),
    "ollama": ("ollama", "llama3.1"),  # also set LLM_BASE_URL=http://localhost:11434/v1
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

    # --- Agent dispatch (Phase 6): "agent, <task>" hands off to a full,
    # MCP-capable Claude Code agent in `agent_workspace`. Disabled until a
    # workspace is set (a capable agent should not run in an arbitrary dir). ---
    agent_trigger: str = "agent"
    agent_workspace: str | None = None
    agent_model: str = "sonnet"

    # --- Wake / capture ---
    wake_phrase: str = "maziko"
    wake_model_path: str = "wakewords/maziko.onnx"
    wake_threshold: float = 0.5
    follow_up_seconds: float = 8.0
    sample_rate: int = 16000
    preroll_seconds: float = 0.3
    max_record_seconds: float = 30.0
    vad_silence_seconds: float = 0.7
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

    # --- Speaker ID ---
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

    debug: bool = False

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
            agent_trigger=env.get("AGENT_TRIGGER", "agent"),
            agent_workspace=env.get("AGENT_WORKSPACE") or None,
            agent_model=env.get("AGENT_MODEL", "sonnet"),
            wake_model_path=env.get("WAKE_MODEL_PATH", "wakewords/maziko.onnx"),
            stt_model=env.get("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3"),
            piper_data_dir=env.get("PIPER_DATA_DIR", "voices"),
            barge_in=env.get("BARGE_IN", "off"),
            aec_mode=env.get("AEC_MODE", "off"),
            turn_analyzer=env.get("TURN_ANALYZER", "smart"),
            smart_turn_model_path=env.get("SMART_TURN_MODEL_PATH", "models/smart-turn-v3.0.onnx"),
            smart_turn_model_url=env.get("SMART_TURN_MODEL_URL", SMART_TURN_MODEL_URL),
            smart_turn_auto_download=_env_bool("SMART_TURN_AUTO_DOWNLOAD", default=True),
            stt_streaming=_env_bool("STT_STREAMING", default=False),
            interrupt_predict=_env_bool("INTERRUPT_PREDICT", default=True),
            transport=env.get("TRANSPORT", "local"),
            transport_host=env.get("TRANSPORT_HOST", "0.0.0.0"),  # noqa: S104 — LAN bind
            transport_token=env.get("TRANSPORT_TOKEN") or None,
            tools_enabled=_env_bool("TOOLS_ENABLED", default=True),
            stt_backend=env.get("STT_BACKEND", "local"),
            stt_cloud_model=env.get("STT_CLOUD_MODEL", "whisper-1"),
            stt_cloud_base_url=env.get("STT_CLOUD_BASE_URL") or None,
            stt_cloud_api_key=env.get("STT_CLOUD_API_KEY") or env.get("OPENAI_API_KEY") or None,
            tts_backend=env.get("TTS_BACKEND", "local"),
            tts_cloud_model=env.get("TTS_CLOUD_MODEL", "gpt-4o-mini-tts"),
            tts_cloud_voice=env.get("TTS_CLOUD_VOICE", "alloy"),
            tts_cloud_base_url=env.get("TTS_CLOUD_BASE_URL") or None,
            tts_cloud_api_key=env.get("TTS_CLOUD_API_KEY") or env.get("OPENAI_API_KEY") or None,
            debug=_env_bool("DEBUG", default=False),
        )
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
        return cfg

    def apply_brain_preset(self, name: str) -> None:
        """Set provider + model from a :data:`BRAIN_PRESETS` key."""
        if name not in BRAIN_PRESETS:
            raise ConfigError(f"unknown brain preset {name!r}; choose from {tuple(BRAIN_PRESETS)}")
        self.llm_provider, self.llm_model = BRAIN_PRESETS[name]

    def validate(self) -> None:
        """Raise :class:`ConfigError` listing every problem (fail-fast)."""
        errors: list[str] = []
        if self.llm_provider not in PROVIDERS:
            errors.append(f"LLM_PROVIDER must be one of {PROVIDERS}; got {self.llm_provider!r}")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required for provider 'anthropic'")
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for provider 'openai'")
        if self.llm_provider in {"openai-compatible", "ollama"} and not self.llm_base_url:
            errors.append(f"LLM_BASE_URL is required for provider {self.llm_provider!r}")
        if self.llm_provider == "claude-cli" and not shutil.which("claude"):
            errors.append("provider 'claude-cli' needs the `claude` CLI on PATH")
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
        if self.stt_backend not in STT_BACKENDS:
            errors.append(f"stt_backend must be one of {STT_BACKENDS}; got {self.stt_backend!r}")
        if self.tts_backend not in TTS_BACKENDS:
            errors.append(f"tts_backend must be one of {TTS_BACKENDS}; got {self.tts_backend!r}")
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
