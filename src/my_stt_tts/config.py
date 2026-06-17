"""Central configuration: load from environment / .env, validate fail-fast."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROVIDERS = ("anthropic", "openai", "openai-compatible", "ollama", "claude-cli")
LANGUAGES = ("de", "fr", "en")

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

    # --- Wake / capture ---
    wake_phrase: str = "maziko"
    sample_rate: int = 16000
    preroll_seconds: float = 0.3
    max_record_seconds: float = 30.0
    vad_silence_seconds: float = 0.7
    mic_gate_tail_seconds: float = 0.2

    # --- STT ---
    stt_model: str = "mlx-community/parakeet-tdt-0.6b-v3"

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
            stt_model=env.get("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3"),
            piper_data_dir=env.get("PIPER_DATA_DIR", "voices"),
            debug=_env_bool("DEBUG", default=False),
        )
        if env.get("TTS_VOICE_EN"):
            cfg.tts_voices["en"] = env["TTS_VOICE_EN"]
        if env.get("TTS_LENGTH_SCALE"):
            cfg.tts_length_scale = float(env["TTS_LENGTH_SCALE"])
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
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
