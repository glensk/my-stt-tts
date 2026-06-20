"""Tests for configuration loading and fail-fast validation."""
# pylint: disable=missing-function-docstring,import-outside-toplevel,protected-access

import pytest

from my_stt_tts.config import Config, ConfigError


def test_defaults_require_anthropic_key():
    with pytest.raises(ConfigError):
        Config().validate()


def test_valid_anthropic():
    Config(anthropic_api_key="sk-test").validate()


def test_openai_requires_key():
    with pytest.raises(ConfigError):
        Config(llm_provider="openai").validate()
    Config(llm_provider="openai", openai_api_key="sk-test").validate()


def test_openai_compatible_requires_base_url():
    with pytest.raises(ConfigError):
        Config(llm_provider="openai-compatible").validate()
    Config(llm_provider="openai-compatible", llm_base_url="http://localhost:11434/v1").validate()


def test_ollama_needs_only_base_url():
    Config(llm_provider="ollama", llm_base_url="http://localhost:11434/v1").validate()


def test_bad_provider_rejected():
    with pytest.raises(ConfigError):
        Config(llm_provider="bogus", anthropic_api_key="x").validate()


def test_codex_cli_provider_is_valid():
    from my_stt_tts.config import PROVIDERS

    assert "codex-cli" in PROVIDERS


def test_codex_cli_requires_codex_on_path(monkeypatch):
    import my_stt_tts.config as config_mod

    monkeypatch.setattr(config_mod.shutil, "which", lambda _name: None)
    with pytest.raises(ConfigError, match="codex-cli"):
        Config(llm_provider="codex-cli").validate()
    monkeypatch.setattr(config_mod.shutil, "which", lambda _name: "/usr/bin/codex")
    Config(llm_provider="codex-cli").validate()  # no key needed when codex is present


def test_codex_brain_preset_sets_provider_and_model():
    cfg = Config()
    cfg.apply_brain_preset("codex")
    assert cfg.llm_provider == "codex-cli"
    assert cfg.llm_model


def test_missing_anthropic_key_message_points_at_fixes():
    with pytest.raises(ConfigError) as exc:
        Config().validate()
    msg = str(exc.value)
    assert "quickstart.sh" in msg
    # The hint now points at the DEFAULT key-free brain (opus-sub) first.
    assert "opus-sub" in msg
    assert "ollama" in msg.lower()


def test_default_brain_preset_is_opus_sub():
    from my_stt_tts.config import BRAIN_PRESETS, DEFAULT_BRAIN_PRESET

    assert DEFAULT_BRAIN_PRESET == "opus-sub"
    assert BRAIN_PRESETS[DEFAULT_BRAIN_PRESET] == ("claude-cli", "opus")


# --- exact model + reasoning-level label (GUI contract) ------------------------


def test_model_label_claude_cli_opus_is_exact_with_reasoning():
    from my_stt_tts.config import model_label

    # The shared contract: claude-cli Opus -> "claude-cli / opus-4.8 · think".
    assert model_label("claude-cli", "opus") == "claude-cli / opus-4.8 · think"


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("claude-cli", "opus", "claude-cli / opus-4.8 · think"),
        ("claude-cli", "sonnet", "claude-cli / sonnet-4.6 · think"),
        ("claude-cli", "haiku", "claude-cli / haiku-4.5 · think"),
        # API ids resolve to the same marketing version (no CLI reasoning suffix).
        ("anthropic", "claude-opus-4-8", "anthropic / opus-4.8"),
        ("anthropic", "claude-haiku-4-5", "anthropic / haiku-4.5"),
        # Non-Anthropic providers pass the model through unchanged.
        ("ollama", "llama3.1", "ollama / llama3.1"),
        ("codex-cli", "gpt-5-codex", "codex-cli / gpt-5-codex"),
    ],
)
def test_model_label_maps_versions_and_reasoning(provider, model, expected):
    from my_stt_tts.config import model_label

    assert model_label(provider, model) == expected


def test_model_version_label_unknown_passes_through():
    from my_stt_tts.config import model_version_label

    assert model_version_label("claude-opus-4-8") == "opus-4.8"
    assert model_version_label("some-future-model") == "some-future-model"


# --- wake-word reliability metadata (GUI contract) -----------------------------


@pytest.mark.parametrize(
    ("word", "tier", "recall"),
    [
        # Official -> always green, recall unmeasured (None).
        ("alexa", "green", None),
        ("hey_jarvis", "green", None),
        ("hey_mycroft", "green", None),
        # Self-trained by recall band: >=0.70 green, [0.50,0.70) orange, <0.50 red.
        ("maziko", "green", 0.76),
        ("orion", "green", 0.70),  # 0.70 is the green boundary (inclusive)
        ("computer", "orange", 0.64),
        ("luna", "orange", 0.52),
        ("sage", "red", 0.45),
        # Self-trained, unrecorded recall -> red.
        ("nexus", "red", None),
        ("jarvis", "red", None),
    ],
)
def test_wake_word_tier_rules(word, tier, recall):
    from my_stt_tts.config import wake_word_tier

    got_tier, note, got_recall = wake_word_tier(word)
    assert got_tier == tier
    assert got_recall == recall
    assert note  # always a non-empty human reason


def test_wake_word_info_shape_for_available_models():
    from my_stt_tts.config import available_wake_words, wake_word_info

    info = wake_word_info()
    # One entry per available model, each with the contract keys.
    assert set(info) == set(available_wake_words())
    for _word, meta in info.items():
        assert set(meta) == {"tier", "note", "recall"}
        assert meta["tier"] in {"green", "orange", "red"}
        assert isinstance(meta["note"], str) and meta["note"]
        assert meta["recall"] is None or isinstance(meta["recall"], float)


def test_wake_word_info_in_settings_dict():
    from my_stt_tts.webui import settings_dict

    d = settings_dict(Config(anthropic_api_key="sk-test"))
    assert "wake_word_info" in d
    # The official models are present and green (they ship in wakewords/).
    assert d["wake_word_info"]["hey_jarvis"]["tier"] == "green"
    assert d["wake_word_info"]["alexa"]["tier"] == "green"


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.from_env()
    assert cfg.llm_provider == "openai"
    assert cfg.llm_model == "gpt-4o-mini"
    cfg.validate()


# --- location + units (wave-g) -------------------------------------------------


def test_location_units_defaults():
    cfg = Config(anthropic_api_key="x")
    assert cfg.location == "Lausanne, Switzerland"
    assert cfg.units == "metric"
    cfg.validate()  # defaults are valid


def test_units_validation_rejects_unknown():
    with pytest.raises(ConfigError):
        Config(anthropic_api_key="x", units="kelvin").validate()
    Config(anthropic_api_key="x", units="imperial").validate()


def test_empty_location_rejected():
    with pytest.raises(ConfigError):
        Config(anthropic_api_key="x", location="   ").validate()


def test_location_units_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LOCATION", "Tokyo, Japan")
    monkeypatch.setenv("UNITS", "imperial")
    cfg = Config.from_env()
    assert cfg.location == "Tokyo, Japan"
    assert cfg.units == "imperial"
    cfg.validate()


# --- system-prompt locale injection (wave-g) -----------------------------------


def test_locale_prompt_line_appends_location_and_units():
    from my_stt_tts.config import locale_prompt_line

    out = locale_prompt_line("BASE PROMPT.", "Lausanne, Switzerland", "metric")
    assert out.startswith("BASE PROMPT.")
    assert "Lausanne, Switzerland" in out
    assert "metric units" in out


def test_locale_prompt_line_blank_location_is_noop():
    from my_stt_tts.config import locale_prompt_line

    assert locale_prompt_line("BASE", "   ", "metric") == "BASE"


def test_brain_system_prompt_contains_locale_line():
    from my_stt_tts.brain import Brain

    cfg = Config(anthropic_api_key="x", location="Geneva, Switzerland", units="imperial")
    brain = Brain(cfg)
    prompt = brain._system_prompt()  # noqa: SLF001 — assert the injection point
    assert cfg.system_prompt in prompt  # editable base preserved
    assert "Geneva, Switzerland" in prompt
    assert "imperial units" in prompt


# --------------------------------------------------------------------------- #
# Wake sensitivity (wake_threshold) — default, env override, validation       #
# --------------------------------------------------------------------------- #
def test_wake_threshold_default_is_0_4():
    assert Config().wake_threshold == 0.4


def test_wake_threshold_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("WAKE_THRESHOLD", "0.65")
    cfg = Config.from_env()
    assert cfg.wake_threshold == 0.65
    cfg.validate()  # in-range value is valid


def test_wake_threshold_validation_rejects_out_of_range():
    with pytest.raises(ConfigError, match="wake_threshold"):
        Config(anthropic_api_key="x", wake_threshold=1.5).validate()
    with pytest.raises(ConfigError, match="wake_threshold"):
        Config(anthropic_api_key="x", wake_threshold=-0.1).validate()
    # Bounds are inclusive.
    Config(anthropic_api_key="x", wake_threshold=0.0).validate()
    Config(anthropic_api_key="x", wake_threshold=1.0).validate()
