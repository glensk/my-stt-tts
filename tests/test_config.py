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
    assert "haiku-sub" in msg
    assert "ollama" in msg.lower()


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
