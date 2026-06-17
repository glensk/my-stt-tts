"""Tests for configuration loading and fail-fast validation."""
# pylint: disable=missing-function-docstring

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


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config.from_env()
    assert cfg.llm_provider == "openai"
    assert cfg.llm_model == "gpt-4o-mini"
    cfg.validate()
