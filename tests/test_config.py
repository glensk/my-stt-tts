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


# --- wake-word DATA-DRIVEN reliability (GUI contract) --------------------------
# Reliability is a 0..1 scalar: the MEASURED chance the word fires for THIS user
# (from his wake tests, server-biased) when tests exist, else a static prior
# (official 0.9, self-trained 0.3). Tier is DERIVED from the scalar. Empirical
# driver: official models fire ~100% on Albert's voice while self-trained maziko
# scores 0% — so its dot must read RED (0.3 prior), not the static training metric.


@pytest.mark.parametrize(
    ("word", "tier", "reliability"),
    [
        # Official -> 0.9 prior -> green (no measured tests).
        ("alexa", "green", 0.9),
        ("hey_jarvis", "green", 0.9),
        ("hey_mycroft", "green", 0.9),
        # Self-trained -> 0.3 prior -> RED (maziko was the empirical proof of 0%).
        ("maziko", "red", 0.3),
        ("nexus", "red", 0.3),
        ("computer", "red", 0.3),
        ("orion", "red", 0.3),
        # An entirely unknown word is treated as self-trained -> red prior.
        ("whatever", "red", 0.3),
    ],
)
def test_wake_word_tier_prior_rules(word, tier, reliability):
    """Static-prior tier from the scalar, no measured tests (empty stats)."""
    from my_stt_tts.config import wake_word_tier

    got_tier, note, got_reliability = wake_word_tier(word, stats={})
    assert got_tier == tier
    assert got_reliability == pytest.approx(reliability)
    assert note  # always a non-empty human reason


def test_tier_from_reliability_thresholds():
    from my_stt_tts.config import _tier_from_reliability

    assert _tier_from_reliability(0.70) == "green"  # green boundary inclusive
    assert _tier_from_reliability(0.99) == "green"
    assert _tier_from_reliability(0.69) == "orange"
    assert _tier_from_reliability(0.40) == "orange"  # orange boundary inclusive
    assert _tier_from_reliability(0.39) == "red"
    assert _tier_from_reliability(0.0) == "red"


def test_measured_overrides_prior_and_drives_tier():
    """When this user's tests exist, the MEASURED mean drives reliability + tier,
    overriding the static prior. A self-trained word that fails for him -> red ~0;
    one that fires for him -> green."""
    from my_stt_tts.config import wake_word_tier

    # maziko fails on his voice (0% fire, near-zero confidence) -> red ~0.0.
    failing = {"maziko": [{"confidence": 0.0, "fired": False, "source": "server"}] * 6}
    tier, note, rel = wake_word_tier("maziko", stats=failing)
    assert tier == "red"
    assert rel == pytest.approx(0.0)
    assert "measured" in note

    # A self-trained word that fires reliably for him -> measured overrides 0.3 -> green.
    firing = {"maziko": [{"confidence": 0.97, "fired": True, "source": "server"}] * 6}
    tier, _note, rel = wake_word_tier("maziko", stats=firing)
    assert tier == "green"
    assert rel == pytest.approx(0.97)


def test_measured_reliability_is_server_biased():
    """Server tests are the live loop, so they win: when ANY server test exists, only
    server tests count; browser-only falls back to browser."""
    from my_stt_tts.config import measured_reliability

    mixed = {
        "w": [
            {"confidence": 0.1, "fired": False, "source": "server"},
            {"confidence": 0.9, "fired": True, "source": "browser"},  # ignored (server exists)
        ]
    }
    rel, tested = measured_reliability("w", mixed)
    assert rel == pytest.approx(0.1)
    assert tested == 1

    browser_only = {"w": [{"confidence": 0.8, "fired": True, "source": "browser"}]}
    rel, tested = measured_reliability("w", browser_only)
    assert rel == pytest.approx(0.8)
    assert tested == 1

    assert measured_reliability("missing", {}) == (None, 0)


def test_measured_reliability_uses_only_recent_window():
    """Only the most recent WAKE_STATS_RECENT tests feed the mean (old failures age out)."""
    from my_stt_tts.config import WAKE_STATS_RECENT, measured_reliability

    old = [{"confidence": 0.0, "fired": False, "source": "server"}] * 5
    recent = [{"confidence": 1.0, "fired": True, "source": "server"}] * WAKE_STATS_RECENT
    rel, tested = measured_reliability("w", {"w": old + recent})
    assert rel == pytest.approx(1.0)  # the 5 old zeros aged out of the window
    assert tested == WAKE_STATS_RECENT


def test_wake_word_info_shape_for_available_models():
    from my_stt_tts.config import available_wake_words, wake_word_info

    info = wake_word_info(stats={})
    # One entry per available model, each with the contract keys.
    assert set(info) == set(available_wake_words())
    for _word, meta in info.items():
        assert set(meta) == {"tier", "note", "reliability", "tested", "measured"}
        assert meta["tier"] in {"green", "orange", "red"}
        assert isinstance(meta["note"], str) and meta["note"]
        assert isinstance(meta["reliability"], float) and 0.0 <= meta["reliability"] <= 1.0
        assert isinstance(meta["tested"], int) and meta["tested"] >= 0
        assert isinstance(meta["measured"], bool)


def test_wake_word_info_in_settings_dict():
    from my_stt_tts.webui import settings_dict

    d = settings_dict(Config(anthropic_api_key="sk-test"))
    assert "wake_word_info" in d
    info = d["wake_word_info"]
    # Official models are present and green (0.9 prior).
    assert info["hey_jarvis"]["tier"] == "green"
    assert info["alexa"]["tier"] == "green"
    # Self-trained maziko reads RED by default (0.3 prior) — the empirical fix.
    assert info["maziko"]["tier"] == "red"
    # Contract carries the data-driven fields.
    assert {"reliability", "tested", "measured"} <= set(info["maziko"])


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
