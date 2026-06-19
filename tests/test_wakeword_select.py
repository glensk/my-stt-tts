"""Pre-shipped wake-word selection: discovery, name->path derivation, surfaces.

The repo ships several trained wake-word models in ``wakewords/`` and a user picks
one by NAME (UI dropdown / ``--wake-word`` / ``WAKE_PHRASE``) without editing paths.
These tests never rely on the real ``wakewords/`` directory — discovery is always
exercised against a temp dir with fake ``.onnx`` files.
"""
# pylint: disable=missing-function-docstring,import-outside-toplevel,protected-access

from __future__ import annotations

from pathlib import Path

from my_stt_tts import __main__ as main_mod
from my_stt_tts import webui as webui_mod
from my_stt_tts.config import Config, available_wake_words, wake_model_for
from my_stt_tts.webui import settings_dict


def _make_models(directory: Path, *names: str) -> None:
    for name in names:
        (directory / f"{name}.onnx").write_bytes(b"\x00")  # fake ONNX payload


# --------------------------------------------------------------------------- #
# Discovery: available_wake_words scans a temp dir (NOT the real wakewords/)    #
# --------------------------------------------------------------------------- #
def test_available_wake_words_lists_onnx_stems_sorted(tmp_path: Path) -> None:
    _make_models(tmp_path, "maziko", "jarvis", "alexa")
    assert available_wake_words(tmp_path) == ["alexa", "jarvis", "maziko"]


def test_available_wake_words_ignores_non_onnx(tmp_path: Path) -> None:
    _make_models(tmp_path, "computer")
    (tmp_path / "readme.md").write_text("not a model")
    (tmp_path / "maziko.tflite").write_bytes(b"\x00")  # different extension
    assert available_wake_words(tmp_path) == ["computer"]


def test_available_wake_words_empty_when_dir_missing(tmp_path: Path) -> None:
    assert available_wake_words(tmp_path / "does-not-exist") == []


def test_available_wake_words_empty_dir_is_empty_list(tmp_path: Path) -> None:
    assert available_wake_words(tmp_path) == []


# --------------------------------------------------------------------------- #
# Name -> path derivation (default + explicit-override precedence)             #
# --------------------------------------------------------------------------- #
def test_wake_model_for_uses_convention() -> None:
    assert wake_model_for("jarvis") == "wakewords/jarvis.onnx"
    assert wake_model_for("nexus", "models/ww") == "models/ww/nexus.onnx"


def test_from_env_default_phrase_derives_path(monkeypatch) -> None:
    for var in ("WAKE_PHRASE", "WAKE_MODEL_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cfg = Config.from_env()
    assert cfg.wake_phrase == "maziko"
    assert cfg.wake_model_path == "wakewords/maziko.onnx"


def test_from_env_wake_phrase_drives_path(monkeypatch) -> None:
    monkeypatch.delenv("WAKE_MODEL_PATH", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("WAKE_PHRASE", "jarvis")
    cfg = Config.from_env()
    assert cfg.wake_phrase == "jarvis"
    assert cfg.wake_model_path == "wakewords/jarvis.onnx"


def test_from_env_explicit_model_path_overrides_phrase(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("WAKE_PHRASE", "jarvis")
    monkeypatch.setenv("WAKE_MODEL_PATH", "/custom/elsewhere.onnx")
    cfg = Config.from_env()
    assert cfg.wake_phrase == "jarvis"  # phrase still tracks the spoken word
    assert cfg.wake_model_path == "/custom/elsewhere.onnx"  # explicit path wins


def test_select_wake_word_sets_phrase_and_path() -> None:
    cfg = Config()
    cfg.select_wake_word("computer")
    assert cfg.wake_phrase == "computer"
    assert cfg.wake_model_path == "wakewords/computer.onnx"


# --------------------------------------------------------------------------- #
# CLI: --wake-word NAME selects; --wake-model-path overrides                   #
# --------------------------------------------------------------------------- #
def test_cli_wake_word_flag_selects(monkeypatch) -> None:
    monkeypatch.delenv("WAKE_MODEL_PATH", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    args = main_mod._parse_args(["--wake-word", "nexus"])
    cfg = main_mod._build_config(args)
    assert cfg.wake_phrase == "nexus"
    assert cfg.wake_model_path == "wakewords/nexus.onnx"


def test_cli_wake_model_path_overrides_wake_word(monkeypatch) -> None:
    monkeypatch.delenv("WAKE_MODEL_PATH", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    args = main_mod._parse_args(["--wake-word", "nexus", "--wake-model-path", "/tmp/custom.onnx"])
    cfg = main_mod._build_config(args)
    assert cfg.wake_phrase == "nexus"
    assert cfg.wake_model_path == "/tmp/custom.onnx"


def test_settings_text_shows_selected_and_available(tmp_path, monkeypatch) -> None:
    _make_models(tmp_path, "maziko", "jarvis")
    # Point discovery at the temp dir so we don't depend on the real wakewords/.
    monkeypatch.setattr(main_mod, "available_wake_words", lambda *_a, **_k: ["jarvis", "maziko"])
    cfg = Config(wake_phrase="jarvis", wake_model_path=str(tmp_path / "jarvis.onnx"))
    text = main_mod.settings_text(cfg, color=False)
    assert "jarvis" in text
    assert "maziko" in text
    assert "exists True" in text  # the selected model file is present


# --------------------------------------------------------------------------- #
# Web UI: settings_dict exposes wake_words; apply_settings re-derives the path #
# --------------------------------------------------------------------------- #
def test_settings_dict_carries_wake_words(monkeypatch) -> None:
    monkeypatch.setattr(
        webui_mod, "available_wake_words", lambda *_a, **_k: ["alexa", "computer", "maziko"]
    )
    s = settings_dict(Config())
    assert s["wake_words"] == ["alexa", "computer", "maziko"]
    assert s["wake_phrase"] == "maziko"


def test_apply_settings_wake_phrase_rederives_path() -> None:
    cfg = Config()
    webui_mod.apply_settings(cfg, {"wake_phrase": "jarvis"})
    assert cfg.wake_phrase == "jarvis"
    assert cfg.wake_model_path == "wakewords/jarvis.onnx"
