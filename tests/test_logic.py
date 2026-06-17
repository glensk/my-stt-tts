"""Tests for speaker matching, LLM routing/memory, and TTS voice selection."""
# pylint: disable=missing-function-docstring,protected-access

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from my_stt_tts.brain import Brain, LLMError, should_use_deep
from my_stt_tts.config import Config
from my_stt_tts.speaker_id import AMBIGUOUS, UNKNOWN, match_speaker
from my_stt_tts.tts import detect_language, select_voice


def test_match_known_speaker():
    centroids = {"alice": np.array([1.0, 0.0, 0.0]), "bob": np.array([0.0, 1.0, 0.0])}
    name, score = match_speaker(np.array([0.9, 0.1, 0.0]), centroids, threshold=0.5, margin=0.05)
    assert name == "alice"
    assert score > 0.9


def test_unknown_below_threshold():
    centroids = {"alice": np.array([1.0, 0.0, 0.0])}
    name, _ = match_speaker(np.array([0.0, 1.0, 0.0]), centroids, threshold=0.5, margin=0.05)
    assert name == UNKNOWN


def test_ambiguous_within_margin():
    centroids = {"alice": np.array([1.0, 0.0, 0.0]), "bob": np.array([1.0, 0.02, 0.0])}
    name, _ = match_speaker(np.array([1.0, 0.01, 0.0]), centroids, threshold=0.3, margin=0.2)
    assert name == AMBIGUOUS


def test_empty_centroids_is_unknown():
    name, score = match_speaker(np.array([1.0]), {}, threshold=0.5, margin=0.05)
    assert name == UNKNOWN
    assert score == 0.0


def test_should_use_deep_trigger():
    cfg = Config(deep_trigger="think hard")
    assert should_use_deep(cfg, "Please think HARD about this")
    assert not should_use_deep(cfg, "what time is it")


def test_history_trim_keeps_recent_turns():
    cfg = Config(max_history_turns=2, anthropic_api_key="x")
    brain = Brain(cfg)
    brain.history.extend({"role": "user", "content": str(i)} for i in range(10))
    brain._trim()
    assert len(brain.history) == 4


def test_select_voice_piper_then_say_fallback():
    cfg = Config()
    assert select_voice(cfg, "de") == ("piper", "de_DE-thorsten-high")
    assert select_voice(cfg, "fr")[0] == "piper"
    assert select_voice(cfg, "it")[0] == "say"


def test_detect_language_falls_back_without_lingua():
    assert detect_language("bonjour le monde", default="en") in {"de", "fr", "en"}


def test_claude_cli_session_then_resume():
    cfg = Config(llm_provider="claude-cli", llm_model="haiku")
    brain = Brain(cfg)
    completed = MagicMock(returncode=0, stdout=json.dumps({"result": "hi", "is_error": False}))
    with (
        patch("my_stt_tts.brain.shutil.which", return_value="/usr/bin/claude"),
        patch("my_stt_tts.brain.subprocess.run", return_value=completed) as run,
    ):
        assert "".join(brain.stream("hello")) == "hi"
        session_id = brain._session_id
        assert session_id is not None
        assert "--session-id" in run.call_args.args[0]

        assert "".join(brain.stream("again")) == "hi"
        assert brain._session_id == session_id  # same session reused
        assert "--resume" in run.call_args.args[0]
        assert session_id in run.call_args.args[0]


def test_claude_cli_error_propagates():
    cfg = Config(llm_provider="claude-cli")
    brain = Brain(cfg)
    failed = MagicMock(returncode=1, stderr="model not found")
    with (
        patch("my_stt_tts.brain.shutil.which", return_value="/usr/bin/claude"),
        patch("my_stt_tts.brain.subprocess.run", return_value=failed),
        pytest.raises(LLMError),
    ):
        list(brain.stream("hi"))
