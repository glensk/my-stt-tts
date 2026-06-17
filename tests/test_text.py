"""Tests for streaming sentence chunking and non-spoken-text stripping."""
# pylint: disable=missing-function-docstring

from my_stt_tts.text import SentenceChunker, strip_non_spoken


def test_chunker_emits_complete_sentences():
    chunker = SentenceChunker()
    assert chunker.feed("Hello world. How are") == ["Hello world."]
    assert chunker.feed(" you? Fine.") == ["How are you?", "Fine."]


def test_chunker_decimal_guard_english():
    chunker = SentenceChunker()
    assert chunker.feed("Pi is 3.14 today. ") == ["Pi is 3.14 today."]


def test_chunker_german_comma_decimal_not_split():
    chunker = SentenceChunker()
    assert chunker.feed("Das sind 3,14 Euro. ") == ["Das sind 3,14 Euro."]


def test_chunker_flush_returns_remainder():
    chunker = SentenceChunker()
    chunker.feed("no terminator yet")
    assert chunker.flush() == "no terminator yet"


def test_strip_removes_think_code_and_markdown():
    raw = "<think>secret</think>Hello **world** [link](http://x) `code` (aside)."
    spoken = strip_non_spoken(raw)
    assert "secret" not in spoken
    assert "aside" not in spoken
    assert "*" not in spoken
    for word in ("Hello", "world", "link", "code"):
        assert word in spoken
