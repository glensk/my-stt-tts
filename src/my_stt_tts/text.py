"""Streaming sentence chunking + non-spoken-text stripping for TTS."""

from __future__ import annotations

import re

_SENTENCE_END = ".!?…。"

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_PAREN = re.compile(r"\([^)]*\)")
_EMPHASIS = re.compile(r"[*_#>~]+")
_WS = re.compile(r"\s+")


def strip_non_spoken(text: str) -> str:
    """Remove content that should not be spoken aloud.

    Drops ``<think>`` reasoning blocks and fenced code, unwraps markdown links
    and inline code to their text, removes parentheticals and emphasis markers,
    and collapses whitespace.
    """
    text = _THINK.sub(" ", text)
    text = _CODE_FENCE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _PAREN.sub(" ", text)
    text = _EMPHASIS.sub("", text)
    return _WS.sub(" ", text).strip()


class SentenceChunker:
    """Accumulate streamed text and emit complete sentences as they finish.

    Guards decimals ("3.14" and German "3,14" — a comma is never a terminator)
    and only treats a terminator as a boundary when it is followed by whitespace
    or end-of-buffer, so abbreviations like "z.B." or "e.g." are not split.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        """Add a text delta; return any complete sentences now available."""
        self._buf += delta
        out: list[str] = []
        while (idx := self._boundary(self._buf)) is not None:
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str:
        """Return and clear any buffered remainder (end of stream)."""
        rest = self._buf.strip()
        self._buf = ""
        return rest

    @staticmethod
    def _boundary(s: str) -> int | None:
        for i, ch in enumerate(s):
            if ch not in _SENTENCE_END:
                continue
            prev = s[i - 1] if i > 0 else ""
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if ch == "." and prev.isdigit() and nxt.isdigit():
                continue  # decimal like 3.14
            if nxt and not nxt.isspace():
                continue  # require whitespace/end after the terminator
            return i
        return None
