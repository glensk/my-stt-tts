"""Data-driven wake reliability: per-word clip folders, the /recordings/ route, and
the debug/wake_stats.json outcome log.

Covers the three backend pillars of the wake-reliability feature:

* :func:`audio.save_recording` routes a ``kind="wake"`` clip into a PER-WORD
  subfolder (``wake/<word>/<file>``) — every test is kept as labelled training data —
  while a mic-check clip stays flat.
* :func:`audio.resolve_recording` / the WebUI ``/recordings/`` route resolve a clip by
  BASENAME across subfolders, traversal-safe (``..`` can never escape).
* :func:`config.record_wake_outcome` appends to ``debug/wake_stats.json`` and
  :func:`config.measured_reliability` reads it back (server-biased, recent window).
"""
# pylint: disable=missing-function-docstring,import-outside-toplevel,protected-access
# pylint: disable=too-few-public-methods,missing-class-docstring

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from my_stt_tts import audio


def _loud_clip(n: int = 32000) -> np.ndarray:
    return (np.sin(np.linspace(0, 100, n)) * 0.4).astype(np.float32)


# --------------------------------------------------------------------------- #
# save_recording: wake clips land in per-word folders; mic clips stay flat     #
# --------------------------------------------------------------------------- #
def test_wake_clip_saved_in_per_word_subfolder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    path, hash8, wav_url = audio.save_recording(
        _loud_clip(), 16000, kind="wake", source="server", word="maziko"
    )
    assert Path(path).is_file()
    # Lives under wake/<word>/, NOT flat in the recordings dir.
    assert Path(path).parent == tmp_path / "wake" / "maziko"
    assert wav_url == f"/recordings/wake/maziko/{Path(path).name}"
    assert hash8 in Path(path).name
    # It is a readable 16 kHz mono WAV.
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0


def test_mic_clip_stays_flat(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    path, _hash, wav_url = audio.save_recording(_loud_clip(), 16000, kind="mic", source="browser")
    assert Path(path).parent == tmp_path  # flat, no subfolder
    assert wav_url == f"/recordings/{Path(path).name}"


def test_every_wake_test_is_kept_not_overwritten(monkeypatch, tmp_path: Path) -> None:
    """Distinct clips for the SAME word accumulate (training data) — never overwrite."""
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    p1, _h1, _u1 = audio.save_recording(_loud_clip(), 16000, kind="wake", source="server", word="x")
    other = (np.sin(np.linspace(0, 50, 32000)) * 0.3).astype(np.float32)
    p2, _h2, _u2 = audio.save_recording(other, 16000, kind="wake", source="server", word="x")
    assert p1 != p2
    word_dir = tmp_path / "wake" / "x"
    assert len(list(word_dir.glob("*.wav"))) == 2


def test_wake_word_name_is_sanitized_no_traversal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    path, _hash, _url = audio.save_recording(
        _loud_clip(), 16000, kind="wake", source="server", word="../../etc"
    )
    real = os.path.realpath(path)
    # Whatever the word, the clip stays under the recordings dir.
    assert os.path.commonpath([os.path.realpath(str(tmp_path)), real]) == os.path.realpath(
        str(tmp_path)
    )


# --------------------------------------------------------------------------- #
# resolve_recording: basename lookup across subfolders, traversal-safe          #
# --------------------------------------------------------------------------- #
def test_resolve_recording_finds_nested_wake_clip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    path, _hash, _url = audio.save_recording(
        _loud_clip(), 16000, kind="wake", source="server", word="nexus"
    )
    base = Path(path).name
    assert audio.resolve_recording(base) == path  # found in wake/nexus/


def test_resolve_recording_finds_flat_mic_clip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    path, _hash, _url = audio.save_recording(_loud_clip(), 16000, kind="mic", source="server")
    assert audio.resolve_recording(Path(path).name) == path


def test_resolve_recording_rejects_traversal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    # A name with path segments is not a bare basename -> refused outright.
    assert audio.resolve_recording("../../etc/passwd") is None
    assert audio.resolve_recording("wake/nexus/x.wav") is None  # base != name
    assert audio.resolve_recording("nope.wav") is None  # no such clip
    assert audio.resolve_recording("notawav.txt") is None  # not a .wav


def test_find_recordings_matches_flat_and_nested(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    p_flat, hflat, _u = audio.save_recording(_loud_clip(), 16000, kind="mic", source="server")
    p_wake, hwake, _u2 = audio.save_recording(
        _loud_clip(), 16000, kind="wake", source="server", word="w"
    )
    assert p_flat in audio.find_recordings(f"*-{hflat}.wav")
    assert p_wake in audio.find_recordings(f"*-{hwake}.wav")


# --------------------------------------------------------------------------- #
# WebUI /recordings/ route resolves nested files (traversal-safe)               #
# --------------------------------------------------------------------------- #
def test_webui_serves_nested_wake_recording(monkeypatch, tmp_path: Path) -> None:
    from my_stt_tts import webui

    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    path, _hash, wav_url = audio.save_recording(
        _loud_clip(), 16000, kind="wake", source="server", word="maziko"
    )
    sent: dict[str, object] = {}

    class _Stub:
        def _send(self, code, ctype, body):  # noqa: ANN001
            sent.update(code=code, ctype=ctype, body=body)

    # The route hands the part after /recordings/ to _serve_recording; the per-word
    # URL still resolves because only the basename is used.
    rel = wav_url[len("/recordings/") :]
    webui._Handler._serve_recording(_Stub(), rel)  # type: ignore[arg-type]
    assert sent["code"] == 200
    assert sent["ctype"] == "audio/wav"
    assert sent["body"] == Path(path).read_bytes()


def test_webui_recording_route_blocks_traversal(monkeypatch, tmp_path: Path) -> None:
    from my_stt_tts import webui

    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    sent: dict[str, object] = {}

    class _Stub:
        def _send(self, code, ctype, body):  # noqa: ANN001
            sent.update(code=code, ctype=ctype, body=body)

    webui._Handler._serve_recording(_Stub(), "../../../../etc/passwd")  # type: ignore[arg-type]
    assert sent["code"] == 404


# --------------------------------------------------------------------------- #
# wake_stats.json: append + read-back, server-biased recent reliability         #
# --------------------------------------------------------------------------- #
def test_record_wake_outcome_appends_keyed_by_word(tmp_path: Path) -> None:
    from my_stt_tts.config import load_wake_stats, record_wake_outcome

    p = str(tmp_path / "wake_stats.json")
    record_wake_outcome("maziko", confidence=0.0, fired=False, source="server", path=p)
    record_wake_outcome("maziko", confidence=0.1, fired=False, source="server", path=p)
    record_wake_outcome("alexa", confidence=0.95, fired=True, source="browser", path=p)
    stats = load_wake_stats(p)
    assert set(stats) == {"maziko", "alexa"}
    assert len(stats["maziko"]) == 2
    assert stats["maziko"][0]["confidence"] == 0.0
    assert stats["maziko"][0]["fired"] is False
    assert stats["maziko"][0]["source"] == "server"
    assert stats["maziko"][0]["ts"]  # an ISO timestamp from the system clock
    assert stats["alexa"][0]["fired"] is True


def test_record_wake_outcome_uses_given_timestamp(tmp_path: Path) -> None:
    from my_stt_tts.config import load_wake_stats, record_wake_outcome

    p = str(tmp_path / "wake_stats.json")
    record_wake_outcome(
        "w", confidence=0.5, fired=True, source="server", path=p, ts="2026-06-21T10:00:00"
    )
    assert load_wake_stats(p)["w"][0]["ts"] == "2026-06-21T10:00:00"


def test_load_wake_stats_missing_or_corrupt_is_empty(tmp_path: Path) -> None:
    from my_stt_tts.config import load_wake_stats

    assert load_wake_stats(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert load_wake_stats(str(bad)) == {}


def test_outcome_log_drives_reliability_end_to_end(tmp_path: Path) -> None:
    """Append failing maziko tests, then reliability reads RED ~0 from the log."""
    from my_stt_tts.config import load_wake_stats, record_wake_outcome, wake_word_tier

    p = str(tmp_path / "wake_stats.json")
    for _ in range(6):
        record_wake_outcome("maziko", confidence=0.0, fired=False, source="server", path=p)
    tier, _note, rel = wake_word_tier("maziko", stats=load_wake_stats(p))
    assert tier == "red"
    assert rel == pytest.approx(0.0)
