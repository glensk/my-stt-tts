"""macOS mic-permission detection + how it shapes the mic-test verdict.

Pure-logic tests: ``mic_permission_status`` is exercised for its contract (returns
one of the known tags, ``unavailable`` off macOS / without pyobjc) and the verdict
mapping is checked for every permission case without touching a real microphone.
"""

from __future__ import annotations

from my_stt_tts import audio
from my_stt_tts.config import Config
from my_stt_tts.webui import settings_dict

_KNOWN = {"authorized", "denied", "notDetermined", "restricted", "unavailable"}


def test_permission_status_returns_a_known_tag() -> None:
    # On macOS with pyobjc it reflects the real TCC grant; core-only/Linux -> unavailable.
    assert audio.mic_permission_status() in _KNOWN


def test_denied_permission_wins_over_a_loud_capture() -> None:
    r = audio.mic_test_verdict(captured=True, rms=0.5, peak=0.5, permission="denied")
    assert not r.ok
    assert r.verdict == "denied"
    assert "DENIED" in r.message
    assert r.permission == "denied"


def test_restricted_permission_is_reported() -> None:
    r = audio.mic_test_verdict(captured=False, rms=0.0, peak=0.0, permission="restricted")
    assert not r.ok
    assert r.verdict == "restricted"


def test_authorized_but_silent_is_a_device_issue_not_permission() -> None:
    r = audio.mic_test_verdict(captured=True, rms=0.0, peak=0.0, permission="authorized")
    assert r.verdict == "silent"
    assert "granted" in r.message.lower()
    assert r.permission == "authorized"


def test_not_determined_silent_mentions_the_prompt() -> None:
    r = audio.mic_test_verdict(captured=True, rms=0.0, peak=0.0, permission="notDetermined")
    assert r.verdict == "silent"
    assert "prompt" in r.message.lower()


def test_authorized_with_audio_is_ok() -> None:
    r = audio.mic_test_verdict(captured=True, rms=0.4, peak=0.6, permission="authorized")
    assert r.ok
    assert r.verdict == "ok"
    assert r.permission == "authorized"


def test_settings_dict_exposes_mic_permission() -> None:
    s = settings_dict(Config.from_env())
    assert s["mic_permission"] in _KNOWN
