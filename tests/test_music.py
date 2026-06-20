"""Tests for play-music-from-YouTube: the intent router, the player, the turn hook.

Everything that would touch the network (yt-dlp) or the speakers (mpv/ffplay,
project playback) is MOCKED — no real search, download, audio, or subprocess.
Coverage:

* :func:`match_music_intent` across EN/DE/FR play / stop / pause / resume, and
  non-music turns that must NOT be hijacked.
* :func:`search` / :class:`MusicPlayer.play` with yt-dlp + the player subprocess
  faked: success starts a tracked process; "no result" + missing-yt-dlp degrade to
  a spoken reason; mpv pause/resume/stop drive the IPC + process.
* graceful missing-deps: yt-dlp absent and no player present both yield a clear,
  speakable reason and never raise.
* the turn hook (:func:`my_stt_tts.__main__.maybe_handle_music`) handles a music
  intent (plays + speaks a confirmation, skips the LLM) and passes non-music
  through; :func:`_respond` short-circuits the LLM on a music turn.
"""
# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name
# pylint: disable=too-few-public-methods,unused-argument

from typing import Any

import pytest

from my_stt_tts import music
from my_stt_tts.config import Config

# --- intent router -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "query"),
    [
        ("play We Will Rock You by Queen", "We Will Rock You by Queen"),
        ("play We Will Rock You by Queen from youtube", "We Will Rock You by Queen"),
        ("Play Bohemian Rhapsody", "Bohemian Rhapsody"),
        ("put on Wonderwall", "Wonderwall"),
        ("please play Africa by Toto", "Africa by Toto"),
        # German
        ("spiele Bohemian Rhapsody", "Bohemian Rhapsody"),
        ("spiel Wonderwall von Oasis", "Wonderwall von Oasis"),
        # French
        ("joue We Will Rock You", "We Will Rock You"),
        ("mets Bohemian Rhapsody", "Bohemian Rhapsody"),
    ],
)
def test_play_with_query(text, query):
    assert music.match_music_intent(text) == {"action": "play", "query": query}


@pytest.mark.parametrize(
    "text",
    [
        "play some music",
        "put on some music",
        "mach Musik an",
        "mach mal Musik",
        "spiele etwas Musik",
        "mets de la musique",
        "joue de la musique",
    ],
)
def test_play_generic_has_empty_query(text):
    assert music.match_music_intent(text) == {"action": "play", "query": ""}


@pytest.mark.parametrize(
    "text",
    [
        "stop",
        "stop the music",
        "Stop the song",
        "stopp",
        "halt die Musik",
        "arrête",
        "arrête la musique",
    ],
)
def test_stop(text):
    assert music.match_music_intent(text) == {"action": "stop"}


@pytest.mark.parametrize("text", ["pause", "pause the music", "pausiere", "mets en pause"])
def test_pause(text):
    assert music.match_music_intent(text) == {"action": "pause"}


@pytest.mark.parametrize("text", ["resume", "continue", "weiter", "mach weiter", "reprends"])
def test_resume(text):
    assert music.match_music_intent(text) == {"action": "resume"}


@pytest.mark.parametrize(
    "text",
    ["", "   ", "what is the weather today", "tell me a joke", "how are you", "set a timer"],
)
def test_non_music_is_none(text):
    assert music.match_music_intent(text) is None


def test_case_insensitive():
    assert music.match_music_intent("PLAY Thriller") == {"action": "play", "query": "Thriller"}
    assert music.match_music_intent("STOP")["action"] == "stop"


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("stop the music please", "stop"),
        ("stop please", "stop"),
        ("stop now", "stop"),
        ("stop, please", "stop"),
        ("pause please", "pause"),
        ("resume please", "resume"),
        ("stopp die Musik bitte", "stop"),
        ("pausiere bitte", "pause"),
        ("weiter bitte", "resume"),
        ("arrête la musique s'il te plaît", "stop"),
        ("pause s'il vous plaît", "pause"),
        ("stop svp", "stop"),
    ],
)
def test_control_intent_strips_trailing_politeness(text, action):
    """The root-cause fix at the router: trailing politeness must not break the
    match (so "stop the music please" routes locally instead of to the LLM)."""
    assert music.match_music_intent(text) == {"action": action}


def test_politeness_does_not_hijack_a_play_query():
    # "please" only LEADS a play ("please play X"); a song literally named with a
    # trailing word is still a play, never mis-classified as a control command.
    assert music.match_music_intent("play stop this train") == {
        "action": "play",
        "query": "stop this train",
    }


# --- search (yt-dlp mocked) ----------------------------------------------------


class _FakeYDL:
    """A minimal yt_dlp.YoutubeDL stand-in returning a canned search result."""

    def __init__(self, info: dict[str, Any] | None) -> None:
        self._info = info

    def __enter__(self) -> "_FakeYDL":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def extract_info(self, query: str, download: bool = False) -> dict[str, Any] | None:
        return self._info

    def download(self, urls: list[str]) -> int:
        return 0


def _patch_ytdlp(monkeypatch, info: dict[str, Any] | None) -> None:
    """Make ``import yt_dlp`` inside music.py resolve to a fake returning ``info``."""
    fake_module = type("yt_dlp", (), {"YoutubeDL": lambda self, opts: _FakeYDL(info)})()
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", fake_module)
    monkeypatch.setattr(music, "yt_dlp_available", lambda: True)


def test_search_returns_track(monkeypatch):
    _patch_ytdlp(
        monkeypatch,
        {"webpage_url": "https://youtu.be/abc", "title": "We Will Rock You"},
    )
    track = music.search("we will rock you")
    assert track is not None
    assert track.url == "https://youtu.be/abc"
    assert track.title == "We Will Rock You"


def test_search_handles_entries_list(monkeypatch):
    _patch_ytdlp(
        monkeypatch,
        {"entries": [{"webpage_url": "https://youtu.be/xyz", "title": "Song"}]},
    )
    track = music.search("song")
    assert track is not None and track.url == "https://youtu.be/xyz"


def test_search_extracts_video_id_from_yt_dlp_id(monkeypatch):
    _patch_ytdlp(
        monkeypatch,
        {"id": "dQw4w9WgXcQ", "webpage_url": "https://youtu.be/dQw4w9WgXcQ", "title": "Song"},
    )
    track = music.search("song")
    assert track is not None and track.video_id == "dQw4w9WgXcQ"


def test_search_extracts_video_id_from_watch_url(monkeypatch):
    # No bare `id`; the 11-char id is parsed out of a watch?v= URL.
    _patch_ytdlp(
        monkeypatch,
        {"webpage_url": "https://www.youtube.com/watch?v=abc12345678", "title": "Song"},
    )
    track = music.search("song")
    assert track is not None and track.video_id == "abc12345678"


def test_search_no_result_returns_none(monkeypatch):
    _patch_ytdlp(monkeypatch, None)
    assert music.search("nonsense query") is None


def test_search_without_ytdlp_returns_none(monkeypatch):
    # Simulate yt-dlp not installed: the lazy import raises ImportError.
    monkeypatch.setitem(__import__("sys").modules, "yt_dlp", None)
    assert music.search("anything") is None


# --- player (subprocess + project playback mocked) -----------------------------


@pytest.fixture(autouse=True)
def _reset_player():
    music.reset_player()
    yield
    music.reset_player()


class _FakeProc:
    """A subprocess.Popen stand-in: starts 'running', stops on terminate/kill."""

    def __init__(self) -> None:
        self._alive = True
        self.terminated = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        return 0


def test_play_success_starts_process(monkeypatch):
    _patch_ytdlp(monkeypatch, {"webpage_url": "https://youtu.be/abc", "title": "Thriller"})
    monkeypatch.setattr(
        music.shutil, "which", lambda name: "/usr/bin/mpv" if name == "mpv" else None
    )
    started = {}

    def _fake_popen(cmd, **kwargs):
        started["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(music.subprocess, "Popen", _fake_popen)
    player = music.MusicPlayer(player="mpv")
    result = player.play("thriller")
    assert result.ok
    assert result.title == "Thriller"
    assert player.is_playing()
    assert player.now_playing() == "Thriller"
    assert started["cmd"][0] == "mpv"
    assert "--no-video" in started["cmd"]
    assert "https://youtu.be/abc" in started["cmd"]


def test_play_no_result_speaks_reason(monkeypatch):
    _patch_ytdlp(monkeypatch, None)
    monkeypatch.setattr(music.shutil, "which", lambda name: "/usr/bin/mpv")
    player = music.MusicPlayer(player="mpv")
    result = player.play("doesnotexist")
    assert not result.ok
    assert "couldn't find" in result.reason.lower()
    assert not player.is_playing()


def test_play_without_ytdlp_speaks_install_reason(monkeypatch):
    monkeypatch.setattr(music, "yt_dlp_available", lambda: False)
    player = music.MusicPlayer()
    result = player.play("anything")
    assert not result.ok
    assert "yt-dlp" in result.reason


def test_stop_terminates_process(monkeypatch):
    _patch_ytdlp(monkeypatch, {"webpage_url": "https://youtu.be/abc", "title": "Song"})
    monkeypatch.setattr(
        music.shutil, "which", lambda name: "/usr/bin/mpv" if name == "mpv" else None
    )
    proc = _FakeProc()
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(music.MusicPlayer, "_mpv_command", lambda self, payload: True)
    player = music.MusicPlayer(player="mpv")
    player.play("song")
    assert player.stop() is True
    assert proc.terminated
    assert not player.is_playing()
    # Stopping when nothing plays is a no-op (False), never an error.
    assert player.stop() is False


def test_pause_resume_use_mpv_ipc(monkeypatch):
    _patch_ytdlp(monkeypatch, {"webpage_url": "https://youtu.be/abc", "title": "Song"})
    monkeypatch.setattr(
        music.shutil, "which", lambda name: "/usr/bin/mpv" if name == "mpv" else None
    )
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: _FakeProc())
    ipc_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        music.MusicPlayer, "_mpv_command", lambda self, payload: ipc_calls.append(payload) or True
    )
    player = music.MusicPlayer(player="mpv")
    player.play("song")
    assert player.pause() is True
    assert player.resume() is True
    assert {"command": ["set_property", "pause", True]} in ipc_calls
    assert {"command": ["set_property", "pause", False]} in ipc_calls


def test_pause_resume_no_op_without_mpv(monkeypatch):
    # ffplay backend: pause/resume are best-effort no-ops (stop only).
    _patch_ytdlp(
        monkeypatch,
        {"webpage_url": "https://youtu.be/abc", "title": "Song", "url": "https://x/audio"},
    )
    monkeypatch.setattr(
        music.shutil, "which", lambda name: "/usr/bin/ffplay" if name == "ffplay" else None
    )
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: _FakeProc())
    player = music.MusicPlayer(player="ffplay")
    player.play("song")
    assert player.pause() is False
    assert player.resume() is False
    assert player.is_playing()  # still playing; pause just isn't supported


def test_play_picks_ffplay_when_no_mpv(monkeypatch):
    _patch_ytdlp(
        monkeypatch,
        {"webpage_url": "https://youtu.be/abc", "title": "Song", "url": "https://x/audio.m4a"},
    )
    monkeypatch.setattr(
        music.shutil, "which", lambda name: "/usr/bin/ffplay" if name == "ffplay" else None
    )
    cmds: list[list[str]] = []
    monkeypatch.setattr(music.subprocess, "Popen", lambda cmd, **k: cmds.append(cmd) or _FakeProc())
    player = music.MusicPlayer(player="auto")
    result = player.play("song")
    assert result.ok
    assert cmds[0][0] == "ffplay"


def test_get_player_is_shared_singleton():
    p1 = music.get_player(player="mpv")
    p2 = music.get_player(player="ffplay")  # args ignored after first build
    assert p1 is p2
    assert p1.player == "mpv"


# --- system-state line + status snapshot ---------------------------------------


def test_music_state_line_when_idle():
    # No player constructed yet -> idle, side-effect-free.
    assert music.music_state_line() == "System state: no music is playing."


def test_music_state_line_reflects_playing(monkeypatch):
    _patch_ytdlp(monkeypatch, {"webpage_url": "https://youtu.be/abc", "title": "Thriller"})
    monkeypatch.setattr(music.shutil, "which", lambda name: "/usr/bin/mpv")
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: _FakeProc())
    player = music.get_player(player="mpv")  # the SAME singleton music_state_line() reads
    player.play("thriller")
    line = music.music_state_line()
    assert "currently playing" in line and "Thriller" in line


def test_music_state_line_reflects_paused(monkeypatch):
    _patch_ytdlp(monkeypatch, {"webpage_url": "https://youtu.be/abc", "title": "Song"})
    monkeypatch.setattr(music.shutil, "which", lambda name: "/usr/bin/mpv")
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(music.MusicPlayer, "_mpv_command", lambda self, payload: True)
    player = music.get_player(player="mpv")
    player.play("song")
    player.pause()
    assert "paused" in music.music_state_line()


def test_status_snapshot_carries_video_id(monkeypatch):
    _patch_ytdlp(
        monkeypatch,
        {"id": "dQw4w9WgXcQ", "webpage_url": "https://youtu.be/dQw4w9WgXcQ", "title": "Song"},
    )
    monkeypatch.setattr(music.shutil, "which", lambda name: "/usr/bin/mpv")
    monkeypatch.setattr(music.subprocess, "Popen", lambda *a, **k: _FakeProc())
    player = music.MusicPlayer(player="mpv")
    player.play("song")
    snap = player.status()
    assert snap["status"] == "playing"
    assert snap["video_id"] == "dQw4w9WgXcQ"
    assert snap["url"] == "https://youtu.be/dQw4w9WgXcQ"


# --- GUI control actions (_music_action) ---------------------------------------


def test_music_action_stop_emits_event(monkeypatch):
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.events import bus

    monkeypatch.setattr(music.MusicPlayer, "stop", lambda self: True)
    sub = bus.subscribe()
    try:
        main_mod._music_action("music_stop")
        events = _drain_bus_events(sub)
    finally:
        bus.unsubscribe(sub)
    assert any(e.get("type") == "music" and e["status"] == "stopped" for e in events)


def test_music_action_pause_resume_emit_events(monkeypatch):
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.events import bus

    monkeypatch.setattr(music.MusicPlayer, "pause", lambda self: True)
    monkeypatch.setattr(music.MusicPlayer, "resume", lambda self: True)
    monkeypatch.setattr(
        music.MusicPlayer,
        "status",
        lambda self: {"status": "paused", "title": "Song", "video_id": "id", "url": "u"},
    )
    sub = bus.subscribe()
    try:
        main_mod._music_action("music_pause")
        main_mod._music_action("music_resume")
        events = _drain_bus_events(sub)
    finally:
        bus.unsubscribe(sub)
    statuses = {e["status"] for e in events if e.get("type") == "music"}
    assert {"paused", "resumed"} <= statuses


# --- the turn hook -------------------------------------------------------------


class _RecordingTTS:
    """A TTSRouter stand-in that records spoken sentences (no audio)."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str, lang: str | None = None) -> None:
        self.spoken.append(text)


class _NullGate:
    def gate(self) -> None:
        pass

    def release(self) -> None:
        pass


def test_turn_hook_handles_play(monkeypatch):
    from my_stt_tts import __main__ as main_mod

    tts = _RecordingTTS()
    played = {}

    def _fake_play(self, q):
        played["q"] = q
        return music.PlayResult(ok=True, title="We Will Rock You", video_id="abc12345678")

    monkeypatch.setattr(music.MusicPlayer, "play", _fake_play)
    cfg = Config(anthropic_api_key="sk-test")
    handled = main_mod.maybe_handle_music(cfg, tts, _NullGate(), "play We Will Rock You by Queen")
    assert handled is True
    assert played["q"] == "We Will Rock You by Queen"
    # The glyph-free SPOKEN line never contains the ▶ symbol (it must not be read aloud).
    assert any("Playing We Will Rock You" in s for s in tts.spoken)
    assert all("▶" not in s for s in tts.spoken)


def test_turn_hook_handles_stop(monkeypatch):
    from my_stt_tts import __main__ as main_mod

    tts = _RecordingTTS()
    monkeypatch.setattr(music.MusicPlayer, "stop", lambda self: True)
    cfg = Config(anthropic_api_key="sk-test")
    assert main_mod.maybe_handle_music(cfg, tts, _NullGate(), "stop the music") is True
    assert any("Stopped the music" in s for s in tts.spoken)


def test_turn_hook_speaks_reason_on_failure(monkeypatch):
    from my_stt_tts import __main__ as main_mod

    tts = _RecordingTTS()
    monkeypatch.setattr(
        music.MusicPlayer,
        "play",
        lambda self, q: music.PlayResult(
            ok=False, reason="I can't play music yet — please install yt-dlp."
        ),
    )
    cfg = Config(anthropic_api_key="sk-test")
    assert main_mod.maybe_handle_music(cfg, tts, _NullGate(), "play anything") is True
    assert any("yt-dlp" in s for s in tts.spoken)


def _drain_bus_events(sub) -> list[dict]:
    """Collect every JSON event currently queued on a bus subscriber."""
    import json
    import queue

    out: list[dict] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


def test_turn_hook_play_emits_assistant_response_and_music_event(monkeypatch):
    """ALWAYS-RESPOND: a play turn must emit a final bus.response (the transcript
    bubble) AND a structured `music` event carrying the video_id/url."""
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.events import bus

    monkeypatch.setattr(
        music.MusicPlayer,
        "play",
        lambda self, q: music.PlayResult(
            ok=True, title="Thriller", video_id="dQw4w9WgXcQ", url="https://youtu.be/dQw4w9WgXcQ"
        ),
    )
    sub = bus.subscribe()
    try:
        cfg = Config(anthropic_api_key="sk-test")
        assert main_mod.maybe_handle_music(cfg, _RecordingTTS(), _NullGate(), "play Thriller")
        events = _drain_bus_events(sub)
    finally:
        bus.unsubscribe(sub)
    # A final assistant response with the glyph + the active model label (the bubble).
    responses = [e for e in events if e.get("type") == "response" and e.get("final")]
    assert responses, "play must emit a final bus.response so the transcript shows a bubble"
    assert "▶ Playing: Thriller." in responses[0]["text"]
    assert responses[0]["model"]  # "ASSISTANT · <model>" label
    # The structured music event the GUI consumes, with the embeddable id + url.
    music_events = [e for e in events if e.get("type") == "music"]
    assert music_events and music_events[0]["status"] == "playing"
    assert music_events[0]["video_id"] == "dQw4w9WgXcQ"
    assert music_events[0]["url"] == "https://youtu.be/dQw4w9WgXcQ"


def test_turn_hook_stop_emits_response_and_stopped_event(monkeypatch):
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.events import bus

    monkeypatch.setattr(music.MusicPlayer, "stop", lambda self: True)
    sub = bus.subscribe()
    try:
        cfg = Config(anthropic_api_key="sk-test")
        assert main_mod.maybe_handle_music(cfg, _RecordingTTS(), _NullGate(), "stop the music")
        events = _drain_bus_events(sub)
    finally:
        bus.unsubscribe(sub)
    responses = [e for e in events if e.get("type") == "response" and e.get("final")]
    assert any("Stopped the music" in e["text"] for e in responses)
    assert any(e.get("type") == "music" and e["status"] == "stopped" for e in events)


@pytest.mark.parametrize(
    "phrase",
    [
        "stop the music please",
        "stop music",
        "stop, please",
        "stop now",
        "pause please",
        "stopp die Musik bitte",
        "arrête la musique s'il te plaît",
        "pause s'il vous plaît",
    ],
)
def test_control_with_trailing_politeness_never_reaches_llm(monkeypatch, phrase):
    """ROOT-CAUSE FIX: a control phrase with trailing politeness ("…please") must
    be handled locally (returns True) and never fall through to the LLM."""
    from my_stt_tts import __main__ as main_mod

    # Player control is faked so nothing real runs; the point is that the router
    # OWNS the turn (returns True) for every politeness-suffixed control phrase.
    monkeypatch.setattr(music.MusicPlayer, "stop", lambda self: True)
    monkeypatch.setattr(music.MusicPlayer, "pause", lambda self: True)
    monkeypatch.setattr(music.MusicPlayer, "is_playing", lambda self: True)
    cfg = Config(anthropic_api_key="sk-test")
    assert main_mod.maybe_handle_music(cfg, _RecordingTTS(), _NullGate(), phrase) is True


def test_stop_with_nothing_playing_is_answered_by_router(monkeypatch):
    """A stop with nothing playing must be answered BY THE ROUTER ("Nothing is
    playing right now."), not handed to the LLM (which hallucinated)."""
    from my_stt_tts import __main__ as main_mod

    tts = _RecordingTTS()
    monkeypatch.setattr(music.MusicPlayer, "stop", lambda self: False)
    cfg = Config(anthropic_api_key="sk-test")
    assert main_mod.maybe_handle_music(cfg, tts, _NullGate(), "stop the music please") is True
    assert any("Nothing is playing" in s for s in tts.spoken)


def test_turn_hook_passes_non_music_through():
    from my_stt_tts import __main__ as main_mod

    tts = _RecordingTTS()
    cfg = Config(anthropic_api_key="sk-test")
    assert main_mod.maybe_handle_music(cfg, tts, _NullGate(), "what's the weather") is False
    assert tts.spoken == []


def test_turn_hook_disabled_when_music_off():
    from my_stt_tts import __main__ as main_mod

    tts = _RecordingTTS()
    cfg = Config(anthropic_api_key="sk-test", music_enabled=False)
    assert main_mod.maybe_handle_music(cfg, tts, _NullGate(), "play Thriller") is False
    assert tts.spoken == []


def test_respond_skips_llm_on_music_intent(monkeypatch):
    from my_stt_tts import __main__ as main_mod

    # If the LLM were reached, brain.stream would be called — assert it is NOT.
    class _Brain:
        def set_speaker(self, name):  # noqa: D401
            pass

        def stream(self, text, deep=None):
            raise AssertionError("LLM must not be called on a music intent")

    monkeypatch.setattr(main_mod, "maybe_handle_music", lambda cfg, tts, gate, text: True)
    cfg = Config(anthropic_api_key="sk-test")
    result = main_mod._respond(cfg, _Brain(), _RecordingTTS(), _NullGate(), "play Thriller")
    assert result.interrupted is False


# --- tool path (API brains) ----------------------------------------------------


def test_default_tools_include_music():
    from my_stt_tts.tools import default_tools

    names = {t.name for t in default_tools(music_enabled=True)}
    assert {"play_music", "stop_music"} <= names


def test_default_tools_omit_music_when_disabled():
    from my_stt_tts.tools import default_tools

    names = {t.name for t in default_tools(music_enabled=False)}
    assert "play_music" not in names and "stop_music" not in names


def test_play_music_tool_runs(monkeypatch):
    from my_stt_tts.tools import make_music_tools

    monkeypatch.setattr(
        music.MusicPlayer, "play", lambda self, q: music.PlayResult(ok=True, title="Thriller")
    )
    tools = {t.name: t for t in make_music_tools()}
    out = tools["play_music"].run({"query": "thriller"})
    assert "Playing Thriller" in out


def test_stop_music_tool_runs(monkeypatch):
    from my_stt_tts.tools import make_music_tools

    monkeypatch.setattr(music.MusicPlayer, "stop", lambda self: True)
    tools = {t.name: t for t in make_music_tools()}
    assert "Stopped" in tools["stop_music"].run({})


def test_play_music_tool_requires_query():
    from my_stt_tts.tools import make_music_tools

    tools = {t.name: t for t in make_music_tools()}
    assert "no song" in tools["play_music"].run({"query": ""})


# --- config --------------------------------------------------------------------


def test_music_config_defaults_valid():
    cfg = Config(anthropic_api_key="sk-test")
    assert cfg.music_enabled is True
    assert cfg.music_player == "auto"
    cfg.validate()


def test_music_player_invalid_rejected():
    from my_stt_tts.config import ConfigError

    with pytest.raises(ConfigError):
        Config(anthropic_api_key="sk-test", music_player="bogus").validate()


def test_music_volume_range_validated():
    from my_stt_tts.config import ConfigError

    Config(anthropic_api_key="sk-test", music_volume=50).validate()
    with pytest.raises(ConfigError):
        Config(anthropic_api_key="sk-test", music_volume=200).validate()


def test_music_env_loading(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MUSIC_ENABLED", "false")
    monkeypatch.setenv("MUSIC_PLAYER", "mpv")
    monkeypatch.setenv("MUSIC_VOLUME", "70")
    cfg = Config.from_env()
    assert cfg.music_enabled is False
    assert cfg.music_player == "mpv"
    assert cfg.music_volume == 70


# --- music_playback (server vs hybrid) -----------------------------------------


def test_music_playback_defaults_hybrid():
    cfg = Config(anthropic_api_key="sk-test")
    assert cfg.music_playback == "hybrid"
    cfg.validate()


def test_music_playback_server_valid():
    Config(anthropic_api_key="sk-test", music_playback="server").validate()


def test_music_playback_invalid_rejected():
    from my_stt_tts.config import ConfigError

    with pytest.raises(ConfigError, match="music_playback"):
        Config(anthropic_api_key="sk-test", music_playback="bogus").validate()


def test_music_playback_env_loading(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MUSIC_PLAYBACK", "server")
    assert Config.from_env().music_playback == "server"


def test_music_playback_env_defaults_hybrid_when_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("MUSIC_PLAYBACK", raising=False)
    assert Config.from_env().music_playback == "hybrid"


def test_music_playback_in_settings_dict():
    from my_stt_tts.webui import settings_dict

    d = settings_dict(Config(anthropic_api_key="sk-test", music_playback="server"))
    assert d["music_playback"] == "server"
    assert d["music_playback_modes"] == ["server", "hybrid"]


def test_music_playback_apply_settings():
    from my_stt_tts.webui import apply_settings

    cfg = Config(anthropic_api_key="sk-test")
    apply_settings(cfg, {"music_playback": "server"})
    assert cfg.music_playback == "server"
    # An unrecognised value is ignored (kept valid for validate()).
    apply_settings(cfg, {"music_playback": "bogus"})
    assert cfg.music_playback == "server"
    cfg.validate()


def test_music_playback_in_settings_text():
    from my_stt_tts.__main__ import settings_text

    text = settings_text(Config(anthropic_api_key="sk-test", music_playback="server"), color=False)
    assert "playback server" in text
