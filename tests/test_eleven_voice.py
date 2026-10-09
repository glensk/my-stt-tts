"""Tests for the ElevenLabs Mac audio bridge (no real audio device, no network)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, ClassVar

import pytest

from my_stt_tts import eleven_voice
from my_stt_tts.attention import AttentionStore
from my_stt_tts.bridge import Authoriser, BridgeController, Problem

pytest.importorskip("elevenlabs")


class _FakeStream:
    """Stands in for sounddevice Raw*Stream: records the callback, never opens a device."""

    created: ClassVar[list[_FakeStream]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.callback = kwargs["callback"]
        self.started = False
        _FakeStream.created.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        pass


class _FakeSd:
    RawOutputStream = _FakeStream
    RawInputStream = _FakeStream


@pytest.fixture(autouse=True)
def _fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeStream.created.clear()
    monkeypatch.setattr(eleven_voice, "_sd", lambda: _FakeSd)


def _play(n: int) -> bytes:
    """Pull ``n`` bytes from the playback stream (always the first stream created)."""
    out = bytearray(n)
    _FakeStream.created[0].callback(out, n // 2, None, None)
    return bytes(out)


def _mic() -> Any:
    """The mic stream's callback (created by ``start``, after the playback stream)."""
    return _FakeStream.created[1].callback


def test_output_is_played_then_padded_with_silence() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.output(b"\x01\x02\x03\x04")
    assert _play(8) == b"\x01\x02\x03\x04" + b"\x00" * 4
    assert _play(4) == b"\x00" * 4


def test_interrupt_drops_buffered_audio() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.output(b"\x05" * 100)
    iface.interrupt()
    assert _play(10) == b"\x00" * 10


def test_speakers_mode_mutes_mic_while_agent_speaks() -> None:
    sent: list[bytes] = []
    iface = eleven_voice.make_audio_interface("speakers", None, None)
    iface.start(sent.append)
    mic = _mic()
    iface.output(b"\x07" * 64)
    mic(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x00" * 8  # agent audio queued -> mic gated
    iface.interrupt()
    iface.playback.last_audio = time.monotonic() - 1.0
    mic(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x11" * 8  # agent silent -> mic passes through


def test_headphones_mode_never_gates_the_mic() -> None:
    sent: list[bytes] = []
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.start(sent.append)
    iface.output(b"\x07" * 64)
    _mic()(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x11" * 8


def test_missing_credentials_exit_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eleven_voice, "_load_env", lambda: None)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_AGENT_ID", raising=False)
    assert eleven_voice.main([]) == 2


class _FakeDuplex:
    """Stands in for aec.VoiceProcessingDuplex (no CoreAudio)."""

    starts_ok = True
    last: _FakeDuplex | None = None

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.played: list[bytes] = []
        self.flushed = 0
        _FakeDuplex.last = self

    def start(self) -> bool:
        return self.starts_ok

    def mic_frames(self) -> Any:
        return iter(())

    def play(self, pcm: bytes) -> None:
        self.played.append(pcm)

    def flush(self) -> None:
        self.flushed += 1

    def close(self) -> None:
        pass


def test_aec_mode_routes_playback_and_interrupt_through_voice_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("my_stt_tts.aec.VoiceProcessingDuplex", _FakeDuplex)
    iface = eleven_voice.make_audio_interface("aec", None, None)
    iface.start(lambda _pcm: None)
    iface.output(b"\x01\x02")
    iface.interrupt()
    vp = _FakeDuplex.last
    assert vp is not None
    assert vp.played == [b"\x01\x02"]
    assert vp.flushed == 1
    assert len(_FakeStream.created) == 1  # only the (unstarted) sounddevice playback object


def test_aec_mode_falls_back_to_gated_speakers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_FakeDuplex, "starts_ok", False)
    monkeypatch.setattr("my_stt_tts.aec.VoiceProcessingDuplex", _FakeDuplex)
    sent: list[bytes] = []
    iface = eleven_voice.make_audio_interface("aec", None, None)
    iface.start(sent.append)
    assert iface.mode == "speakers"
    iface.output(b"\x07" * 64)
    _mic()(b"\x11" * 8, 4, None, None)
    assert sent[-1] == b"\x00" * 8


def test_start_after_stop_does_not_open_the_mic() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.stop()  # end_session() before the SDK's late audio start
    iface.start(lambda _pcm: None)
    assert len(_FakeStream.created) == 1  # only the playback object; no input stream


def test_stop_is_idempotent() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.start(lambda _pcm: None)
    iface.stop()
    iface.stop()


def test_loud_mic_audio_marks_the_user_as_speaking() -> None:
    iface = eleven_voice.make_audio_interface("headphones", None, None)
    iface.start(lambda _pcm: None)
    _mic()(b"\x00\x00" * 4, 4, None, None)
    assert iface.last_user_audio == 0.0
    _mic()((3000).to_bytes(2, "little", signed=True) * 4, 4, None, None)
    assert iface.last_user_audio > 0.0


def test_every_log_line_gets_a_symbol() -> None:
    fmt = eleven_voice.EmojiFormatter("%(message)s")

    def line(name: str, level: int, msg: str) -> str:
        return fmt.format(logging.LogRecord(name, level, __file__, 1, msg, (), None))

    assert (
        line("elevenlabs", logging.ERROR, "Error receiving message") == "❌ Error receiving message"
    )
    assert line("x", logging.WARNING, "careful").startswith("⚠️")
    assert line("my_stt_tts.wake", logging.INFO, "wake detector: on") == "👂 wake detector: on"
    assert line("x", logging.INFO, "🔔 wake: hey_jarvis") == "🔔 wake: hey_jarvis"  # kept as is


@pytest.mark.parametrize(
    ("text", "stop"),
    [
        ("Voice off.", True),
        ("voice off", True),
        ("Weiß auf.", True),
        ("Weis aus", True),
        ("Stimme aus!", True),
        ("Voice on.", False),
        ("Can you turn the voice off later?", False),
        ("Weiß auf dem Papier", False),
    ],
)
def test_voice_off_phrase(text: str, stop: bool) -> None:
    assert eleven_voice.is_voice_off(text) is stop


# -- Phase 7: the bridge in a live VoiceSession (fake SDK Conversation, real ClientTools) ----
class _FakeConversation:  # pylint: disable=too-many-instance-attributes
    """Stands in for the SDK ``Conversation``: ``_run`` runs on ``_thread`` like the real one."""

    last: _FakeConversation | None = None
    behaviour = "open"  # open | quota | refused | silent

    def __init__(self, _client: Any, _agent_id: str, **kwargs: Any) -> None:
        self.config = kwargs["config"]
        self.client_tools = kwargs["client_tools"]
        self.on_user = kwargs["callback_user_transcript"]
        self.on_agent = kwargs["callback_agent_response"]
        self._thread: threading.Thread | None = None
        self._conversation_id: str | None = None
        self._stop = threading.Event()
        self.ended = 0
        self.sent: list[str] = []
        _FakeConversation.last = self

    def _run(self, _ws_url: str) -> None:
        if self.behaviour == "quota":
            raise RuntimeError("received 3000 (private use) [quota_exceeded]; then sent 3000")
        if self.behaviour == "refused":
            logging.getLogger("elevenlabs.conversational_ai.conversation").error(
                "Error receiving message: 1008 Override for field 'first_message' is not allowed"
            )
            return
        if self.behaviour == "open":
            self._conversation_id = "conv_test"
        self._stop.wait(10)

    def start_session(self) -> None:
        self._thread = threading.Thread(target=self._run, args=("wss://fake",), daemon=True)
        self._thread.start()

    def end_session(self) -> None:
        self.ended += 1
        self._stop.set()

    def wait_for_session_end(self) -> str | None:
        assert self._thread is not None
        self._thread.join(5)
        return self._conversation_id

    def send_user_message(self, text: str) -> None:
        self.sent.append(text)


class _Vad:
    def is_speech(self, _frame: Any) -> bool:
        return False


class _Briefings:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.marked = 0

    def briefing(self) -> str | None:
        return self.text

    def mark_briefed(self) -> int:
        self.marked += 1
        return 1


def _wait(pred: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@pytest.fixture(name="sdk")
def _sdk(monkeypatch: pytest.MonkeyPatch) -> type[_FakeConversation]:
    monkeypatch.setattr("elevenlabs.conversational_ai.conversation.Conversation", _FakeConversation)
    monkeypatch.setattr("elevenlabs.client.ElevenLabs", lambda **_kw: object())
    monkeypatch.setattr(_FakeConversation, "behaviour", "open")
    return _FakeConversation


@pytest.fixture(name="banners")
def _banners(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    posted: list[str] = []

    def post(kind: str, runner: Any = None) -> bool:
        del runner
        posted.append(kind)
        return True

    monkeypatch.setattr("my_stt_tts.attention.post_banner", post)
    return posted


def _controller(briefing: str | None = None, store: Any = None) -> Any:
    ctl = BridgeController(Authoriser(None, None), vad_factory=_Vad, problems=store)
    ctl.briefings = _Briefings(briefing)
    return ctl


def _session(ctl: Any = None, **kwargs: Any) -> eleven_voice.VoiceSession:
    return eleven_voice.VoiceSession(
        "agent_test", "key_test", mode="headphones", echo=False, bridge=ctl, **kwargs
    )


def test_every_controller_tool_is_registered_on_the_client_tools(sdk) -> None:
    ctl = _controller()
    ctl.register_tool("open_url", lambda target: f"ok: {target}")
    session = _session(ctl)
    assert set(session.client_tools.tools) == {"confirm_action", "open_url"}
    assert sdk.last is not None and sdk.last.client_tools is session.client_tools


def test_tool_wrapper_strips_the_call_id_and_voids_unused_rights(sdk) -> None:
    del sdk
    ctl = _controller()
    seen: list[Any] = []
    contexts: list[str] = []

    def open_url(target: str) -> str:
        seen.append((target, ctl.tool_in_flight("open_url")))
        return "ok: opened"

    ctl.register_tool("open_url", open_url)
    ctl.note_context = contexts.append
    session = _session(ctl)
    handler = session.client_tools.tools["open_url"][0]
    assert handler({"tool_call_id": "t1", "target": "youtube"}) == "ok: opened"
    assert seen == [("youtube", True)]  # in flight while it runs
    assert not ctl.tool_in_flight("open_url")
    assert contexts == ["tool result"]


def test_tool_errors_become_speakable_answers(sdk) -> None:
    del sdk
    ctl = _controller()

    def broken(target: str) -> str:
        raise RuntimeError(f"secret detail {target}")

    def slow() -> str:
        time.sleep(2)
        return "ok"

    ctl.register_tool("open_url", broken)
    ctl.register_tool("youtube_play_first", slow)
    contexts: list[str] = []
    ctl.note_context = contexts.append
    session = _session(ctl, tool_guard=0.2)
    tools = session.client_tools.tools
    assert tools["open_url"][0]({"target": "x"}) == "open_url failed: internal error"
    assert tools["open_url"][0]({}) == "open_url: missing or unexpected arguments"
    assert tools["youtube_play_first"][0]({}) == "youtube_play_first failed: it took too long"
    assert contexts == ["tool result"] * 3


def test_briefing_is_the_first_message_and_marked_only_once_spoken(sdk) -> None:
    text = "One problem needs your attention. voice bridge: message failed."
    ctl = _controller(text)
    session = _session(ctl)
    assert sdk.last is not None
    assert sdk.last.config.conversation_config_override == {"agent": {"first_message": text}}
    session.start()
    _wait(lambda: sdk.last._conversation_id is not None)
    assert ctl.briefings.marked == 0  # connected, but nothing spoken yet
    sdk.last.on_agent("One problem needs your attention.")
    sdk.last.on_agent("Anything else?")
    assert ctl.briefings.marked == 1
    session.end()
    session.wait()


def test_no_briefing_no_override(sdk) -> None:
    _session(_controller(None))
    assert sdk.last is not None and sdk.last.config.conversation_config_override == {}


def test_quota_startup_failure_reports_banners_and_ends(sdk, banners, tmp_path, caplog) -> None:
    store = AttentionStore(tmp_path / "attention.db")
    ctl = _controller(None, store)
    ctl.briefings = store
    sdk.behaviour = "quota"
    session = _session(ctl)
    with caplog.at_level(logging.ERROR, logger="my_stt_tts.eleven_voice"):
        session.start()
        _wait(lambda: bool(session.start_failure))
        assert session.wait() is None  # the session ended: the daemon is not left hanging
    assert session.start_failure == "ElevenLabs quota exceeded"
    assert "❌ ElevenLabs: quota exceeded" in caplog.text
    assert banners == ["voice startup failed"]
    assert sdk.last is not None and sdk.last.ended >= 1
    item = store.get("startup:voice")
    assert item is not None and item.reason == "ElevenLabs quota exceeded"
    assert item.briefed_at is None  # retried at the next call
    assert ctl.turns is None  # end_call ran
    store.close()


def test_startup_failure_keeps_the_briefing_unbriefed(sdk, banners, tmp_path) -> None:
    del banners
    store = AttentionStore(tmp_path / "attention.db")
    store.record(Problem("delivery_failed", "scratch-dead", "message failed"))
    ctl = _controller(None, store)
    ctl.briefings = store
    sdk.behaviour = "quota"
    session = _session(ctl)
    assert sdk.last is not None and sdk.last.config.conversation_config_override
    session.start()
    _wait(lambda: bool(session.start_failure))
    session.wait()
    assert all(item.briefed_at is None for item in store.items())
    store.close()


def test_startup_failure_without_a_bridge_still_banners(sdk, banners) -> None:
    sdk.behaviour = "quota"
    session = _session(None)
    session.start()
    _wait(lambda: bool(session.start_failure))
    session.wait()
    assert banners == ["voice startup failed"]


def test_no_conversation_within_the_start_timeout_is_a_failure(sdk, banners) -> None:
    del banners
    sdk.behaviour = "silent"
    session = _session(None, start_timeout=0.2)
    session.start()
    _wait(lambda: bool(session.start_failure))
    session.wait()
    assert session.start_failure == "voice agent did not start"


def test_refused_override_turns_briefings_off(sdk, banners) -> None:
    del banners
    ctl = _controller("One problem needs your attention.")
    sdk.behaviour = "refused"
    session = _session(ctl)
    session.start()
    _wait(lambda: bool(session.start_failure))
    session.wait()
    assert session.start_failure == "agent refused the start settings"
    assert ctl.first_message_ok is False
    assert ctl.briefings.marked == 0
    _session(ctl)  # the next call starts without the override
    assert sdk.last is not None and sdk.last.config.conversation_config_override == {}


def test_notify_in_call_sends_a_system_notice_and_voids_rights(sdk) -> None:
    ctl = _controller()
    injected: list[str] = []
    real = ctl.on_injected

    def spy(text: str, **kwargs: Any) -> Any:
        injected.append(text)
        return real(text, **kwargs)

    ctl.on_injected = spy
    assert ctl.notify_in_call("before the call") is False
    session = _session(ctl)
    session.start()
    _wait(lambda: sdk.last._conversation_id is not None)
    assert ctl.notify_in_call("scratch-dead: message failed") is True
    assert sdk.last.sent == ["[system notice] scratch-dead: message failed"]
    assert injected == ["[system notice] scratch-dead: message failed"]
    session.end()
    session.wait()
    assert ctl.notify_in_call("after the call") is False
    assert len(sdk.last.sent) == 1
