"""Tests for G2 — typed, prioritized, non-droppable event model.

Covers: event classification (system vs data), SYSTEM frames bypassing queued
data, system frames flushing stale data, non-drop under back-pressure (a full
data queue still delivers every system frame), consistent ordering across many
subscribers (= every transport drains the same way), and back-compat of the
public bus API (subscribe / get / get_nowait / empty / the emitters).
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,import-outside-toplevel

import json
import logging
import queue
import threading

import pytest

import my_stt_tts.events as events_mod
from my_stt_tts.events import (
    EventBus,
    Frame,
    LogBusHandler,
    Priority,
    classify,
    install_log_bridge,
    render_console_line,
)


def _types(sub, limit=1000):
    out = []
    for _ in range(limit):
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_classify_system_vs_data():
    assert classify({"type": "interrupt", "phase": "start"}) == Priority.SYSTEM
    assert classify({"type": "barge_in"}) == Priority.SYSTEM
    assert classify({"type": "error"}) == Priority.SYSTEM
    assert classify({"type": "end_of_turn"}) == Priority.SYSTEM
    assert classify({"type": "bot_stopped_speaking"}) == Priority.SYSTEM
    assert classify({"type": "state", "state": "idle"}) == Priority.DATA
    assert classify({"type": "transcript", "text": "hi"}) == Priority.DATA
    assert classify({"type": "response", "text": "x"}) == Priority.DATA


def test_classify_error_log_promoted_to_system():
    assert classify({"type": "log", "level": "error", "message": "boom"}) == Priority.SYSTEM
    assert classify({"type": "log", "level": "info", "message": "hi"}) == Priority.DATA


def test_frame_of_serializes_and_classifies():
    f = Frame.of({"type": "interrupt", "phase": "start"})
    assert f.priority == Priority.SYSTEM
    assert f.type == "interrupt"
    assert json.loads(f.data)["phase"] == "start"


# ---------------------------------------------------------------------------
# system frames bypass + flush queued data
# ---------------------------------------------------------------------------


def test_system_frame_delivered_before_queued_data():
    bus = EventBus()
    sub = bus.subscribe()
    bus.transcript("partial 1", partial=True)
    bus.transcript("partial 2", partial=True)
    bus.interrupt_start()  # SYSTEM: must jump the queue
    first = json.loads(sub.get_nowait())
    assert first["type"] == "interrupt" and first["phase"] == "start"


def test_system_frame_flushes_stale_data():
    bus = EventBus()
    sub = bus.subscribe()
    bus.response("hello ", final=False)
    bus.response("world ", final=False)
    bus.interrupt_start()  # flushes the two stale response deltas
    events = _types(sub)
    # Only the interrupt remains; the stale partial deltas were flushed.
    assert [e["type"] for e in events] == ["interrupt"]


def test_data_after_system_still_flows():
    bus = EventBus()
    sub = bus.subscribe()
    bus.interrupt_start()
    bus.state("recording")  # new data AFTER the system frame survives
    events = _types(sub)
    assert events[0]["type"] == "interrupt"
    assert any(e["type"] == "state" and e["state"] == "recording" for e in events)


# ---------------------------------------------------------------------------
# non-drop under back-pressure
# ---------------------------------------------------------------------------


def test_system_frames_never_dropped_under_full_data_queue():
    bus = EventBus()
    sub = bus.subscribe()
    # Overflow the bounded data queue far past its maxsize WITHOUT draining.
    for i in range(sub._data.maxsize + 200):
        bus.transcript(f"p{i}", partial=True)
    # Now fire several system frames — none may be lost.
    bus.interrupt_start()
    bus.error("kaboom")
    bus.interrupt_stop()
    events = _types(sub)
    sys_types = [(e.get("type"), e.get("phase")) for e in events if e.get("type") != "transcript"]
    assert ("interrupt", "start") in sys_types
    assert ("error", None) in sys_types
    assert ("interrupt", "stop") in sys_types


def test_data_may_drop_but_system_survives_concurrent_load():
    bus = EventBus()
    sub = bus.subscribe()
    stop = threading.Event()

    def _flood():
        i = 0
        while not stop.is_set():
            bus.transcript(f"x{i}", partial=True)
            i += 1

    t = threading.Thread(target=_flood, daemon=True)
    t.start()
    try:
        for _ in range(50):
            bus.interrupt_start()
    finally:
        stop.set()
        t.join(timeout=1.0)
    # Drain and count delivered interrupts; every published system frame survives.
    seen = _types(sub, limit=100000)
    interrupts = [e for e in seen if e.get("type") == "interrupt"]
    assert len(interrupts) == 50


# ---------------------------------------------------------------------------
# consistent ordering across subscribers (= across transports)
# ---------------------------------------------------------------------------


def test_consistent_ordering_across_subscribers():
    bus = EventBus()
    subs = [bus.subscribe() for _ in range(4)]
    bus.state("recording")
    bus.transcript("hi", partial=False)
    bus.bot_stopped_speaking()  # SYSTEM end-of-turn
    bus.state("idle")
    orderings = []
    for sub in subs:
        orderings.append([e["type"] for e in _types(sub)])
    # Every subscriber/transport drains the SAME ordered sequence.
    assert all(o == orderings[0] for o in orderings)
    # end_of_turn (bot_stopped_speaking) precedes the data frames queued before it.
    assert orderings[0][0] == "bot_stopped_speaking"


# ---------------------------------------------------------------------------
# back-compat of the public API
# ---------------------------------------------------------------------------


def test_subscribe_replays_last_state():
    bus = EventBus()
    bus.state("speaking", "detail")
    sub = bus.subscribe()  # late subscriber gets the last state
    first = json.loads(sub.get_nowait())
    assert first["type"] == "state" and first["state"] == "speaking"


def test_get_blocks_then_times_out():
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.raises(queue.Empty):
        sub.get(timeout=0.05)


def test_empty_and_qsize_shims():
    bus = EventBus()
    sub = bus.subscribe()
    assert sub.empty()
    bus.transcript("x")
    assert not sub.empty()
    assert sub.qsize() >= 1


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    sub = bus.subscribe()
    bus.unsubscribe(sub)
    bus.interrupt_start()
    assert sub.empty()


def test_publish_dict_passthrough_for_metrics():
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish({"type": "metrics", "speech_id": "t1", "stages_ms": {}})
    evt = json.loads(sub.get_nowait())
    assert evt["type"] == "metrics" and evt["speech_id"] == "t1"


def test_file_sink_explicit_writes_jsonl(tmp_path):
    path = tmp_path / "ev.jsonl"
    bus = EventBus()
    bus.attach_file_sink(str(path))
    bus.state("listening")
    bus.transcript("hi", source="wake")
    bus.debug("wake", wake_score=0.42)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    objs = [json.loads(line) for line in lines]
    assert all("ts" in o for o in objs)  # every line carries a wall-clock timestamp
    assert objs[0]["type"] == "state" and objs[0]["state"] == "listening"
    assert objs[1]["source"] == "wake"
    assert objs[2]["type"] == "debug" and objs[2]["wake_score"] == 0.42


def test_file_sink_auto_attaches_from_env(tmp_path, monkeypatch):
    path = tmp_path / "auto.jsonl"
    monkeypatch.setenv("MSTT_EVENT_LOG", str(path))
    bus = EventBus()  # lazy attach happens on first publish
    bus.wake()
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["type"] == "wake"


def test_no_sink_is_a_noop(monkeypatch):
    monkeypatch.delenv("MSTT_EVENT_LOG", raising=False)
    monkeypatch.delenv("MSTT_EVENT_CONSOLE", raising=False)
    bus = EventBus()
    bus.log("nothing is written anywhere")  # must not raise without a sink
    assert bus._sink is None


# ---------------------------------------------------------------------------
# Sink A — bus -> stderr console (so quickstart.log captures every EVENT-LOG event)
# ---------------------------------------------------------------------------


def test_render_console_line_human_strings():
    assert render_console_line({"type": "state", "state": "listening"}) == "[event:state] listening"
    assert (
        render_console_line({"type": "state", "state": "tts", "detail": "piper"})
        == "[event:state] tts (piper)"
    )
    assert (
        render_console_line({"type": "music", "status": "playing", "title": "Song"})
        == '[event:music] playing "Song"'
    )
    assert (
        render_console_line({"type": "response", "text": "hello", "final": True})
        == "[event:response] [final] hello"
    )
    # Unknown type -> compact key=value dump (nothing silently lost).
    assert "k=v" in render_console_line({"type": "weird", "k": "v"})


def test_console_sink_on_when_event_log_set(monkeypatch, capsys):
    monkeypatch.setenv("MSTT_EVENT_LOG", "")  # blank path: no file sink, but...
    monkeypatch.setenv("MSTT_EVENT_CONSOLE", "1")  # ...console forced on
    bus = EventBus()
    bus.state("listening")
    bus.transcript("hi there", source="wake")
    err = capsys.readouterr().err
    assert "[event:state] listening" in err
    assert "[event:transcript]" in err and "hi there" in err


def test_console_sink_on_by_default_when_file_sink_active(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MSTT_EVENT_CONSOLE", raising=False)
    monkeypatch.setenv("MSTT_EVENT_LOG", str(tmp_path / "ev.jsonl"))
    bus = EventBus()
    bus.state("recording")  # file sink auto-attaches -> console default ON
    assert "[event:state] recording" in capsys.readouterr().err


def test_console_sink_off_by_default_without_file_sink(monkeypatch, capsys):
    monkeypatch.delenv("MSTT_EVENT_LOG", raising=False)
    monkeypatch.delenv("MSTT_EVENT_CONSOLE", raising=False)
    bus = EventBus()
    bus.state("idle")  # library/test default: silent
    assert capsys.readouterr().err == ""


def test_console_sink_explicit_off_overrides_file_sink(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MSTT_EVENT_LOG", str(tmp_path / "ev.jsonl"))
    monkeypatch.setenv("MSTT_EVENT_CONSOLE", "0")  # explicit off beats the file-sink default
    bus = EventBus()
    bus.state("idle")
    assert capsys.readouterr().err == ""


def test_console_sink_skips_debug_events(monkeypatch, capsys):
    monkeypatch.setenv("MSTT_EVENT_CONSOLE", "1")
    bus = EventBus()
    bus.debug("audio capture", sample_rate=16000)  # _AudioDebug already prints these
    assert capsys.readouterr().err == ""


def test_console_sink_skips_log_bridge_events(monkeypatch, capsys):
    monkeypatch.setenv("MSTT_EVENT_CONSOLE", "1")
    bus = EventBus()
    # A bridged log event (already on stderr via the logging library) must NOT re-print.
    bus.publish({"type": "log", "level": "warning", "message": "x", "_log_bridge": True})
    assert capsys.readouterr().err == ""
    # ...but a normal (non-bridged) log event DOES reach the console sink.
    bus.publish({"type": "log", "level": "info", "message": "shown"})
    assert "[event:log] info: shown" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Sink B — Python logging -> bus (so the EVENT LOG captures library/app logs)
# ---------------------------------------------------------------------------


def test_log_bus_handler_publishes_record():
    bus = EventBus()
    sub = bus.subscribe()
    handler = LogBusHandler(bus)
    record = logging.LogRecord(
        name="my_stt_tts.x",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    handler.emit(record)
    evt = json.loads(sub.get_nowait())
    assert evt["type"] == "log"
    assert evt["level"] == "warning"
    assert evt["message"] == "hello world"
    assert evt["_log_bridge"] is True


def test_log_bus_handler_no_infinite_recursion(monkeypatch):
    # Force the console sink ON: publishing inside emit() must NOT loop back into emit().
    monkeypatch.setenv("MSTT_EVENT_CONSOLE", "1")
    bus = EventBus()
    handler = LogBusHandler(bus)
    calls = {"n": 0}
    real_publish = bus.publish

    def _counting_publish(event):
        calls["n"] += 1
        return real_publish(event)

    monkeypatch.setattr(bus, "publish", _counting_publish)
    record = logging.LogRecord(
        name="root",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    handler.emit(record)
    assert calls["n"] == 1  # exactly one publish, no recursive re-entry


def test_log_bus_handler_suppresses_publish_failure():
    class _BoomBus(EventBus):
        def publish(self, event):  # noqa: ARG002
            raise RuntimeError("bus down")

    handler = LogBusHandler(_BoomBus())
    handler.handleError = lambda record: None  # swallow the fallback so the test is quiet
    record = logging.LogRecord(
        name="x",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="m",
        args=(),
        exc_info=None,
    )
    handler.emit(record)  # must not raise


@pytest.fixture
def _isolated_log_bridge():
    """Give each bridge test a clean root logger, then restore it.

    Strips ANY pre-existing :class:`LogBusHandler` up front — another test (e.g.
    ``test_audio_preflight`` calling the real ``main()``) may have leaked one onto
    the real ``bus`` — and resets the install-once global, so these tests are
    deterministic in the full suite. The original handler set + levels are restored
    on teardown so we don't perturb the rest of the run."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_app_level = logging.getLogger("my_stt_tts").level
    # Clean slate: drop any leaked bridge handler + reset the install-once flag.
    root.handlers[:] = [h for h in root.handlers if not isinstance(h, LogBusHandler)]
    events_mod._log_bridge_installed = False
    try:
        yield
    finally:
        for h in list(root.handlers):
            if isinstance(h, LogBusHandler):
                root.removeHandler(h)
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
        logging.getLogger("my_stt_tts").setLevel(saved_app_level)
        logging.captureWarnings(False)
        events_mod._log_bridge_installed = False


def test_install_log_bridge_routes_logs_and_warnings(_isolated_log_bridge):
    bus = EventBus()
    sub = bus.subscribe()
    handler = install_log_bridge(bus)
    assert handler is not None
    # (1) a normal app log record at WARNING+ reaches the bus...
    logging.getLogger("my_stt_tts.test").warning("a warning")
    # (2) ...and captureWarnings(True) was enabled, so warnings.warn() lands on the
    # 'py.warnings' logger. We emit ON that logger directly (what captureWarnings'
    # showwarning hook does) — deterministic regardless of pytest's per-test
    # warnings-plugin override of warnings.showwarning.
    assert logging.captureWarnings.__module__ == "logging"  # sanity: real captureWarnings
    logging.getLogger("py.warnings").warning("a userwarning")
    seen = []
    for _ in range(50):
        try:
            seen.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    msgs = [e["message"] for e in seen if e.get("type") == "log"]
    assert any("a warning" in m for m in msgs)
    assert any("a userwarning" in m for m in msgs)
    assert all(e.get("_log_bridge") for e in seen if e.get("type") == "log")


def test_install_log_bridge_enables_capture_warnings(_isolated_log_bridge):
    # install_log_bridge must turn warnings.warn(...) into log records (the onnxruntime
    # / Hugging Face warnings the EVENT LOG should show). Assert showwarning is rerouted
    # to logging — done OUTSIDE a pytest warnings-plugin context here so it sticks.
    import warnings as _warnings  # noqa: PLC0415

    before = _warnings.showwarning
    install_log_bridge(EventBus())
    assert _warnings.showwarning is not before  # captureWarnings(True) replaced it


def test_install_log_bridge_is_idempotent(_isolated_log_bridge):
    bus = EventBus()
    h1 = install_log_bridge(bus)
    h2 = install_log_bridge(bus)
    assert h1 is h2  # second call is a no-op, returns the same handler
    root = logging.getLogger()
    assert sum(isinstance(h, LogBusHandler) for h in root.handlers) == 1


def test_install_log_bridge_no_double_print_to_stderr(_isolated_log_bridge, monkeypatch, capsys):
    # End-to-end: a bridged log must appear in the EVENT LOG (bus) but NOT be re-printed
    # to stderr by sink A (it's _log_bridge-tagged). Console sink forced on.
    monkeypatch.setenv("MSTT_EVENT_CONSOLE", "1")
    bus = EventBus()
    sub = bus.subscribe()
    install_log_bridge(bus)
    logging.getLogger("my_stt_tts.test").warning("bridge once")
    # bus has it...
    evt = json.loads(sub.get_nowait())
    assert evt["type"] == "log" and "bridge once" in evt["message"]
    # ...and sink A did NOT echo it to stderr.
    assert "[event:log]" not in capsys.readouterr().err


def _bridged_messages(sub):
    out = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return [(e["level"], e["message"]) for e in out if e.get("type") == "log"]


def test_install_log_bridge_level_policy(_isolated_log_bridge):
    # WARNING+ from ANY logger AND INFO+ from my_stt_tts.* are bridged; other INFO is not.
    bus = EventBus()
    sub = bus.subscribe()
    install_log_bridge(bus)
    logging.getLogger("my_stt_tts.feat").info("app info")  # app INFO -> bridged
    logging.getLogger("thirdparty").info("lib info")  # third-party INFO -> NOT bridged
    logging.getLogger("thirdparty").warning("lib warn")  # any WARNING -> bridged
    msgs = _bridged_messages(sub)
    assert ("info", "app info") in msgs
    assert ("warning", "lib warn") in msgs
    assert all("lib info" not in m for _lvl, m in msgs)


def test_install_log_bridge_noisy_info_diverges(_isolated_log_bridge):
    # The single intentional divergence: httpx per-request INFO is kept OUT of the
    # EVENT LOG (too chatty) though it still reaches stderr; its WARNING+ still bridges.
    bus = EventBus()
    sub = bus.subscribe()
    install_log_bridge(bus)
    logging.getLogger("httpx").info("HTTP Request: POST /v1/messages 200 OK")
    logging.getLogger("httpx").warning("httpx problem")
    msgs = _bridged_messages(sub)
    assert all("HTTP Request" not in m for _lvl, m in msgs)  # diverges (not in EVENT LOG)
    assert ("warning", "httpx problem") in msgs
