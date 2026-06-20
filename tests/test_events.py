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
import queue
import threading

import pytest

from my_stt_tts.events import EventBus, Frame, Priority, classify


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
    bus = EventBus()
    bus.log("nothing is written anywhere")  # must not raise without a sink
    assert bus._sink is None
