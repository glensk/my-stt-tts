"""Attention inbox, monitor and briefing of the Claude/Mac bridge (Phase 6).

Fakes only: a manual wall clock, a fake ``ccc`` runner that serves JSON envelopes and
records ``osascript`` banners, and the bridge's fake VAD + speaker scorer for the voice
checks of ``acknowledge_problem``. Every store lives under ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import attention
from my_stt_tts.attention import (
    EXPIRY_S,
    GRACE_S,
    AttentionStore,
    EnvelopeError,
    Monitor,
    banner_argv,
    compose_briefing,
    first_message_override,
    parse_envelope,
    problem_id,
    register,
    report_startup_failure,
    startup_failure_reason,
)
from my_stt_tts.bridge import Authoriser, BridgeController, Problem

T0 = 1_800_000_000.0
SESSION = "sess-1"
FRAME = 1600
ALBERT, OTHER = 0.5, 0.3
SCORES = {ALBERT: 0.52, OTHER: 0.18}
RLO = chr(0x202E)  # right-to-left override, a format control character


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def envelope(data: Any = None, *, ok: bool = True, code: str = "", version: int = 1) -> str:
    error = None if ok else {"code": code or "failed", "message": "x"}
    return json.dumps({"schema_version": version, "ok": ok, "data": data, "error": error})


class FakeCcc:
    """``ccc events`` / ``ccc delivery`` envelopes; ``osascript`` calls are banners."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.deliveries: dict[str, str] = {}
        self.calls: list[list[str]] = []
        self.banners: list[list[str]] = []
        self.events_reply: tuple[int, str] | None = None
        self.ignore_after = False

    def add_event(self, kind: str, at: float, session_id: str = SESSION, **detail: Any) -> None:
        self.events.append(
            {
                "cursor": len(self.events) + 1,
                "kind": kind,
                "session_id": session_id,
                "at": iso(at),
                "detail": detail,
            }
        )

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str]:
        argv = list(argv)
        assert timeout > 0
        if argv[0] == "osascript":
            self.banners.append(argv)
            return 0, ""
        self.calls.append(argv)
        if argv[1] == "events":
            return self._events(argv)
        if argv[1] == "delivery":
            state = self.deliveries.get(argv[3])
            if state is None:
                return 1, envelope(ok=False, code="not_found")
            return 0, envelope({"delivery_id": argv[3], "state": state})
        return 2, envelope(ok=False, code="usage")

    @staticmethod
    def _cursor(event: Any, index: int) -> int:
        return int(event["cursor"]) if isinstance(event, dict) and "cursor" in event else index

    def _events(self, argv: list[str]) -> tuple[int, str]:
        if self.events_reply is not None:
            return self.events_reply
        after = int(argv[argv.index("--after") + 1]) if "--after" in argv else 0
        if self.ignore_after:
            after = 0
        batch = [e for i, e in enumerate(self.events, 1) if self._cursor(e, i) > after]
        next_cursor = len(self.events) if batch else after
        return 0, envelope({"events": batch, "next_cursor": next_cursor})


@pytest.fixture(autouse=True)
def _ccc_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The monitor resolves ccc via ``claude_sessions.resolve_ccc``: a stable ``ccc`` here."""
    monkeypatch.setattr(attention, "resolve_ccc", lambda: "ccc")


@pytest.fixture(name="clock")
def clock_fixture() -> Clock:
    return Clock()


@pytest.fixture(name="db")
def db_fixture(tmp_path: Path) -> Path:
    return tmp_path / "state" / "mac-voice" / "attention.db"


@pytest.fixture(name="store")
def store_fixture(db: Path, clock: Clock) -> AttentionStore:
    return AttentionStore(db, clock=clock)


def problem(subject: str = "scratch-dead", kind: str = "delivery_failed", **kw: Any) -> Problem:
    return Problem(kind, subject, kw.pop("reason", "message failed"), kw.pop("severity", "error"))


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# -- store -----------------------------------------------------------------------------------
def test_dedupe_bumps_last_seen(store: AttentionStore, clock: Clock) -> None:
    assert store.record(problem()) == "opened"
    clock.now += 30
    assert store.record(problem()) == "bumped"
    [item] = store.open_items()
    assert item.count == 2
    assert item.opened_at == T0
    assert item.last_seen == T0 + 30
    assert item.id == problem_id(item.dedupe_key) == problem_id("delivery failed:scratch dead")


def test_report_is_the_problem_sink_and_stores_no_message_text(store: AttentionStore) -> None:
    store.report(problem(reason="x" * 500))
    [item] = store.open_items()
    assert len(item.reason) <= attention.REASON_CAP


def test_replay_of_the_same_source_is_a_noop_even_after_ack(
    store: AttentionStore, clock: Clock
) -> None:
    assert store.record(problem(), source_cursor="event:1") == "opened"
    assert store.record(problem(), source_cursor="event:1") == "noop"
    assert store.acknowledge("scratch-dead") == 1
    assert store.record(problem(), source_cursor="event:1") == "noop"
    assert store.open_items() == []
    clock.now += 60
    assert store.record(problem(), source_cursor="event:7") == "reopened"
    [item] = store.open_items()
    assert item.acknowledged_at is None and item.briefed_at is None


def test_seven_day_expiry(store: AttentionStore, clock: Clock) -> None:
    store.record(problem())
    clock.now += EXPIRY_S - 1
    assert len(store.open_items()) == 1
    clock.now += 2
    assert store.open_items() == []
    assert store.expire() == 1
    assert store.items() == []


def test_file_modes_are_created_and_fixed(db: Path, clock: Clock) -> None:
    store = AttentionStore(db, clock=clock)
    store.record(problem())
    assert mode(db.parent) == 0o700
    for suffix in ("", "-wal", "-shm"):
        path = db.with_name(db.name + suffix)
        assert path.exists(), suffix
        assert mode(path) == 0o600, suffix
    os.chmod(db.parent, 0o755)
    os.chmod(db, 0o644)
    os.chmod(db.with_name(db.name + "-wal"), 0o644)
    os.chmod(db.with_name(db.name + "-shm"), 0o666)
    fixed = store.fix_modes()
    assert len(fixed) == 4
    assert mode(db.parent) == 0o700
    assert all(mode(db.with_name(db.name + s)) == 0o600 for s in ("", "-wal", "-shm"))
    store.close()
    os.chmod(db, 0o644)
    AttentionStore(db, clock=clock).close()  # verified + fixed on open too
    assert mode(db) == 0o600


def test_corrupt_db_is_quarantined_and_rebuilt(db: Path, clock: Clock) -> None:
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database at all" * 100)
    os.chmod(db, 0o600)
    store = AttentionStore(db, clock=clock)
    quarantined = list(db.parent.glob("attention.db.corrupt-*"))
    assert [p.name for p in quarantined if not p.name.endswith(("-wal", "-shm"))]
    [item] = store.open_items()
    assert item.kind == "attention_db_corrupt"
    assert store.record(problem()) == "opened"  # rebuilt and usable
    store.close()
    again = AttentionStore(db, clock=clock)
    assert len(again.open_items()) == 2  # no second corruption problem


# -- briefing ----------------------------------------------------------------------------------
def test_briefing_single_item(store: AttentionStore) -> None:
    store.record(problem())
    assert store.briefing() == (
        "One problem needs your attention. scratch-dead: message failed. "
        "Say got it and the name to clear one."
    )


def test_briefing_cap_sanitising_and_ordering(store: AttentionStore, clock: Clock) -> None:
    store.record(problem("warn-one", kind="needs_input", reason="needs input", severity="warning"))
    for i in range(30):
        clock.now += 1
        store.record(
            problem(
                f"session\x1b[31m-{i}\n" + "y" * 40,
                reason="token ghp_abcdefghijklmnopqrstuvwx\x07 leaked" + RLO,
            )
        )
    text = store.briefing()
    assert text is not None
    assert len(text) <= 600
    assert "And " in text and " more." in text
    assert "ghp_" not in text and "[redacted]" in text
    assert not any(ord(c) < 32 or 0x7F <= ord(c) < 0xA0 or c == RLO for c in text)
    assert "warn-one" not in text  # errors first, the warning is among the "more"
    assert text.startswith("31 problems need your attention. session")
    # the newest error comes first among equals
    assert text.index("-29") < text.index("-28")


def test_compose_briefing_counts_the_rest(store: AttentionStore) -> None:
    for i in range(40):
        store.record(problem(f"name{i:02d}" + "z" * 50))
    items = store.open_items()
    text, used = compose_briefing(items)
    assert text is not None and len(text) <= 600
    assert f"And {len(items) - len(used)} more." in text
    assert compose_briefing([]) == (None, [])


def test_briefed_only_after_the_conversation_started(store: AttentionStore) -> None:
    store.record(problem())
    override = first_message_override(store)
    assert override is not None
    assert override["agent"]["first_message"].startswith("One problem")
    assert store.open_items()[0].briefed_at is None  # not yet: the call may still fail
    assert store.mark_briefed() == 1
    item = store.open_items()[0]
    assert item.briefed_at is not None
    # briefed items stay until acknowledged, so the next "voice on" briefs them again
    assert first_message_override(store) is not None


def test_startup_failure_keeps_items_unbriefed(store: AttentionStore) -> None:
    store.record(problem())
    assert first_message_override(store) is not None
    report_startup_failure(store, startup_failure_reason("3000 [quota_exceeded] closed"))
    items = store.open_items()
    assert {i.kind for i in items} == {"delivery_failed", "voice_startup_failed"}
    assert all(i.briefed_at is None for i in items)
    startup = store.get("startup:voice")
    assert startup is not None and startup.reason == "ElevenLabs quota exceeded"
    text = store.briefing()
    assert text is not None and "scratch-dead" in text and "quota" in text


def test_startup_failure_banner_is_opt_in(store: AttentionStore) -> None:
    fake = FakeCcc()
    report_startup_failure(store, "x", banner=True, runner=fake)
    assert fake.banners == [banner_argv("voice startup failed")]


def test_no_override_without_problems(store: AttentionStore) -> None:
    assert first_message_override(store) is None


def test_startup_failure_reasons() -> None:
    assert startup_failure_reason(RuntimeError("received 3000 [quota_exceeded]")) == (
        "ElevenLabs quota exceeded"
    )
    assert startup_failure_reason("1008 Override for field not allowed") == (
        "agent refused the start settings"
    )
    assert startup_failure_reason("boom") == "voice agent did not start"


# -- monitor -----------------------------------------------------------------------------------
def monitor_for(store: AttentionStore, fake: FakeCcc, **kw: Any) -> Monitor:
    return Monitor(store, fake, **kw)


def test_correlated_stop_failure_alerts_and_cursor_persists(
    db: Path, store: AttentionStore, clock: Clock
) -> None:
    fake = FakeCcc()
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    fake.add_event("stop_failure", T0 + 10)  # already there at start: catch-up
    clock.now += 20
    mon = monitor_for(store, fake)
    mon.poll_once()
    assert fake.calls[0] == ["ccc", "events", "-j"]
    [item] = store.open_items()
    assert (item.kind, item.subject, item.reason) == (
        "stop_failure",
        "scratch-busy",
        "stopped with an error",
    )
    assert store.get_meta(attention.CURSOR_KEY) == "1"
    assert len(fake.banners) == 1
    mon.poll_once()
    assert ["ccc", "events", "--after", "1", "-j"] in fake.calls
    store.close()

    # daemon restart with a ccc that replays everything: idempotent
    fake.ignore_after = True
    again = AttentionStore(db, clock=clock)
    mon2 = monitor_for(again, fake)
    mon2.poll_once()
    assert len(again.items()) == 1
    assert again.items()[0].count == 1
    assert len(fake.banners) == 1


def test_uncorrelated_events_and_long_running_work_are_silent(
    store: AttentionStore, clock: Clock
) -> None:
    fake = FakeCcc()
    fake.add_event("stop_failure", T0, session_id="not-voice")
    fake.add_event("needs_input", T0, session_id="not-voice")
    fake.add_event("something_new", T0)
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    fake.deliveries["d1"] = "accepted"
    mon = monitor_for(store, fake)
    for _ in range(10):  # hours of busy work
        clock.now += 3600
        mon.poll_once()
    assert store.items() == []
    # ccc gives up after 6 h without a turn end: a delivery_failed event, still no alert
    fake.deliveries["d1"] = "timed_out"
    fake.add_event("delivery_failed", clock.now, delivery_id="d1", state="timed_out",
                   reason="no_turn_end")  # fmt: skip
    mon.poll_once()
    assert store.items() == []
    assert store.open_deliveries() == []
    assert not fake.banners


def test_needs_input_from_event_and_delivery_dedupes(store: AttentionStore, clock: Clock) -> None:
    fake = FakeCcc()
    store.track_delivery("d1", SESSION, "scratch-idle", "accepted")
    clock.now += 5
    fake.add_event("needs_input", clock.now)
    fake.deliveries["d1"] = "needs_input"
    monitor_for(store, fake).poll_once()
    [item] = store.open_items()
    assert item.kind == "needs_input" and item.subject == "scratch-idle"
    assert item.severity == "warning"
    assert store.open_deliveries() == []


def test_failed_delivery_alerts(store: AttentionStore) -> None:
    fake = FakeCcc()
    store.track_delivery("d9", "dead", "scratch-dead", "accepted")
    fake.deliveries["d9"] = "failed"
    monitor_for(store, fake).poll_once()
    [item] = store.open_items()
    assert (item.kind, item.subject, item.reason) == (
        "delivery_failed",
        "scratch-dead",
        "message failed",
    )


def test_unknown_delivery_grace_then_auto_resolve(store: AttentionStore, clock: Clock) -> None:
    fake = FakeCcc()
    store.track_delivery("d2", SESSION, "scratch-idle", "unknown")
    fake.deliveries["d2"] = "unknown"
    mon = monitor_for(store, fake)
    clock.now += 60
    mon.poll_once()
    assert store.open_items() == []
    clock.now += GRACE_S
    mon.poll_once()
    [item] = store.open_items()
    assert item.kind == "delivery_unknown" and item.reason == "message not confirmed"
    clock.now += 15
    mon.poll_once()  # still unknown: no duplicate
    assert len(store.items()) == 1
    fake.deliveries["d2"] = "completed"  # later evidence
    mon.poll_once()
    assert store.open_items() == []
    assert store.get("unconfirmed:d2").resolved_at is not None  # type: ignore[union-attr]
    assert store.open_deliveries() == []


def test_unknown_delivery_confirmed_inside_grace_never_alerts(
    store: AttentionStore, clock: Clock
) -> None:
    fake = FakeCcc()
    store.track_delivery("d3", SESSION, "scratch-idle", "unknown")
    mon = monitor_for(store, fake)  # ccc does not know it yet → not_found = no evidence
    clock.now += 100
    mon.poll_once()
    fake.deliveries["d3"] = "accepted"
    clock.now += 100
    mon.poll_once()
    clock.now += GRACE_S * 3
    mon.poll_once()
    assert store.items() == []
    assert [d.state for d in store.open_deliveries()] == ["accepted"]


def test_the_store_is_the_delivery_tracker(store: AttentionStore) -> None:
    store.track_delivery("d5", SESSION, "scratch-idle", "unknown")
    [delivery] = store.open_deliveries()
    assert (delivery.delivery_id, delivery.state) == ("d5", "unknown")
    assert delivery.unknown_since is not None


@pytest.mark.parametrize(
    ("reply", "code"),
    [
        ((0, "not json"), "bad_json"),
        ((0, envelope({"events": []}, version=2)), "bad_envelope"),
        ((1, envelope(ok=False, code="internal")), "internal"),
        ((3, envelope({"events": []})), "exit_mismatch"),
        ((0, envelope(None)), "bad_envelope"),
    ],
)
def test_parse_envelope_errors(reply: tuple[int, str], code: str) -> None:
    with pytest.raises(EnvelopeError) as err:
        parse_envelope(*reply)
    assert err.value.code == code


def test_envelope_errors_keep_the_cursor_and_raise_one_problem(
    store: AttentionStore, clock: Clock
) -> None:
    fake = FakeCcc()
    store.set_meta(attention.CURSOR_KEY, "4")
    fake.events_reply = (0, envelope({"events": "nope", "next_cursor": 9}))
    mon = monitor_for(store, fake)
    for _ in range(5):
        clock.now += 15
        mon.poll_once()
    assert store.get_meta(attention.CURSOR_KEY) == "4"
    [item] = store.open_items()
    assert item.kind == "monitor_failed" and item.count == 3
    fake.events_reply = None
    mon.poll_once()
    assert store.open_items() == []


def test_malformed_events_are_skipped(store: AttentionStore) -> None:
    fake = FakeCcc()
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    fake.events = [
        "junk",
        {"cursor": 2, "kind": "stop_failure"},
        {"cursor": 3, "kind": "stop_failure", "session_id": SESSION, "at": "garbage"},
    ]
    monitor_for(store, fake).poll_once()
    [item] = store.open_items()
    assert item.kind == "stop_failure"
    assert store.get_meta(attention.CURSOR_KEY) == "3"


def test_old_events_are_ignored_on_catch_up(store: AttentionStore) -> None:
    fake = FakeCcc()
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    fake.add_event("stop_failure", T0 - EXPIRY_S - 10)
    monitor_for(store, fake).poll_once()
    assert store.items() == []


# -- in-call delivery and banners --------------------------------------------------------------
def test_in_call_problems_are_spoken_not_bannered(store: AttentionStore) -> None:
    fake = FakeCcc()
    spoken: list[str] = []
    store.track_delivery("d9", "dead", "scratch-dead", "accepted")
    fake.deliveries["d9"] = "failed"
    mon = monitor_for(store, fake, call_active=lambda: True, notify_in_call=spoken.append)
    mon.poll_once()
    assert spoken == ["scratch-dead: message failed"]
    assert not fake.banners
    assert store.open_items()[0].briefed_at is not None  # spoken, still unacknowledged


def test_without_a_call_problems_wait_in_the_inbox(store: AttentionStore) -> None:
    fake = FakeCcc()
    spoken: list[str] = []
    store.track_delivery("d9", "dead", "scratch-dead", "accepted")
    fake.deliveries["d9"] = "failed"
    mon = monitor_for(store, fake, call_active=lambda: False, notify_in_call=spoken.append)
    mon.poll_once()
    assert not spoken
    assert store.open_items()[0].briefed_at is None
    assert fake.banners == [banner_argv("delivery_failed")]


def test_failed_notice_stays_unbriefed(store: AttentionStore) -> None:
    def broken(text: str) -> None:
        raise RuntimeError(text)

    mon = Monitor(store, FakeCcc(), call_active=lambda: True, notify_in_call=broken)
    assert mon.notify_in_call is broken
    store.record(problem(), announce=True)
    assert store.open_items()[0].briefed_at is None


def test_direct_reports_in_a_call_are_announced_only_on_request(store: AttentionStore) -> None:
    spoken: list[str] = []
    Monitor(store, FakeCcc(), call_active=lambda: True, notify_in_call=spoken.append)
    store.report(problem("open_url"))  # the failing tool already told the agent
    store.report(problem("operator", reason="operator request failed"), announce=True)
    assert spoken == ["operator: operator request failed"]


def test_banner_is_content_free(store: AttentionStore) -> None:
    fake = FakeCcc()
    mon = Monitor(store, fake, call_active=lambda: False)
    store.record(problem("secret-project", kind='evil" & do shell script "x'))
    store.record(problem("scratch-dead"))
    assert fake.banners[0] == [
        "osascript",
        "-e",
        'display notification "evil do shell script x needs attention" with title "mac-voice"',
    ]
    assert banner_argv("delivery_failed") == [
        "osascript",
        "-e",
        'display notification "delivery failed needs attention" with title "mac-voice"',
    ]
    assert not any(
        "secret-project" in " ".join(b) or "scratch" in " ".join(b) for b in fake.banners
    )
    mon.banner = False
    store.record(problem("other"))
    assert len(fake.banners) == 2


def test_banners_are_coalesced_per_tick(store: AttentionStore) -> None:
    fake = FakeCcc()
    for i in range(3):
        store.track_delivery(f"d{i}", f"s{i}", f"name{i}", "accepted")
        fake.deliveries[f"d{i}"] = "failed"
    monitor_for(store, fake).poll_once()
    assert len(store.open_items()) == 3
    assert fake.banners == [banner_argv("delivery_failed")]


def test_monitor_thread_starts_and_stops(store: AttentionStore) -> None:
    fake = FakeCcc()
    mon = Monitor(store, fake, interval=0.01)
    mon.start()
    for _ in range(200):
        if fake.calls:
            break
        mon._stop.wait(0.01)  # pylint: disable=protected-access
    mon.stop()
    assert fake.calls
    assert mon._thread is None  # pylint: disable=protected-access


# -- acknowledge_problem + register -------------------------------------------------------------
class FakeVad:
    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


class FakeScorer:
    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del timeout
        if name != "albert":
            return None
        return SCORES[round(float(np.max(np.abs(audio))), 2)]


class Call:
    def __init__(self) -> None:
        self.now = 1000.0
        self.ctl = BridgeController(
            Authoriser(FakeScorer(), "albert"), vad_factory=FakeVad, clock=lambda: self.now
        )
        self.ctl.begin_call()
        self.seq = 0

    def say(self, text: str, level: float = ALBERT) -> None:
        for amp in [level] * 10 + [0.0] * 6:
            self.now += 0.1
            self.ctl.feed_audio(np.full(FRAME, amp, dtype=np.float32))
        self.now += 1.5
        self.seq += 1
        self.ctl.on_transcript(self.seq, text, self.now)


ENV_ON = {"MAC_VOICE_CLAUDE_BRIDGE": "1"}


@pytest.fixture(name="wired")
def wired_fixture(store: AttentionStore) -> tuple[Call, Monitor]:
    call = Call()
    mon = register(call.ctl, store, FakeCcc(), env=ENV_ON, start=False)
    assert mon is not None
    store.record(problem("scratch-dead"))
    store.record(problem("voice bridge", kind="needs_input", reason="needs your input"))
    return call, mon


def test_register_wires_store_tool_and_shutdown(wired: tuple[Call, Monitor]) -> None:
    call, mon = wired
    assert call.ctl.problems is mon.store
    assert call.ctl.deliveries is mon.store
    assert call.ctl.briefings is mon.store
    assert "acknowledge_problem" in call.ctl.tools
    mon.start()
    call.ctl.shutdown()
    assert mon._thread is None  # pylint: disable=protected-access


def test_register_needs_a_bridge_flag(store: AttentionStore) -> None:
    call = Call()
    assert register(call.ctl, store, FakeCcc(), env={}) is None
    assert "acknowledge_problem" not in call.ctl.tools


def test_register_default_store_path_under_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    call = Call()
    mon = register(call.ctl, None, FakeCcc(), env=ENV_ON, start=False)
    assert mon is not None
    assert mon.store.path == tmp_path / ".local/state/mac-voice/attention.db"
    assert mode(mon.store.path.parent) == 0o700
    mon.store.close()


def test_monitor_follows_the_call_state(wired: tuple[Call, Monitor]) -> None:
    call, mon = wired
    assert mon.call_active() is True
    call.ctl.end_call()
    assert mon.call_active() is False


def test_acknowledge_is_per_item_and_voice_verified(wired: tuple[Call, Monitor]) -> None:
    call, mon = wired
    ack = call.ctl.tools["acknowledge_problem"]
    assert ack("scratch-dead") == "refused: nothing said yet"
    call.say("got it, scratch dead", level=OTHER)
    assert ack("scratch-dead").startswith("refused: voice not verified")
    call.say("got it, scratch dead")
    assert ack("voice bridge").startswith("refused:")  # not what was said
    call.say("got it, scratch dead")
    assert ack("scratch-dead") == "ok: cleared 1"
    assert [i.subject for i in mon.store.open_items()] == ["voice bridge"]
    assert ack("scratch-dead").startswith("refused:")  # capability is single use
    call.say("got it, voice bridge")
    assert ack("voice bridge") == "ok: cleared 1"
    assert mon.store.open_items() == []
    call.say("got it, nothing here")
    assert ack("nothing here") == "nothing open with that name"


# -- replay, corruption, shutdown and delivery polling --------------------------------------------
def test_replay_after_a_crash_never_reopens_an_acknowledged_item(
    store: AttentionStore, clock: Clock
) -> None:
    fake = FakeCcc()
    spoken: list[str] = []
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    fake.add_event("stop_failure", T0 + 5)
    fake.add_event("stop_failure", T0 + 10)
    clock.now += 20
    mon = monitor_for(store, fake, call_active=lambda: True, notify_in_call=spoken.append)
    mon.poll_once()
    [item] = store.open_items()
    assert item.event_cursor == 2 and spoken == ["scratch-busy: stopped with an error"]
    clock.now += 60
    assert store.acknowledge("scratch-busy") == 1
    store.set_meta(attention.CURSOR_KEY, "0")  # the cursor write was lost (crash)
    mon.poll_once()
    [after] = store.items()
    assert after.acknowledged_at is not None and after.count == item.count
    assert store.open_items() == [] and len(spoken) == 1


def test_an_event_from_before_the_acknowledgement_does_not_reopen(
    store: AttentionStore, clock: Clock
) -> None:
    store.record(problem(), source_cursor="delivery:d1:delivery_failed")
    clock.now += 100
    store.acknowledge("scratch-dead")
    assert store.record(problem(), source_cursor="event:9", at=T0 + 50) == "noop"
    assert store.record(problem(), source_cursor="event:10", at=clock.now + 1) == "reopened"


def test_older_event_cursor_is_a_replay_even_while_open(store: AttentionStore) -> None:
    assert store.record(problem(), source_cursor="event:5") == "opened"
    assert store.record(problem(), source_cursor="event:3") == "noop"
    assert store.record(problem(), source_cursor="event:6") == "bumped"
    assert store.open_items()[0].event_cursor == 6


def test_an_inbox_without_the_event_cursor_column_is_migrated(db: Path, clock: Clock) -> None:
    db.parent.mkdir(parents=True, mode=0o700)
    conn = sqlite3.connect(db)
    conn.executescript(
        attention._SCHEMA.replace(",\n    event_cursor INTEGER", "")  # pylint: disable=protected-access
    )
    conn.close()
    os.chmod(db, 0o600)
    store = AttentionStore(db, clock=clock)
    assert store.record(problem(), source_cursor="event:4") == "opened"
    assert store.open_items()[0].event_cursor == 4
    store.close()


@pytest.mark.parametrize("message", ["database is locked", "unable to open database file"])
def test_a_locked_inbox_is_never_quarantined(
    db: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    AttentionStore(db, clock=clock).close()

    def locked(_self: AttentionStore) -> sqlite3.Connection:
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(AttentionStore, "_connect", locked)
    with pytest.raises(sqlite3.OperationalError):
        AttentionStore(db, clock=clock)
    assert db.exists()
    assert not list(db.parent.glob("attention.db.corrupt-*"))


def test_a_long_running_timeout_is_silent_an_unconfirmed_one_alerts(
    store: AttentionStore, clock: Clock
) -> None:
    fake = FakeCcc()
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    store.track_delivery("d2", "sess-2", "scratch-lost", "unknown")
    fake.deliveries.update({"d1": "accepted", "d2": "unknown"})
    clock.now += 10
    fake.add_event("delivery_failed", clock.now, delivery_id="d1", state="timed_out",
                   reason="no_turn_end")  # fmt: skip
    fake.add_event("delivery_failed", clock.now, session_id="sess-2", delivery_id="d2",
                   state="timed_out", reason="unconfirmed")  # fmt: skip
    monitor_for(store, fake).poll_once()
    [item] = store.open_items()
    assert (item.kind, item.subject) == ("delivery_failed", "scratch-lost")
    assert store.open_deliveries() == []  # both closed by the events


def test_monitor_failure_keeps_the_last_delivery_state(store: AttentionStore, clock: Clock) -> None:
    fake = FakeCcc()
    store.track_delivery("d2", SESSION, "scratch-idle", "unknown")
    fake.deliveries["d2"] = "accepted"
    mon = monitor_for(store, fake)
    mon.poll_once()
    assert [d.state for d in store.open_deliveries()] == ["accepted"]
    fake.deliveries.clear()  # ccc unreachable for this delivery from now on
    clock.now += GRACE_S * 2
    mon.poll_once()
    assert store.items() == []
    assert [d.state for d in store.open_deliveries()] == ["accepted"]


def test_only_unknown_deliveries_are_polled_every_tick(store: AttentionStore, clock: Clock) -> None:
    fake = FakeCcc()
    store.track_delivery("d1", SESSION, "scratch-busy", "accepted")
    store.track_delivery("d2", "sess-2", "scratch-idle", "unknown")
    fake.deliveries.update({"d1": "accepted", "d2": "unknown"})
    mon = monitor_for(store, fake)
    for _ in range(4):
        clock.now += attention.POLL_INTERVAL_S
        mon.poll_once()
    polled = [c[3] for c in fake.calls if c[1] == "delivery"]
    assert polled.count("d2") == 4 and polled.count("d1") == 1
    clock.now += attention.ACCEPTED_POLL_S
    fake.deliveries["d1"] = "completed"
    mon.poll_once()
    assert [d.delivery_id for d in store.open_deliveries()] == ["d2"]


def test_a_rebuilt_events_table_resets_the_cursor(store: AttentionStore) -> None:
    fake = FakeCcc()
    store.record(problem("scratch-busy", kind="stop_failure"), source_cursor="event:50")
    store.set_meta(attention.CURSOR_KEY, "50")
    fake.ignore_after = True  # the new table only has cursors 1..2
    fake.add_event("needs_input", T0)
    fake.add_event("needs_input", T0)
    monitor_for(store, fake).poll_once()
    assert store.get_meta(attention.CURSOR_KEY) == "2"
    assert store.items()[0].event_cursor is None  # old numbering forgotten


def test_the_monitor_finds_ccc_via_resolve_ccc(
    store: AttentionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeCcc()
    monkeypatch.setattr(attention, "resolve_ccc", lambda: "/opt/bin/ccc")
    mon = Monitor(store, fake)
    mon.poll_once()
    assert fake.calls[0] == ["/opt/bin/ccc", "events", "-j"]
    monkeypatch.setattr(attention, "resolve_ccc", lambda: None)
    for _ in range(attention.FAILURES_BEFORE_PROBLEM):
        mon.poll_once()
    assert [i.kind for i in store.open_items()] == ["monitor_failed"]


class BlockingCcc(FakeCcc):
    """``ccc events`` that blocks until released (a slow ccc during shutdown)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, argv: Sequence[str], timeout: float) -> tuple[int, str]:
        if argv[1] == "events":
            self.entered.set()
            self.release.wait(5.0)
        return super().__call__(argv, timeout)


def test_stop_joins_the_thread_and_checks_between_pages(store: AttentionStore) -> None:
    fake = BlockingCcc()
    mon = Monitor(store, fake, interval=0.01)
    mon.start()
    assert fake.entered.wait(2.0)
    threading.Timer(0.2, fake.release.set).start()
    started = time.monotonic()
    assert mon.stop() is True
    assert time.monotonic() - started >= 0.1  # waited for the running tick
    assert len([c for c in fake.calls if c[1] == "events"]) == 1  # no page after the stop


def test_shutdown_leaves_the_store_open_while_the_thread_runs(
    store: AttentionStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = Call()
    fake = BlockingCcc()
    mon = register(call.ctl, store, fake, env=ENV_ON)
    assert mon is not None
    assert fake.entered.wait(2.0)
    monkeypatch.setattr(attention, "JOIN_TIMEOUT_S", 0.05)
    call.ctl.shutdown()
    assert store.items() == []  # still open: the monitor thread is using it
    fake.release.set()
    assert mon.stop(timeout=5.0) is True
    store.close()


def test_run_command_kills_the_whole_process_group() -> None:
    started = time.monotonic()
    assert attention.run_command(["sh", "-c", "sleep 30 & sleep 30"], 0.3) == (124, "")
    assert time.monotonic() - started < 5.0
    assert attention.run_command(["/nonexistent/ccc"], 1.0) == (127, "")
