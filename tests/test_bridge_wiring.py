"""Bridge wiring (Phase 7): components registered, operator notices spoken when due."""

from __future__ import annotations

from typing import Any

import pytest

from my_stt_tts import bridge_wiring
from my_stt_tts.bridge import Authoriser, BridgeController
from my_stt_tts.mac_operator import OperatorResult


class _Vad:
    def is_speech(self, _frame: Any) -> bool:
        return False


class _Call:
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id


class _Operator:
    def __init__(self) -> None:
        self.current: _Call | None = None


@pytest.fixture(name="ctl")
def _ctl() -> BridgeController:
    ctl = BridgeController(Authoriser(None, None), vad_factory=_Vad)
    ctl.begin_call()
    return ctl


def _notices(ctl: BridgeController, reply: str) -> tuple[bridge_wiring.OperatorNotices, list[str]]:
    sent: list[str] = []
    ctl.set_notice_sink(sent.append)
    operator = _Operator()

    def do_on_mac(request: str, background: bool = False) -> str:
        del request, background
        operator.current = _Call("op-1")
        return reply

    ctl.register_tool("do_on_mac", do_on_mac)
    notices = bridge_wiring.OperatorNotices(ctl)
    notices.attach(operator)  # type: ignore[arg-type]
    return notices, sent


def _result(status: str, reason: str = "", summary: str = "") -> OperatorResult:
    return OperatorResult("op-1", status, summary, 3.0, reason)


def test_late_failure_is_spoken(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "started: working on it; the result follows")
    assert ctl.tools["do_on_mac"]("close the tabs").startswith("started")
    assert notices.on_result(_result("failed", "timeout")) is True  # spoken in the call
    assert sent == ["[system notice] Mac task failed: timeout"]


def test_late_success_is_spoken_with_its_summary(ctl: BridgeController) -> None:
    """Phase 8 run 5: the Trash move finished and Albert heard nothing."""
    notices, sent = _notices(ctl, "started: working on it; the result follows")
    ctl.tools["do_on_mac"]("delete the file delete-me")
    summary = "moved delete-me.txt to the Trash"
    assert notices.on_result(_result("done", summary=summary)) is True
    assert sent == ["[system notice] done: moved delete-me.txt to the Trash"]


def test_a_success_notice_is_capped_and_redacted(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "started: working on it")
    ctl.tools["do_on_mac"]("x")
    secret = "ghp_" + "A" * 30
    notices.on_result(_result("done", summary=f"used {secret} " + "word " * 200))
    assert len(sent) == 1
    notice = sent[0]
    assert secret not in notice and "[redacted]" in notice
    assert len(notice) <= len("[system notice] ") + 300


def test_a_success_without_summary_still_says_done(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "started: working on it")
    ctl.tools["do_on_mac"]("x")
    assert notices.on_result(_result("done")) is True
    assert sent == ["[system notice] done: the Mac task finished"]


def test_a_success_the_tool_already_returned_is_not_repeated(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "done: Calculator shows 4.")
    ctl.tools["do_on_mac"]("compute 2+2")
    assert notices.on_result(_result("done", summary="Calculator shows 4.")) is True
    assert not sent


def test_the_one_repeat_of_a_confirmation_is_spoken_as_is(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "needs confirmation: x")
    notices.confirm_needed("still waiting: move delete-me.txt to the Trash — say yes, do it")
    assert sent == [
        "[system notice] Mac task still waiting: move delete-me.txt to the Trash — say yes, do it"
    ]


def test_result_already_returned_by_the_tool_is_not_repeated(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "failed: could not find the window")
    ctl.tools["do_on_mac"]("do it")
    assert notices.on_result(_result("failed", "no window")) is True  # the tool said it
    assert not sent


def test_confirmation_request_waits_while_the_tool_answers(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "started: working on it; the result follows")
    ctl.tool_started("do_on_mac")
    notices.confirm_needed("close 3 tabs — say confirm 42")
    notices.on_result(_result("failed", "x"))
    assert not sent  # the pending do_on_mac answer carries both
    ctl.tool_finished("do_on_mac")
    notices.confirm_needed("close 3 tabs — say confirm 42")
    assert sent == ["[system notice] Mac task needs confirmation: close 3 tabs — say confirm 42"]


def test_outside_a_call_nothing_is_sent(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "started: working on it")
    ctl.tools["do_on_mac"]("x")
    ctl.end_call()
    assert notices.on_result(_result("failed", "x")) is False  # briefed at the next voice-on
    assert notices.on_result(_result("done", summary="ok")) is False
    assert not sent


def test_wire_registers_every_component_and_the_notifier(
    ctl: BridgeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class Monitor:
        notifier: Any = None

        def set_notifier(self, fn: Any) -> None:
            Monitor.notifier = fn

    def fake(name: str, result: Any = None) -> Any:
        def register(controller: Any, **kwargs: Any) -> Any:
            assert controller is ctl
            calls.append(name)
            if name == "operator":
                assert callable(kwargs["notify"]) and callable(kwargs["on_result"])
            return result

        return register

    monkeypatch.setattr("my_stt_tts.attention.register", fake("attention", Monitor()))
    monkeypatch.setattr("my_stt_tts.claude_sessions.register", fake("sessions"))
    monkeypatch.setattr("my_stt_tts.mac_control.register", fake("mac"))
    monkeypatch.setattr("my_stt_tts.mac_operator.register", fake("operator"))
    parts = bridge_wiring.wire(ctl, {"MAC_VOICE_CLAUDE_BRIDGE": "1"})
    assert calls == ["attention", "sessions", "mac", "operator"]
    assert Monitor.notifier == ctl.notify_in_call
    assert parts["operator"] is None and "monitor" in ctl.components


def test_a_failing_component_does_not_stop_the_others(
    ctl: BridgeController, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("locked")

    seen: list[str] = []
    monkeypatch.setattr("my_stt_tts.attention.register", boom)
    monkeypatch.setattr(
        "my_stt_tts.claude_sessions.register", lambda c, **k: seen.append("sessions")
    )
    monkeypatch.setattr("my_stt_tts.mac_control.register", lambda c, **k: seen.append("mac"))
    monkeypatch.setattr("my_stt_tts.mac_operator.register", lambda c, **k: None)
    bridge_wiring.wire(ctl, {})
    assert seen == ["sessions", "mac"]
