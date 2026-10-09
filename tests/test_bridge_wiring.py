"""Bridge wiring (Phase 7): components registered, operator notices spoken only when due."""

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


def _result(status: str, reason: str = "") -> OperatorResult:
    return OperatorResult("op-1", status, "", 3.0, reason)


def test_late_failure_is_spoken_success_is_silent(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "started: working on it; the result follows")
    assert ctl.tools["do_on_mac"]("close the tabs").startswith("started")
    notices.on_result(_result("done"))
    assert not sent
    notices.on_result(_result("failed", "timeout"))
    assert sent == ["[system notice] Mac task failed: timeout"]


def test_result_already_returned_by_the_tool_is_not_repeated(ctl: BridgeController) -> None:
    notices, sent = _notices(ctl, "failed: could not find the window")
    ctl.tools["do_on_mac"]("do it")
    notices.on_result(_result("failed", "no window"))
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
    notices.on_result(_result("failed", "x"))
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
