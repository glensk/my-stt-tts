"""Register the bridge components on a :class:`~my_stt_tts.bridge.BridgeController`.

Called once from :func:`my_stt_tts.bridge.build_bridge` (``voice_control.daemon_main``).
Each component gates itself on its own flag and a failing one never stops the others:

* :func:`my_stt_tts.attention.register` (any bridge flag) — the inbox becomes the
  controller's problem sink, delivery tracker and briefing provider; its monitor speaks
  new problems in a running call through :meth:`BridgeController.notify_in_call`;
* :func:`my_stt_tts.claude_sessions.register` (``MAC_VOICE_CLAUDE_BRIDGE``);
* :func:`my_stt_tts.mac_control.register` (``MAC_VOICE_MAC_CONTROL``);
* :func:`my_stt_tts.mac_operator.register` (``MAC_VOICE_OPERATOR``) with
  :class:`OperatorNotices` — confirmation requests and LATE failures are spoken in the
  call; a result the ``do_on_mac`` tool already returned is never spoken twice, and
  successes stay silent (D6).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import BridgeController
    from .mac_operator import MacOperator, OperatorResult

log = logging.getLogger("my_stt_tts.bridge")

OPERATOR_TOOL = "do_on_mac"
#: Tool answers that leave the outcome open: the final result arrives later (on_result).
_OPEN_ANSWERS = ("started", "needs confirmation")


class OperatorNotices:
    """Speak the Mac operator's confirmation requests and late failures in the call."""

    def __init__(self, controller: BridgeController) -> None:
        self.controller = controller
        self.operator: MacOperator | None = None
        self._answered: set[str] = set()  # call ids whose final result the tool returned
        self._lock = threading.Lock()

    def attach(self, operator: MacOperator) -> None:
        """Wrap the registered ``do_on_mac`` so a final result it returns is remembered."""
        self.operator = operator
        inner = self.controller.tools.get(OPERATOR_TOOL)
        if inner is None:
            return

        def do_on_mac(request: str, background: bool = False) -> str:
            reply = inner(request, background)
            self._note_reply(str(reply))
            return reply

        do_on_mac.__doc__ = inner.__doc__
        self.controller.register_tool(OPERATOR_TOOL, do_on_mac)

    def _note_reply(self, reply: str) -> None:
        from .mac_operator import FINAL_STATES  # pylint: disable=import-outside-toplevel

        operator = self.operator
        current = operator.current if operator is not None else None
        if current is None or reply.startswith(_OPEN_ANSWERS):
            return
        if reply.split(":", 1)[0].strip() in FINAL_STATES:
            with self._lock:
                self._answered.add(current.call_id)

    def confirm_needed(self, message: str) -> None:
        """``notify`` of the operator: a blocked step needs ``confirm <code>``."""
        if self.controller.tool_in_flight(OPERATOR_TOOL):
            return  # the do_on_mac answer itself carries the request
        self.controller.notify_in_call(f"Mac task needs confirmation: {message}")

    def on_result(self, result: OperatorResult) -> None:
        """``on_result`` of the operator: failures spoken in the call, successes silent."""
        from .mac_operator import DONE  # pylint: disable=import-outside-toplevel

        with self._lock:
            answered = result.call_id in self._answered
            self._answered.discard(result.call_id)
        if result.status == DONE or answered:
            return
        if self.controller.tool_in_flight(OPERATOR_TOOL):
            return  # the pending do_on_mac answer returns this result
        what = result.reason or result.summary or result.status.replace("_", " ")
        self.controller.notify_in_call(f"Mac task {result.status.replace('_', ' ')}: {what}")


def _safely(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # pylint: disable=broad-exception-caught  # the others still register
        log.exception("❌ bridge component %s failed to register", name)
        return None


def wire(controller: BridgeController, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Register every enabled component; returns them by name (also ``controller.components``)."""
    # pylint: disable=import-outside-toplevel  # heavy / cyclic modules, loaded on demand
    from . import attention, claude_sessions, mac_control, mac_operator

    monitor = _safely("attention", lambda: attention.register(controller, env=env))
    if monitor is not None:
        monitor.set_notifier(controller.notify_in_call)
    sessions = _safely("claude_sessions", lambda: claude_sessions.register(controller, env=env))
    mac = _safely("mac_control", lambda: mac_control.register(controller, env=env))
    notices = OperatorNotices(controller)
    operator = _safely(
        "mac_operator",
        lambda: mac_operator.register(
            controller, env=env, notify=notices.confirm_needed, on_result=notices.on_result
        ),
    )
    if operator is not None:
        notices.attach(operator)
    parts = {
        "monitor": monitor,
        "sessions": sessions,
        "mac": mac,
        "operator": operator,
        "operator_notices": notices,
    }
    controller.components.update({k: v for k, v in parts.items() if v is not None})
    log.info("🌉 bridge tools: %s", ", ".join(sorted(controller.tools)))
    return parts
