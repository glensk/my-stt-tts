"""Register the bridge components on a :class:`~my_stt_tts.bridge.BridgeController`.

:func:`build_bridge` (called once from ``voice_control.daemon_main``) builds the controller
and calls :func:`wire`. It lives here, not in :mod:`my_stt_tts.bridge`, because this module is
the composition root: it imports every component, and the components import the bridge.
Each component gates itself on its own flag and a failing one never stops the others:

* :func:`my_stt_tts.attention.register` (any bridge flag) — the inbox becomes the
  controller's problem sink, delivery tracker and briefing provider; its monitor speaks
  new problems in a running call through :meth:`BridgeController.notify_in_call`;
* :func:`my_stt_tts.claude_sessions.register` (``MAC_VOICE_CLAUDE_BRIDGE``);
* :func:`my_stt_tts.mac_control.register` (``MAC_VOICE_MAC_CONTROL``);
* :func:`my_stt_tts.decide_tools.register` (``MAC_VOICE_DECIDE``) — the decide hub's
  plate, briefs and read-back answers (executor ``decide_answer`` behind "yes, do it");
* :func:`my_stt_tts.mac_operator.register` (``MAC_VOICE_OPERATOR``) with
  :class:`OperatorNotices` — confirmation requests (and their one "still waiting"
  repeat) and every LATE final result are spoken in the call: a success as "done:
  <summary>", a failure as "Mac task failed: <reason>". A result the ``do_on_mac`` tool
  already returned is never spoken twice. A failure the call heard is reported as
  spoken, so the next "voice on" does not brief it again.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .bridge import AUTH_ENV, Authoriser, BridgeController, SpeakerScorer

if TYPE_CHECKING:
    from .mac_operator import MacOperator, OperatorResult

log = logging.getLogger("my_stt_tts.bridge")

OPERATOR_TOOL = "do_on_mac"
#: Tool answers that leave the outcome open: the final result arrives later (on_result).
_OPEN_ANSWERS = ("started", "needs confirmation")
#: What a success says when the operator gave no summary.
DONE_FALLBACK = "the Mac task finished"


class OperatorNotices:
    """Speak the Mac operator's confirmation requests and late results in the call."""

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
        """``notify`` of the operator: a blocked step needs Albert's "yes, do it"."""
        if self.controller.tool_in_flight(OPERATOR_TOOL):
            return  # the do_on_mac answer itself carries the request
        from .mac_operator import STILL_WAITING  # pylint: disable=import-outside-toplevel

        if message.startswith(STILL_WAITING):  # the one repeat after a silence
            self.controller.notify_in_call(f"Mac task {message}")
            return
        self.controller.notify_in_call(f"Mac task needs confirmation: {message}")

    def on_result(self, result: OperatorResult) -> bool:
        """``on_result`` of the operator: every final result is spoken in the running call.

        A success becomes "done: <summary>" (the operator's one-line summary; capped and
        redacted by :meth:`~my_stt_tts.bridge.BridgeController.notify_in_call`), a failure
        "Mac task <status>: <reason>". True when the agent heard (or will hear) the result
        in the running call — the ``do_on_mac`` answer carried it, or the notice went out —
        so the operator reports a failure as already spoken and no later "voice on" briefs
        it again.
        """
        from .mac_operator import DONE  # pylint: disable=import-outside-toplevel

        with self._lock:
            answered = result.call_id in self._answered
            self._answered.discard(result.call_id)
        if answered or self.controller.tool_in_flight(OPERATOR_TOOL):
            return self.controller.in_call  # the do_on_mac answer returns this result
        if result.status == DONE:
            return self.controller.notify_in_call(f"done: {result.summary or DONE_FALLBACK}")
        what = result.reason or result.summary or result.status.replace("_", " ")
        return self.controller.notify_in_call(f"Mac task {result.status.replace('_', ' ')}: {what}")


def _safely(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # pylint: disable=broad-exception-caught  # the others still register
        log.exception("❌ bridge component %s failed to register", name)
        return None


def wire(controller: BridgeController, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Register every enabled component; returns them by name (also ``controller.components``)."""
    # pylint: disable=import-outside-toplevel  # heavy / cyclic modules, loaded on demand
    from . import attention, claude_sessions, decide_tools, mac_control, mac_operator

    monitor = _safely("attention", lambda: attention.register(controller, env=env))
    if monitor is not None:
        monitor.set_notifier(controller.notify_in_call)
    sessions = _safely("claude_sessions", lambda: claude_sessions.register(controller, env=env))
    mac = _safely("mac_control", lambda: mac_control.register(controller, env=env))
    decide = _safely("decide_tools", lambda: decide_tools.register(controller, env=env))
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
        "decide": decide,
        "operator": operator,
        "operator_notices": notices,
    }
    controller.components.update({k: v for k, v in parts.items() if v is not None})
    log.info("🌉 bridge tools: %s", ", ".join(sorted(controller.tools)))
    return parts


def build_bridge(
    scorer: SpeakerScorer | None,
    env: Mapping[str, str] | None = None,
    *,
    wire_components: bool = True,
) -> BridgeController:
    """The daemon's controller: identity from ``MAC_VOICE_AUTHORIZED`` + the voice gate.

    With ``wire_components`` the components register themselves, each gated by its own flag
    (:func:`wire`): attention inbox + monitor, Claude sessions, Mac control, the decide hub
    and the Mac operator — plus the interference capture (``interference.attach``, on unless
    ``MAC_VOICE_INTERFERENCE_CAPTURE=0``).
    """
    authoriser = Authoriser.from_env(scorer, env)
    if authoriser.authorized is None:
        log.warning("⚠️  %s unset — every bridge action will be refused", AUTH_ENV)
    controller = BridgeController(authoriser)
    if wire_components:
        wire(controller, env)
        # pylint: disable-next=import-outside-toplevel  # only the daemon keeps clips
        from .interference import attach

        _safely("interference", lambda: attach(controller, scorer, env))
    controller.preload()
    return controller
