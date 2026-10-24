"""The decide hub as mac-voice client tools (``MAC_VOICE_DECIDE=1``).

Registered on the :class:`~my_stt_tts.bridge.BridgeController` by :func:`register`. The
conversation state lives in one :class:`~my_stt_tts.decide.DecideDesk` for the daemon's
lifetime; a new call keeps the focus but forgets the decision in discussion and the skips.

Tools (the voice agent's client tools):

* ``decide_plate()`` — "what's on my plate": counts per section, then the top three with one
  sentence of context each.
* ``decide_next()`` — the next decision: who asks, the question, one sentence of context, the
  numbered options and the recommendation. It becomes the decision in discussion.
* ``decide_more()`` — "tell me more": the full brief (background, building, fit, the options
  with their trade-offs, the recommendation).
* ``decide_answer(choice, comment, free_text)`` — never sends. Albert's verified sentence must
  carry the choice (and the comment); the tool then makes the answer THE pending proposal and
  returns the read-back that ends with "say yes, do it". Only ``confirm_action`` after his
  new, voice-checked "yes, do it" posts it (executor ``decide_answer``), with ``channel:
  voice``. The hub then holds it for its answer grace (currently 3 s) before the relay may
  deliver it.
* ``decide_comment(text)`` — adds a comment to the pending answer and reads it back again.
* ``decide_undo()`` — takes the last voice answer back while the hub still holds it.
* ``decide_skip()`` / ``decide_snooze(until)`` — next decision; a snooze hides the decision
  from voice until tomorrow 08:00 (kept in this client only: the hub lets the voice token
  answer and undo, nothing else).
* ``decide_focus(scope)`` — work, private or everything, for every later read.

Nothing destructive or bulk is exposed: one decision at a time, every answer read back and
confirmed. Reads count as external content (they void the sentence's unused capabilities),
like the session briefings.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping

from .bridge import CONFIRM_PHRASE, BridgeController, PendingAction, Refusal, log_refused, log_tool
from .claude_sessions import _guarded
from .decide import DecideDesk, HubClient, Proposal, build_client, tomorrow_morning

log = logging.getLogger("my_stt_tts.decide")

FLAG = "MAC_VOICE_DECIDE"
EXECUTOR = "decide_answer"


def _refused(reason: str) -> str:
    log_refused(reason)
    return f"refused: {reason}"


class DecideTools:
    """The decide client tools on the bridge."""

    def __init__(self, controller: BridgeController, desk: DecideDesk) -> None:
        self.ctl = controller
        self.desk = desk
        self._call: object | None = None
        self._lock = threading.Lock()

    def register(self) -> None:
        tools: dict[str, Callable[..., str]] = {
            "decide_plate": self.plate,
            "decide_next": self.next,
            "decide_more": self.more,
            "decide_answer": self.answer,
            "decide_comment": self.comment,
            "decide_undo": self.undo,
            "decide_skip": self.skip,
            "decide_snooze": self.snooze,
        }
        for name, fn in tools.items():
            self.ctl.register_tool(name, _guarded(name, fn, self.ctl))
        focus = _guarded("decide_focus", self.focus, self.ctl)
        self.ctl.register_tool("decide_focus", focus, own_outcome=True)
        self.ctl.register_executor(EXECUTOR, self.execute)

    def _sync_call(self) -> None:
        """A new call forgets the decision in discussion and the skips (the focus stays)."""
        with self._lock:
            turns = self.ctl.turns
            if turns is self._call:
                return
            self._call = turns
        with self.desk.lock:
            self.desk.current = None
            self.desk.skipped.clear()
            self.desk.queue = []

    # -- reads ----------------------------------------------------------------------------
    def plate(self) -> str:
        """What is on Albert's plate: counts per section and the top three decisions."""
        self._sync_call()
        log_tool("decide_plate")
        return self.desk.summary()

    def next(self) -> str:
        """The next decision with its options and recommendation (it is now in discussion)."""
        self._sync_call()
        log_tool("decide_next")
        return self.desk.next_decision()

    def more(self) -> str:
        """The full brief of the decision in discussion."""
        self._sync_call()
        log_tool("decide_more")
        return self.desk.tell_more()

    def skip(self) -> str:
        """Leave the decision in discussion for now and go to the next one."""
        self._sync_call()
        log_tool("decide_skip")
        return self.desk.skip()

    def snooze(self, until: str = "tomorrow") -> str:
        """Hide the decision in discussion from voice until tomorrow 08:00 (voice only)."""
        self._sync_call()
        log_tool("decide_snooze")
        del until  # only "tomorrow" is offered by voice; anything else would need a date
        return self.desk.snooze(tomorrow_morning())

    def focus(self, scope: str = "all") -> str:
        """Focus every later read on work, private, or all."""
        self._sync_call()
        log_tool("decide_focus", str(scope or "all")[:10])
        return self.desk.set_focus(scope)

    # -- the answer: read back, confirm, send -----------------------------------------------
    def answer(self, choice: str = "", comment: str = "", free_text: str = "") -> str:
        """Read back the answer for the decision in discussion; ``confirm_action`` sends it."""
        self._sync_call()
        log_tool("decide_answer")
        said = {k: v for k, v in (("choice", choice), ("text", free_text)) if str(v).strip()}
        if comment.strip():
            said["comment_text"] = comment
        refusal = self._authorise("decide_answer", said)
        if refusal is not None:
            return refusal
        proposal = self.desk.prepare(choice or free_text, comment=comment, free_text=free_text)
        if isinstance(proposal, str):
            return proposal
        self.ctl.propose(EXECUTOR, proposal.as_args())
        return proposal.readback(CONFIRM_PHRASE)

    def comment(self, text: str) -> str:
        """Add Albert's comment to the answer waiting for his yes, and read it back again."""
        self._sync_call()
        log_tool("decide_comment")
        pending = [p for p in self.ctl.pending.live() if p.kind == EXECUTOR]
        if not pending:
            return "No answer is waiting — pick an option first."
        refusal = self._authorise("decide_comment", {"comment_text": text})
        if refusal is not None:
            return refusal
        proposal = Proposal.from_args(pending[0].args).with_comment(text)
        self.ctl.propose(EXECUTOR, proposal.as_args())
        return proposal.readback(CONFIRM_PHRASE)

    def _authorise(self, action: str, said: Mapping[str, str]) -> str | None:
        """Albert's voice-checked sentence must carry ``said`` (the choice, the comment).

        Each value is checked on its own (the free text and the comment are both word by word,
        :func:`~my_stt_tts.bridge_text.derivable` with key ``text``); the capability is then
        spent. The decision itself is the one in discussion, never taken from the sentence.
        """
        cap = self.ctl.mint()
        if isinstance(cap, Refusal):
            return _refused(cap.reason)
        for key, value in said.items():
            check = {("text" if key == "comment_text" else key): value}
            refusal = self.ctl.caps.check(cap, action, check)
            if refusal is not None:
                return _refused(refusal.reason)
        refusal = self.ctl.caps.use(cap, action, {})
        return _refused(refusal.reason) if refusal is not None else None

    def execute(self, pending: PendingAction) -> str:
        """``decide_answer`` executor: post the confirmed answer (``channel: voice``)."""
        proposal = Proposal.from_args(pending.args)
        log_tool("decide_answer", "confirmed")
        return self.desk.send(proposal)

    def undo(self) -> str:
        """Take back the last voice answer while the hub still holds it."""
        self._sync_call()
        log_tool("decide_undo")
        refusal = self._authorise("decide_undo", {})
        if refusal is not None:
            return refusal
        return self.desk.undo()


def register(
    controller: BridgeController,
    client: HubClient | None = None,
    env: Mapping[str, str] | None = None,
    desk: DecideDesk | None = None,
) -> DecideTools | None:
    """Add the decide tools + executor when ``MAC_VOICE_DECIDE=1`` (else None)."""
    env = os.environ if env is None else env
    if env.get(FLAG, "").strip() != "1":
        return None
    if desk is None:
        client = client or build_client(env)
        if client is None:
            log.warning(
                "⚠️  %s=1 but DECIDE_VOICE_CF_CLIENT_ID or secret-run is missing:"
                " decide tools answer 'not set up'",
                FLAG,
            )
        desk = DecideDesk(client)
    tools = DecideTools(controller, desk)
    tools.register()
    return tools


__all__ = ["EXECUTOR", "FLAG", "DecideTools", "register"]
