"""The decide hub in the local pipeline's intent router (``DECIDE_VOICE=1``).

Runs in the turn path before the LLM, next to the music router
(:func:`my_stt_tts.__main__.maybe_handle_decide`), so it works for every brain. A turn that is
a decide intent is answered here and never reaches the model; any other turn returns None.

Intents (English, with common German and French forms):

* "what's on my plate", "what do I need to decide" → the plate summary;
* "next decision" (bare "next" while a decision is being discussed) → its short brief;
* "tell me more" → the full brief of the decision in discussion;
* "answer option two", "go with the recommendation", "answer: <free text>", "done" for a
  step → the read-back, then "Say yes to send it, or no to cancel.";
* "add comment …" → the comment joins the waiting answer, read back again;
* "yes" / "no" while an answer waits → send it (``channel: voice``) / drop it;
* "undo" → the last voice answer is taken back while the hub still holds it (its grace,
  currently 3 s — often already over, so the read-back before "yes" is the safety);
* "skip", "snooze until tomorrow", "work only" / "private only" / "show everything".

Answers, comments, confirmations and undo are taken only from the voice named by
``DECIDE_VOICE_SPEAKER`` (else ``MAC_VOICE_AUTHORIZED``) as speaker identification resolved
it for the turn; with neither set, or another or an unknown voice, they are refused. Reads are
open to anyone in the room. An answer waits at most :data:`CONFIRM_TTL_S` for its yes.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Mapping

from .bridge_text import normalise
from .decide import DecideDesk, Proposal, build_client

SPEAKER_ENVS = ("DECIDE_VOICE_SPEAKER", "MAC_VOICE_AUTHORIZED")
CONFIRM_TTL_S = 120.0
UNDO_WINDOW_S = 60.0

_PLATE = re.compile(
    r"\b(?:on my plate|what (?:do|should|must) i (?:need to |have to )?decide|my (?:open )?"
    r"decisions|open decisions|decisions? (?:for|waiting for) me|was (?:muss|soll) ich "
    r"entscheiden|meine entscheidungen|offene entscheidungen|mes decisions)\b"
)
_NEXT = re.compile(
    r"^(?:(?:the )?next decision|next one|nachste entscheidung|naechste entscheidung|"
    r"decision suivante)(?: please| bitte)?$"
)
_BARE_NEXT = re.compile(r"^(?:next|weiter|nachste|suivante|go on)(?: please| bitte)?$")
_MORE = re.compile(
    r"^(?:tell me more|more|more details|details|what is it about|mehr dazu|erzahl mehr|"
    r"erzaehl mehr|dis m en plus)(?: please| bitte)?$"
)
_ANSWER = re.compile(
    r"^(?:answer|my answer is|reply|antworte|antwort|meine antwort ist|reponds|ma reponse est)"
    r"\b\s*(?P<rest>.*)$"
)
_PICK = re.compile(
    r"^(?:(?:i ll |i d |let s |we ll )?(?:go with|take|pick|choose|option|options|nimm|"
    r"nehme|ich nehme|wahle|prends|je prends))\b"
)
_RECOMMEND = re.compile(r"\b(?:recommend\w*|empfehlung|empfohlen\w*|recommande\w*)\b")
_DONE = re.compile(r"^(?:done|i did it|it s done|mark (?:it )?done|erledigt|fait)(?: please)?$")
_COMMENT = re.compile(
    r"^(?:add (?:a )?comment|comment|with (?:the )?comment|kommentar|commentaire)\b\s*(?P<rest>.*)$"
)
_UNDO = re.compile(
    r"^(?:undo|undo that|undo it|take (?:it|that) back|ruckgangig|rueckgaengig|annule)$"
)
_SKIP = re.compile(r"^(?:skip|skip it|skip this one|later|uberspringen|spater|passe|plus tard)$")
_SNOOZE = re.compile(r"\b(?:snooze|remind me tomorrow|morgen wieder|demain)\b")
_FOCUS = re.compile(
    r"\b(?:(?:show |focus on )?(?P<a>work|private) only|only (?P<b>work|private)|nur "
    r"(?P<c>arbeit|privat)|(?:show |focus on )(?P<all>everything|all decisions|all))\b"
)
_YES = re.compile(r"^(?:yes|yeah|yep|yes please|yes send it|send it|ja|ja bitte|oui|go ahead)$")
_NO = re.compile(r"^(?:no|nope|cancel|don t|do not|stop|nein|abbrechen|non|annuler)\b")


def _authorized_name(env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    for key in SPEAKER_ENVS:
        value = env.get(key, "").strip()
        if value:
            return value
    return None


class LocalDecide:  # pylint: disable=too-many-instance-attributes  # one conversation's state
    """Decide intents for the local pipeline: read, read back, confirm, send, undo."""

    def __init__(
        self,
        desk: DecideDesk,
        authorized: str | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.desk = desk
        self.authorized = authorized
        self.clock = clock
        self._pending: tuple[Proposal, float] | None = None
        self._sent_at: float | None = None
        self._lock = threading.Lock()

    # -- state ------------------------------------------------------------------------------
    def _live_pending(self) -> Proposal | None:
        with self._lock:
            if self._pending is None:
                return None
            proposal, at = self._pending
            if self.clock() - at > CONFIRM_TTL_S:
                self._pending = None
                return None
            return proposal

    def _set_pending(self, proposal: Proposal | None) -> None:
        with self._lock:
            self._pending = None if proposal is None else (proposal, self.clock())

    def _in_discussion(self) -> bool:
        return self.desk.current is not None

    def _may_act(self, speaker: str | None) -> str | None:
        """None when ``speaker`` may answer, else the spoken refusal."""
        if self.authorized is None:
            return "Answers by voice are off: set DECIDE_VOICE_SPEAKER to your enrolled name."
        if speaker is None or speaker.casefold() != self.authorized.casefold():
            return "I only take decision answers from Albert's voice."
        return None

    # -- the router ------------------------------------------------------------------------
    def handle(  # pylint: disable=too-many-return-statements  # one reply per intent
        self, text: str, speaker: str | None = None
    ) -> str | None:
        """The spoken reply when ``text`` is a decide intent, else None (the LLM answers)."""
        norm = normalise(text)
        if not norm:
            return None
        pending = self._live_pending()
        if pending is not None:
            reply = self._while_pending(norm, text, pending, speaker)
            if reply is not None:
                return reply
        if _PLATE.search(norm):
            return self.desk.summary()
        focus = _FOCUS.search(norm)
        if focus and (self._in_discussion() or "decision" in norm or "show" in norm):
            return self._focus(focus)
        if _NEXT.match(norm) or (_BARE_NEXT.match(norm) and self._in_discussion()):
            return self.desk.next_decision()
        if self._recent_answer() and _UNDO.match(norm):
            return self._undo(speaker)
        if not self._in_discussion():
            return None
        return self._about_current(norm, text, speaker)

    def _about_current(self, norm: str, text: str, speaker: str | None) -> str | None:
        if _MORE.match(norm):
            return self.desk.tell_more()
        if _SKIP.match(norm):
            return self.desk.skip()
        if _SNOOZE.search(norm):
            return self.desk.snooze()
        answer = _ANSWER.match(norm)
        if answer:
            raw_rest = _rest_of(text, len(norm) - len(answer.group("rest")))
            return self._propose(answer.group("rest"), raw_rest, speaker)
        if _PICK.match(norm) or _RECOMMEND.search(norm) or _DONE.match(norm):
            return self._propose(norm, "", speaker)
        return None

    def _while_pending(  # pylint: disable=too-many-return-statements  # one reply per intent
        self, norm: str, text: str, pending: Proposal, speaker: str | None
    ) -> str | None:
        if _YES.match(norm):
            refusal = self._may_act(speaker)
            if refusal is not None:
                return refusal
            self._set_pending(None)
            reply = self.desk.send(pending)
            if reply.startswith("Sent"):
                with self._lock:
                    self._sent_at = self.clock()
            return reply
        if _NO.match(norm):
            self._set_pending(None)
            return "Cancelled — nothing was sent."
        comment = _COMMENT.match(norm)
        if comment:
            refusal = self._may_act(speaker)
            if refusal is not None:
                return refusal
            words = _rest_of(text, len(norm) - len(comment.group("rest")))
            if not words.strip():
                return "What is the comment?"
            updated = pending.with_comment(words)
            self._set_pending(updated)
            return updated.readback("yes")
        return None

    def _propose(self, said: str, raw: str, speaker: str | None) -> str:
        refusal = self._may_act(speaker)
        if refusal is not None:
            return refusal
        current = self.desk.current or {}
        free = raw.strip().lstrip(":,.- ").rstrip(" ,:")
        if free and not current.get("options") and current.get("kind") != "action":
            proposal = self.desk.prepare("", free_text=free)  # his own words, as spelled
        else:
            proposal = self.desk.prepare(said)
            if isinstance(proposal, str) and free and current.get("allow_other", True):
                # "answer: <words>" that names no option is Albert's own answer
                proposal = self.desk.prepare("", free_text=free)
        if isinstance(proposal, str):
            return proposal
        self._set_pending(proposal)
        return proposal.readback("yes")

    def _focus(self, m: re.Match[str]) -> str:
        scope = m.group("a") or m.group("b") or m.group("c")
        scope = {"arbeit": "work", "privat": "private"}.get(scope or "", scope)
        reply = self.desk.set_focus(scope if not m.group("all") else None)
        return reply + self.desk.summary()

    def _recent_answer(self) -> bool:
        with self._lock:
            at = self._sent_at
        return at is not None and self.clock() - at <= UNDO_WINDOW_S

    def _undo(self, speaker: str | None) -> str:
        refusal = self._may_act(speaker)
        if refusal is not None:
            return refusal
        with self._lock:
            self._sent_at = None
        return self.desk.undo()


def _rest_of(text: str, skip_norm_chars: int) -> str:
    """The original words of ``text`` after its first normalised ``skip_norm_chars`` characters.

    Free text and comments keep Albert's own spelling and punctuation; the intent itself was
    matched on the normalised form.
    """
    words = text.split()
    taken = 0
    for i, word in enumerate(words):
        if taken >= skip_norm_chars:
            return " ".join(words[i:]).lstrip(" :,.-")
        taken += len(normalise(word)) + (1 if normalise(word) else 0)
    return ""


_ROUTER: LocalDecide | None = None
_ROUTER_LOCK = threading.Lock()


def local_router(env: Mapping[str, str] | None = None) -> LocalDecide:
    """The process-wide router (built on first use from the environment)."""
    global _ROUTER  # pylint: disable=global-statement  # one conversation per process
    with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = LocalDecide(DecideDesk(build_client(env)), _authorized_name(env))
        return _ROUTER


__all__ = ["CONFIRM_TTL_S", "LocalDecide", "local_router"]
