"""Typed, prioritized, non-droppable event bus (G2).

The voice loop publishes events (state / transcript / response / metrics) and any
number of subscribers (the web UI's SSE endpoint, network satellites) consume
them. Historically this was a flat string-fanout queue where a full subscriber
queue silently *dropped* events — including interruptions and errors, the very
events that must never be lost or arrive late.

This refactor introduces a **typed frame model** with two priority classes:

* **SYSTEM** frames — interruption, error, and end-of-turn. They are *control*
  signals: they must (1) **never be dropped**, even under back-pressure, (2)
  **bypass** any queued data frames so they are delivered immediately, and (3)
  **flush** the queued data ahead of them (stale partial transcripts / response
  deltas are pointless once an interruption fires). They are also ordered
  **consistently** relative to each other across *every* transport — local, ws,
  webrtc, telephony — because every transport drains the same per-subscriber
  ordering.
* **DATA** frames — state, transcript, response, wake, log, metrics, … the normal
  high-volume stream. These ride a bounded queue and *may* be dropped under
  back-pressure (a missed partial transcript is harmless).

A :class:`_Subscriber` holds a small system deque (unbounded — system frames are
rare and must not be lost) plus a bounded data queue; :meth:`_Subscriber.get`
always returns a pending **system** frame first, and a system frame **clears** the
data queue when enqueued (flush-on-interrupt). The public API is unchanged:
``publish(dict)`` and the convenience emitters still exist, and subscribers still
receive JSON **strings** via ``.get()`` / ``.get_nowait()`` — so the SSE endpoint
and satellites need no changes. The ad-hoc interruption emitters are now backed
by this typed model.
"""

from __future__ import annotations

import collections
import contextlib
import datetime
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import IO, Any

log = logging.getLogger("my_stt_tts.events")

# Env var naming a file to append every published event to as JSON Lines (one
# event per line, prefixed with a wall-clock timestamp). quickstart.sh sets this
# to logs/events-<ts>.jsonl so a full EVENT LOG of each run is captured to disk
# for after-the-fact investigation (e.g. why a wake word didn't fire). Empty /
# unset -> no file sink (default; tests + library use write nothing).
_EVENT_LOG_ENV = "MSTT_EVENT_LOG"

# State machine the UI renders (the order a turn flows through):
STATES = (
    "idle",
    "listening",  # waiting for the wake word
    "recording",  # mic is hot
    "stt",  # transcribing
    "llm_request",  # sending the prompt
    "llm_wait",  # awaiting first token
    "llm_response",  # tokens arriving
    "tts",  # synthesizing speech
    "speaking",  # playing audio
    "interrupted",  # user barged in: TTS/LLM aborted, recording their new turn
)


class Priority(IntEnum):
    """Delivery priority of a frame. Higher is delivered first and never dropped."""

    DATA = 0  # normal high-volume stream (state/transcript/response/log/metrics)
    SYSTEM = 1  # control: interruption / error / end-of-turn — never dropped


# Event ``type`` values that are SYSTEM priority (control plane). Everything else
# is DATA. ``interrupt`` covers both start/stop phases; ``barge_in`` is the
# user-facing interruption notice; ``bot_stopped_speaking`` / ``end_of_turn`` mark
# end-of-turn; ``error`` is a failure. The wire ``type`` strings are unchanged for
# back-compat — only their *priority* is promoted.
_SYSTEM_TYPES = frozenset({"interrupt", "barge_in", "error", "end_of_turn", "bot_stopped_speaking"})


def classify(event: dict[str, Any]) -> Priority:
    """Classify an event dict into a delivery :class:`Priority` (G2).

    Interruption / error / end-of-turn are SYSTEM (control, non-droppable); a
    ``log`` event with ``level == "error"`` is promoted to SYSTEM too. Everything
    else (state / transcript / response / wake / metrics / …) is DATA.
    """
    etype = event.get("type")
    if etype in _SYSTEM_TYPES:
        return Priority.SYSTEM
    if etype == "log" and event.get("level") == "error":
        return Priority.SYSTEM
    return Priority.DATA


@dataclass(slots=True)
class Frame:
    """A typed event frame: the serialized payload + its delivery priority (G2)."""

    data: str  # the JSON-serialized event (what subscribers receive)
    priority: Priority
    type: str = ""

    @classmethod
    def of(cls, event: dict[str, Any]) -> Frame:
        """Build a :class:`Frame` from an event dict (classifies + serializes once)."""
        return cls(
            data=json.dumps(event, ensure_ascii=False),
            priority=classify(event),
            type=str(event.get("type", "")),
        )


@dataclass
class _Subscriber:
    """Per-consumer delivery state: a non-droppable system lane + a bounded data queue.

    System frames go on an unbounded deque (rare, must never be lost) and, on
    arrival, **flush** the bounded data queue — so an interruption is delivered
    immediately and isn't stuck behind a backlog of stale partials. :meth:`get`
    always drains the system lane before the data lane, giving consistent ordering
    across every transport that pulls from this subscriber.
    """

    maxsize: int = 512
    _system: collections.deque[str] = field(default_factory=collections.deque, init=False)
    _data: queue.Queue[str] = field(init=False)
    _wake: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._data = queue.Queue(maxsize=self.maxsize)

    def offer(self, frame: Frame) -> None:
        """Enqueue ``frame``. SYSTEM frames flush queued data and never drop."""
        if frame.priority >= Priority.SYSTEM:
            with self._lock:
                self._drain_data_locked()  # stale data is pointless ahead of a control signal
                self._system.append(frame.data)
            self._wake.set()
            return
        with contextlib.suppress(queue.Full):  # DATA may drop under back-pressure
            self._data.put_nowait(frame.data)
            self._wake.set()

    def _drain_data_locked(self) -> None:
        while True:
            try:
                self._data.get_nowait()
            except queue.Empty:
                break

    def _take_system(self) -> str | None:
        with self._lock:
            if self._system:
                return self._system.popleft()
        return None

    def get(self, timeout: float | None = None) -> str:
        """Return the next frame (system first), blocking up to ``timeout`` seconds.

        Raises :class:`queue.Empty` on timeout — same contract as a plain Queue, so
        the SSE endpoint / satellites need no change.
        """
        sysframe = self._take_system()
        if sysframe is not None:
            return sysframe
        # Block on the data queue, but wake early if a system frame arrives.
        try:
            data = self._data.get(timeout=timeout)
        except queue.Empty:
            sysframe = self._take_system()
            if sysframe is not None:
                return sysframe
            raise
        return data

    def get_nowait(self) -> str:
        """Return the next frame (system first) without blocking; raise if none."""
        sysframe = self._take_system()
        if sysframe is not None:
            return sysframe
        return self._data.get_nowait()

    def put_nowait(self, data: str) -> None:
        """Back-compat shim: enqueue a raw JSON string as a DATA frame."""
        with contextlib.suppress(queue.Full):
            self._data.put_nowait(data)

    def empty(self) -> bool:
        """True when neither the system lane nor the data queue has a pending frame."""
        with self._lock:
            if self._system:
                return False
        return self._data.empty()

    def qsize(self) -> int:
        """Approximate number of pending frames (system + data)."""
        with self._lock:
            n_sys = len(self._system)
        return n_sys + self._data.qsize()


class EventBus:
    """Typed, prioritized fan-out of event frames to any number of subscribers (G2).

    System events (interruption / error / end-of-turn) are never dropped, bypass
    queued data, and flush it; data events ride a bounded queue. Publishing is a
    cheap no-op when nobody is listening, so the loop can call it unconditionally.
    A module-level :data:`bus` singleton is shared.
    """

    def __init__(self) -> None:
        self._subs: list[_Subscriber] = []
        self._lock = threading.Lock()
        self._last_state: str | None = None
        self._sink: IO[str] | None = None
        self._sink_lock = threading.Lock()
        self._sink_checked = False  # lazy one-time auto-attach from $MSTT_EVENT_LOG

    def attach_file_sink(self, path: str) -> None:
        """Append every published event (as JSON Lines) to ``path`` from now on.

        Captures the full EVENT LOG to disk for after-the-fact investigation. Opened
        line-buffered in append mode so the file is readable live (``tail -f``). Called
        automatically when :data:`MSTT_EVENT_LOG` is set (see :meth:`_write_sink`), or
        explicitly. Failures are logged, never raised — logging must not break the bus.
        """
        with self._sink_lock:
            with contextlib.suppress(OSError):
                if self._sink is not None:
                    self._sink.close()
            try:
                self._sink = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
            except OSError as exc:  # pragma: no cover - unwritable path
                self._sink = None
                log.warning("event log sink %s could not be opened: %s", path, exc)

    def _write_sink(self, event: dict[str, Any]) -> None:
        """Append one event to the file sink (lazy auto-attach from env on first call)."""
        if not self._sink_checked:
            self._sink_checked = True
            path = os.environ.get(_EVENT_LOG_ENV, "").strip()
            if path:
                self.attach_file_sink(path)
        if self._sink is None:
            return
        line = json.dumps(
            {"ts": datetime.datetime.now().isoformat(timespec="milliseconds"), **event},
            ensure_ascii=False,
        )
        with self._sink_lock, contextlib.suppress(OSError, ValueError):
            self._sink.write(line + "\n")

    def publish(self, event: dict[str, Any]) -> None:
        """Classify + fan out an event dict to all subscribers (G2 priority rules)."""
        if event.get("type") == "state":
            self._last_state = json.dumps(event, ensure_ascii=False)
        self._write_sink(event)
        frame = Frame.of(event)
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            sub.offer(frame)

    def subscribe(self) -> _Subscriber:
        """Register a new subscriber; immediately replays the last state event."""
        sub = _Subscriber()
        with self._lock:
            self._subs.append(sub)
        if self._last_state is not None:
            sub.put_nowait(self._last_state)
        return sub

    def unsubscribe(self, sub: _Subscriber) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    # --- convenience emitters (DATA priority) ---

    def state(self, state: str, detail: str = "") -> None:
        self.publish({"type": "state", "state": state, "detail": detail})

    def transcript(self, text: str, *, partial: bool = False, source: str = "") -> None:
        """Publish a transcript. ``partial=True`` marks an in-progress streaming
        transcript (G6); the UI can replace it when the final arrives.

        ``source`` (optional) tags WHERE this turn's text came from so the UI can
        show e.g. "YOU · push-to-talk": one of ``"typed"`` / ``"push_to_talk"`` /
        ``"wake"`` / ``"live_audio"``. Default ``""`` (unset) keeps back-compat for
        callers that don't tag — the wire field is only added when non-empty."""
        event: dict[str, Any] = {"type": "transcript", "text": text, "partial": partial}
        if source:
            event["source"] = source
        self.publish(event)

    def response(self, text: str, *, final: bool = False, model: str = "") -> None:
        """Publish a response delta. ``model`` (when set) names the active model that
        produced the reply (e.g. ``"claude-cli / haiku"``) so the page can show an
        "ASSISTANT · <model>" label; it also rides the ``llm_request`` state detail."""
        event: dict[str, Any] = {"type": "response", "text": text, "final": final}
        if model:
            event["model"] = model
        self.publish(event)

    def wake(self) -> None:
        self.publish({"type": "wake", "fired": True})

    def mic_result(
        self, *, ok: bool, verdict: str, message: str, level: int = 0, permission: str = "unknown"
    ) -> None:
        """Publish the outcome of a server-side mic test (GUI "Test mic").

        DATA priority — the UI shows it prominently (green/red status chip) and
        also logs it. ``verdict`` is the machine tag (``ok`` / ``silent`` /
        ``no_device`` / ``error`` / ``denied`` / ``restricted``); ``level`` is the
        measured loudness 0–100; ``permission`` is the macOS mic authorization
        (``authorized`` / ``denied`` / ``notDetermined`` / ``restricted`` / ``unavailable``)."""
        self.publish(
            {
                "type": "mic_result",
                "ok": ok,
                "verdict": verdict,
                "message": message,
                "level": level,
                "permission": permission,
            }
        )

    def speaker(self, name: str | None) -> None:
        """Publish the identified speaker for the current turn (G7).

        ``name`` is the enrolled person, or ``None`` for a guest / unrecognized /
        typed turn. DATA priority — purely informational for the UI."""
        self.publish({"type": "speaker", "name": name or "", "known": bool(name)})

    # --- interruption / control lifecycle (G2: SYSTEM priority, non-droppable) ---
    # These are the control plane: they bypass queued data, flush it, and are never
    # dropped — across every transport. Formerly ad-hoc DATA events.

    def interrupted(self, spoken_chars: int = 0) -> None:
        """User barged in: the in-flight reply was aborted after ``spoken_chars``
        characters were actually voiced (G1). SYSTEM priority."""
        self.publish({"type": "barge_in", "spoken_chars": spoken_chars})

    def interrupt_start(self) -> None:
        """The user has been confirmed as taking the floor; abort TTS + the LLM
        stream. Stages should stop producing/queuing speech immediately (R2-6).
        SYSTEM priority — flushes any queued partials/deltas."""
        self.publish({"type": "interrupt", "phase": "start"})

    def interrupt_stop(self) -> None:
        """The interruption has been handled and the captured audio handed off; it
        is safe for stages to resume from a clean state (R2-6). SYSTEM priority."""
        self.publish({"type": "interrupt", "phase": "stop"})

    def bot_stopped_speaking(self) -> None:
        """Playback for the current reply has ended (cancelled or completed). The
        chunker and TTS queue should flush any residual state (R2-6). End-of-turn
        control signal — SYSTEM priority (wire type kept for back-compat)."""
        self.publish({"type": "bot_stopped_speaking"})

    def error(self, message: str, *, detail: str = "") -> None:
        """Publish an error control event (SYSTEM priority, never dropped)."""
        self.publish({"type": "error", "message": message, "detail": detail})

    def log(self, message: str, level: str = "info") -> None:
        self.publish({"type": "log", "level": level, "message": message})

    def debug(self, message: str, **fields: Any) -> None:
        """Publish a verbose audio-pipeline debug trace (the GUI "debugger").

        DATA priority — high-volume diagnostics the GUI EVENT LOG renders so it is
        obvious WHERE audio is lost (sample rate / #samples / rms / peak per capture,
        VAD + endpoint decisions, wake max-score, STT input length + transcript).
        ``message`` is the human line; ``fields`` carry the structured numbers (e.g.
        ``sample_rate=16000, samples=24000, rms=0.04``). Only emitted when audio
        debugging is on (``cfg.debug_audio``), so it is a no-op in normal runs."""
        self.publish({"type": "debug", "message": message, **fields})


# Shared singleton used by the loop and the web UI.
bus = EventBus()
