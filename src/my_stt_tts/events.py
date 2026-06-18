"""Tiny thread-safe pub/sub event bus for the live web UI (``--browser``).

The voice loop publishes state / transcript / response events; the web UI's SSE
endpoint subscribes. Publishing is a cheap no-op when nobody is listening, so the
loop can call it unconditionally. A module-level :data:`bus` singleton is shared.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from typing import Any

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


class EventBus:
    """Fan-out of JSON event strings to any number of SSE subscribers."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue[str]] = []
        self._lock = threading.Lock()
        self._last_state: str | None = None

    def publish(self, event: dict[str, Any]) -> None:
        """Broadcast an event dict to all subscribers (drops if a queue is full)."""
        if event.get("type") == "state":
            self._last_state = json.dumps(event, ensure_ascii=False)
        data = json.dumps(event, ensure_ascii=False)
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            with contextlib.suppress(queue.Full):
                sub.put_nowait(data)

    def subscribe(self) -> queue.Queue[str]:
        """Register a new subscriber; immediately replays the last state event."""
        sub: queue.Queue[str] = queue.Queue(maxsize=512)
        with self._lock:
            self._subs.append(sub)
        if self._last_state is not None:
            with contextlib.suppress(queue.Full):
                sub.put_nowait(self._last_state)
        return sub

    def unsubscribe(self, sub: queue.Queue[str]) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    # --- convenience emitters ---

    def state(self, state: str, detail: str = "") -> None:
        self.publish({"type": "state", "state": state, "detail": detail})

    def transcript(self, text: str, *, partial: bool = False) -> None:
        """Publish a transcript. ``partial=True`` marks an in-progress streaming
        transcript (G6); the UI can replace it when the final arrives."""
        self.publish({"type": "transcript", "text": text, "partial": partial})

    def response(self, text: str, *, final: bool = False) -> None:
        self.publish({"type": "response", "text": text, "final": final})

    def wake(self) -> None:
        self.publish({"type": "wake", "fired": True})

    def interrupted(self, spoken_chars: int = 0) -> None:
        """User barged in: the in-flight reply was aborted after ``spoken_chars``
        characters were actually voiced (G1)."""
        self.publish({"type": "barge_in", "spoken_chars": spoken_chars})

    # --- interruption lifecycle (R2-6): formalised as bus events so the chunker,
    # TTS queue and partial-STT state flush consistently, without per-sentence
    # coupling. Every stage can subscribe and react.

    def interrupt_start(self) -> None:
        """The user has been confirmed as taking the floor; abort TTS + the LLM
        stream. Stages should stop producing/queuing speech immediately (R2-6)."""
        self.publish({"type": "interrupt", "phase": "start"})

    def interrupt_stop(self) -> None:
        """The interruption has been handled and the captured audio handed off; it
        is safe for stages to resume from a clean state (R2-6)."""
        self.publish({"type": "interrupt", "phase": "stop"})

    def bot_stopped_speaking(self) -> None:
        """Playback for the current reply has ended (cancelled or completed). The
        chunker and TTS queue should flush any residual state (R2-6)."""
        self.publish({"type": "bot_stopped_speaking"})

    def log(self, message: str, level: str = "info") -> None:
        self.publish({"type": "log", "level": level, "message": message})


# Shared singleton used by the loop and the web UI.
bus = EventBus()
