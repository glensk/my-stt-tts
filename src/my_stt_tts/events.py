"""Tiny thread-safe pub/sub event bus for the live web UI (``--browser``).

The voice loop publishes state / transcript / response events; the web UI's SSE
endpoint subscribes. Publishing is a cheap no-op when nobody is listening, so the
loop can call it unconditionally. A module-level :data:`bus` singleton is shared.
"""

from __future__ import annotations

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
            try:
                sub.put_nowait(data)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue[str]:
        """Register a new subscriber; immediately replays the last state event."""
        sub: queue.Queue[str] = queue.Queue(maxsize=512)
        with self._lock:
            self._subs.append(sub)
        if self._last_state is not None:
            try:
                sub.put_nowait(self._last_state)
            except queue.Full:
                pass
        return sub

    def unsubscribe(self, sub: queue.Queue[str]) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    # --- convenience emitters ---

    def state(self, state: str, detail: str = "") -> None:
        self.publish({"type": "state", "state": state, "detail": detail})

    def transcript(self, text: str) -> None:
        self.publish({"type": "transcript", "text": text})

    def response(self, text: str, *, final: bool = False) -> None:
        self.publish({"type": "response", "text": text, "final": final})

    def wake(self) -> None:
        self.publish({"type": "wake", "fired": True})

    def log(self, message: str, level: str = "info") -> None:
        self.publish({"type": "log", "level": level, "message": message})


# Shared singleton used by the loop and the web UI.
bus = EventBus()
