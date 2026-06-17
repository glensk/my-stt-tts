"""Per-turn latency + transcript telemetry, keyed by a shared ``speech_id``."""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger("my_stt_tts.metrics")

_counter = itertools.count(1)


def next_speech_id() -> str:
    """Return a process-unique, monotonically increasing turn id."""
    return f"turn-{next(_counter):05d}"


@dataclass(slots=True)
class TurnMetrics:
    """Per-stage durations (ms) and notes for one conversational turn.

    The ``speech_id`` ties every stage's timing together so an end-to-end turn
    can be reconstructed from the logs.
    """

    speech_id: str = field(default_factory=next_speech_id)
    stages: dict[str, float] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a stage; record elapsed milliseconds under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000.0, 1)

    def note(self, **kwargs: object) -> None:
        """Attach free-form notes (transcript, speaker, language, …)."""
        self.notes.update(kwargs)

    def total_ms(self) -> float:
        """Wall-clock milliseconds since this turn was created."""
        return round((time.perf_counter() - self._t0) * 1000.0, 1)

    def as_dict(self) -> dict[str, object]:
        """Flat dict for logging / inspection."""
        return {
            "speech_id": self.speech_id,
            "total_ms": self.total_ms(),
            "stages_ms": self.stages,
            **self.notes,
        }

    def log(self) -> None:
        """Emit one structured info-level log line for this turn."""
        log.info("turn %s", json.dumps(self.as_dict(), ensure_ascii=False))
