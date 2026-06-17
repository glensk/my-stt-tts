"""Threaded producer-consumer pipeline spine.

Each stage runs in its own thread, reading from an input queue and writing to an
output queue. A stage is either a plain callable (one output per input) or a
generator callable (many outputs per input — e.g. LLM tokens -> sentences). Two
control sentinels travel through the pipeline:

* ``SESSION_END``  — per-turn boundary; forwarded downstream so stages can flush.
* ``PIPELINE_END`` — shutdown; each stage forwards it, then exits.

The native model/LLM calls release the GIL, so threads genuinely overlap.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import cast

log = logging.getLogger("my_stt_tts.spine")


class _Signal:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<{self.name}>"


SESSION_END = _Signal("SESSION_END")
PIPELINE_END = _Signal("PIPELINE_END")


@dataclass
class Stage:
    """One pipeline stage. ``generator=True`` if ``fn`` yields multiple outputs."""

    name: str
    fn: Callable[[object], object]
    generator: bool = False


@dataclass
class Pipeline:
    """Wire ``stages`` into a chain of queues, one worker thread per stage."""

    stages: list[Stage]
    maxsize: int = 64
    _queues: list[queue.Queue] = field(default_factory=list, init=False)
    _threads: list[threading.Thread] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._queues = [queue.Queue(self.maxsize) for _ in range(len(self.stages) + 1)]

    def _worker(self, index: int) -> None:
        stage = self.stages[index]
        q_in = self._queues[index]
        q_out = self._queues[index + 1]
        while True:
            item = q_in.get()
            if item is PIPELINE_END:
                q_out.put(PIPELINE_END)
                return
            if item is SESSION_END:
                q_out.put(SESSION_END)
                continue
            try:
                if stage.generator:
                    for out in cast("Iterable[object]", stage.fn(item)):
                        q_out.put(out)
                else:
                    result = stage.fn(item)
                    if result is not None:
                        q_out.put(result)
            except Exception:
                log.exception("stage %r failed on %r", stage.name, item)

    def start(self) -> None:
        """Launch one daemon worker thread per stage."""
        for i, stage in enumerate(self.stages):
            thread = threading.Thread(
                target=self._worker, args=(i,), name=f"stage-{stage.name}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def feed(self, item: object) -> None:
        """Put an item onto the first stage's input queue."""
        self._queues[0].put(item)

    @property
    def output(self) -> queue.Queue:
        """The final output queue."""
        return self._queues[-1]

    def shutdown(self, timeout: float = 5.0) -> None:
        """Send ``PIPELINE_END`` and join all workers."""
        self._queues[0].put(PIPELINE_END)
        for thread in self._threads:
            thread.join(timeout=timeout)
