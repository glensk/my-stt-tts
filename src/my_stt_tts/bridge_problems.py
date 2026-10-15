"""Problems and briefings of the Claude/Mac bridge: what went wrong, and who keeps it.

* :class:`Problem` — one thing worth telling Albert (kind, subject, short reason, severity;
  never message text).
* :class:`ProblemSink` — where problems go (the attention inbox in the daemon,
  :class:`MemoryProblemSink` in tests); :func:`report_problem` passes the options a sink
  understands: ``spoken`` (the problem already reached the agent in the running call, so
  no later "voice on" briefs it again) and ``dedupe_key``.
* :class:`DeliveryTracker` / :class:`BriefingProvider` and their in-memory versions.

Re-exported by :mod:`my_stt_tts.bridge`, which most callers import them from.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .bridge_text import redact

log = logging.getLogger("my_stt_tts.bridge")

BRIEFING_CAP = 600
LOG_DETAIL_CAP = 80


def _detail(text: str) -> str:
    return redact(" ".join(str(text).split()), LOG_DETAIL_CAP)


@dataclass(frozen=True)
class Problem:
    """A problem worth telling Albert about (no message text in ``reason``)."""

    kind: str
    subject: str
    reason: str
    severity: str = "error"


class ProblemSink(Protocol):
    """Where problems go; ``resolve`` closes an open one once later evidence settled it.

    A sink may take keyword options in ``report`` (``spoken``, ``dedupe_key``);
    :func:`report_problem` passes only those it accepts.
    """

    def report(self, problem: Problem) -> None: ...

    def resolve(self, kind: str, subject: str) -> bool: ...


def accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """True when ``fn`` takes the keyword ``name`` (or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


def report_problem(
    sink: ProblemSink, problem: Problem, *, spoken: bool = False, dedupe_key: str | None = None
) -> None:
    """``sink.report(problem)`` with the options the sink understands.

    ``spoken``: the problem already reached the agent in the running call (a tool result
    or a ``[system notice]``) — the attention inbox then marks it briefed and acknowledged
    at once, so no later "voice on" briefs it again.
    """
    report: Callable[..., None] = sink.report
    options: dict[str, Any] = {}
    if spoken and accepts_keyword(report, "spoken"):
        options["spoken"] = True
    if dedupe_key is not None and accepts_keyword(report, "dedupe_key"):
        options["dedupe_key"] = dedupe_key
    report(problem, **options)


class BriefingProvider(Protocol):
    def briefing(self) -> str | None: ...


class MemoryProblemSink:
    """Keeps problems in a list (Phase 6 replaces it with the attention inbox)."""

    def __init__(self) -> None:
        self.problems: list[Problem] = []
        #: The problems reported as already spoken in the call (also in :attr:`problems`).
        self.spoken: list[Problem] = []
        self._lock = threading.Lock()

    def report(self, problem: Problem, *, spoken: bool = False) -> None:
        with self._lock:
            self.problems.append(problem)
            if spoken:
                self.spoken.append(problem)
        log.info("⚠️  problem: %s · %s", _detail(problem.kind), _detail(problem.subject))

    def resolve(self, kind: str, subject: str) -> bool:
        """Drop every kept problem ``kind`` · ``subject``; True when there was one."""
        with self._lock:
            kept = [p for p in self.problems if (p.kind, p.subject) != (kind, subject)]
            dropped = len(self.problems) - len(kept)
            self.problems[:] = kept
        if dropped:
            log.info("✅ problem resolved: %s · %s", _detail(kind), _detail(subject))
        return dropped > 0


class DeliveryTracker(Protocol):
    """Who follows a ccc delivery after the voice sent it (Phase 6's attention monitor)."""

    def track_delivery(
        self, delivery_id: str, session_id: str, subject: str, outcome: str
    ) -> None: ...


class MemoryDeliveryTracker:
    """In-memory :class:`DeliveryTracker` (tests, and until the attention store is wired)."""

    def __init__(self) -> None:
        self.tracked: list[tuple[str, str, str, str]] = []

    def track_delivery(self, delivery_id: str, session_id: str, subject: str, outcome: str) -> None:
        self.tracked.append((delivery_id, session_id, subject, outcome))


class MemoryBriefingProvider:
    """A fixed briefing (or none), redacted and capped at 600 characters."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def briefing(self) -> str | None:
        return redact(self.text, BRIEFING_CAP) if self.text else None
