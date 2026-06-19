"""Per-turn latency + transcript telemetry, keyed by a shared ``speech_id`` (R3-7).

Measuring latency is the prerequisite for tuning "responsiveness". Each
conversational turn gets a :class:`TurnMetrics` that times the key stages —
``stt`` (transcription), ``llm_first_token`` (time to the model's first delta),
``tts`` (synthesis), ``first_audio`` (time-to-first-audio, the headline number) —
all tied together by one ``speech_id`` so an end-to-end turn can be reconstructed
from the logs. When the turn finishes it is emitted three ways:

* to :data:`events.bus` as a ``metrics`` event (the live web UI can render it),
* to a structured **JSON-lines** log via :class:`MetricsLog` (one record per
  line, machine-parseable, off unless a path is configured), and
* optionally to an **OpenTelemetry** span (lazy import, OFF by default).

:class:`MetricsAggregator` rolls many turns into count / mean / p50 / p95 per
stage for a session summary. The clock is **injectable** (``clock=`` defaults to
:func:`time.perf_counter`) so the whole module is testable with a fake clock —
no real wall-clock, no sleeps, deterministic assertions.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger("my_stt_tts.metrics")

_counter = itertools.count(1)

# The canonical per-turn stages, in pipeline order. Free-form stage names are
# allowed too (recorded as-is), but these are the ones the aggregator/UI expect.
STAGES = ("stt", "llm_first_token", "llm", "tts", "first_audio")

Clock = Callable[[], float]


def next_speech_id() -> str:
    """Return a process-unique, monotonically increasing turn id."""
    return f"turn-{next(_counter):05d}"


@dataclass(slots=True)
class TurnMetrics:
    """Per-stage durations (ms) and notes for one conversational turn (R3-7).

    The ``speech_id`` ties every stage's timing together so an end-to-end turn
    can be reconstructed from the logs. Time stages either by wrapping them in
    :meth:`stage` (a context manager) or by calling :meth:`mark` once per stage
    boundary (handy for streaming, where the "first token" / "first audio"
    instants are points, not spans). The ``clock`` is injectable for tests.
    """

    speech_id: str = field(default_factory=next_speech_id)
    stages: dict[str, float] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)
    clock: Clock = time.perf_counter
    _t0: float = field(default=0.0)

    def __post_init__(self) -> None:
        self._t0 = self.clock()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a stage; record elapsed milliseconds under ``name``."""
        start = self.clock()
        try:
            yield
        finally:
            self.stages[name] = round((self.clock() - start) * 1000.0, 1)

    def mark(self, name: str) -> float:
        """Record the time *since the turn started* (ms) as stage ``name``.

        For point-in-time milestones in a streaming turn — e.g. the LLM's first
        token or the first audio sample — where what matters is the latency from
        the start of the turn, not the duration of a span. Returns the value.
        """
        elapsed = round((self.clock() - self._t0) * 1000.0, 1)
        self.stages[name] = elapsed
        return elapsed

    def note(self, **kwargs: object) -> None:
        """Attach free-form notes (transcript, speaker, language, …)."""
        self.notes.update(kwargs)

    def total_ms(self) -> float:
        """Wall-clock milliseconds since this turn was created."""
        return round((self.clock() - self._t0) * 1000.0, 1)

    def as_dict(self) -> dict[str, object]:
        """Flat dict for logging / inspection."""
        return {
            "speech_id": self.speech_id,
            "total_ms": self.total_ms(),
            "stages_ms": dict(self.stages),
            **self.notes,
        }

    def log(self) -> None:
        """Emit one structured info-level log line for this turn."""
        log.info("turn %s", json.dumps(self.as_dict(), ensure_ascii=False))

    def emit(self, sink: TelemetrySink | None = None) -> dict[str, object]:
        """Finalize the turn: log it, publish to the bus, and feed ``sink``.

        Returns the record dict (also handy for tests). When a ``sink`` is given
        (built from config by :func:`make_sink`) the record is appended to the
        JSON-lines file, the aggregator, and the optional OpenTelemetry span.
        Always cheap + non-throwing: a sink/IO failure never breaks the turn.
        """
        record = self.as_dict()
        self.log()
        _publish_bus(record)
        if sink is not None:
            sink.record(self)
        return record


def _publish_bus(record: dict[str, object]) -> None:
    """Best-effort publish a metrics record to the shared event bus."""
    try:
        from .events import bus

        bus.publish({"type": "metrics", **record})
    except Exception:  # the bus must never break a turn  # noqa: BLE001
        log.debug("metrics bus publish failed", exc_info=True)


# --- structured JSON-lines log -------------------------------------------------


class MetricsLog:
    """Append per-turn metrics records to a JSON-lines file (one record per line).

    Thread-safe (turns can be emitted from the network loop's worker thread and
    the local loop concurrently). Each line is a self-contained JSON object so the
    file is trivially parseable (``jq -c`` / pandas ``read_json(lines=True)``). A
    write failure is logged and swallowed — telemetry never takes down the loop.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            log.warning("metrics log write failed (%s)", self.path, exc_info=True)


# --- aggregation ---------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of ``values`` (``pct`` in [0, 100]); 0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 1)


class MetricsAggregator:
    """Roll many :class:`TurnMetrics` into count / mean / p50 / p95 per stage (R3-7).

    Records each turn's per-stage durations and, on demand, summarizes them so a
    session can report e.g. "median first-audio latency 312 ms, p95 540 ms". Pure
    + thread-safe; no clock, no IO — just arithmetic over the recorded numbers.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_stage: dict[str, list[float]] = {}
        self._count = 0

    def add(self, turn: TurnMetrics) -> None:
        """Fold one turn's stage timings into the running aggregate."""
        with self._lock:
            self._count += 1
            for name, ms in turn.stages.items():
                self._by_stage.setdefault(name, []).append(float(ms))

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-stage ``{count, mean, p50, p95, min, max}`` over all recorded turns."""
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for name, values in self._by_stage.items():
                if not values:
                    continue
                out[name] = {
                    "count": float(len(values)),
                    "mean": round(sum(values) / len(values), 1),
                    "p50": _percentile(values, 50),
                    "p95": _percentile(values, 95),
                    "min": round(min(values), 1),
                    "max": round(max(values), 1),
                }
            return out


# --- optional OpenTelemetry span hook (lazy, OFF by default) -------------------


def _otel_span(record: dict[str, object]) -> None:
    """Emit one OpenTelemetry span for a turn, if the SDK is importable (R3-7).

    Lazy-imported and best-effort: when ``opentelemetry`` is not installed (the
    common case — it is OFF by default) this is a clean no-op. Each per-stage
    duration becomes a span attribute so a turn shows up as a single span in any
    OTLP-compatible backend without coupling the core package to the SDK.
    """
    try:
        from opentelemetry import trace
    except Exception:  # SDK absent / misconfigured -> silently skip  # noqa: BLE001
        log.debug("opentelemetry not available; skipping span", exc_info=True)
        return
    tracer = trace.get_tracer("my_stt_tts.metrics")
    speech_id = str(record.get("speech_id", "turn"))
    with tracer.start_as_current_span(f"turn:{speech_id}") as span:
        span.set_attribute("speech_id", speech_id)
        total = record.get("total_ms", 0.0)
        span.set_attribute("total_ms", float(total) if isinstance(total, (int, float)) else 0.0)
        stages = record.get("stages_ms", {})
        if isinstance(stages, dict):
            for name, ms in stages.items():
                if isinstance(ms, (int, float)):
                    span.set_attribute(f"stage.{name}_ms", float(ms))


# --- the wired-up sink ---------------------------------------------------------


class TelemetrySink:
    """Bundles the JSON-lines log, the aggregator, and the optional OTel span.

    A single object the loop hands each finished :class:`TurnMetrics` to via
    :meth:`record`; it fans the turn out to whichever destinations are enabled.
    Built from :class:`~my_stt_tts.config.Config` by :func:`make_sink` (returns
    ``None`` when telemetry is disabled, so the loop can pass ``None`` cheaply).
    """

    def __init__(self, *, log_file: str | Path | None = None, otel: bool = False) -> None:
        self.aggregator = MetricsAggregator()
        self._jsonl = MetricsLog(log_file) if log_file else None
        self._otel = otel

    def record(self, turn: TurnMetrics) -> None:
        """Fan one finished turn out to the file, the aggregator, and OTel."""
        record = turn.as_dict()
        self.aggregator.add(turn)
        if self._jsonl is not None:
            self._jsonl.write(record)
        if self._otel:
            _otel_span(record)

    def summary(self) -> dict[str, dict[str, float]]:
        """Aggregated per-stage stats over every turn recorded so far."""
        return self.aggregator.summary()


def make_sink(cfg: Config) -> TelemetrySink | None:
    """Build a :class:`TelemetrySink` from config, or ``None`` when telemetry is off.

    Telemetry is opt-in (``cfg.telemetry``): the headline log line + the ``bus``
    metrics event are always emitted by :meth:`TurnMetrics.emit`, but the
    JSON-lines file, the session aggregator, and the OpenTelemetry span only spin
    up when telemetry is enabled — keeping the default path allocation-free.
    """
    if not getattr(cfg, "telemetry", False):
        return None
    return TelemetrySink(
        log_file=getattr(cfg, "telemetry_log_file", None) or None,
        otel=getattr(cfg, "telemetry_otel", False),
    )
