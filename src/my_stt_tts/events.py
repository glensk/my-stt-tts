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
import sys
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

# Env var forcing the stderr CONSOLE sink on/off (see ``EventBus._write_console``).
# The console sink mirrors every published bus event to stderr as a concise human
# one-liner so quickstart.sh's stderr tee (logs/quickstart-<ts>.log) captures the
# SAME events the GUI EVENT LOG shows. Default: ON whenever the file sink is active
# (MSTT_EVENT_LOG set, which quickstart sets) so quickstart runs get it for free;
# library / test use stays silent. Set MSTT_EVENT_CONSOLE to "1"/"true"/"yes"/"on"
# to force on regardless of the file sink, or "0"/"false"/"no"/"off" to force off.
_EVENT_CONSOLE_ENV = "MSTT_EVENT_CONSOLE"

# A truthy ``_log_bridge`` field on an event marks it as having ORIGINATED from the
# Python logging library (via the LogBusHandler bridge, sink B): it is already on
# stderr through the root/library log handlers, so the console sink (A) must SKIP it
# to avoid a double-print. The field is internal plumbing (it still rides the wire
# + file sink, harmlessly) — the console sink is the only consumer that branches on it.
_LOG_BRIDGE_FIELD = "_log_bridge"


def _env_flag(name: str) -> bool | None:
    """Parse an env var as an explicit on/off flag, or ``None`` when unset/blank.

    ``"1"/"true"/"yes"/"on"`` -> True; ``"0"/"false"/"no"/"off"`` -> False (case-
    insensitive); anything else (incl. unset / empty) -> ``None`` (no opinion, fall
    back to the default policy). Used to gate the console sink via MSTT_EVENT_CONSOLE.
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def render_console_line(event: dict[str, Any]) -> str:
    """Render an event dict as a concise one-line human string for the stderr sink.

    Mirrors what the GUI EVENT LOG shows so the terminal log is readable: e.g.
    ``[event:state] listening``, ``[event:response] hello`` (final responses only),
    ``[event:music] playing "Song"``. Falls back to a compact key=value dump for
    types without a bespoke formatter so nothing is silently lost.
    """
    etype = str(event.get("type", "?"))
    body = _CONSOLE_FORMATTERS.get(etype, _console_default)(event)
    return f"[event:{etype}] {body}".rstrip()


def _console_default(event: dict[str, Any]) -> str:
    """Compact ``key=value`` dump for event types without a bespoke formatter."""
    skip = {"type", _LOG_BRIDGE_FIELD}
    return " ".join(f"{k}={v}" for k, v in event.items() if k not in skip)


def _fmt_state(e: dict[str, Any]) -> str:
    state = str(e.get("state", ""))
    detail = str(e.get("detail", "")).strip()
    return f"{state} ({detail})" if detail else state


def _fmt_transcript(e: dict[str, Any]) -> str:
    tag = "partial" if e.get("partial") else "final"
    src = str(e.get("source", "")).strip()
    head = f"{tag}:{src}" if src else tag
    return f"[{head}] {e.get('text', '')}"


def _fmt_response(e: dict[str, Any]) -> str:
    tag = "final" if e.get("final") else "delta"
    model = str(e.get("model", "")).strip()
    head = f"{tag}:{model}" if model else tag
    return f"[{head}] {e.get('text', '')}"


def _fmt_music(e: dict[str, Any]) -> str:
    status = str(e.get("status", ""))
    title = str(e.get("title", "")).strip()
    return f'{status} "{title}"' if title else status


def _fmt_log(e: dict[str, Any]) -> str:
    return f"{e.get('level', 'info')}: {e.get('message', '')}"


def _fmt_speaker(e: dict[str, Any]) -> str:
    name = str(e.get("name", "")).strip()
    return name if name else "(guest)"


def _fmt_mic_result(e: dict[str, Any]) -> str:
    return f"{e.get('verdict', '')} level={e.get('level', 0)} — {e.get('message', '')}".rstrip()


def _fmt_wake_test_result(e: dict[str, Any]) -> str:
    fired = "FIRED" if e.get("fired") else "no-fire"
    return f"{e.get('word', '')} {fired} conf={e.get('confidence', 0)} ({e.get('source', '')})"


def _fmt_metrics(e: dict[str, Any]) -> str:
    return f"speech_id={e.get('speech_id', '')} stages_ms={e.get('stages_ms', {})}"


def _fmt_wake(_e: dict[str, Any]) -> str:
    return "fired"


# Per-type one-line renderers for the stderr console sink (everything else falls
# back to ``_console_default``). ``debug`` is intentionally absent — those events
# are skipped wholesale (already printed to stderr by ``_AudioDebug``).
_CONSOLE_FORMATTERS = {
    "state": _fmt_state,
    "transcript": _fmt_transcript,
    "response": _fmt_response,
    "music": _fmt_music,
    "log": _fmt_log,
    "speaker": _fmt_speaker,
    "mic_result": _fmt_mic_result,
    "wake_test_result": _fmt_wake_test_result,
    "metrics": _fmt_metrics,
    "wake": _fmt_wake,
}

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
        self._console_lock = threading.Lock()

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

    def _console_enabled(self) -> bool:
        """Whether the stderr console sink (A) is active for this run.

        Explicit ``MSTT_EVENT_CONSOLE`` wins (on/off); otherwise default to ON
        whenever the file sink is active (``MSTT_EVENT_LOG`` set / a sink attached),
        so quickstart runs mirror every EVENT-LOG event to the stderr tee for free
        while library / test use (no env, no sink) stays silent. Read every publish
        (cheap) so a late ``attach_file_sink`` / env change takes effect."""
        flag = _env_flag(_EVENT_CONSOLE_ENV)
        if flag is not None:
            return flag
        return self._sink is not None or bool(os.environ.get(_EVENT_LOG_ENV, "").strip())

    def _write_console(self, event: dict[str, Any]) -> None:
        """Mirror one event to stderr as a concise human one-liner (sink A).

        SKIPS ``debug`` events (``_AudioDebug`` already prints those to stderr) and
        events tagged ``_log_bridge`` (already on stderr via the logging library, see
        sink B) so nothing double-prints. Thread-safe + failure-suppressed like the
        file sink — the console mirror must never break the bus."""
        if event.get("type") == "debug" or event.get(_LOG_BRIDGE_FIELD):
            return
        if not self._console_enabled():
            return
        line = render_console_line(event)
        with self._console_lock, contextlib.suppress(Exception):
            print(line, file=sys.stderr)

    def publish(self, event: dict[str, Any]) -> None:
        """Classify + fan out an event dict to all subscribers (G2 priority rules)."""
        if event.get("type") == "state":
            self._last_state = json.dumps(event, ensure_ascii=False)
        self._write_sink(event)  # may lazily attach the file sink -> gates the console sink
        self._write_console(event)
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

    def mic_check_result(
        self,
        *,
        source: str,
        peak: float,
        level: int,
        rms: float,
        duration_s: float,
        sample_rate: int,
        levels: list[float],
        processing: dict[str, Any],
        hash: str,  # noqa: A002 — wire field name per the shared GUI contract
        wav_url: str,
        message: str,
    ) -> None:
        """Publish the outcome of a unified 2.0 s microphone check (GUI "mic check").

        DATA priority — the GUI shows the live level meter + a level-over-time graph
        and links the saved WAV. ``source`` is ``"server"`` (the server mic, after the
        software ``mic_gain``) or ``"browser"`` (a clip the page recorded + POSTed).
        ``peak`` / ``rms`` are the raw 0..1 amplitudes; ``level`` is ``round(peak*100)``
        (0..100); ``levels`` is ~48 evenly-spaced per-window peak magnitudes (0..1) for
        the graph; ``processing`` carries ``{agc, ns, ec, gain}`` (browser echo/AGC/NS
        flags or null + the applied server ``gain``); ``hash`` is the 8-hex content id
        and ``wav_url`` the ``/recordings/<file>.wav`` link to the saved clip.
        """
        self.publish(
            {
                "type": "mic_check_result",
                "source": source,
                "peak": round(float(peak), 4),
                "level": int(level),
                "rms": round(float(rms), 4),
                "duration_s": round(float(duration_s), 3),
                "sample_rate": int(sample_rate),
                "levels": [round(float(v), 4) for v in levels],
                "processing": processing,
                "hash": hash,
                "wav_url": wav_url,
                "message": message,
            }
        )

    def music(
        self,
        status: str,
        *,
        title: str = "",
        video_id: str = "",
        url: str = "",
    ) -> None:
        """Publish a structured music-playback event for the GUI (DATA priority).

        ``status`` is one of ``"playing"`` / ``"stopped"`` / ``"paused"`` /
        ``"resumed"``. ``title`` names the track; ``video_id`` is the 11-char
        YouTube id and ``url`` the page URL so the GUI can embed / link the video.
        Emitted alongside the spoken confirmation whenever the intent router or a
        GUI button drives playback, so the page can show + control what's playing.
        """
        self.publish(
            {
                "type": "music",
                "status": status,
                "title": title,
                "video_id": video_id,
                "url": url,
            }
        )

    def wake_test_result(
        self,
        *,
        word: str,
        source: str,
        confidence: float,
        fired: bool,
        message: str,
        wav_path: str = "",
        peak: float = 0.0,
        level: int = 0,
        rms: float = 0.0,
        duration_s: float = 0.0,
        sample_rate: int = 16000,
        levels: list[float] | None = None,
        processing: dict[str, Any] | None = None,
        hash: str = "",  # noqa: A002 — wire field name per the shared GUI contract
        wav_url: str = "",
    ) -> None:
        """Publish the outcome of a wake-word test (GUI "Wake test"); DATA priority.

        Fired when the user diagnoses whether a wake word would trigger on a ~2 s
        clip recorded either by the SERVER mic (``source="server"``) or supplied by
        the BROWSER (``source="browser"``). ``confidence`` is the max openWakeWord
        score over the clip (0..1); ``fired`` is whether it cleared the threshold;
        ``message`` is the human one-liner the UI shows; ``wav_path`` is the legacy
        on-disk path (kept for back-compat). The remaining fields mirror
        :meth:`mic_check_result` so the GUI shows the SAME level meter + graph +
        saved-WAV link for a wake test: ``peak``/``rms`` (0..1), ``level``
        (``round(peak*100)``), ``levels`` (~48 per-window peaks), ``duration_s`` /
        ``sample_rate``, ``processing`` (``{agc, ns, ec, gain}``), the ``hash`` content
        id, and the ``wav_url`` ``/recordings/<file>.wav`` link. Mirrors the live wake
        path so a never-firing word is diagnosable.
        """
        self.publish(
            {
                "type": "wake_test_result",
                "word": word,
                "source": source,
                "confidence": confidence,
                "fired": fired,
                "message": message,
                "wav_path": wav_path,
                "peak": round(float(peak), 4),
                "level": int(level),
                "rms": round(float(rms), 4),
                "duration_s": round(float(duration_s), 3),
                "sample_rate": int(sample_rate),
                "levels": [round(float(v), 4) for v in (levels or [])],
                "processing": processing or {},
                "hash": hash,
                "wav_url": wav_url,
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


# --- Sink B: Python logging -> bus bridge ----------------------------------------
# Library output that currently only reaches stderr — the onnxruntime
# CUDAExecutionProvider UserWarning, Hugging Face "unauthenticated requests" notes,
# httpx per-request lines, and the app's own ``logging`` — never appears in the GUI
# EVENT LOG. ``LogBusHandler`` republishes each LogRecord as a bus ``log`` event so
# the EVENT LOG becomes a superset of those messages, while the ``_log_bridge`` tag
# keeps the stderr console sink (A) from re-printing them (they're already on stderr
# via the root/library handlers).

# Logger-name prefix for the app's own loggers — bridged at INFO+ (everything else
# only at WARNING+, see ``_BusBridgeFilter`` / ``install_log_bridge``).
_APP_LOGGER_PREFIX = "my_stt_tts"

# Third-party loggers whose INFO stream is too chatty for the EVENT LOG: their INFO
# is NOT mirrored into the bus (only their WARNING+). This is the SINGLE intentional
# divergence between the two streams — httpx's per-request "HTTP Request: POST ... 200
# OK" INFO lines stay on stderr (raw quickstart.log via the root handler) but are kept
# OUT of the EVENT LOG to keep it readable. We do NOT lower these loggers' own level,
# so the lines still reach stderr — only the bridge filter drops them. Documented in PLAN.md.
_NOISY_INFO_LOGGERS = ("httpx", "httpcore")


class _BusBridgeFilter(logging.Filter):
    """Per-handler filter implementing the bridge's level policy (sink B).

    Pass a record onto the bus when EITHER it is WARNING+ (from *any* logger) OR it is
    INFO+ from one of the app's own ``my_stt_tts.*`` loggers — EXCEPT records from the
    chatty third-party INFO loggers below WARNING (``httpx`` / ``httpcore``), which are
    dropped from the bus while still reaching stderr (the one intentional divergence).
    Records the bridge itself produces are tagged ``my_stt_tts.events`` WARNING and are
    handled by the handler's own re-entrancy guard, not here.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno < logging.INFO:
            return False
        name = record.name
        if any(name == n or name.startswith(n + ".") for n in _NOISY_INFO_LOGGERS):
            return False  # chatty third-party INFO: stderr only, kept out of the EVENT LOG
        return name == _APP_LOGGER_PREFIX or name.startswith(_APP_LOGGER_PREFIX + ".")


class LogBusHandler(logging.Handler):
    """A ``logging.Handler`` that republishes each record onto the event bus (sink B).

    Each emitted record becomes ``bus.publish({"type":"log","level":<lvl>,
    "message":<msg>,"_log_bridge":True})`` so library/app logs + captured warnings
    show up in the GUI EVENT LOG / file sink. The ``_log_bridge`` tag stops sink (A)
    from re-printing it to stderr (already there via the library handlers).

    Recursion guard: events the bus emits while handling a record (the file/console
    sinks, or a subscriber) must NOT loop back through this handler. A thread-local
    re-entrancy flag drops any record produced while we are already publishing, and
    every failure is suppressed — a logging bridge must never raise into the caller
    or wedge the logging machinery.
    """

    def __init__(self, target: EventBus) -> None:
        super().__init__()
        self._bus = target
        self._busy = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._busy, "active", False):
            return  # re-entrant: a record produced while we publish -> drop, no loop
        self._busy.active = True
        try:
            self._bus.publish(
                {
                    "type": "log",
                    "level": record.levelname.lower(),
                    "message": record.getMessage(),
                    _LOG_BRIDGE_FIELD: True,
                }
            )
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            # Never let a logging bridge raise into the caller / logging machinery.
            with contextlib.suppress(Exception):
                self.handleError(record)
        finally:
            self._busy.active = False


_log_bridge_installed = False
_log_bridge_lock = threading.Lock()


def install_log_bridge(target: EventBus | None = None) -> LogBusHandler | None:
    """Install the logging->bus bridge ONCE (idempotent); return the handler (B).

    Wires :class:`LogBusHandler` onto the root logger so library + app logs flow into
    the bus, and calls :func:`logging.captureWarnings` so ``warnings.warn(...)``
    (onnxruntime's CUDAExecutionProvider note, Hugging Face's unauthenticated-request
    warning, …) becomes ``py.warnings`` log records that are bridged too.

    Level policy (the streams' agreed contract), enforced by :class:`_BusBridgeFilter`:
      * WARNING+ from *every* logger is bridged into the EVENT LOG;
      * INFO+ from the app's own ``my_stt_tts.*`` loggers is bridged too;
      * chatty third-party INFO (``httpx`` / ``httpcore`` per-request lines) is the
        SINGLE intentional divergence — kept on stderr (raw quickstart.log) but OUT of
        the EVENT LOG. To see those records arrive the root logger must pass INFO, so
        we lower it to INFO (without lowering it below an existing DEBUG setting).

    Idempotent: a second call is a no-op (returns the existing handler). Called once
    at app startup from the run/browser entrypoint; library/test imports never wire it.
    """
    global _log_bridge_installed  # pylint: disable=global-statement
    sink = target if target is not None else bus
    with _log_bridge_lock:
        if _log_bridge_installed:
            return next(
                (h for h in logging.getLogger().handlers if isinstance(h, LogBusHandler)),
                None,
            )
        handler = LogBusHandler(sink)
        handler.setLevel(logging.INFO)  # INFO+ reaches the handler; the filter narrows it
        handler.addFilter(_BusBridgeFilter())
        root = logging.getLogger()
        # INFO records must reach the handler: drop the root threshold to INFO unless it
        # is already more verbose (DEBUG via --debug). basicConfig sets it to INFO/DEBUG.
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        # The app's own loggers surface INFO (basicConfig already lets root pass it, but
        # be explicit so an app INFO line is captured regardless of root threshold).
        logging.getLogger(_APP_LOGGER_PREFIX).setLevel(logging.INFO)
        root.addHandler(handler)
        logging.captureWarnings(True)  # warnings.warn(...) -> 'py.warnings' log records
        _log_bridge_installed = True
        return handler
