"""Transport-driven turn loop (R2-5/R3-2): run the pipeline over an :class:`AudioTransport`.

The local loop in :mod:`__main__` owns :mod:`sounddevice` directly. This module
drives the *same* pipeline stages — VAD-gated capture, end-of-turn analysis,
streaming STT, the streaming :class:`~my_stt_tts.brain.Brain`, and TTS — but
sources mic frames from a :class:`~my_stt_tts.transport.AudioTransport` and sinks
synthesized TTS PCM back through it. So a WebSocket satellite or the browser GUI
runs the exact production pipeline, just with the device boundary moved onto the
wire.

R3-2 makes the reply **full-duplex over the wire**: while TTS is being synthesized
and sunk to the transport, the inbound mic queue stays live and the same barge-in
machinery as the local loop (VAD + :class:`~my_stt_tts.interrupt.InterruptGate` +
AEC + :class:`~my_stt_tts.interrupt.InterruptPredictor`) runs on every mic frame.
On a confirmed interruption the outbound TTS frames AND the in-flight LLM stream
are cancelled and the captured audio is handed to the next turn — so a satellite /
browser user can interrupt, exactly like a local user.

The capture/transcribe/respond functions take an injectable VAD + analyzer +
transcriber so the whole loop is unit-testable with fakes (no mic, no model, no
network) — see ``tests/test_transport.py``.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .aec import make_echo_canceller
from .brain import Brain, LLMError
from .config import Config
from .denoise import make_denoiser
from .events import bus
from .interrupt import InterruptGate, frame_energy, make_interrupt_predictor
from .metrics import TelemetrySink, TurnMetrics, make_sink
from .stt import StreamingTranscriber, Transcriber
from .text import SentenceChunker, strip_non_spoken
from .transport import AudioTransport
from .tts import TTSRouter

log = logging.getLogger("my_stt_tts.net_loop")


@dataclass
class TransportResult:
    """Outcome of one transport-side reply (R3-2).

    ``interrupted`` is True if the remote user barged in; ``captured`` holds the
    echo-cancelled/denoised audio from the barge-in onward, ``spoken`` the text
    actually voiced before the interruption (to keep history honest)."""

    interrupted: bool = False
    captured: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    spoken: str = ""


class _MicSource:
    """A single, shared, non-blocking mic-frame reader over a transport (R3-2).

    Both the capture phase and the barge-in monitor during playout must read from
    the *same* inbound mic stream — but a transport exposes only one
    ``mic_frames()`` generator. This wraps that generator in a background thread
    feeding a small queue, so either phase can pull frames with a timeout (and the
    monitor can poll without blocking the playout). ``closed`` once the underlying
    source is exhausted (the client disconnected)."""

    def __init__(self, transport: AudioTransport) -> None:
        self._frames = transport.mic_frames()
        self._q: queue.Queue[Any] = queue.Queue(maxsize=1024)
        self._eof = object()
        self._done = threading.Event()
        self._reader = threading.Thread(target=self._run, daemon=True)
        self._reader.start()

    def _run(self) -> None:
        try:
            for frame in self._frames:
                self._q.put(np.asarray(frame, dtype=np.float32).ravel())
        finally:
            self._q.put(self._eof)
            self._done.set()

    def get(self, timeout: float = 0.1) -> np.ndarray | None:
        """Next mic frame, ``None`` on timeout, or marks closed on EOF."""
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._eof:
            self._done.set()
            return None
        return item  # type: ignore[no-any-return]

    @property
    def closed(self) -> bool:
        return self._done.is_set() and self._q.empty()


def capture_turn(
    transport: AudioTransport,
    cfg: Config,
    vad: Any,
    analyzer: Any,
    transcriber: Transcriber,
    *,
    max_frames: int | None = None,
    source: _MicSource | None = None,
    denoiser: Any = None,
) -> str:
    """Pull mic frames from ``transport`` until end-of-turn, then transcribe.

    Convenience wrapper over :func:`capture_turn_clip` that returns only the
    transcript (kept for back-compat with callers/tests that don't need the audio).
    """
    return capture_turn_clip(
        transport,
        cfg,
        vad,
        analyzer,
        transcriber,
        max_frames=max_frames,
        source=source,
        denoiser=denoiser,
    )[0]


def capture_turn_clip(
    transport: AudioTransport,
    cfg: Config,
    vad: Any,
    analyzer: Any,
    transcriber: Transcriber,
    *,
    max_frames: int | None = None,
    source: _MicSource | None = None,
    denoiser: Any = None,
) -> tuple[str, np.ndarray]:
    """Pull mic frames until end-of-turn; return ``(transcript, captured_clip)``.

    Drives the supplied VAD + :class:`~my_stt_tts.turn.TurnAnalyzer` exactly like
    the local capture loop and feeds a :class:`StreamingTranscriber` so partial
    transcripts are published to the bus during the turn (and the final at the
    end). ``max_frames`` bounds the loop for tests; in production the analyzer
    ends the turn. ``source`` shares one mic reader across capture + barge-in
    monitoring (R3-2); ``denoiser`` (R3-6) cleans each frame before VAD/STT.

    The (denoised) captured PCM is also accumulated and returned so the speaker-ID
    pipeline can embed the same audio (G7) — the local loops keep the clip too. On
    silence returns ``("", empty)``.
    """
    streamer = StreamingTranscriber(
        transcriber,
        cfg.sample_rate,
        partial_interval_ms=cfg.stt_partial_interval_ms,
        window_s=cfg.stt_window_s,
    )
    if denoiser is not None:
        denoiser.reset()
    analyzer.reset()
    spoke = False
    captured: list[np.ndarray] = []
    bus.state("recording")
    for seen, frame in enumerate(_iter_source(transport, source), start=1):
        arr = np.asarray(frame, dtype=np.float32).ravel()
        if denoiser is not None:
            arr = denoiser.process(arr)
        captured.append(arr)
        is_speech = vad.is_speech(arr)
        spoke = spoke or is_speech
        partial = streamer.feed(arr)
        if partial is not None:
            bus.transcript(partial, partial=True, source="live_audio")
        if analyzer.update(arr, is_speech):
            break
        if max_frames is not None and seen >= max_frames:
            break
    if not spoke:
        bus.state("idle")
        return "", np.zeros(0, dtype=np.float32)
    bus.state("stt")
    clip = np.concatenate(captured) if captured else np.zeros(0, dtype=np.float32)
    return str(streamer.final().text).strip(), clip


def _iter_source(transport: AudioTransport, source: _MicSource | None) -> Iterator[np.ndarray]:
    """Yield mic frames from a shared :class:`_MicSource` or directly from ``transport``."""
    if source is None:
        yield from transport.mic_frames()
        return
    while not source.closed:
        frame = source.get(timeout=0.1)
        if frame is not None:
            yield frame


def _set_speaker(brain: Brain, speaker_id: Any, clip: np.ndarray | None) -> None:
    """Resolve a captured clip to an enrolled name and set it on the brain (G7).

    Mirrors the local loop: gated + defensive. With no pipeline (disabled / no
    enrollment / no speechbrain) or no clip the speaker is ``None`` (shared guest
    bucket) and nothing is embedded. The identified name is published to the bus so
    a remote/browser UI can show who is talking."""
    name = speaker_id.identify(clip) if speaker_id is not None and clip is not None else None
    brain.set_speaker(name)
    bus.speaker(name)


def respond_over_transport(
    transport: AudioTransport,
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    text: str,
    *,
    source: _MicSource | None = None,
    vad: Any | None = None,
    denoiser: Any = None,
    sink: TelemetrySink | None = None,
    speaker_id: Any = None,
    clip: np.ndarray | None = None,
) -> TransportResult:
    """Stream the LLM reply for ``text``, synthesize each sentence, sink it to ``transport``.

    **Full-duplex (R3-2)** when ``cfg.barge_in`` is on and a live mic ``source`` +
    ``vad`` are supplied: the inbound mic stays live during TTS playout and a
    confirmed interruption (VAD + gate + AEC + predictor) cancels the outbound TTS
    and the in-flight LLM stream, returning the captured audio for the next turn.
    Otherwise it is half-duplex over the wire (the original R2-5 behaviour). The
    streamed text is also published to the bus for the GUI. On an LLM error it
    sends a short spoken apology, mirroring the local loop.

    ``speaker_id`` + ``clip`` (G7): when both are supplied the clip is embedded and
    resolved to an enrolled name before streaming, so memory is per-person — the
    same wiring as the local loop, just over the wire. The barge-in chaining in
    :func:`run_transport_session` passes each follow-up's captured clip so a
    different remote speaker is re-identified.
    """
    _set_speaker(brain, speaker_id, clip)
    metrics = TurnMetrics()
    metrics.note(transcript=text, transport=True)
    voiced_any = False
    # Turns over the wire (browser-audio / satellite) are live-mic audio.
    bus.transcript(text, source="live_audio")
    chunker = SentenceChunker()
    barge = (
        _TransportBargeIn.build(cfg, source, vad, denoiser)
        if (cfg.barge_in != "off" and source is not None and vad is not None)
        else None
    )
    result = TransportResult()
    spoken: list[str] = []
    stream = brain.stream(text)
    try:
        bus.state("llm_response")
        first = True
        for delta in stream:
            if first:
                metrics.mark("llm_first_token")  # R3-7
                first = False
            bus.response(delta, final=False)
            for sentence in chunker.feed(delta):
                voiced = _voice_to_transport(transport, tts, sentence, barge)
                spoken.append(voiced)
                if voiced and not voiced_any:
                    metrics.mark("first_audio")  # R3-7
                    voiced_any = True
                if barge is not None and barge.interrupted:
                    break
            if barge is not None and barge.interrupted:
                break
        else:
            tail = chunker.flush()
            if tail:
                voiced = _voice_to_transport(transport, tts, tail, barge)
                spoken.append(voiced)
                if voiced and not voiced_any:
                    metrics.mark("first_audio")
                    voiced_any = True
        bus.response("", final=True)
    except LLMError as exc:
        log.error("LLM error over transport: %s", exc)
        bus.log(str(exc), "error")
        _voice_to_transport(transport, tts, "Sorry, I had a problem.", None)
    finally:
        with contextlib.suppress(Exception):  # best-effort cancel of in-flight tokens
            stream.close()
        if barge is not None and barge.interrupted:
            result.interrupted = True
            result.captured = barge.captured
            voiced = "".join(spoken)
            result.spoken = voiced
            brain.commit_spoken(voiced)
            bus.interrupted(len(voiced))
            bus.interrupt_start()
    result.spoken = result.spoken or "".join(spoken)
    bus.state("idle")
    metrics.emit(sink)  # R3-7: per-turn latency telemetry over the wire
    return result


def _voice_to_transport(
    transport: AudioTransport,
    tts: TTSRouter,
    sentence: str,
    barge: _TransportBargeIn | None,
) -> str:
    """Synthesize one sentence to PCM, forward it, and monitor for barge-in (R3-2).

    With streamed TTS the sentence is synthesized clause-by-clause; each PCM chunk
    is sunk to the transport and — when a barge-in monitor is supplied — the live
    mic is polled between chunks so a confirmed interruption stops the send mid-
    sentence. Returns the chars voiced (empty if blank or cut off before audio)."""
    text = strip_non_spoken(sentence)
    if not text:
        return ""
    bus.state("speaking")
    if barge is None:
        pcm, sr = tts.synth_pcm(text)
        if pcm.size:
            transport.send_tts(pcm, sr)
        return sentence
    if barge.interrupted:
        return ""
    sent_any = False
    for pcm, sr in tts.synth_pcm_stream(text):
        if barge.poll():  # confirmed interruption -> stop sinking this sentence
            return sentence if sent_any else ""
        if pcm.size:
            transport.send_tts(pcm, sr)
            barge.note_reference(pcm, sr)
            sent_any = True
        # Drain a slice of mic frames per chunk so playout stays full-duplex.
        if barge.monitor_window():
            return sentence
    return sentence


@dataclass
class _TransportBargeIn:
    """Barge-in monitor for the wire (R3-2): VAD + gate + AEC + predictor on the mic.

    Mirrors :func:`~my_stt_tts.audio.monitor_during_playback` but over a transport's
    shared :class:`_MicSource` instead of a local ``InputStream``. :meth:`poll`
    /:meth:`monitor_window` consume buffered mic frames, run the echo canceller +
    VAD + the false-interrupt gate + the acoustic predictor, and latch
    :attr:`interrupted` (with the captured speech) the moment an interruption is
    confirmed. Reference TTS PCM is fed via :meth:`note_reference` so the AEC can
    subtract the assistant's own voice."""

    source: _MicSource
    vad: Any
    gate: InterruptGate
    sample_rate: int
    energy_floor: float
    aec: Any = None
    predictor: Any = None
    denoiser: Any = None
    interrupted: bool = field(default=False, init=False)
    captured: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), init=False)
    _buf: list[np.ndarray] = field(default_factory=list, init=False)

    @classmethod
    def build(cls, cfg: Config, source: Any, vad: Any, denoiser: Any) -> _TransportBargeIn:
        """Construct the monitor with config-derived gate/AEC/predictor (R3-2)."""
        frame_ms = 512 / cfg.sample_rate * 1000.0
        gate = InterruptGate(
            min_speech_ms=cfg.interrupt_min_speech_ms,
            min_words=cfg.interrupt_min_words,
            frame_ms=frame_ms,
        )
        monitor = cls(
            source=source,
            vad=vad,
            gate=gate,
            sample_rate=cfg.sample_rate,
            energy_floor=cfg.barge_in_energy,
            aec=make_echo_canceller(cfg),
            predictor=make_interrupt_predictor(cfg, frame_ms),
            denoiser=denoiser,
        )
        gate.reset()
        if monitor.aec is not None:
            monitor.aec.reset()
        if monitor.predictor is not None:
            monitor.predictor.reset()
        return monitor

    def note_reference(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Feed sunk TTS PCM to the AEC reference (resampled to capture rate)."""
        if self.aec is None:
            return
        ref = np.asarray(pcm, dtype=np.float32).ravel()
        if sample_rate != self.sample_rate and ref.size:
            idx = (
                np.arange(int(ref.size * self.sample_rate / sample_rate))
                * sample_rate
                / self.sample_rate
            ).astype(int)
            idx = idx[idx < ref.size]
            ref = ref[idx]
        with contextlib.suppress(Exception):
            self.aec.push_reference(ref)

    def _floor(self) -> float:
        return (
            0.0
            if (self.aec is not None and getattr(self.aec, "active", False))
            else (self.energy_floor)
        )

    def _consume(self, frame: np.ndarray) -> bool:
        """Run one mic frame through the chain; latch + return True on interruption."""
        clean = self.aec.process(frame) if self.aec is not None else frame
        if self.denoiser is not None:
            clean = self.denoiser.process(clean)
        loud_enough = frame_energy(clean) >= self._floor()
        is_speech = bool(loud_enough and self.vad.is_speech(clean))
        if is_speech or self._buf:
            self._buf.append(clean)
        predicted = self.predictor.update(clean, is_speech) if self.predictor is not None else False
        if self.gate.update(is_speech) or predicted:
            self.interrupted = True
            self.captured = (
                np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
            )
            return True
        return False

    def poll(self) -> bool:
        """Drain one buffered mic frame (if any) without blocking; latch on interrupt."""
        if self.interrupted:
            return True
        frame = self.source.get(timeout=0.0)
        if frame is not None and self._consume(frame):
            return True
        return self.interrupted

    def monitor_window(self, *, max_frames: int = 8) -> bool:
        """Drain up to ``max_frames`` buffered mic frames; True if interrupted."""
        if self.interrupted:
            return True
        for _ in range(max_frames):
            frame = self.source.get(timeout=0.0)
            if frame is None:
                break
            if self._consume(frame):
                return True
        return self.interrupted


def run_transport_session(
    transport: AudioTransport,
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    transcriber: Transcriber,
    *,
    vad: Any | None = None,
    speaker_id: Any = None,
) -> None:
    """Run turns over ``transport`` until its mic source ends (one satellite session).

    Builds the VAD + turn analyzer from config (lazily, so tests can inject), then
    loops: capture a turn, respond over the wire. Full-duplex barge-in (R3-2) is
    armed when ``cfg.barge_in`` is on — the reply is interruptible and the captured
    audio seeds the next turn. ``speaker_id`` (G7), when supplied, identifies the
    captured clip before each reply so memory is per-person over the wire. Ends when
    ``mic_frames`` is exhausted (the client disconnected). Used as the ``on_session``
    callback by the WebSocket server.
    """
    if vad is None:
        from .vad import SileroVad

        vad = SileroVad(cfg.sample_rate, cfg.vad_threshold)
    from .turn import make_turn_analyzer

    frame_seconds = 512 / cfg.sample_rate
    barge_on = cfg.barge_in != "off"
    source = _MicSource(transport)
    denoiser = make_denoiser(cfg)
    sink = make_sink(cfg)  # R3-7: one telemetry sink per session (aggregates turns)
    bus.state("idle")
    while not _transport_closed(transport) and not source.closed:
        analyzer = make_turn_analyzer(cfg, frame_seconds)
        text, clip = capture_turn_clip(
            transport, cfg, vad, analyzer, transcriber, source=source, denoiser=denoiser
        )
        if not text:
            break  # silence / client gone
        result = respond_over_transport(
            transport,
            cfg,
            brain,
            tts,
            text,
            source=source if barge_on else None,
            vad=vad if barge_on else None,
            denoiser=denoiser,
            sink=sink,
            speaker_id=speaker_id,
            clip=clip,
        )
        # Chain barge-in follow-ups: the captured audio becomes the next turn (and is
        # re-identified — the interrupter may be a different person) (G7).
        while result.interrupted and result.captured.size and not source.closed:
            barge_clip = result.captured
            follow = _transcribe_captured(cfg, transcriber, result.captured)
            bus.interrupt_stop()
            if not follow:
                break
            result = respond_over_transport(
                transport,
                cfg,
                brain,
                tts,
                follow,
                source=source if barge_on else None,
                vad=vad if barge_on else None,
                denoiser=denoiser,
                sink=sink,
                speaker_id=speaker_id,
                clip=barge_clip,
            )


def _transcribe_captured(cfg: Config, transcriber: Transcriber, clip: np.ndarray) -> str:
    """Transcribe a barge-in clip as the next turn's text (no from-scratch re-record)."""
    if clip.size == 0:
        return ""
    streamer = StreamingTranscriber(
        transcriber,
        cfg.sample_rate,
        partial_interval_ms=cfg.stt_partial_interval_ms,
        window_s=cfg.stt_window_s,
    )
    streamer.feed_clip(clip)
    return str(streamer.final().text).strip()


def _transport_closed(transport: AudioTransport) -> bool:
    return bool(getattr(transport, "closed", False))


def mic_frame_chunks(pcm: np.ndarray, frame_samples: int = 512) -> Iterator[np.ndarray]:
    """Split a PCM array into fixed-size frames (used by the satellite client)."""
    arr = np.asarray(pcm, dtype=np.float32).ravel()
    for start in range(0, arr.size, frame_samples):
        yield arr[start : start + frame_samples]
