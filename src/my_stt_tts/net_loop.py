"""Transport-driven turn loop (R2-5): run the pipeline over an :class:`AudioTransport`.

The local loop in :mod:`__main__` owns :mod:`sounddevice` directly. This module
drives the *same* pipeline stages — VAD-gated capture, end-of-turn analysis,
streaming STT, the streaming :class:`~my_stt_tts.brain.Brain`, and TTS — but
sources mic frames from a :class:`~my_stt_tts.transport.AudioTransport` and sinks
synthesized TTS PCM back through it. So a WebSocket satellite or the browser GUI
runs the exact production pipeline, just with the device boundary moved onto the
wire.

The capture/transcribe/respond functions take an injectable VAD + analyzer +
transcriber so the whole loop is unit-testable with fakes (no mic, no model, no
network) — see ``tests/test_transport.py``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from .brain import Brain, LLMError
from .config import Config
from .events import bus
from .stt import StreamingTranscriber, Transcriber
from .text import SentenceChunker, strip_non_spoken
from .transport import AudioTransport
from .tts import TTSRouter

log = logging.getLogger("my_stt_tts.net_loop")


def capture_turn(
    transport: AudioTransport,
    cfg: Config,
    vad: Any,
    analyzer: Any,
    transcriber: Transcriber,
    *,
    max_frames: int | None = None,
) -> str:
    """Pull mic frames from ``transport`` until end-of-turn, then transcribe.

    Drives the supplied VAD + :class:`~my_stt_tts.turn.TurnAnalyzer` exactly like
    the local capture loop and feeds a :class:`StreamingTranscriber` so partial
    transcripts are published to the bus during the turn (and the final at the
    end). ``max_frames`` bounds the loop for tests; in production the analyzer
    ends the turn. Returns the final transcript (possibly empty on silence).
    """
    streamer = StreamingTranscriber(
        transcriber,
        cfg.sample_rate,
        partial_interval_ms=cfg.stt_partial_interval_ms,
        window_s=cfg.stt_window_s,
    )
    analyzer.reset()
    spoke = False
    bus.state("recording")
    for seen, frame in enumerate(transport.mic_frames(), start=1):
        arr = np.asarray(frame, dtype=np.float32).ravel()
        is_speech = vad.is_speech(arr)
        spoke = spoke or is_speech
        partial = streamer.feed(arr)
        if partial is not None:
            bus.transcript(partial, partial=True)
        if analyzer.update(arr, is_speech):
            break
        if max_frames is not None and seen >= max_frames:
            break
    if not spoke:
        bus.state("idle")
        return ""
    bus.state("stt")
    return str(streamer.final().text).strip()


def respond_over_transport(
    transport: AudioTransport,
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    text: str,
) -> str:
    """Stream the LLM reply for ``text``, synthesize each sentence, sink it to ``transport``.

    Half-duplex over the wire: the reply is chunked into sentences, each sentence
    is synthesized to PCM and forwarded via :meth:`AudioTransport.send_tts`, and
    the streamed text is also published to the bus for the GUI. Returns the full
    spoken text (so the caller can keep history honest). On an LLM error it sends a
    short spoken apology, mirroring the local loop.
    """
    bus.transcript(text)
    chunker = SentenceChunker()
    spoken: list[str] = []
    stream = brain.stream(text)
    try:
        bus.state("llm_response")
        for delta in stream:
            bus.response(delta, final=False)
            for sentence in chunker.feed(delta):
                spoken.append(_voice_to_transport(transport, tts, sentence))
        tail = chunker.flush()
        if tail:
            spoken.append(_voice_to_transport(transport, tts, tail))
        bus.response("", final=True)
    except LLMError as exc:
        log.error("LLM error over transport: %s", exc)
        bus.log(str(exc), "error")
        _voice_to_transport(transport, tts, "Sorry, I had a problem.")
    finally:
        with contextlib.suppress(Exception):  # best-effort cancel of in-flight tokens
            stream.close()
    bus.state("idle")
    return "".join(spoken)


def _voice_to_transport(transport: AudioTransport, tts: TTSRouter, sentence: str) -> str:
    """Synthesize one sentence to PCM and forward it; return the chars voiced."""
    text = strip_non_spoken(sentence)
    if not text:
        return ""
    bus.state("speaking")
    pcm, sr = tts.synth_pcm(text)
    if pcm.size:
        transport.send_tts(pcm, sr)
    return sentence


def run_transport_session(
    transport: AudioTransport,
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    transcriber: Transcriber,
    *,
    vad: Any | None = None,
) -> None:
    """Run turns over ``transport`` until its mic source ends (one satellite session).

    Builds the VAD + turn analyzer from config (lazily, so tests can inject), then
    loops: capture a turn, respond over the wire. Ends when ``mic_frames`` is
    exhausted (the client disconnected). Used as the ``on_session`` callback by the
    WebSocket server.
    """
    if vad is None:
        from .vad import SileroVad

        vad = SileroVad(cfg.sample_rate)
    from .turn import make_turn_analyzer

    frame_seconds = 512 / cfg.sample_rate
    bus.state("idle")
    while not _transport_closed(transport):
        analyzer = make_turn_analyzer(cfg, frame_seconds)
        text = capture_turn(transport, cfg, vad, analyzer, transcriber)
        if not text:
            break  # silence / client gone
        respond_over_transport(transport, cfg, brain, tts, text)


def _transport_closed(transport: AudioTransport) -> bool:
    return bool(getattr(transport, "closed", False))


def mic_frame_chunks(pcm: np.ndarray, frame_samples: int = 512) -> Iterator[np.ndarray]:
    """Split a PCM array into fixed-size frames (used by the satellite client)."""
    arr = np.asarray(pcm, dtype=np.float32).ravel()
    for start in range(0, arr.size, frame_samples):
        yield arr[start : start + frame_samples]
