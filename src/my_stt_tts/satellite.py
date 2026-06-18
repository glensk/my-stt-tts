"""Satellite client (R2-5): capture mic + play TTS over a WebSocket to the server.

This is the small, end-to-end-usable remote endpoint: run it on a Raspberry Pi /
spare Mac in another room, point it at the host running ``my-stt-tts --transport
websocket``, and it streams mic PCM up and plays the TTS PCM that comes back. The
full pipeline (STT/LLM/TTS) stays on the server; the satellite is just ears +
mouth.

The wire protocol is :mod:`my_stt_tts.transport` (JSON ``hello`` handshake, then
binary int16 LE PCM frames). ``websockets`` + ``sounddevice`` are needed at
runtime (the ``transport`` + ``audio`` extras); the framing helpers used here are
the same pure functions the server uses, so the encode/decode path is covered by
``tests/test_transport.py`` without a real socket or mic.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import threading

from .transport import decode_frame, encode_frame, make_handshake

log = logging.getLogger("my_stt_tts.satellite")


def _mic_reader(sample_rate: int, frame_samples: int, out_q: queue.Queue[bytes]) -> threading.Event:
    """Start a background mic-capture thread that enqueues encoded PCM frames."""
    stop = threading.Event()

    def _run() -> None:
        from . import audio

        sd = audio._sd()  # noqa: SLF001 — lazy sounddevice accessor

        def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
            with _ignore_full():
                out_q.put_nowait(encode_frame(indata[:, 0].copy()))

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=_callback,
        ):
            stop.wait()

    threading.Thread(target=_run, daemon=True).start()
    return stop


class _ignore_full:  # noqa: N801 — small queue.Full swallow
    def __enter__(self) -> _ignore_full:
        return self

    def __exit__(self, exc_type: type | None, *_exc: object) -> bool:
        return exc_type is not None and issubclass(exc_type, queue.Full)


async def run_satellite(
    url: str,
    *,
    sample_rate: int = 16000,
    token: str | None = None,
    frame_samples: int = 512,
) -> None:
    """Connect to ``url``, stream mic PCM up and play TTS PCM coming back.

    Sends the ``hello`` handshake (with the optional shared ``token``), then runs
    two concurrent pumps: mic→socket and socket→speaker. Requires ``websockets``
    and ``sounddevice``; raises a clear error if either is missing.
    """
    try:
        import sounddevice as sd
        import websockets
    except ImportError as exc:  # pragma: no cover - needs the extras
        raise RuntimeError(
            "satellite needs the 'transport' + 'audio' extras: "
            "uv sync --extra transport --extra audio"
        ) from exc

    log.info("connecting to %s ...", url)
    async with websockets.connect(url, max_size=None) as conn:
        await conn.send(make_handshake(sample_rate=sample_rate, token=token, role="satellite"))
        ready = await conn.recv()
        log.info("server ready: %s", ready)
        mic_q: queue.Queue[bytes] = queue.Queue(maxsize=512)
        stop = _mic_reader(sample_rate, frame_samples, mic_q)
        loop = asyncio.get_running_loop()

        async def _send_mic() -> None:
            while True:
                data = await loop.run_in_executor(None, _drain, mic_q)
                if data:
                    await conn.send(data)

        async def _play_back() -> None:
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out:
                async for message in conn:
                    if isinstance(message, bytes):
                        pcm = decode_frame(message).reshape(-1, 1)
                        await loop.run_in_executor(None, out.write, pcm)

        send_task = asyncio.ensure_future(_send_mic())
        try:
            await _play_back()
        finally:
            send_task.cancel()
            stop.set()


def _drain(q: queue.Queue[bytes], timeout: float = 0.1) -> bytes:
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return b""


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m my_stt_tts.satellite ws://HOST:PORT [--token T]``."""
    parser = argparse.ArgumentParser(
        prog="my-stt-tts-satellite",
        description="Remote mic+speaker satellite for my-stt-tts over WebSocket.",
    )
    parser.add_argument("url", help="WebSocket URL of the server, e.g. ws://192.168.1.10:8770")
    parser.add_argument("--token", help="Shared auth token (must match the server).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="PCM sample rate (Hz).")
    parser.add_argument("--frame-samples", type=int, default=512, help="Mic frame size in samples.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        asyncio.run(
            run_satellite(
                args.url,
                sample_rate=args.sample_rate,
                token=args.token,
                frame_samples=args.frame_samples,
            )
        )
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    except RuntimeError as exc:
        print(exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_satellite"]
