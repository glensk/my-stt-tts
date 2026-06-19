"""Text-to-speech router: per-language voice selection + pluggable backends.

The default engine is **Piper**, invoked as a SUBPROCESS (its CLI binary) so this
Apache-2.0 project never links the GPL-3.0 library in-process. macOS ``say`` is
the always-available fallback. Language detection (``lingua``) is optional/lazy
and the detector is cached so per-sentence detection is cheap. English voices can
be swapped via :data:`VOICE_PRESETS`; missing Piper voices are downloaded on first
use, and ``tts_length_scale`` slows delivery for a calmer cadence.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .text import ClauseChunker, strip_non_spoken

log = logging.getLogger("my_stt_tts.tts")


def _resample_to(pcm: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample float32 mono PCM to ``to_sr`` by linear interpolation (no deps).

    A small, dependency-free resampler so streamed clauses (Piper renders at its
    own native rate) can be fed into a single ``OutputStream`` opened at the
    pipeline ``sample_rate``. Identity when the rates already match.
    """
    arr = np.asarray(pcm, dtype=np.float32).ravel()
    if from_sr == to_sr or arr.size == 0:
        return arr
    n_out = max(1, int(round(arr.size * to_sr / from_sr)))
    x_old = np.arange(arr.size, dtype=np.float64)
    x_new = np.linspace(0.0, arr.size - 1, n_out)
    return np.interp(x_new, x_old, arr).astype(np.float32)


# A small menu of English Piper voices to pick from (`--voice <name>`).
VOICE_PRESETS: dict[str, str] = {
    "lessac": "en_US-lessac-medium",
    "amy": "en_US-amy-medium",
    "ryan": "en_US-ryan-medium",
    "kristin": "en_US-kristin-medium",
    "hfc-female": "en_US-hfc_female-medium",
    "hfc-male": "en_US-hfc_male-medium",
    "joe": "en_US-joe-medium",
    "alba": "en_GB-alba-medium",
    "alan": "en_GB-alan-medium",
}

_VOICE_NOTES: dict[str, str] = {
    "lessac": "neutral, clear (default)",
    "amy": "calm female",
    "ryan": "calm male",
    "kristin": "warm female",
    "hfc-female": "natural female",
    "hfc-male": "natural male",
    "joe": "deep male",
    "alba": "British female",
    "alan": "British male",
}


def list_voice_presets() -> str:
    """Return a printable menu of the English voice presets."""
    return "\n".join(
        f"  {name:11} {VOICE_PRESETS[name]:24} {_VOICE_NOTES.get(name, '')}"
        for name in VOICE_PRESETS
    )


@lru_cache(maxsize=4)
def _detector(langs: tuple[str, ...]):  # noqa: ANN202 — lingua type is lazy-imported
    from lingua import IsoCode639_1, LanguageDetectorBuilder

    codes = [getattr(IsoCode639_1, c.upper()) for c in langs]
    return LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()


def detect_language(
    text: str, default: str = "en", langs: tuple[str, ...] = ("de", "fr", "en")
) -> str:
    """Detect the language of ``text`` (lingua); fall back to ``default``."""
    try:
        detected = _detector(langs).detect_language_of(text)
    except ImportError:
        return default  # lingua not installed (the `lang` extra)
    except Exception:  # any build/detection error must never break the loop
        log.debug("language detection unavailable; using default %r", default, exc_info=True)
        return default
    if detected is None:
        return default
    return detected.iso_code_639_1.name.lower()


def select_voice(cfg: Config, lang: str) -> tuple[str, str]:
    """Return ``(engine, voice)``: Piper voice for ``lang`` if mapped, else ``say``."""
    if lang in cfg.tts_voices:
        return "piper", cfg.tts_voices[lang]
    if lang in cfg.say_voices:
        return "say", cfg.say_voices[lang]
    return "say", cfg.say_voices.get(cfg.default_language, "")


class Playback:
    """A cancellable handle around a running player subprocess (G1 barge-in).

    The player (``afplay`` / ``say``) is launched with :class:`subprocess.Popen`
    so the loop can :meth:`cancel` it mid-utterance — killing the process aborts
    playback immediately. :meth:`wait` blocks until it finishes (or is cancelled).
    A finished/never-started handle is a harmless no-op for both methods.
    """

    def __init__(
        self,
        proc: subprocess.Popen | None = None,
        *,
        reference: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self._proc = proc
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        # The synthesized PCM being played (the AEC reference signal, R2-1) and its
        # sample rate. None when the player wasn't fed a known waveform (e.g. `say`).
        self.reference = (
            np.asarray(reference, dtype=np.float32).ravel() if reference is not None else None
        )
        self.reference_sr = sample_rate

    def cancel(self) -> None:
        """Abort playback now (kill the subprocess). Idempotent + thread-safe."""
        self._cancelled.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):  # already gone
                proc.kill()

    def wait(self) -> None:
        """Block until the player exits (or has been cancelled)."""
        with self._lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.wait()
            except Exception:  # killed mid-wait is fine
                log.debug("playback wait interrupted", exc_info=True)

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` was called on this handle."""
        return self._cancelled.is_set()

    @property
    def done(self) -> bool:
        """Whether the player has finished (or there was nothing to play)."""
        with self._lock:
            proc = self._proc
        return proc is None or proc.poll() is not None


# A sentinel pushed onto the PCM queue to mark the end of a streamed utterance.
_PCM_EOF = object()


class StreamingPlayback:
    """Cancellable playout that consumes PCM *chunks* as they arrive (R3-3).

    Unlike :class:`Playback` (which launches a player on a fully-rendered WAV),
    this presents the SAME barge-in surface (``done`` / :meth:`cancel` /
    :meth:`wait` / ``reference`` / ``reference_sr``) but is fed clause-sized PCM
    chunks via :meth:`feed` and plays them through a :mod:`sounddevice`
    ``OutputStream`` so time-to-first-audio is the synthesis time of the *first
    clause*, not the whole sentence. Each fed chunk is also appended to
    :attr:`reference` so the AEC front-end can still subtract the assistant's
    own voice during barge-in.

    The player runs in a background thread reading a bounded PCM queue;
    :meth:`cancel` stops it immediately (mid-chunk) and aborts the underlying
    stream, so a barge-in still cuts the utterance off. Construct, start the
    feeder, then hand the handle to ``monitor_during_playback``.
    """

    def __init__(self, sample_rate: int, *, frame_samples: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.reference_sr: int | None = sample_rate
        self.reference = np.zeros(0, dtype=np.float32)
        self._frame_samples = frame_samples
        self._q: queue.Queue[Any] = queue.Queue(maxsize=256)
        self._cancelled = threading.Event()
        self._finished = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        self._player: threading.Thread | None = None
        self._open_stream = _open_output_stream

    def feed(self, pcm: np.ndarray) -> None:
        """Queue one PCM chunk for playout (no-op once cancelled/closed)."""
        arr = np.asarray(pcm, dtype=np.float32).ravel()
        if arr.size == 0 or self._cancelled.is_set():
            return
        with self._lock:
            self.reference = np.concatenate([self.reference, arr])
        self._ensure_player()
        with contextlib.suppress(queue.Full):
            self._q.put(arr, timeout=0.5)

    def end_feed(self) -> None:
        """Signal that no more chunks will be fed (the utterance is complete)."""
        if not self._started:
            # Nothing was ever played -> a finished, inert handle.
            self._finished.set()
            return
        with contextlib.suppress(queue.Full):
            self._q.put(_PCM_EOF, timeout=0.5)

    def _ensure_player(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._player = threading.Thread(target=self._run, daemon=True)
            self._player.start()

    def _run(self) -> None:
        stream = None
        try:
            stream = self._open_stream(self.sample_rate)
            stream.start()
            while not self._cancelled.is_set():
                try:
                    item = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _PCM_EOF:
                    break
                if self._cancelled.is_set():
                    break
                with contextlib.suppress(Exception):
                    stream.write(np.asarray(item, dtype=np.float32).reshape(-1, 1))
        except Exception:  # any device error must not crash the loop
            log.warning("streaming playout failed", exc_info=True)
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.stop()
                    stream.close()
            self._finished.set()

    def cancel(self) -> None:
        """Abort playout now (drop queued chunks, stop the stream). Idempotent."""
        self._cancelled.set()
        with contextlib.suppress(queue.Full):
            self._q.put_nowait(_PCM_EOF)

    def wait(self) -> None:
        """Block until the queued audio has finished playing (or was cancelled)."""
        if not self._started:
            return
        self._finished.wait()

    @property
    def cancelled(self) -> bool:
        """Whether :meth:`cancel` was called on this handle."""
        return self._cancelled.is_set()

    @property
    def done(self) -> bool:
        """Whether playout has finished (or nothing was ever fed)."""
        return self._finished.is_set() or (not self._started)


def _open_output_stream(sample_rate: int) -> Any:
    """Open a mono float32 :mod:`sounddevice` ``OutputStream`` (lazy import)."""
    from . import audio

    sd = audio._sd()  # noqa: SLF001 — same package lazy accessor
    return sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")


def _popen(cmd: list[str], *, stdin: bytes | None = None) -> subprocess.Popen:
    """Launch ``cmd`` detached so it can be killed; feed ``stdin`` if given."""
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if stdin is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin)
            proc.stdin.close()
        except BrokenPipeError:  # player exited early
            pass
    return proc


def _player_argv(cfg: Config | None = None) -> tuple[str, ...]:
    """The argv prefix for a cancellable WAV player (G8: afplay on macOS, aplay on Linux).

    The WAV path is appended by the caller. Falls back to ``afplay`` when no player
    is found on PATH (the macOS default), so behaviour is unchanged on a Mac.
    """
    from .platform import select_player

    return select_player(cfg) or ("afplay",)


def _afplay(path: str) -> None:
    subprocess.run([*_player_argv(), path], check=False)  # noqa: S603


def _read_wav(path: str) -> tuple[np.ndarray | None, int | None]:
    """Read a mono PCM16 WAV to float32 (-1..1) + sample rate, for the AEC reference.

    Returns ``(None, None)`` if the file can't be read; never raises (the AEC
    reference is best-effort and must not break playback).
    """
    try:
        with wave.open(path, "rb") as handle:
            sr = handle.getframerate()
            channels = handle.getnchannels()
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error):
        return None, None
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, sr


def _ensure_piper_voice(data_dir: str, voice: str) -> bool:
    """Make sure ``<data_dir>/<voice>.onnx`` exists, downloading it if needed."""
    if (Path(data_dir) / f"{voice}.onnx").exists():
        return True
    log.info("downloading Piper voice %s ...", voice)
    subprocess.run(  # noqa: S603, S607
        [
            "uv",
            "tool",
            "run",
            "--from",
            "piper-tts",
            "python",
            "-m",
            "piper.download_voices",
            "--download-dir",
            data_dir,
            voice,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return (Path(data_dir) / f"{voice}.onnx").exists()


class CloudTTS:
    """Optional cloud TTS (R2-7): an OpenAI-compatible speech endpoint.

    Renders text to a WAV via a ``/audio/speech``-style API (OpenAI TTS, or any
    compatible gateway) — useful for a **high-quality cloud German voice**, since
    the local German TTS is the weak spot. **Local-first**: selected only when
    ``tts_backend=cloud`` *and* an API key is present; otherwise the router uses
    Piper / ``say``. The ``openai`` client is lazy-imported from the ``llm`` extra.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        *,
        voice: str = "alloy",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.base_url = base_url
        self._client: object | None = None

    def available(self) -> bool:
        """True when an API key is configured (so cloud TTS can actually be used)."""
        return bool(self.api_key)

    def _ensure(self) -> object:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key or "not-needed", base_url=self.base_url)
        return self._client

    def render(self, text: str) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to PCM via the cloud endpoint (WAV response → float32)."""
        client = self._ensure()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = handle.name
        try:
            response = client.audio.speech.create(  # type: ignore[attr-defined]
                model=self.model, voice=self.voice, input=text, response_format="wav"
            )
            response.stream_to_file(out)
            return _read_wav(out)
        except Exception:  # never break the loop on a cloud hiccup
            log.warning("cloud TTS render failed", exc_info=True)
            return None, None
        finally:
            Path(out).unlink(missing_ok=True)


class TTSRouter:
    """Synthesize and play text in the right voice for its language.

    Use :meth:`speak` for blocking playback (typed mode, error clips), or
    :meth:`start_speaking` to get a cancellable :class:`Playback` handle so the
    loop can abort the utterance mid-stream on barge-in (G1).
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Cloud renderer (any backend exposing ``render(text) -> (pcm, sr)``):
        # OpenAI/ElevenLabs/Cartesia, selected by ``cfg.tts_backend`` via the
        # registry (G1). None => local Piper / ``say`` (key-gated graceful fallback).
        from .registry import select_tts_backend

        self._cloud: Any = select_tts_backend(cfg)

    def speak(self, text: str, lang: str | None = None) -> None:
        """Synthesize ``text`` and play it, blocking until done."""
        self.start_speaking(text, lang).wait()

    def synth_pcm(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` to float32 mono PCM + sample rate (no playback).

        Used by the network transport (R2-5): the TTS audio is forwarded to a
        remote satellite / browser instead of being played on the local speaker.
        Uses Piper when available (it already renders to a WAV we can read back);
        falls back to macOS ``say`` rendering to an AIFF. Returns ``(empty, sr)``
        for blank text or when no renderer is available (the caller then has
        nothing to forward — graceful, never raises).
        """
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), self.cfg.sample_rate
        if self._cloud is not None:
            pcm, sr = self._cloud.render(text)
            if pcm is not None and sr is not None:
                return pcm, sr
        lang = lang or detect_language(text, self.cfg.default_language)
        engine, voice = select_voice(self.cfg, lang)
        if (
            engine == "piper"
            and shutil.which("piper")
            and _ensure_piper_voice(self.cfg.piper_data_dir, voice)
        ):
            pcm, sr = self._render_piper(text, voice)
            if pcm is not None and sr is not None:
                return pcm, sr
        say_voice = self.cfg.say_voices.get(lang) or self.cfg.say_voices.get(
            self.cfg.default_language, ""
        )
        pcm, sr = self._render_say(text, voice=say_voice)
        if pcm is not None and sr is not None:
            return pcm, sr
        return np.zeros(0, dtype=np.float32), self.cfg.sample_rate

    def synth_pcm_stream(
        self, text: str, lang: str | None = None
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield ``(pcm, sample_rate)`` clause-by-clause for low-latency playout (R3-3).

        Splits ``text`` into clauses (``ClauseChunker``) and synthesizes each one
        independently, so the **first** chunk is ready after the first clause is
        rendered rather than after the whole sentence — time-to-first-audio scales
        with the first clause, not the utterance length. Each clause reuses
        :meth:`synth_pcm`, so it inherits the cloud → Piper → ``say`` fallback. The
        per-clause language is detected once for the whole ``text`` (cheap + stable
        prosody). Empty/blank clauses are skipped; a blank input yields nothing.
        """
        spoken = strip_non_spoken(text)
        if not spoken:
            return
        lang = lang or detect_language(spoken, self.cfg.default_language)
        chunker = ClauseChunker(min_chars=self.cfg.tts_stream_min_chars)
        for clause in [*chunker.feed(spoken), chunker.flush()]:
            clause = clause.strip()
            if not clause:
                continue
            pcm, sr = self.synth_pcm(clause, lang)
            if pcm.size:
                yield pcm, sr

    def start_speaking_stream(self, text: str, lang: str | None = None) -> StreamingPlayback:
        """Stream-synthesize ``text`` into a cancellable :class:`StreamingPlayback` (R3-3).

        Synthesis runs clause-by-clause in a background thread (via
        :meth:`synth_pcm_stream`) and each rendered chunk is fed into a
        :class:`StreamingPlayback` that plays through a ``sounddevice``
        ``OutputStream`` — so the first audio plays as soon as the first clause is
        synthesized, while later clauses render concurrently. The returned handle
        has the same barge-in surface as :class:`Playback` (``done`` / ``cancel`` /
        ``wait`` / ``reference``), so ``monitor_during_playback`` works unchanged.
        Returns an inert handle for blank text.
        """
        handle = StreamingPlayback(self.cfg.sample_rate, frame_samples=self.cfg.tts_stream_frame)
        if not text.strip():
            handle.end_feed()
            return handle

        def _synth() -> None:
            try:
                for pcm, sr in self.synth_pcm_stream(text, lang):
                    if handle.cancelled:
                        break
                    handle.feed(_resample_to(pcm, sr, self.cfg.sample_rate))
            finally:
                handle.end_feed()

        threading.Thread(target=_synth, daemon=True).start()
        return handle

    def start_speaking(self, text: str, lang: str | None = None) -> Playback:
        """Synthesize ``text`` and start playing it; return a cancellable handle.

        Synthesis (Piper) runs to a temp WAV first, then the **player** subprocess
        (``afplay`` / ``say``) is launched via :class:`Playback` so it can be
        killed mid-utterance. Returns an inert handle for empty/blank text.
        """
        text = text.strip()
        if not text:
            return Playback()
        if self._cloud is not None:
            playback = self._play_cloud(text)
            if playback is not None:
                return playback
        lang = lang or detect_language(text, self.cfg.default_language)
        engine, voice = select_voice(self.cfg, lang)
        if (
            engine == "piper"
            and shutil.which("piper")
            and _ensure_piper_voice(self.cfg.piper_data_dir, voice)
        ):
            return self._play_piper(text, voice)
        say_voice = self.cfg.say_voices.get(lang) or self.cfg.say_voices.get(
            self.cfg.default_language, ""
        )
        return self._play_say(text, voice=say_voice)

    def _play_cloud(self, text: str) -> Playback | None:
        """Render via the cloud TTS and play the WAV with the cancellable player.

        Returns ``None`` (so the caller falls back to local TTS) when the cloud
        render fails — never raises, so a transient API error can't break the loop.
        """
        assert self._cloud is not None
        pcm, sr = self._cloud.render(text)
        if pcm is None or sr is None or not pcm.size:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = handle.name
        # pylint: disable=no-member  # wave.open(..., "wb") -> Wave_write
        pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(out, "wb") as wh:
            wh.setnchannels(1)
            wh.setsampwidth(2)
            wh.setframerate(sr)
            wh.writeframes(pcm16.tobytes())
        proc = _popen([*_player_argv(self.cfg), out])
        threading.Thread(target=self._cleanup_after, args=(proc, out), daemon=True).start()
        return Playback(proc, reference=pcm, sample_rate=sr)

    def _play_piper(self, text: str, voice: str) -> Playback:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = handle.name
        cmd = [
            "piper",
            "-m",
            voice,
            "-f",
            out,
            "--length-scale",
            str(self.cfg.tts_length_scale),
        ]
        if self.cfg.piper_data_dir:
            cmd += ["--data-dir", self.cfg.piper_data_dir]
        try:
            subprocess.run(cmd, input=text.encode(), check=True)  # noqa: S603, S607
        except (subprocess.CalledProcessError, OSError):
            Path(out).unlink(missing_ok=True)
            log.warning("Piper synthesis failed for %r; using `say`.", voice, exc_info=True)
            return self._play_say(text, voice="")
        # Read the synthesized PCM back as the AEC reference signal (R2-1) so the
        # echo canceller can subtract exactly what the speaker emits.
        reference, ref_sr = _read_wav(out)
        # afplay is the cancellable player; unlink the temp WAV once it exits.
        proc = _popen([*_player_argv(self.cfg), out])
        threading.Thread(target=self._cleanup_after, args=(proc, out), daemon=True).start()
        return Playback(proc, reference=reference, sample_rate=ref_sr)

    def _play_say(self, text: str, *, voice: str) -> Playback:
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        return Playback(_popen(cmd))

    def _render_piper(self, text: str, voice: str) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to a WAV via Piper and read it back as PCM (no playback)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = handle.name
        cmd = ["piper", "-m", voice, "-f", out, "--length-scale", str(self.cfg.tts_length_scale)]
        if self.cfg.piper_data_dir:
            cmd += ["--data-dir", self.cfg.piper_data_dir]
        try:
            subprocess.run(cmd, input=text.encode(), check=True)  # noqa: S603, S607
            return _read_wav(out)
        except (subprocess.CalledProcessError, OSError):
            log.warning("Piper render failed for %r", voice, exc_info=True)
            return None, None
        finally:
            Path(out).unlink(missing_ok=True)

    def _render_say(self, text: str, *, voice: str) -> tuple[np.ndarray | None, int | None]:
        """Render ``text`` to PCM via macOS ``say`` (AIFF -> WAV read-back)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = handle.name
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd += ["--data-format=LEI16@22050", "-o", out, "--file-format=WAVE", text]
        try:
            subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603, S607
            return _read_wav(out)
        except (subprocess.CalledProcessError, OSError):
            log.warning("`say` render failed", exc_info=True)
            return None, None
        finally:
            Path(out).unlink(missing_ok=True)

    @staticmethod
    def _cleanup_after(proc: subprocess.Popen, path: str) -> None:
        with contextlib.suppress(Exception):
            proc.wait()
        Path(path).unlink(missing_ok=True)
