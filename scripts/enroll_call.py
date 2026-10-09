#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Enroll a CALL-DOMAIN voice profile: sentences recorded the way a mac-voice call hears them.

The Claude bridge checks every in-call instruction against the speaker's profile. The
wake profile (``enroll/<name>.npy``) comes from short raw-mic "hey jarvis" clips, but a
call's audio runs through Apple's VoiceProcessingIO (echo cancellation, AGC), so real
in-call sentences score too low against it. This script records sentences through the
same VoiceProcessingIO duplex at the same 16 kHz / 250 ms framing the daemon uses, trims
them with the bridge's Silero VAD segmentation, embeds them with ECAPA and saves the L2
centroid to ``<out>/call/<name>.npy`` — which ``VoiceGate.score_against`` then prefers.

Each clip asks for any sentence, cycling the language hint (German / English / French by
default); clips with under 1.0 s of speech are skipped, at least 3 usable clips are needed.
The calibration table shows, per clip, its leave-one-out cosine against the centroid of
the OTHER clips and its cosine against the wake profile, then min / median of the
leave-one-out scores and whether the minimum clears the bridge's 0.35 threshold.

Stop the daemon first — it owns the mic (``launchctl bootout gui/$(id -u)/com.albert.mac-voice``
or ``mac-voice -U``; ``mac-voice -I`` brings it back).

Usage:
    scripts/enroll_call.py albert                 # 8 clips of 4 s, saves enroll/call/albert.npy
    scripts/enroll_call.py albert -c 10 -s 5      # 10 clips of 5 s
    scripts/enroll_call.py albert -n              # dry run: record + table, write nothing
    scripts/enroll_call.py albert -o /tmp/enroll -l de,fr

Exit codes: 0 done, 1 too few usable clips, 2 usage / daemon running / audio unavailable.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import re
import statistics
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from _bootstrap import ensure_venv

ensure_venv(["audio", "speaker", "vad", "aec"])

import numpy as np  # noqa: E402  # pylint: disable=wrong-import-position  # after the venv re-exec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "enroll"
DEFAULT_LANGUAGES = ("de", "en", "fr")
LANGUAGE_NAMES = {"de": "German", "en": "English", "fr": "French", "it": "Italian"}
MIN_SPEECH_S = 1.0
MIN_CLIPS = 3
THRESHOLD = 0.35  # bridge.AUTH_THRESHOLD
SAMPLE_RATE = 16000  # eleven_voice.SAMPLE_RATE
FRAME_SAMPLES = 4000  # eleven_voice.INPUT_CHUNK: the frames the bridge's VAD sees in a call
NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
STOP_HINT = (
    "❌ the mac-voice daemon is running and owns the mic — stop it first:\n"
    "   launchctl bootout gui/$(id -u)/com.albert.mac-voice   (or: mac-voice -U)\n"
    "   then re-run this script; afterwards start it again with mac-voice -I"
)


class Duplex(Protocol):
    """What this script needs of ``aec.VoiceProcessingDuplex``."""

    def start(self) -> bool: ...

    def mic_frames(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class SpeechDetector(Protocol):
    def is_speech(self, frame: Any) -> bool: ...


def _default_duplex() -> Duplex:
    from my_stt_tts.aec import VoiceProcessingDuplex
    from my_stt_tts.eleven_voice import INPUT_CHUNK
    from my_stt_tts.eleven_voice import SAMPLE_RATE as CALL_RATE

    return VoiceProcessingDuplex(CALL_RATE, frame_samples=INPUT_CHUNK)


def _default_vad() -> SpeechDetector:
    from my_stt_tts.vad import SileroVad

    return SileroVad()  # what bridge._default_vad builds


def _default_embedder() -> Callable[[np.ndarray], np.ndarray]:
    from my_stt_tts.voice_gate import default_embedder

    return default_embedder()


def _daemon_running() -> bool:
    from my_stt_tts.voice_control import send

    return send("status", timeout=1.0) is not None


@dataclass
class Deps:
    """The hardware / model seams (tests replace them)."""

    duplex: Callable[[], Duplex] = _default_duplex
    vad: Callable[[], SpeechDetector] = _default_vad
    embedder: Callable[[], Callable[[np.ndarray], np.ndarray]] = _default_embedder
    daemon_running: Callable[[], bool] = _daemon_running
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = print


class Recorder:  # pylint: disable=too-many-instance-attributes  # capture-thread state
    """Pumps the duplex's mic frames on a thread; :meth:`record` keeps the next N seconds."""

    def __init__(self, duplex: Duplex, sample_rate: int = SAMPLE_RATE) -> None:
        self.duplex = duplex
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._want = 0
        self._have = 0
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._pump, name="enroll-capture", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        for frame in self.duplex.mic_frames():
            with self._lock:
                if self._want <= 0:
                    continue  # between clips: drop the audio
                pcm = np.asarray(frame, dtype=np.float32).ravel()
                self._frames.append(pcm)
                self._have += pcm.size
                if self._have >= self._want:
                    self._want = 0
                    self._done.set()
        self._done.set()  # the source ended

    def record(self, seconds: float, slack_s: float = 3.0) -> np.ndarray:
        """The next ``seconds`` of mic audio (shorter if the source stops)."""
        want = int(round(seconds * self.sample_rate))
        with self._lock:
            self._frames, self._have, self._want = [], 0, want
            self._done.clear()
        if self._thread.is_alive():
            self._done.wait(seconds + slack_s)
        with self._lock:
            self._want = 0
            frames = self._frames
            self._frames = []
        pcm = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        return pcm[:want]


def speech_only(clip: np.ndarray, vad: SpeechDetector) -> tuple[np.ndarray, float]:
    """(speech PCM, seconds) — the utterances the bridge's segmentation finds in ``clip``."""
    from my_stt_tts.turns import TurnSource

    turns = TurnSource(vad, sample_rate=SAMPLE_RATE, ring_s=1e9)
    at = 0.0
    padded = np.concatenate([clip, np.zeros(3 * FRAME_SAMPLES, dtype=np.float32)])
    for start in range(0, padded.size, FRAME_SAMPLES):
        frame = padded[start : start + FRAME_SAMPLES]
        at += frame.size / SAMPLE_RATE
        turns.feed(frame, at=at)
    utts = turns.utterances()
    if not utts:
        return np.zeros(0, dtype=np.float32), 0.0
    return np.concatenate([u.pcm for u in utts]), sum(u.duration for u in utts)


def _l2(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    return vec / (float(np.linalg.norm(vec)) + 1e-9)


def call_centroid(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalised mean of the L2-normalised embeddings."""
    return _l2(np.mean([_l2(e) for e in embeddings], axis=0))


def leave_one_out(embeddings: list[np.ndarray]) -> list[float]:
    """Per clip: cosine against the centroid of all OTHER clips."""
    return [
        float(_l2(e) @ call_centroid(embeddings[:i] + embeddings[i + 1 :]))
        for i, e in enumerate(embeddings)
    ]


def parse_languages(raw: str) -> list[str]:
    """``de,en,fr`` → codes in order, de-duplicated; the default set when empty."""
    seen: list[str] = []
    for part in raw.split(","):
        code = part.strip().lower()
        if code and code not in seen:
            seen.append(code)
    return seen or list(DEFAULT_LANGUAGES)


def clip_prompt(index: int, total: int, languages: list[str]) -> str:
    """``Clip 3/8 — say any sentence in French, press Enter to start``."""
    code = languages[index % len(languages)]
    name = LANGUAGE_NAMES.get(code, code.upper())
    return f"Clip {index + 1}/{total} — say any sentence in {name}, press Enter to start "


def calibration_table(
    labels: list[str], loo: list[float], wake: list[float | None], threshold: float = THRESHOLD
) -> tuple[list[str], bool]:
    """The printed table + summary lines, and whether min(leave-one-out) ≥ ``threshold``."""
    lines = [
        "| clip | speech | leave-one-out | vs wake |",
        "| ---: | -----: | ------------: | ------: |",
    ]
    for label, own, vs_wake in zip(labels, loo, wake, strict=True):
        wake_txt = "-" if vs_wake is None else f"{vs_wake:.2f}"
        lines.append(f"| {label} | {own:13.2f} | {wake_txt:>7} |")
    low, mid = min(loo), statistics.median(loo)
    ok = low >= threshold
    lines.append(f"leave-one-out: min {low:.2f}, median {mid:.2f}")
    mark = "✅" if ok else "❌"
    verdict = "clears" if ok else "is below"
    lines.append(f"{mark} min leave-one-out {low:.2f} {verdict} the bridge threshold {threshold}")
    return lines, ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="whose profile (file name: <out>/call/<name>.npy)")
    parser.add_argument("-c", "--clips", type=int, default=8, help="clips to record (default 8)")
    parser.add_argument(
        "-s", "--seconds", type=float, default=4.0, help="seconds per clip (default 4)"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="profile directory; writes <out>/call/<name>.npy, reads the wake profile "
        "<out>/<name>.npy (default: the repo's enroll/)",
    )
    parser.add_argument(
        "-l",
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="language hints cycled across the clips (default de,en,fr)",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="record and print the table, write nothing"
    )
    return parser


def _record_clips(
    args: argparse.Namespace, deps: Deps, recorder: Recorder, vad: SpeechDetector
) -> tuple[list[np.ndarray], list[str]]:
    """Record, trim and keep the usable clips; (speech PCM per clip, table labels)."""
    languages = parse_languages(args.languages)
    clips: list[np.ndarray] = []
    labels: list[str] = []
    for index in range(args.clips):
        deps.ask(clip_prompt(index, args.clips, languages))
        deps.out(f"  🎙️  recording {args.seconds:g} s …")
        speech, seconds = speech_only(recorder.record(args.seconds), vad)
        if seconds < MIN_SPEECH_S:
            deps.out(f"  ❌ only {seconds:.2f} s of speech — skipped")
            continue
        deps.out(f"  ✅ {seconds:.2f} s of speech")
        clips.append(speech)
        labels.append(f"{index + 1:4d} | {seconds:5.2f}s")
    return clips, labels


def run(args: argparse.Namespace, deps: Deps) -> int:
    """Record, calibrate and (unless ``--dry-run``) save; the exit code."""
    if deps.daemon_running():
        deps.out(STOP_HINT)
        return 2
    deps.out("⏳ loading the speaker model and the VAD …")
    embed = deps.embedder()
    vad = deps.vad()
    duplex = deps.duplex()
    if not duplex.start():
        deps.out("❌ VoiceProcessingIO did not start (aec extra installed? mic permission?)")
        return 2
    try:
        deps.out(f"✅ recording through VoiceProcessingIO ({SAMPLE_RATE} Hz) for '{args.name}'")
        clips, labels = _record_clips(args, deps, Recorder(duplex), vad)
    finally:
        duplex.close()
    if len(clips) < MIN_CLIPS:
        deps.out(f"❌ {len(clips)} usable clip(s), need {MIN_CLIPS} — nothing saved")
        return 1
    embeddings = [np.asarray(embed(clip), dtype=np.float32) for clip in clips]
    wake_path = args.out / f"{args.name}.npy"
    wake = _l2(np.load(wake_path)) if wake_path.is_file() else None
    vs_wake = [None if wake is None else float(_l2(e) @ wake) for e in embeddings]
    lines, _ = calibration_table(labels, leave_one_out(embeddings), vs_wake)
    for line in lines:
        deps.out(line)
    if wake is None:
        deps.out(f"   (no wake profile at {wake_path})")
    if args.dry_run:
        deps.out("✅ dry run — nothing written")
        return 0
    target = args.out / "call" / f"{args.name}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, call_centroid(embeddings).astype(np.float32))
    deps.out(f"✅ saved {target} ({len(embeddings)} clips); restart mac-voice to use it")
    return 0


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if not NAME_RE.fullmatch(args.name):
        print(f"❌ name must be letters, digits, '_' or '-': {args.name!r}", file=sys.stderr)
        return 2
    if args.clips < 1 or args.seconds <= 0:
        print("❌ --clips must be ≥ 1 and --seconds > 0", file=sys.stderr)
        return 2
    return run(args, deps or Deps())


if __name__ == "__main__":
    raise SystemExit(main())
