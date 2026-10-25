#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Guided voice re-enrollment: read 25 sentences aloud, rebuild the voice profiles (tp#1235).

mac-voice decides from the voice alone whether a sentence is yours (who may start a call,
who may send, answer or confirm). This script trains that check on more and better audio:

1. It shows 25 sentences, German and English, short commands ("Answer option two.") and
   longer sentences, one at a time. Press Enter, read the sentence aloud, wait for the next.
   A take with under 1 s of speech can be repeated (Enter) or skipped (s); q finishes early.
2. Each take is recorded through VoiceProcessingIO, the echo-cancelled path a mac-voice call
   uses, and at the same time from the plain mic (the wake word's path) when macOS allows a
   second input stream. Without VoiceProcessingIO it records the plain mic only and says so.
3. Every take is kept, privately (0600, gitignored): ``enroll/<who>/<stamp>/call/*.wav``,
   ``raw/*.wav`` and ``session.json`` (sentence, language, seconds of speech).
4. It rebuilds the call profile ``enroll/call/<who>.npy`` from the call takes of ALL sessions
   and the wake profile ``enroll/<who>.npy`` from the wake-word clips plus the plain-mic
   takes. The old profiles are copied to ``enroll/<who>/backup/`` first.
5. It prints a before/after table: each new take scored against the OLD call profile, and
   leave-one-out against the NEW one (the profile built from all other takes), with how many
   short commands and long sentences reach the voice-check threshold.

The mac-voice daemon holds the mic: when it runs, the script offers to stop it and starts it
again at the end (``-y`` without asking), so the new profiles are used right away.

Usage:
    scripts/enroll_voice.py                  # Albert: all 25 sentences (~15 min)
    scripts/enroll_voice.py -c 6             # only the first 6 sentences (a quick top-up)
    scripts/enroll_voice.py -l de            # German sentences only
    scripts/enroll_voice.py -R               # no recording: rebuild from the kept takes
    scripts/enroll_voice.py -w kid -y        # another person, stop the daemon without asking

Next step: ``scripts/calibrate_voice.py`` recommends the threshold for the new profile.

Exit codes: 0 profiles rebuilt, 1 too few usable takes (takes kept, profiles unchanged),
2 usage error / daemon not stoppable / audio unavailable.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _bootstrap import ensure_venv

ensure_venv(["audio", "speaker", "vad", "aec"])

# pylint: disable=wrong-import-position,wrong-import-order  # after the venv re-exec
import numpy as np  # noqa: E402, RUF100  # E402: the commit hook's ruff selects it
from enroll_call import Duplex, Recorder, SpeechDetector  # noqa: E402, RUF100

from my_stt_tts import voice_profiles as vp  # noqa: E402, RUF100
from my_stt_tts.voice_control import restart_hint  # noqa: E402, RUF100

# pylint: enable=wrong-import-position,wrong-import-order

log = logging.getLogger("enroll_voice")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "enroll"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 4000
MAX_ATTEMPTS = 3
NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
LANGUAGE_NAMES = {"de": "German", "en": "English"}
LAUNCH_LABEL = "com.albert.mac-voice"  # voice_control.LAUNCH_LABEL
LAUNCH_AGENT = Path.home() / "Library/LaunchAgents" / f"{LAUNCH_LABEL}.plist"


@dataclass(frozen=True)
class Sentence:
    """One sentence to read: language code, ``short`` (a command) or ``long``, the text."""

    lang: str
    kind: str
    text: str


SENTENCES: tuple[Sentence, ...] = (
    Sentence("en", "short", "Yes, please do it now."),
    Sentence("de", "short", "Ja, bitte mach das jetzt."),
    Sentence("en", "short", "Answer option two."),
    Sentence("de", "short", "Antworte mit Option zwei."),
    Sentence("en", "short", "Send that to Peter."),
    Sentence("de", "short", "Schick das an Peter."),
    Sentence("en", "long", "Tell the session in the voice repo to run the tests again."),
    Sentence("de", "long", "Sag der Session im Home-Assistant-Repo, sie soll das Licht prüfen."),
    Sentence("en", "short", "What's on my plate today?"),
    Sentence("de", "short", "Was steht heute auf meiner Liste?"),
    Sentence("en", "short", "Go with the recommendation."),
    Sentence("de", "short", "Nein, brich das bitte ab."),
    Sentence(
        "en",
        "long",
        "Send a message to the infra session: the backup finished, please check the logs.",
    ),
    Sentence(
        "de",
        "long",
        "Schreib eine Nachricht, dass ich in etwa zwanzig Minuten zu Hause bin.",
    ),
    Sentence("en", "short", "Read me the next decision."),
    Sentence("de", "short", "Lies mir die nächste Entscheidung vor."),
    Sentence("en", "short", "Open YouTube and play the first video."),
    Sentence("de", "short", "Mach den Bildschirm etwas heller."),
    Sentence(
        "en",
        "long",
        "I'd like to hear the brief for the second decision, then snooze the rest until tomorrow.",
    ),
    Sentence(
        "de",
        "long",
        "Ich nehme die Empfehlung an und füge hinzu, dass wir das nächste Woche anschauen.",
    ),
    Sentence("en", "short", "Stop, wait a moment."),
    Sentence("de", "short", "Ja, genau so, schick es ab."),
    Sentence(
        "en",
        "long",
        "Open Safari, go to the Jellyfin page and continue the last episode where I stopped.",
    ),
    Sentence(
        "de",
        "long",
        "Erinnere mich morgen früh um neun daran, die Rechnung der Versicherung zu bezahlen.",
    ),
    Sentence("en", "long", "Please draft an email saying that the meeting moves to Thursday."),
)


def record_seconds(text: str) -> float:
    """How long to record a sentence: 2 s plus ~0.45 s per word, 3.5–10 s."""
    return min(10.0, max(3.5, 2.0 + 0.45 * len(text.split())))


# -- audio sources ---------------------------------------------------------------------------
class SoundDeviceMic:
    """The plain mic (no echo cancellation) as a :class:`Duplex`: 16 kHz, 250 ms frames."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_samples: int = FRAME_SAMPLES):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._closed = threading.Event()
        self._stream: Any = None

    def start(self) -> bool:
        try:
            import sounddevice as sd

            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_samples,
                callback=lambda data, *_: self._queue.put(
                    np.frombuffer(bytes(data), dtype="<i2").astype(np.float32) / 32768.0
                ),
            )
            self._stream.start()
        except Exception:  # pylint: disable=broad-exception-caught  # no second stream
            log.debug("plain mic unavailable", exc_info=True)
            self._stream = None
            return False
        return True

    def mic_frames(self) -> Iterator[np.ndarray]:
        while not self._closed.is_set():
            try:
                yield self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed.set()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()


def _default_duplex() -> Duplex:
    from my_stt_tts.aec import VoiceProcessingDuplex

    return VoiceProcessingDuplex(SAMPLE_RATE, frame_samples=FRAME_SAMPLES)


def _default_vad() -> SpeechDetector:
    from my_stt_tts.vad import SileroVad

    return SileroVad()


def _default_embedder() -> Callable[[np.ndarray], np.ndarray]:
    from my_stt_tts.voice_gate import default_embedder

    return default_embedder()


class LaunchdDaemon:
    """Stop / start the mac-voice LaunchAgent (the daemon owns the mic during a call).

    A daemon started by hand (``./mac-voice -d`` in a tab) is detected (``managed()`` is False)
    and never killed: the script prints how to stop and start it there instead.
    """

    def running(self) -> bool:
        from my_stt_tts.voice_control import send

        return send("status", timeout=1.0) is not None

    def managed(self) -> bool:
        from my_stt_tts.voice_control import launch_agent_loaded

        return launch_agent_loaded()

    def stop_hint(self) -> str:
        from my_stt_tts.voice_control import send, stop_hint

        reply = send("status", timeout=1.0) or {}
        pid = reply.get("pid")
        return stop_hint(pid if isinstance(pid, int) else None, loaded=self.managed())

    def stop(self) -> bool:
        res = subprocess.run(
            ["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCH_LABEL}"],
            check=False,
            capture_output=True,
        )
        if res.returncode != 0:
            return False
        for _ in range(50):  # the daemon closes its socket on the way out
            if not self.running():
                return True
            time.sleep(0.1)
        return not self.running()

    def start(self) -> bool:
        if not LAUNCH_AGENT.is_file():
            return False
        res = subprocess.run(
            ["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(LAUNCH_AGENT)],
            check=False,
            capture_output=True,
        )
        return res.returncode == 0


@dataclass
class Deps:  # pylint: disable=too-many-instance-attributes  # the hardware / model seams
    """What the script needs from the outside world (tests replace it)."""

    duplex: Callable[[], Duplex] = _default_duplex
    raw_mic: Callable[[], Duplex] = SoundDeviceMic
    vad: Callable[[], SpeechDetector] = _default_vad
    embedder: Callable[[], Callable[[np.ndarray], np.ndarray]] = _default_embedder
    daemon: Any = field(default_factory=LaunchdDaemon)
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = print
    stamp: Callable[[], str] = lambda: time.strftime("%Y%m%d-%H%M%S")
    #: Where the wake-word enrollment clips live (part of the wake profile).
    wake_clips: Path = vp.WAKE_CLIPS
    #: Where the voice-check threshold is read from (dotenv file, then the environment).
    env_file: Path = vp.ENV_FILE
    env: Mapping[str, str] | None = None


# -- recording -------------------------------------------------------------------------------
@dataclass
class Sources:
    """The open audio sources of one session."""

    primary: Recorder
    domain: str  # "call" (VoiceProcessingIO) or "raw"
    twin: Recorder | None = None
    closers: list[Callable[[], None]] = field(default_factory=list)

    def capture(self, seconds: float) -> tuple[np.ndarray, np.ndarray | None]:
        """(primary take, plain-mic twin or None), recorded at the same time."""
        if self.twin is None:
            return self.primary.record(seconds), None
        box: dict[str, np.ndarray] = {}
        twin = self.twin
        thread = threading.Thread(
            target=lambda: box.setdefault("raw", twin.record(seconds)), daemon=True
        )
        thread.start()
        take = self.primary.record(seconds)
        thread.join(seconds + 5.0)
        return take, box.get("raw")

    def close(self) -> None:
        for close in self.closers:
            with contextlib.suppress(Exception):
                close()


def open_sources(args: argparse.Namespace, deps: Deps) -> Sources | None:
    """VoiceProcessingIO (+ the plain-mic twin), or the plain mic alone; None when neither."""
    if not args.raw:
        duplex = deps.duplex()
        if duplex.start():
            sources = Sources(Recorder(duplex), "call", closers=[duplex.close])
            if not args.no_twin:
                twin = deps.raw_mic()
                if twin.start():
                    sources.twin = Recorder(twin)
                    sources.closers.append(twin.close)
                    deps.out("✅ recording through VoiceProcessingIO and the plain mic")
                else:
                    deps.out(
                        "✅ recording through VoiceProcessingIO · ⚠️  no second (plain-mic)"
                        " stream: the wake profile keeps using the wake-word clips"
                    )
            else:
                deps.out("✅ recording through VoiceProcessingIO")
            return sources
        deps.out(
            "⚠️  VoiceProcessingIO did not start — recording the plain mic instead. In-call"
            " audio runs through VoiceProcessingIO, so the call profile will match calls less"
            " well; re-run without -r once it works (aec extra, mic permission)."
        )
    mic = deps.raw_mic()
    if not mic.start():
        deps.out("❌ no microphone could be opened (mic permission for this terminal?)")
        return None
    if args.raw:
        deps.out("✅ recording the plain mic (-r)")
    return Sources(Recorder(mic), "raw", closers=[mic.close])


def pick_sentences(args: argparse.Namespace) -> list[Sentence]:
    """The sentences for this session: language filter, then the first ``--count``."""
    langs = {code.strip().lower() for code in args.languages.split(",") if code.strip()}
    chosen = [s for s in SENTENCES if not langs or s.lang in langs]
    return chosen[: args.count] if args.count else chosen


@dataclass
class Session:
    """One recording session's folder and ``session.json`` entries."""

    folder: Path
    who: str
    domain: str
    twin: bool
    entries: list[dict[str, Any]] = field(default_factory=list)

    def save(self) -> None:
        vp.write_json(
            self.folder / "session.json",
            {
                "who": self.who,
                "started": self.folder.name,
                "domain": "vpio" if self.domain == "call" else "raw",
                "plain_mic_twin": self.twin,
                "sample_rate": SAMPLE_RATE,
                "takes": self.entries,
            },
        )


def _record_take(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    sources: Sources,
    vad: SpeechDetector,
    sentence: Sentence,
    number: int,
    attempt: int,
    seconds: float,
) -> dict[str, Any]:
    """Record one take, keep its files, return its ``session.json`` entry."""
    take, twin = sources.capture(seconds)
    _, speech_s = vp.speech_only(take, vad)
    name = f"{number:02d}-a{attempt}.wav"
    entry: dict[str, Any] = {
        "n": number,
        "attempt": attempt,
        "lang": sentence.lang,
        "kind": sentence.kind,
        "text": sentence.text,
        "seconds": round(take.size / SAMPLE_RATE, 2),
        "speech_s": round(speech_s, 2),
        "usable": speech_s >= vp.MIN_TAKE_SPEECH_S,
    }
    if sources.domain == "call":
        vp.write_wav(session.folder / "call" / name, take)
        entry["call_file"] = f"call/{name}"
        if twin is not None and twin.size:
            vp.write_wav(session.folder / "raw" / name, twin)
            entry["raw_file"] = f"raw/{name}"
    else:
        vp.write_wav(session.folder / "raw" / name, take)
        entry["raw_file"] = f"raw/{name}"
    session.entries.append(entry)
    session.save()
    return entry


def record_session(  # pylint: disable=too-many-locals
    args: argparse.Namespace, deps: Deps, sources: Sources, vad: SpeechDetector
) -> Session:
    """Walk through the sentences; every take is kept as soon as it is recorded."""
    folder = vp.private_dir(args.out / args.who / deps.stamp())
    session = Session(folder, args.who, sources.domain, sources.twin is not None)
    sentences = pick_sentences(args)
    deps.out(
        f"\n{len(sentences)} sentences. Read each one aloud in your normal voice, at your"
        " normal distance from the Mac, as you would talk to mac-voice.\n"
    )
    for number, sentence in enumerate(sentences, 1):
        language = LANGUAGE_NAMES.get(sentence.lang, sentence.lang)
        deps.out(f"Sentence {number}/{len(sentences)} ({language}, {sentence.kind}):")
        deps.out(f'    "{sentence.text}"')
        seconds = args.seconds or record_seconds(sentence.text)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            answer = deps.ask("  Enter = record, s = skip, q = finish now: ").strip().lower()
            if answer == "q":
                return session
            if answer == "s":
                break
            deps.out(f"  🎙️  recording {seconds:.0f} s — read it now …")
            entry = _record_take(session, sources, vad, sentence, number, attempt, seconds)
            if entry["usable"]:
                deps.out(f"  ✅ {entry['speech_s']:.1f} s of speech")
                break
            more = " — Enter to try again" if attempt < MAX_ATTEMPTS else ""
            deps.out(f"  ❌ only {entry['speech_s']:.1f} s of speech{more}")
    return session


# -- scores ----------------------------------------------------------------------------------
def score_table(  # pylint: disable=too-many-locals
    rows: list[tuple[str, str, float, float | None, float]],
    threshold: float,
    min_s: float,
) -> list[str]:
    """The before/after table and per-kind summary lines.

    ``rows``: (label, kind, seconds of speech, score vs the OLD call profile or None,
    leave-one-out score vs the NEW one).
    """
    lines = [
        "| take             | speech | before | after |",
        "| :--------------- | -----: | -----: | ----: |",
    ]
    for label, _kind, speech_s, before, after in rows:
        before_txt = "-" if before is None else f"{before:.2f}"
        flag = " (too short for actions)" if speech_s < min_s else ""
        lines.append(f"| {label:<16} | {speech_s:5.1f}s | {before_txt:>6} | {after:5.2f} |{flag}")
    lines.append(
        f"before = against the old call profile, after = leave-one-out against the new one;"
        f" threshold {threshold:.2f}, actions need ≥ {min_s:.1f} s of speech"
    )
    for kind in ("short", "long"):
        picked = [r for r in rows if r[1] == kind and r[2] >= min_s]
        if not picked:
            continue
        after_ok = sum(r[4] >= threshold for r in picked)
        befores = [r[3] for r in picked if r[3] is not None]
        before_txt = (
            f"before {sum(b >= threshold for b in befores)}/{len(befores)}" if befores else ""
        )
        mid_after = statistics.median(r[4] for r in picked)
        mark = "✅" if after_ok == len(picked) else "⚠️ "
        lines.append(
            f"{mark} {kind} sentences ≥ {threshold:.2f}: {before_txt}"
            f"{', ' if before_txt else ''}after {after_ok}/{len(picked)}"
            f" (after: min {min(r[4] for r in picked):.2f}, median {mid_after:.2f})"
        )
    return lines


def _rows(
    session_takes: list[vp.Take],
    rebuild: vp.Rebuild,
    old_call: np.ndarray | None,
    vad: SpeechDetector,
) -> list[tuple[str, str, float, float | None, float]]:
    """Table rows for the takes that built the new call profile and are in ``session_takes``."""
    sources = [p for p in rebuild.embeddings if _in_call_profile(p, rebuild)]
    loo = dict(
        zip(sources, vp.leave_one_out([rebuild.embeddings[p] for p in sources]), strict=True)
    )
    rows = []
    for take in session_takes:
        if take.path not in loo:
            continue
        emb = rebuild.embeddings[take.path]
        _, speech_s = vp.speech_only(vp.read_wav(take.path), vad)
        before = None if old_call is None else float(emb @ old_call)
        label = f"{take.session[-6:]}/{take.path.stem} {take.lang}"
        rows.append((label, take.kind, speech_s, before, loo[take.path]))
    return rows


def _in_call_profile(path: Path, rebuild: vp.Rebuild) -> bool:
    return path.parent.name == rebuild.call_domain


def thresholds(
    env_file: Path = vp.ENV_FILE, env: Mapping[str, str] | None = None
) -> tuple[float, float]:
    """(voice-check threshold, minimum seconds) as the daemon would read them."""
    from my_stt_tts import bridge

    threshold = vp.env_float(bridge.AUTH_THRESHOLD_ENV, bridge.AUTH_THRESHOLD, env_file, env)
    min_s = vp.env_float(bridge.AUTH_MIN_S_ENV, bridge.AUTH_MIN_S, env_file, env)
    return threshold, max(min_s, 0.4)


# -- main flow -------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-w", "--who", default="albert", help="whose profiles (default albert; file names)"
    )
    parser.add_argument(
        "-c", "--count", type=int, default=0, help="only the first N sentences (default: all 25)"
    )
    parser.add_argument(
        "-l", "--languages", default="de,en", help="sentence languages (default de,en)"
    )
    parser.add_argument(
        "-s",
        "--seconds",
        type=float,
        default=0.0,
        help="seconds per take (default: 3.5–10 s by sentence length)",
    )
    parser.add_argument(
        "-r", "--raw", action="store_true", help="record the plain mic, not VoiceProcessingIO"
    )
    parser.add_argument(
        "-T", "--no-twin", action="store_true", help="no second plain-mic copy of each take"
    )
    parser.add_argument(
        "-R", "--rebuild", action="store_true", help="no recording: rebuild from the kept takes"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="stop and restart the daemon without asking"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=DEFAULT_OUT, help="profile folder (default: enroll/)"
    )
    return parser


def _stop_daemon(args: argparse.Namespace, deps: Deps) -> bool | None:
    """None: not running; True: stopped (start it again later); False: refused / failed."""
    if not deps.daemon.running():
        return None
    if not deps.daemon.managed():  # started by hand: launchctl cannot stop or restart it
        deps.out(deps.daemon.stop_hint())
        return False
    if not args.yes:
        answer = deps.ask(
            "mac-voice is running and holds the mic. Stop it now and start it again at the"
            " end? [Y/n] "
        )
        if answer.strip().lower() not in ("", "y", "yes", "j", "ja"):
            deps.out(deps.daemon.stop_hint())
            return False
    if not deps.daemon.stop():
        deps.out(deps.daemon.stop_hint())
        return False
    deps.out("✅ mac-voice stopped for the enrollment")
    return True


def _rebuild_and_report(
    args: argparse.Namespace, deps: Deps, vad: SpeechDetector, session: str | None
) -> int:
    """Rebuild both profiles and print the before/after table; the exit code."""
    embed = deps.embedder()
    old_call = vp.load_profile(args.out / "call" / f"{args.who}.npy")
    rebuild = vp.rebuild_profiles(
        args.who, embed, vad, root=args.out, wake_clips=deps.wake_clips, stamp=deps.stamp()
    )
    takes = [
        t for t in vp.usable_takes(args.who, args.out) if session is None or t.session == session
    ]
    threshold, min_s = thresholds(deps.env_file, deps.env)
    rows = _rows(takes, rebuild, old_call, vad)
    if rows:
        deps.out("")
        for line in score_table(rows, threshold, min_s):
            deps.out(line)
    for note in rebuild.notes:
        deps.out(f"⚠️  {note}")
    if rebuild.call_path is None:
        deps.out("❌ too few usable takes — the takes are kept, the profiles are unchanged")
        return 1
    deps.out(
        f"✅ call profile {rebuild.call_path} ({rebuild.call_used} takes,"
        f" {'VoiceProcessingIO' if rebuild.call_domain == 'call' else 'plain mic'})"
    )
    if rebuild.wake_path is not None:
        deps.out(f"✅ wake profile {rebuild.wake_path} ({rebuild.wake_used} clips and takes)")
    for backup in rebuild.backups:
        deps.out(f"   previous profile saved as {backup}")
    deps.out("Next: scripts/calibrate_voice.py   (dry run: recommends the threshold)")
    return 0


def run(args: argparse.Namespace, deps: Deps) -> int:
    """Record (unless ``--rebuild``), rebuild, report; the exit code."""
    if args.rebuild:
        code = _rebuild_and_report(args, deps, deps.vad(), None)
        if code == 0:
            deps.out(restart_hint("them"))
        return code
    stopped = _stop_daemon(args, deps)
    if stopped is False:
        return 2
    try:
        deps.out("⏳ loading the voice activity detector …")
        vad = deps.vad()
        sources = open_sources(args, deps)
        if sources is None:
            return 2
        try:
            session = record_session(args, deps, sources, vad)
        finally:
            sources.close()
        usable = sum(1 for e in session.entries if e["usable"])
        deps.out(f"\n✅ kept {len(session.entries)} take(s) in {session.folder} ({usable} usable)")
        deps.out("⏳ loading the speaker model and rebuilding the profiles …")
        return _rebuild_and_report(args, deps, vad, session.folder.name)
    finally:
        if stopped:
            if deps.daemon.start():
                deps.out("✅ mac-voice started again with the new profiles")
            else:
                deps.out("❌ could not start mac-voice again — run: mac-voice -I")


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if not NAME_RE.fullmatch(args.who):
        print(f"❌ name must be letters, digits, '_' or '-': {args.who!r}", file=sys.stderr)
        return 2
    if args.count < 0 or args.seconds < 0:
        print("❌ --count and --seconds must be ≥ 0", file=sys.stderr)
        return 2
    try:
        return run(args, deps or Deps())
    except KeyboardInterrupt:
        print("\n❌ interrupted — takes recorded so far are kept; rebuild with -R")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
