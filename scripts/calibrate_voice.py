#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Recommend the voice-check threshold and margin from your takes and labelled clips (tp#1235).

mac-voice lets a sentence send, answer or confirm only when it scores at least
``MAC_VOICE_AUTH_THRESHOLD`` against Albert's call profile — and, with ``MAC_VOICE_AUTH_MARGIN``
set, beats its best score against known other voices by that margin. This script measures
both on everything there is:

* positives: the enrollment takes that built the call profile (``scripts/enroll_voice.py``),
  each scored leave-one-out, and in-call clips labelled ``albert``
  (``scripts/label_voice_clips.py``);
* negatives: in-call clips labelled ``other`` and synthetic text-to-speech voices
  (``enroll/negatives/synthetic/*.wav``; ``-g`` makes them with macOS ``say`` and Piper).

It prints the score spread per group, how the current settings do, and a recommendation
aiming for no negative passing while ≥ 95% of Albert's short sentences (long enough to
authorise anything) still pass — see ``my_stt_tts.voice_calibration``. The margin rule is
only recommended once at least 10 clips are labelled ``other``.

Dry run by default. ``-a`` writes ``MAC_VOICE_AUTH_THRESHOLD`` (and ``MAC_VOICE_AUTH_MARGIN``,
or removes it) to the repo ``.env`` and the negative embeddings to
``enroll/negatives/call.npy``; it refuses until the call profile was built by
``enroll_voice.py`` (so the current values stay until Albert re-enrolled). Restart mac-voice
afterwards.

Usage:
    scripts/calibrate_voice.py            # report + recommendation, writes nothing
    scripts/calibrate_voice.py -g         # first (re)make the synthetic negative voices
    scripts/calibrate_voice.py -a         # write the recommendation to .env
    scripts/calibrate_voice.py -t 0.9 -N 5   # 90% target, margin from 5 'other' clips

Exit codes: 0 done, 1 nothing to recommend or -a refused, 2 usage error / no call profile.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from _bootstrap import ensure_venv

ensure_venv(["speaker", "vad"])

# pylint: disable=wrong-import-position,wrong-import-order  # after the venv re-exec
import numpy as np  # noqa: E402, RUF100

from my_stt_tts import bridge  # noqa: E402, RUF100
from my_stt_tts import interference as itf  # noqa: E402, RUF100
from my_stt_tts import voice_calibration as vc  # noqa: E402, RUF100
from my_stt_tts import voice_profiles as vp  # noqa: E402, RUF100
from my_stt_tts.voice_control import restart_hint  # noqa: E402, RUF100

# pylint: enable=wrong-import-position,wrong-import-order

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "enroll"
SYNTHETIC_SUBDIR = Path("negatives") / "synthetic"
NEGATIVES_FILE = Path("negatives") / "call.npy"  # voice_gate.NEGATIVES_FILE
SAY_SENTENCES = {
    "de": ("Schick das bitte an Peter.", "Ja, mach das jetzt sofort, bitte."),
    "en": ("Answer option two, please.", "Yes, please do it now and send it."),
    "fr": ("Réponds avec l'option deux.", "Oui, fais-le maintenant, s'il te plaît."),
}
MAX_SAY_VOICES = 24
LANGUAGE_ORDER = ("de", "en", "fr")
NOVELTY_VOICES = frozenset(
    {
        "Albert",
        "Bad News",
        "Bahh",
        "Bells",
        "Boing",
        "Bubbles",
        "Cellos",
        "Good News",
        "Jester",
        "Junior",
        "Organ",
        "Superstar",
        "Trinoids",
        "Whisper",
        "Wobble",
        "Zarvox",
    }
)
SAY_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2})_[A-Z]{2}\s+#")


def _default_embedder() -> Callable[[np.ndarray], np.ndarray]:
    from my_stt_tts.voice_gate import default_embedder

    return default_embedder()


def _default_vad() -> Any:
    from my_stt_tts.vad import SileroVad

    return SileroVad()


def _say_list() -> str:
    """``say -v '?'``: the installed macOS voices ('' without ``say``)."""
    if not shutil.which("say"):
        return ""
    res = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, check=False)
    return res.stdout


def _which_piper() -> str | None:
    return shutil.which("piper")


def _run(cmd: list[str], stdin: str | None = None) -> bool:
    try:
        res = subprocess.run(
            cmd, input=stdin, text=True, capture_output=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return res.returncode == 0


@dataclass
class Deps:
    """The models and tools (tests replace them)."""

    embedder: Callable[[], Callable[[np.ndarray], np.ndarray]] = _default_embedder
    vad: Callable[[], Any] = _default_vad
    run: Callable[..., bool] = _run
    say_list: Callable[[], str] = _say_list
    piper: Callable[[], str | None] = _which_piper
    out: Callable[[str], None] = print
    voices_dir: Path = REPO_ROOT / "voices"


# -- synthetic negatives ---------------------------------------------------------------------
def say_voices(listing: str) -> list[tuple[str, str]]:
    """(voice name, language) of real German / English / French voices in ``say -v '?'``.

    Novelty voices (bells, whisper, robots) are left out, and a voice that exists in several
    languages ("Eddy (German (Germany))", "Eddy (English (UK))") counts once, German first.
    """
    found: dict[str, tuple[str, str]] = {}
    for line in listing.splitlines():
        match = SAY_LINE.match(line)
        if not match or match["locale"] not in SAY_SENTENCES:
            continue
        name = match["name"].strip()
        base = name.split(" (", 1)[0]
        if base in NOVELTY_VOICES:
            continue
        rank = LANGUAGE_ORDER.index(match["locale"])
        if base not in found or rank < LANGUAGE_ORDER.index(found[base][1]):
            found[base] = (name, match["locale"])
    ordered = sorted(found.values(), key=lambda v: (LANGUAGE_ORDER.index(v[1]), v[0]))
    return ordered[:MAX_SAY_VOICES]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def _to_16k(deps: Deps, src: Path, dest: Path) -> bool:
    ok = deps.run(
        ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(dest)]
    )
    if ok and dest.is_file():
        dest.chmod(vp.PRIVATE_FILE)
    return ok and dest.is_file()


def generate_synthetic(folder: Path, deps: Deps) -> int:
    """(Re)make the synthetic negatives with ``say`` and Piper; returns how many."""
    vp.private_dir(folder)
    for old in [*folder.glob("say-*.wav"), *folder.glob("piper-*.wav")]:
        old.unlink()
    made = 0
    listing = deps.say_list()
    with tempfile.TemporaryDirectory() as tmp:
        for name, lang in say_voices(listing):
            for n, sentence in enumerate(SAY_SENTENCES[lang], 1):
                raw = Path(tmp) / "say.aiff"
                if deps.run(["/usr/bin/say", "-v", name, "-o", str(raw), sentence]) and _to_16k(
                    deps, raw, folder / f"say-{_slug(name)}-{n}.wav"
                ):
                    made += 1
        piper = deps.piper()
        for model in sorted(deps.voices_dir.glob("*.onnx")) if piper else []:
            lang = model.name[:2]
            for n, sentence in enumerate(SAY_SENTENCES.get(lang, SAY_SENTENCES["en"]), 1):
                raw = Path(tmp) / "piper.wav"
                if deps.run([str(piper), "-m", str(model), "-f", str(raw)], stdin=sentence) and (
                    _to_16k(deps, raw, folder / f"piper-{model.stem}-{n}.wav")
                ):
                    made += 1
    return made


# -- scoring ---------------------------------------------------------------------------------
@dataclass
class Clip:  # pylint: disable=too-many-instance-attributes  # one embedded clip
    """One embedded clip before it becomes a :class:`vc.Sample`."""

    source: str
    positive: bool
    emb: np.ndarray
    seconds: float
    short: bool
    name: str
    score: float = 0.0
    negative_index: int | None = None  # its row in the negatives matrix


@dataclass
class Collected:
    """Everything :func:`collect` found."""

    clips: list[Clip] = field(default_factory=list)
    enrolled_ok: bool = False
    enrolled_why: str = ""
    counts: dict[str, int] = field(default_factory=dict)


def _speech(path: Path, vad: Any) -> tuple[np.ndarray, float]:
    pcm = vp.read_wav(path)
    speech, seconds = vp.speech_only(pcm, vad)
    if seconds <= 0:
        return pcm, pcm.size / vp.SAMPLE_RATE
    return speech, seconds


def _enrolled(
    who: str, root: Path, embed: Callable[[np.ndarray], np.ndarray], vad: Any
) -> tuple[list[Clip], str]:
    """The takes that built the current call profile ([] + why when they cannot be trusted)."""
    manifest = vp.read_json(root / who / "profiles.json") or {}
    call = manifest.get("call") or {}
    profile = root / "call" / f"{who}.npy"
    if not call:
        return [], "no enrollment yet — run scripts/enroll_voice.py first"
    if call.get("sha256") != vp.sha256_file(profile):
        return [], f"{profile} changed since enroll_voice.py built it — re-run enroll_voice.py -R"
    kinds = {}
    for folder in vp.session_dirs(who, root):
        for entry in (vp.read_json(folder / "session.json") or {}).get("takes", []):
            for key in ("call_file", "raw_file"):
                if entry.get(key):
                    kinds[str((folder / entry[key]).relative_to(root))] = entry.get("kind", "")
    clips = []
    for rel in call.get("takes", []):
        path = root / rel
        if not path.is_file():
            continue
        speech, seconds = _speech(path, vad)
        if seconds < vp.MIN_TAKE_SPEECH_S:
            continue
        short = kinds.get(rel) == "short" if rel in kinds else seconds < vc.SHORT_S
        clips.append(Clip("enrolled", True, vp.l2(embed(speech)), seconds, short, rel))
    if len(clips) < 2:
        return [], "fewer than 2 usable enrollment takes"
    for clip, score in zip(clips, vp.leave_one_out([c.emb for c in clips]), strict=True):
        clip.score = score
    return clips, ""


def collect(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    who: str,
    root: Path,
    clips_dir: Path,
    profile: np.ndarray,
    deps: Deps,
    embed: Callable[[np.ndarray], np.ndarray],
) -> Collected:
    """Embed and score every positive and negative clip."""
    vad = deps.vad()
    found = Collected()
    enrolled, why = _enrolled(who, root, embed, vad)
    found.enrolled_ok, found.enrolled_why = bool(enrolled), why
    found.clips += enrolled
    for kept in itf.ClipStore(clips_dir).clips():
        if kept.label is None:
            continue
        emb = vp.l2(embed(vp.read_wav(kept.wav)))
        positive = kept.label == "albert"
        found.clips.append(
            Clip(
                "labelled",
                positive,
                emb,
                kept.duration,
                kept.duration < vc.SHORT_S,
                kept.id,
                float(emb @ profile),
            )
        )
    for wav in sorted((root / SYNTHETIC_SUBDIR).glob("*.wav")):
        speech, seconds = _speech(wav, vad)
        emb = vp.l2(embed(speech))
        found.clips.append(
            Clip(
                "synthetic",
                False,
                emb,
                seconds,
                seconds < vc.SHORT_S,
                wav.stem,
                float(emb @ profile),
            )
        )
    for clip in found.clips:
        key = f"{clip.source} {'albert' if clip.positive else 'other'}"
        found.counts[key] = found.counts.get(key, 0) + 1
    return found


def to_samples(clips: list[Clip]) -> tuple[list[vc.Sample], np.ndarray | None]:
    """Samples with ``best_negative`` (a negative never compared with itself) + the matrix."""
    negatives = [c for c in clips if not c.positive]
    for index, clip in enumerate(negatives):
        clip.negative_index = index
    matrix = np.stack([c.emb for c in negatives]) if negatives else None
    samples = []
    for clip in clips:
        best = None
        if matrix is not None:
            sims = matrix @ clip.emb
            if clip.negative_index is not None:
                sims = np.delete(sims, clip.negative_index)
            best = float(np.max(sims)) if sims.size else None
        samples.append(
            vc.Sample(
                clip.source, clip.positive, clip.score, best, clip.seconds, clip.short, clip.name
            )
        )
    return samples, matrix


# -- report ----------------------------------------------------------------------------------
def spread_table(samples: list[vc.Sample], min_s: float) -> list[str]:
    """Score spread per group."""
    groups = [
        (
            "Albert, short sentences",
            [s for s in samples if s.positive and s.short and s.seconds >= min_s],
        ),
        (
            "Albert, long sentences",
            [s for s in samples if s.positive and not s.short and s.seconds >= min_s],
        ),
        (
            f"Albert, under {min_s:.1f} s (refused by length)",
            [s for s in samples if s.positive and s.seconds < min_s],
        ),
        ("others, labelled", [s for s in samples if not s.positive and s.source == "labelled"]),
        ("synthetic voices", [s for s in samples if s.source == "synthetic"]),
    ]
    lines = [
        "| group                               |   n |   min |    p5 | median |   max |",
        "| :---------------------------------- | --: | ----: | ----: | -----: | ----: |",
    ]
    for name, picked in groups:
        if not picked:
            continue
        scores = sorted(s.score for s in picked)
        p5 = scores[int(0.05 * (len(scores) - 1))]
        lines.append(
            f"| {name:<35} | {len(scores):3d} | {scores[0]:5.2f} | {p5:5.2f} |"
            f" {statistics.median(scores):6.2f} | {scores[-1]:5.2f} |"
        )
    return lines


def describe(outcome: vc.Outcome) -> str:
    """``short 22/23 (96%), all 40/41, 0/32 negatives pass``."""
    margin = "off" if outcome.margin is None else f"{outcome.margin:.2f}"
    rate = f"{outcome.short_rate:.0%}" if outcome.short_total else "-"
    text = (
        f"threshold {outcome.threshold:.2f}, margin {margin}: Albert's short sentences"
        f" {outcome.short_accepted}/{outcome.short_total} ({rate}), all"
        f" {outcome.accepted}/{outcome.total}; negatives passing"
        f" {outcome.false_accepts}/{outcome.negatives}"
    )
    if outcome.false_accept_names:
        text += " (" + ", ".join(outcome.false_accept_names[:5]) + ")"
    return text


def current_settings(env_file: Path) -> tuple[float, float | None, float]:
    """(threshold, margin or None, minimum seconds) as the daemon reads them."""
    threshold = vp.env_float(bridge.AUTH_THRESHOLD_ENV, bridge.AUTH_THRESHOLD, env_file)
    min_s = max(vp.env_float(bridge.AUTH_MIN_S_ENV, bridge.AUTH_MIN_S, env_file), 0.4)
    raw = vp.env_value(bridge.AUTH_MARGIN_ENV, env_file)
    try:
        margin: float | None = float(raw) if raw else None
    except ValueError:
        margin = None
    return threshold, margin, min_s


def apply(
    env_file: Path, root: Path, rec: vc.Recommendation, matrix: np.ndarray | None
) -> list[str]:
    """Write the recommendation; the lines saying what changed."""
    from dotenv import set_key, unset_key

    lines = []
    set_key(str(env_file), bridge.AUTH_THRESHOLD_ENV, f"{rec.threshold:.2f}", quote_mode="never")
    lines.append(f"✅ {bridge.AUTH_THRESHOLD_ENV}={rec.threshold:.2f} in {env_file}")
    if rec.margin is not None and matrix is not None:
        target = root / NEGATIVES_FILE
        vp.save_profile(target, matrix)
        set_key(str(env_file), bridge.AUTH_MARGIN_ENV, f"{rec.margin:.2f}", quote_mode="never")
        lines.append(
            f"✅ {bridge.AUTH_MARGIN_ENV}={rec.margin:.2f}, {len(matrix)} negatives in {target}"
        )
    elif vp.env_value(bridge.AUTH_MARGIN_ENV, env_file, env={}):
        unset_key(str(env_file), bridge.AUTH_MARGIN_ENV)
        lines.append(f"✅ {bridge.AUTH_MARGIN_ENV} removed (margin rule off)")
    return lines


# -- main ------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-w", "--who", default="albert", help="whose call profile (default albert)")
    parser.add_argument("-a", "--apply", action="store_true", help="write the result to .env")
    parser.add_argument(
        "-g", "--generate", action="store_true", help="(re)make the synthetic negatives first"
    )
    parser.add_argument(
        "-t",
        "--target",
        type=float,
        default=vc.DEFAULT_TARGET,
        help="share of short sentences that must pass (default 0.95)",
    )
    parser.add_argument(
        "-N",
        "--min-others",
        type=int,
        default=vc.DEFAULT_MIN_OTHERS,
        help="labelled 'other' clips needed for the margin rule (default 10)",
    )
    parser.add_argument("-o", "--out", type=Path, default=DEFAULT_ROOT, help="profile folder")
    parser.add_argument("-D", "--dir", type=Path, default=itf.ROOT, help="interference clips")
    parser.add_argument("-E", "--env-file", type=Path, default=vp.ENV_FILE, help=".env to write")
    return parser


def run(args: argparse.Namespace, deps: Deps) -> int:  # pylint: disable=too-many-locals
    """Measure, recommend and (with ``--apply``) write; the exit code."""
    if args.generate:
        made = generate_synthetic(args.out / SYNTHETIC_SUBDIR, deps)
        deps.out(f"✅ {made} synthetic voice clip(s) in {args.out / SYNTHETIC_SUBDIR}")
    profile = vp.load_profile(args.out / "call" / f"{args.who}.npy")
    if profile is None:
        deps.out(
            f"❌ no call profile {args.out / 'call' / (args.who + '.npy')} — run enroll_voice.py"
        )
        return 2
    deps.out("⏳ loading the speaker model and scoring every clip …")
    found = collect(args.who, args.out, args.dir, profile, deps, deps.embedder())
    samples, matrix = to_samples(found.clips)
    threshold, margin, min_s = current_settings(args.env_file)
    deps.out(
        f"\nVoice calibration for {args.who}: "
        + ", ".join(f"{n} {k}" for k, n in sorted(found.counts.items()))
    )
    if not found.enrolled_ok:
        deps.out(f"⚠️  enrollment takes not used: {found.enrolled_why}")
    for line in spread_table(samples, min_s):
        deps.out(line)
    deps.out(f"Now:         {describe(vc.evaluate(samples, threshold, margin, min_s))}")
    rec = vc.recommend(samples, min_s=min_s, target=args.target, min_others=args.min_others)
    if rec is None:
        deps.out(
            f"❌ too few of Albert's short sentences (≥ {min_s:.1f} s) to recommend anything —"
            f" need {vc.MIN_SHORT_POSITIVES}; run scripts/enroll_voice.py"
        )
        return 1
    mark = "✅" if rec.feasible else "⚠️ "
    deps.out(f"Recommended: {describe(rec.outcome)} {mark}")
    for note in rec.notes:
        deps.out(f"   {note}")
    if rec.threshold < threshold:
        deps.out(f"⚠️  that lowers the threshold from {threshold:.2f}")
    if not args.apply:
        deps.out("dry run — nothing written; apply with: scripts/calibrate_voice.py -a")
        return 0
    if not found.enrolled_ok:
        deps.out(
            "❌ not applied: the current values stay until the call profile comes from"
            " enroll_voice.py"
        )
        return 1
    for line in apply(args.env_file, args.out, rec, matrix):
        deps.out(line)
    deps.out(restart_hint("it"))
    return 0


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if not 0.5 <= args.target <= 1.0 or args.min_others < 1:
        print("❌ --target must be 0.5–1.0 and --min-others ≥ 1", file=sys.stderr)
        return 2
    return run(args, deps or Deps())


if __name__ == "__main__":
    raise SystemExit(main())
