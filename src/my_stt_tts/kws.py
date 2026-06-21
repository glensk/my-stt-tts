"""sherpa-onnx KeywordSpotter (KWS) — a second, OR'd wake detector for CUSTOM words.

Round-1 of the wake-detection checker loop. openWakeWord (oWW) needs a GPU retrain per
new word and empirically fails on a non-native accent (the self-trained ``maziko`` scores
~0 on Albert's voice). sherpa-onnx's :class:`~sherpa_onnx.KeywordSpotter` is
**open-vocabulary**: any phrase becomes a wake word just by typing its tokens — no
training — with a per-keyword boost + threshold and multiple spellings (accent variants)
all mapping to one logical word. It is therefore added as a SECOND detector that runs
**in parallel** with oWW and fires if EITHER fires (the OR-combine lives in
:mod:`my_stt_tts.wake`).

CRITICAL GUARDRAIL — official words are NEVER touched
-----------------------------------------------------
openWakeWord's official models (``hey_jarvis`` / ``alexa`` / ``hey_mycroft``) fire
99-100% on Albert's voice. KWS is an OR'd path for **custom / self-trained** words only;
the caller gates it on :func:`my_stt_tts.config.is_official_wake_word` so an official
word's behaviour is byte-identical to before.

Zero new dependency + coexistence
----------------------------------
This reuses the EXACT ``sherpa-onnx==1.10.46`` already pinned for the ``diarize`` extra
(see ``pyproject.toml``) — no new package. That wheel self-bundles its onnxruntime
(``libonnxruntime.1.17.1.dylib``), whose leaf name is distinct from the standalone
``onnxruntime`` that openWakeWord loads, so both run in ONE process with no dlopen clash
(verified). The GigaSpeech English zipformer-transducer KWS model (int8 encoder/decoder/
joiner ONNX + ``tokens.txt`` + ``bpe.model``) is auto-downloaded ONCE and
checksum-verified into the gitignored ``models/`` — MIRRORING :mod:`my_stt_tts.diarize`.

Everything is **gated + fully defensive**, like :mod:`diarize`: sherpa unavailable, the
model un-fetchable, or inference raising all degrade to a no-op detector (``detect`` →
``False``) — the wake loop NEVER dies because KWS hiccuped. The keyword vocabulary is
**UPPERCASE GigaSpeech BPE**: a phrase is uppercased then segmented with the model's own
``bpe.model`` via ``sentencepiece`` (which sherpa bundles) — NOT ``sherpa_onnx.text2token``
(it eagerly imports ``pypinyin``, which is not a dependency, even in pure-BPE mode).

A/B on Albert's real ``maziko`` clips (honest)
----------------------------------------------
On the 6 saved clips, oWW fires 1/6 (one clip @0.67; the other 5 are ~0.001-0.002 dead).
KWS at an aggressive boost/threshold recovers ONE of the oWW-dead clips (``d03f2ad3``) but
NOT the rest. So KWS adds the zero-train custom-word capability and recovers one otherwise
-missed activation — it does NOT fully "fix" maziko (GigaSpeech is English; a non-native
accent on a non-English word stays hard). See ``PLAN_kws_detector.md`` for the table.
"""

from __future__ import annotations

import bz2
import contextlib
import logging
import os
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .turn import verify_checksum

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from .config import Config

log = logging.getLogger("my_stt_tts.kws")

# The KWS model expects 16 kHz mono float32 in [-1, 1] (its own native input — no int16
# conversion needed, unlike openWakeWord). Trailing-silence flush so the LAST word in a
# capture is decoded: the streaming transducer needs a bit of post-roll to emit.
KWS_SAMPLE_RATE = 16000
TRAILING_SILENCE_S = 0.66

# The five model files inside the GigaSpeech KWS release dir we actually load (the int8
# variants — smaller + plenty accurate for keyword spotting). Mapped to per-file SHA-256
# so each is checksum-verified after extraction, mirroring diarize.py.
_MODEL_FILES: dict[str, str] = {
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx": (
        "1e721676515bcd42a186979733981213c66c80db680e1cc582dfedf3be76e678"
    ),
    "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx": (
        "e40ff43297abe815e8898494c17e71bba2152d9d40fa3eb803f75d0f7533329a"
    ),
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx": (
        "eae9da0c7e1e6c6a3f4cc42d167899c388f6c6701b94cb96320e4f55df79624c"
    ),
    "tokens.txt": "fd2ded4050a55d2b1578870ba8697d02371980217806b7558bd0a5cc60f3ba53",
    "bpe.model": "c8a2a0129c4ab8e463164c142f82d25649661b122c8cd0b7aab5c9e80b90ad24",
}
_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"


def _sherpa_importable() -> bool:
    """True if ``import sherpa_onnx`` actually succeeds (not just that it's installed).

    Identical probe to :func:`my_stt_tts.diarize._sherpa_importable` — ``find_spec`` only
    proves the package dir exists; the native C-extension can still fail to ``dlopen``.
    Any import-time error reads as "not usable" so the gating skips a broken install
    rather than downloading the model for it. (The 1.10.46 pin is the macOS arm64 wheel
    that self-bundles onnxruntime and coexists with openWakeWord's — see the module
    docstring + the ``diarize`` pin comment in ``pyproject.toml``.)
    """
    try:
        import sherpa_onnx  # noqa: F401 — import probe only
    except Exception:  # noqa: BLE001 — any load failure means sherpa is unusable here
        return False
    return True


def kws_model_present(model_dir: str) -> bool:
    """True if every required KWS model file is present + checksum-valid (no download)."""
    directory = Path(model_dir)
    return all(verify_checksum(directory / name, sha) for name, sha in _MODEL_FILES.items())


def kws_available(cfg: Config) -> bool:
    """True if the sherpa KeywordSpotter could serve a CUSTOM word for this config.

    The lightweight probe behind the ``kws_available`` GUI contract field: ``kws_enabled``
    AND sherpa-onnx importable AND (the model files already present OR auto-download is on,
    so the first custom-word use can fetch them). Does NOT download — it only reports
    capability so the GUI can show whether the OR'd KWS path is wired. Official words still
    never use it; this is purely "is the KWS backend usable at all".
    """
    if not getattr(cfg, "kws_enabled", True):
        return False
    if not _sherpa_importable():
        return False
    return kws_model_present(cfg.kws_model_dir) or bool(getattr(cfg, "kws_auto_download", True))


def _http_get(url: str, timeout: int = 180) -> bytes:
    """Fetch ``url`` and return its bytes (pinned HTTPS release URL only)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — pinned HTTPS
        data: bytes = resp.read()
    return data


def _extract_model_files(archive_bytes: bytes, dest_dir: Path) -> bool:
    """Extract the needed KWS model files from the release ``.tar.bz2`` into ``dest_dir``.

    The release is a ``.tar.bz2`` whose ``<dir>/<file>`` members include the int8
    encoder/decoder/joiner, ``tokens.txt`` and ``bpe.model``. We unpack to a temp dir,
    copy ONLY those members (flattened into ``dest_dir``), and atomically place each.
    Returns whether every required file now exists. Defensive: a malformed archive /
    missing member / IO error is logged and reported as failure (caller degrades to a
    no-op detector). Mirrors :func:`my_stt_tts.diarize._extract_segmentation_onnx`.
    """
    wanted = set(_MODEL_FILES)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = bz2.decompress(archive_bytes)
            tar_path = Path(tmpdir) / "kws.tar"
            tar_path.write_bytes(raw)
            dest_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path) as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    name = Path(member.name).name
                    if name not in wanted:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    payload = extracted.read()
                    out = dest_dir / name
                    tmp = out.with_suffix(out.suffix + ".part")
                    tmp.write_bytes(payload)
                    tmp.replace(out)
    except (OSError, tarfile.TarError, EOFError, ValueError):
        log.warning("failed to extract the KWS model files", exc_info=True)
        return False
    return all((dest_dir / name).is_file() for name in _MODEL_FILES)


def ensure_kws_model(
    model_dir: str,
    url: str,
    *,
    auto_download: bool = True,
    expected_sha256: str = "",
) -> bool:
    """Ensure the KWS model files exist under ``model_dir``, downloading once if needed.

    Returns ``True`` when every required file (encoder/decoder/joiner/tokens/bpe) is
    present AND passes its per-file SHA-256. A present-but-corrupt file triggers a
    re-download of the whole archive; the downloaded archive is checksum-verified
    (``expected_sha256``) BEFORE extraction, and each extracted file is then verified
    against :data:`_MODEL_FILES`. Network/IO failures are swallowed (the caller degrades
    to a no-op detector). Mirrors :func:`my_stt_tts.diarize.ensure_segmentation_model`.
    """
    directory = Path(model_dir)
    if all(verify_checksum(directory / name, sha) for name, sha in _MODEL_FILES.items()):
        return True
    if not auto_download or not url:
        return False
    log.info("downloading sherpa KWS model %s ...", url)
    try:
        archive = _http_get(url)
    except (urllib.error.URLError, OSError, ValueError):
        log.warning("KWS model download failed.", exc_info=True)
        return False
    if not archive:
        return False
    if expected_sha256:
        import hashlib

        if hashlib.sha256(archive).hexdigest().lower() != expected_sha256.strip().lower():
            log.warning("KWS model archive checksum mismatch; discarding.")
            return False
    if not _extract_model_files(archive, directory):
        return False
    if not all(verify_checksum(directory / name, sha) for name, sha in _MODEL_FILES.items()):
        log.warning("KWS model file checksum mismatch after extract; discarding.")
        return False
    return True


def _bpe_keyword_tokens(phrase: str, sp: Any) -> str:  # noqa: ANN401 — opaque SentencePieceProcessor
    """Encode ``phrase`` to the model's UPPERCASE BPE token string for a keyword line.

    The GigaSpeech KWS vocab is uppercase BPE (e.g. ``maziko`` -> ``▁MA Z I K O``). We
    uppercase then segment with the model's OWN ``bpe.model`` via ``sentencepiece`` —
    NOT ``sherpa_onnx.text2token``, which eagerly imports ``pypinyin`` (not a dependency)
    even for pure-BPE input. Returns a space-joined piece string (empty if it can't be
    encoded into in-vocab pieces).
    """
    pieces = sp.encode(phrase.strip().upper(), out_type=str)
    return " ".join(p for p in pieces if p)


def build_keywords(
    words: dict[str, list[str]],
    sp: Any,  # noqa: ANN401 — opaque SentencePieceProcessor
    *,
    boost: float,
    threshold: float,
) -> str:
    """Build a sherpa keywords string from ``{label: [spelling, …]}`` (all map to label).

    Each spelling becomes one keyword line ``<bpe tokens> :<boost> #<threshold> @<label>``
    so MULTIPLE spellings of a logical word ALL fire that one ``@label`` (the accent-variant
    feature). A spelling that can't be BPE-encoded into in-vocab pieces is skipped (logged),
    never aborting the whole build. ``boost``/``threshold`` are the sherpa per-keyword knobs
    (a higher boost fires more easily; a lower threshold accepts a weaker match). Returns the
    newline-joined keywords string (``""`` when nothing encodable was produced).
    """
    lines: list[str] = []
    for label, spellings in words.items():
        for spelling in spellings:
            toks = _bpe_keyword_tokens(spelling, sp)
            if not toks:
                log.warning(
                    "KWS: spelling %r for %r is not BPE-encodable; skipping", spelling, label
                )
                continue
            lines.append(f"{toks} :{boost} #{threshold} @{label}")
    return "\n".join(lines)


def spellings_for(word: str, extra: dict[str, list[str]] | None = None) -> list[str]:
    """The spellings to register for a logical ``word``: the word itself + any variants.

    ``extra`` is the optional ``{word: [variant, …]}`` accent map (``cfg.kws_spellings``).
    The word is always first; variants are appended, de-duplicated (case-insensitive),
    preserving order — so a config with no variants still registers the bare word.
    """
    out: list[str] = []
    seen: set[str] = set()
    for spelling in [word, *(extra or {}).get(word, [])]:
        key = spelling.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(spelling.strip())
    return out


class SherpaKws:
    """Lazy sherpa-onnx :class:`KeywordSpotter` detector matching the :class:`WakeWord` surface.

    Exposes ``detect(frame) -> bool`` + ``last_score`` + ``reset()`` so the wake loop calls
    it identically to openWakeWord. Loads nothing until the first :meth:`detect` (engine +
    stream built lazily, cached). On ANY failure (sherpa unavailable / model un-built /
    inference raises) it latches unavailable and :meth:`detect` returns ``False`` forever —
    a no-op, never an exception. ``last_score`` is 1.0 on a fire (the transducer KWS reports
    a matched-keyword label, not a continuous score) and 0.0 otherwise, so the same
    ``last_score >= threshold`` contract the GUI reads still holds.
    """

    def __init__(
        self,
        model_dir: str,
        word: str,
        *,
        boost: float = 1.5,
        threshold: float = 0.25,
        spellings: dict[str, list[str]] | None = None,
        sample_rate: int = KWS_SAMPLE_RATE,
    ) -> None:
        self.model_dir = model_dir
        self.word = word
        self.boost = boost
        self.threshold = threshold
        self.spellings = spellings or {}
        self.sample_rate = sample_rate
        self._spotter: Any = None
        self._stream: Any = None
        self._keywords: str = ""
        self._keywords_path: str = ""  # temp keywords file backing the spotter
        self._unavailable = False  # latched once sherpa/model can't load
        # Matches WakeWord: 1.0 when the keyword fired on the latest frame(s), else 0.0.
        self.last_score: float = 0.0
        self.model_name: str = word

    @classmethod
    def from_config(cls, cfg: Config, word: str | None = None) -> SherpaKws | None:
        """Build a KWS detector for ``word`` (default ``cfg.wake_phrase``), or ``None``.

        Returns ``None`` — KWS disabled / official word / sherpa unimportable / model
        un-fetchable — so the caller cleanly falls back to openWakeWord-only. NEVER builds
        for an official word (the guardrail). Construction is wrapped so a broken install
        degrades to ``None`` rather than crashing; the heavy ONNX engine is still lazy.
        """
        from .config import is_official_wake_word

        target = word or cfg.wake_phrase
        if not getattr(cfg, "kws_enabled", True):
            return None
        if is_official_wake_word(target):
            return None  # official words are openWakeWord-ONLY (byte-identical)
        try:
            if not _sherpa_importable():
                return None
            ok = ensure_kws_model(
                cfg.kws_model_dir,
                cfg.kws_model_url,
                auto_download=getattr(cfg, "kws_auto_download", True),
                expected_sha256=getattr(cfg, "kws_model_sha256", ""),
            )
            if not ok:
                log.warning("KWS enabled but model unavailable; openWakeWord-only for %r", target)
                return None
            return cls(
                cfg.kws_model_dir,
                target,
                boost=getattr(cfg, "kws_boost", 1.5),
                threshold=getattr(cfg, "kws_threshold", 0.25),
                spellings=getattr(cfg, "kws_spellings", {}),
                sample_rate=getattr(cfg, "sample_rate", KWS_SAMPLE_RATE),
            )
        except Exception as exc:  # noqa: BLE001 — never let KWS setup break the wake path
            log.warning("KWS disabled (setup failed): %s", exc)
            return None

    def available(self) -> bool:
        """True if the KWS model files are all present + checksum-valid on disk."""
        directory = Path(self.model_dir)
        return all(verify_checksum(directory / name, sha) for name, sha in _MODEL_FILES.items())

    def _ensure(self) -> bool:
        """Build (once) the KeywordSpotter + keyword stream; return whether it's usable."""
        if self._unavailable:
            return False
        if self._spotter is not None and self._stream is not None:
            return True
        try:
            import sentencepiece as spm
            import sherpa_onnx

            directory = Path(self.model_dir)
            sp = spm.SentencePieceProcessor()
            sp.load(str(directory / "bpe.model"))
            self._keywords = build_keywords(
                {self.word: spellings_for(self.word, self.spellings)},
                sp,
                boost=self.boost,
                threshold=self.threshold,
            )
            if not self._keywords:
                log.warning("KWS: no encodable keyword for %r; disabling", self.word)
                self._unavailable = True
                return False
            # The KeywordSpotter ctor REQUIRES a valid keywords_file (it parses every line
            # as `<tokens> :boost #thr @label`), so write THIS detector's keywords to a
            # temp file and pass it — never the model's tokens.txt (whose `<piece> <id>`
            # lines are not keyword lines and make InitKeywords fail). create_stream then
            # re-uses the same string per stream.
            fd, self._keywords_path = tempfile.mkstemp(suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as kw_file:
                kw_file.write(self._keywords + "\n")
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(directory / "tokens.txt"),
                encoder=str(directory / _ENCODER),
                decoder=str(directory / _DECODER),
                joiner=str(directory / _JOINER),
                keywords_file=self._keywords_path,
                num_threads=1,
                provider="cpu",
            )
            self._stream = self._spotter.create_stream(self._keywords)
        except Exception as exc:  # noqa: BLE001 — degrade to no-op, keep the wake path alive
            log.warning("KWS engine load failed for %r: %s", self.word, exc)
            self._unavailable = True
            self._spotter = None
            self._stream = None
            return False
        return True

    def _decode(self) -> None:
        """Drain the spotter's ready queue and latch ``last_score`` on a keyword hit."""
        spotter, stream = self._spotter, self._stream
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
        if spotter.get_result(stream):  # a non-empty matched-keyword label == fired
            self.last_score = 1.0
            spotter.reset_stream(stream)  # transient result; clear so it can re-fire

    def detect(self, frame: np.ndarray) -> bool:
        """Return ``True`` if the keyword fired after feeding this float32 16 kHz frame.

        Accepts the SAME native float32 [-1, 1] frame the wake loop feeds openWakeWord (no
        int16 conversion — the KWS feature extractor wants float). Streams the frame in,
        decodes, and polls the (transient) result. Never raises: any inference error latches
        the detector unavailable and returns ``False`` thereafter (a no-op). ``last_score``
        is 1.0 once fired (sticky until :meth:`reset`), matching the WakeWord contract that
        the GUI reads as ``last_score >= threshold``.
        """
        if self._unavailable or not self._ensure():
            return False
        try:
            samples = np.asarray(frame, dtype=np.float32).ravel()
            self._stream.accept_waveform(self.sample_rate, samples)
            self._decode()
        except Exception as exc:  # noqa: BLE001 — a per-frame failure is terminal for KWS
            log.warning("KWS detect failed for %r: %s", self.word, exc)
            self._unavailable = True
            return False
        return self.last_score >= 1.0

    def flush(self) -> bool:
        """Feed trailing silence so the LAST word in a capture decodes; return fired.

        The streaming transducer needs ~0.66 s of post-roll to emit the final keyword;
        :func:`my_stt_tts.wake.score_wake_clip`'s clip path calls this once after the audio
        so a word at the very end of a recording still fires. The live loop does not need it
        (it streams continuously). No-op when unavailable.
        """
        if self._unavailable or self._spotter is None or self._stream is None:
            return False
        try:
            pad = np.zeros(int(TRAILING_SILENCE_S * self.sample_rate), dtype=np.float32)
            self._stream.accept_waveform(self.sample_rate, pad)
            self._decode()
        except Exception as exc:  # noqa: BLE001 — degrade to no-op
            log.warning("KWS flush failed for %r: %s", self.word, exc)
            self._unavailable = True
            return False
        return self.last_score >= 1.0

    def reset(self) -> None:
        """Clear detector state between activations (fresh stream + zeroed score)."""
        self.last_score = 0.0
        if self._spotter is not None and not self._unavailable:
            with contextlib.suppress(Exception):
                self._stream = self._spotter.create_stream(self._keywords)
