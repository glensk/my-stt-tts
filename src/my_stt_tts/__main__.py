"""CLI entrypoint: a push-to-talk voice loop (Phases 1-2).

Press Enter to start/stop recording. Each turn: chime -> capture -> STT ->
stream the LLM -> sentence-chunk -> speak each sentence (mic gated during
playback). ``--debug`` adds spoken stage cues and verbose logs.

Wake-word + VAD endpointing (Phase 4) and speaker ID (Phase 5) build on this.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import audio, chimes
from .brain import Brain, LLMError
from .config import Config, ConfigError
from .metrics import TurnMetrics
from .text import SentenceChunker, strip_non_spoken
from .tts import TTSRouter

log = logging.getLogger("my_stt_tts")
_CHIME_SR = chimes.DEFAULT_SR


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="my-stt-tts", description="Local push-to-talk voice assistant (Anthropic by default)."
    )
    parser.add_argument("--once", action="store_true", help="Run a single turn, then exit.")
    parser.add_argument("--debug", action="store_true", help="Speak stage cues + verbose logs.")
    parser.add_argument("--provider", help="Override LLM_PROVIDER (anthropic/openai/ollama/...).")
    parser.add_argument("--model", help="Override LLM_MODEL.")
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if args.provider:
        cfg.llm_provider = args.provider
    if args.model:
        cfg.llm_model = args.model
    if args.debug:
        cfg.debug = True
    return cfg


def _play(samples: object) -> None:
    audio.play(samples, _CHIME_SR)  # type: ignore[arg-type]


def run_turn(cfg: Config, brain: Brain, stt: object, tts: TTSRouter, gate: audio.MicGate) -> None:
    """Run one capture -> STT -> LLM -> TTS turn."""
    metrics = TurnMetrics()
    gate.gate()
    _play(chimes.chime_listening())
    gate.release()

    with metrics.stage("capture"):
        clip = audio.record_push_to_talk(cfg.sample_rate, cfg.max_record_seconds)
    gate.gate()
    _play(chimes.chime_done())
    gate.release()
    if clip.size == 0:
        return

    with metrics.stage("stt"):
        result = stt.transcribe(clip, cfg.sample_rate)  # type: ignore[attr-defined]
    metrics.note(transcript=result.text, language=result.language)
    if not result.text:
        return
    if cfg.debug:
        tts.speak("recorded")

    lang = result.language or cfg.default_language
    chunker = SentenceChunker()
    try:
        with metrics.stage("llm_tts"):
            for delta in brain.stream(result.text):
                for sentence in chunker.feed(delta):
                    _speak(tts, gate, sentence, lang)
            _speak(tts, gate, chunker.flush(), lang)
    except LLMError as exc:
        log.error("LLM error: %s", exc)
        _play(chimes.chime_error())
        tts.speak("Sorry, I had a problem.")
    metrics.log()


def _speak(tts: TTSRouter, gate: audio.MicGate, sentence: str, lang: str) -> None:
    spoken = strip_non_spoken(sentence)
    if not spoken:
        return
    gate.gate()
    tts.speak(spoken, lang)
    gate.release()


def main(argv: list[str] | None = None) -> int:
    """Validate config, build components, run the loop. Returns an exit code."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    try:
        cfg = _build_config(args)
        cfg.validate()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    from .stt import ParakeetSTT  # heavy (MLX); import only after config validates

    brain = Brain(cfg)
    tts = TTSRouter(cfg)
    stt = ParakeetSTT()
    gate = audio.MicGate(cfg.mic_gate_tail_seconds)
    print(f"my-stt-tts ready (provider={cfg.llm_provider}, model={cfg.llm_model}). Ctrl-C to quit.")
    try:
        while True:
            run_turn(cfg, brain, stt, tts, gate)
            if args.once:
                break
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
