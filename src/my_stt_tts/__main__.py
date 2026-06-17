"""CLI entrypoint: push-to-talk OR typed voice loop (Phases 1-2).

Modes:
  (default)     push-to-talk: press Enter to start/stop recording each turn
  --type        interactive typed input (no mic/STT) — for testing/dev
  --text "..."  run one typed turn, then exit

Brain presets (``--brain``) switch provider+model in one word, e.g.
``haiku-sub`` (subscription via the Claude CLI, no API key) or ``opus-api``.
``--voice`` picks an English Piper voice; ``--list-voices`` shows the menu.
Each turn: [chime] -> capture/typed -> [STT] -> stream the LLM -> sentence-chunk
-> speak each sentence (mic gated during playback). ``--debug`` adds spoken cues.
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import audio, chimes
from .brain import Brain, LLMError
from .config import BRAIN_PRESETS, Config, ConfigError
from .metrics import TurnMetrics
from .text import SentenceChunker, strip_non_spoken
from .tts import VOICE_PRESETS, TTSRouter, list_voice_presets

log = logging.getLogger("my_stt_tts")
_CHIME_SR = chimes.DEFAULT_SR


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="my-stt-tts",
        description="Local voice assistant: wake/typed -> STT -> LLM -> TTS (Anthropic by default).",
    )
    parser.add_argument(
        "--type",
        dest="type_mode",
        action="store_true",
        help="Interactive typed input (no mic/STT).",
    )
    parser.add_argument("--text", help="Run one typed turn with this text, then exit.")
    parser.add_argument("--once", action="store_true", help="Run a single mic turn, then exit.")
    parser.add_argument("--debug", action="store_true", help="Speak stage cues + verbose logs.")
    parser.add_argument(
        "--brain",
        choices=sorted(BRAIN_PRESETS),
        help="One-word provider+model preset (e.g. haiku-sub, opus-api). Overrides --provider/--model.",
    )
    parser.add_argument(
        "--provider", help="Override LLM_PROVIDER (anthropic/openai/ollama/claude-cli/...)."
    )
    parser.add_argument("--model", help="Override LLM_MODEL.")
    parser.add_argument(
        "--voice", choices=sorted(VOICE_PRESETS), help="English TTS voice (see --list-voices)."
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List the English voice presets and exit."
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if args.brain:
        cfg.apply_brain_preset(args.brain)
    if args.provider:
        cfg.llm_provider = args.provider
    if args.model:
        cfg.llm_model = args.model
    if args.voice:
        cfg.tts_voices["en"] = VOICE_PRESETS[args.voice]
    if args.debug:
        cfg.debug = True
    return cfg


def _play(samples: object) -> None:
    audio.play(samples, _CHIME_SR)  # type: ignore[arg-type]


def _speak(tts: TTSRouter, gate: audio.MicGate, sentence: str) -> None:
    spoken = strip_non_spoken(sentence)
    if not spoken:
        return
    gate.gate()
    tts.speak(spoken)  # lang auto-detected from the answer text
    gate.release()


def _capture_transcript(cfg: Config, stt: object, gate: audio.MicGate, metrics: TurnMetrics) -> str:
    gate.gate()
    _play(chimes.chime_listening())
    gate.release()
    with metrics.stage("capture"):
        clip = audio.record_push_to_talk(cfg.sample_rate, cfg.max_record_seconds)
    gate.gate()
    _play(chimes.chime_done())
    gate.release()
    if clip.size == 0:
        return ""
    with metrics.stage("stt"):
        result = stt.transcribe(clip, cfg.sample_rate)  # type: ignore[attr-defined]
    return str(result.text)


def run_turn(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    *,
    stt: object | None = None,
    typed_text: str | None = None,
) -> None:
    """Run one capture/typed -> LLM -> TTS turn."""
    metrics = TurnMetrics()
    text = (
        typed_text.strip()
        if typed_text is not None
        else _capture_transcript(cfg, stt, gate, metrics)
    )
    if not text:
        return
    metrics.note(transcript=text)
    if cfg.debug and typed_text is None:
        tts.speak("recorded")
    chunker = SentenceChunker()
    try:
        with metrics.stage("llm_tts"):
            for delta in brain.stream(text):
                for sentence in chunker.feed(delta):
                    _speak(tts, gate, sentence)
            _speak(tts, gate, chunker.flush())
    except LLMError as exc:
        log.error("LLM error: %s", exc)
        _play(chimes.chime_error())
        tts.speak("Sorry, I had a problem.")
    metrics.log()


def main(argv: list[str] | None = None) -> int:
    """Validate config, build components, run the chosen loop. Returns exit code."""
    args = _parse_args(argv)
    if args.list_voices:
        print("English voice presets (use --voice <name>):")
        print(list_voice_presets())
        return 0
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    try:
        cfg = _build_config(args)
        cfg.validate()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    brain = Brain(cfg)
    tts = TTSRouter(cfg)
    gate = audio.MicGate(cfg.mic_gate_tail_seconds)
    typed_mode = args.type_mode or args.text is not None
    stt: object | None = None
    if not typed_mode:
        from .stt import ParakeetSTT  # heavy (MLX); only needed for mic mode

        stt = ParakeetSTT(cfg.stt_model)

    print(f"my-stt-tts ready (provider={cfg.llm_provider}, model={cfg.llm_model}).")
    try:
        if args.text is not None:
            run_turn(cfg, brain, tts, gate, typed_text=args.text)
        elif args.type_mode:
            print("Type a message ('quit' or blank to exit).")
            while True:
                line = input("you> ").strip()
                if line in {"", "quit", "exit"}:
                    break
                run_turn(cfg, brain, tts, gate, typed_text=line)
        else:
            print("Push-to-talk: Enter to start/stop each turn. Ctrl-C to quit.")
            while True:
                run_turn(cfg, brain, tts, gate, stt=stt)
                if args.once:
                    break
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
