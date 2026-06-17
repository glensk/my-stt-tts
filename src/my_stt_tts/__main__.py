"""CLI entrypoint: wake-word / push-to-talk / typed voice loop (Phases 1-6).

Modes:
  (default)     push-to-talk: press Enter to start/stop recording each turn
  --wake        always-listening: say the wake phrase ("maziko"), then speak
  --type        interactive typed input (no mic/STT) — for testing/dev
  --text "..."  run one typed turn, then exit

Brain presets (``--brain``) switch provider+model in one word, e.g. ``haiku-sub``
(subscription via the Claude CLI, no API key) or ``opus-api``. Say "agent, <task>"
to delegate to a full MCP-capable Claude agent (set AGENT_WORKSPACE to enable).
``--voice`` picks an English Piper voice; ``--list-voices`` shows the menu.
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
    parser.add_argument("--wake", action="store_true", help="Always-listen for the wake phrase.")
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


def _respond(cfg: Config, brain: Brain, tts: TTSRouter, gate: audio.MicGate, text: str) -> None:
    """Stream the LLM for ``text`` and speak it sentence-by-sentence."""
    metrics = TurnMetrics()
    metrics.note(transcript=text)
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


def _capture_ptt(cfg: Config, stt: object, gate: audio.MicGate) -> str:
    gate.gate()
    _play(chimes.chime_listening())
    gate.release()
    clip = audio.record_push_to_talk(cfg.sample_rate, cfg.max_record_seconds)
    gate.gate()
    _play(chimes.chime_done())
    gate.release()
    if clip.size == 0:
        return ""
    return str(stt.transcribe(clip, cfg.sample_rate).text)  # type: ignore[attr-defined]


def run_turn(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    *,
    stt: object | None = None,
    typed_text: str | None = None,
) -> None:
    """Run one push-to-talk or typed turn."""
    text = typed_text.strip() if typed_text is not None else _capture_ptt(cfg, stt, gate)
    if not text:
        return
    if cfg.debug and typed_text is None:
        tts.speak("recorded")
    _respond(cfg, brain, tts, gate, text)


def run_wake_loop(
    cfg: Config, brain: Brain, tts: TTSRouter, gate: audio.MicGate, stt: object
) -> int:
    """Always-listen: on the wake phrase, capture (VAD), respond, then follow up."""
    from .vad import SilenceEndpointer, SileroVad
    from .wake import WakeWord

    wake = WakeWord.from_config(cfg)
    if not wake.available():
        print(
            f'Wake model not found at {cfg.wake_model_path}. Train "{cfg.wake_phrase}" first '
            "— see wakewords/WAKEWORD.md.",
            file=sys.stderr,
        )
        return 2
    vad = SileroVad(cfg.sample_rate)
    frame_seconds = 512 / cfg.sample_rate
    print(f'Listening for "{cfg.wake_phrase}". Ctrl-C to quit.')
    while True:
        audio.listen_for_wake(wake, cfg.sample_rate)
        gate.gate()
        _play(chimes.chime_listening())
        gate.release()
        follow_up = False
        while True:
            endpointer = SilenceEndpointer(cfg.vad_silence_seconds, frame_seconds=frame_seconds)
            max_s = cfg.follow_up_seconds if follow_up else cfg.max_record_seconds
            clip = audio.record_with_vad(cfg.sample_rate, vad, endpointer, max_seconds=max_s)
            if clip.size == 0:
                break  # silence -> back to listening for the wake word
            text = str(stt.transcribe(clip, cfg.sample_rate).text).strip()  # type: ignore[attr-defined]
            if not text:
                break
            _respond(cfg, brain, tts, gate, text)
            follow_up = True  # subsequent turns are short follow-ups (no re-wake)


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
        from .stt import ParakeetSTT  # heavy (MLX); only needed for mic modes

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
        elif args.wake:
            return run_wake_loop(cfg, brain, tts, gate, stt)
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
