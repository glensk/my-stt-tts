"""CLI entrypoint: wake-word / push-to-talk / typed / browser voice loop.

Modes:
  (default)     push-to-talk: press Enter to start/stop recording each turn
  --wake        always-listening: say the wake phrase ("maziko"), then speak
  --type        interactive typed input (no mic/STT)
  --text "..."  run one typed turn, then exit
  --browser     serve a live web GUI (state, transcript, settings, controls)

Brain presets (``--brain``) switch provider+model in one word, e.g. ``haiku-sub``
(subscription via the Claude CLI, no API key) or ``opus-api``. Say "agent, <task>"
to delegate to a full MCP-capable Claude agent (set AGENT_WORKSPACE to enable).
``--voice`` picks a Piper voice; ``--list-voices``/``--settings``/``-h`` print info.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

from . import audio, chimes
from .brain import Brain, LLMError
from .config import BRAIN_PRESETS, Config, ConfigError
from .events import bus
from .metrics import TurnMetrics
from .text import SentenceChunker, strip_non_spoken
from .tts import VOICE_PRESETS, TTSRouter, list_voice_presets

log = logging.getLogger("my_stt_tts")
_CHIME_SR = chimes.DEFAULT_SR
_BLUE = "\033[34m"
_RESET = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def settings_text(cfg: Config, *, color: bool | None = None) -> str:
    """A human-readable summary of the resolved settings (brain, voice, etc.)."""
    use = _use_color() if color is None else color
    blue = _BLUE if use else ""
    reset = _RESET if use else ""
    prompt_head = (cfg.system_prompt.strip().splitlines() or [""])[0][:68]
    agent_ws = cfg.agent_workspace or "(disabled — set AGENT_WORKSPACE)"
    rows = [
        "current settings (override via .env or flags):",
        f"  brain      {blue}{cfg.llm_provider} / {cfg.llm_model}{reset}  (deep: {cfg.llm_model_deep})",
        f"  voice      en={blue}{cfg.tts_voices.get('en')}{reset}"
        f"  de={cfg.tts_voices.get('de')}  fr={cfg.tts_voices.get('fr')}"
        f"  length-scale {cfg.tts_length_scale}",
        f"  stt        {blue}{cfg.stt_model}{reset}",
        f"  wake       phrase {blue}{cfg.wake_phrase}{reset}  model {cfg.wake_model_path}",
        f"  agent      trigger '{cfg.agent_trigger}'  workspace {blue}{agent_ws}{reset}"
        f"  model {cfg.agent_model}",
        f"  prompt     {blue}{prompt_head}…{reset}  (edit prompts/system_prompt.md)",
    ]
    return "\n".join(rows)


def _epilog() -> str:
    try:
        return settings_text(Config.from_env())
    except Exception:  # never let a bad .env break --help
        return "current settings: (unavailable — check your .env)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="my-stt-tts",
        description="Local voice assistant: wake/typed -> STT -> LLM -> TTS (Anthropic by default).",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument("--browser", action="store_true", help="Serve the live web GUI.")
    parser.add_argument("--port", type=int, default=8765, help="Web GUI port (default 8765).")
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
    parser.add_argument(
        "--settings", action="store_true", help="Print the resolved settings and exit."
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
    """Stream the LLM for ``text``, speak it, and publish live state to the bus."""
    metrics = TurnMetrics()
    metrics.note(transcript=text)
    bus.transcript(text)
    chunker = SentenceChunker()
    try:
        with metrics.stage("llm_tts"):
            bus.state("llm_request", f"{cfg.llm_provider} / {cfg.llm_model}")
            bus.state("llm_wait")
            first = True
            for delta in brain.stream(text):
                if first:
                    bus.state("llm_response")
                    first = False
                bus.response(delta, final=False)
                for sentence in chunker.feed(delta):
                    bus.state("speaking")
                    _speak(tts, gate, sentence)
            tail = chunker.flush()
            if tail:
                bus.state("speaking")
                _speak(tts, gate, tail)
        bus.response("", final=True)
    except LLMError as exc:
        log.error("LLM error: %s", exc)
        bus.log(str(exc), "error")
        _play(chimes.chime_error())
        tts.speak("Sorry, I had a problem.")
    bus.state("idle")
    metrics.log()


def _capture_ptt(cfg: Config, stt: object, gate: audio.MicGate) -> str:
    gate.gate()
    _play(chimes.chime_listening())
    gate.release()
    bus.state("recording")
    clip = audio.record_push_to_talk(cfg.sample_rate, cfg.max_record_seconds)
    gate.gate()
    _play(chimes.chime_done())
    gate.release()
    if clip.size == 0:
        bus.state("idle")
        return ""
    bus.state("stt")
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
        msg = (
            f'Wake model not found at {cfg.wake_model_path}. Train "{cfg.wake_phrase}" first '
            "— see wakewords/WAKEWORD.md."
        )
        print(msg, file=sys.stderr)
        bus.log(msg, "error")
        return 2
    vad = SileroVad(cfg.sample_rate)
    frame_seconds = 512 / cfg.sample_rate
    print(f'Listening for "{cfg.wake_phrase}". Ctrl-C to quit.')
    while True:
        bus.state("listening", cfg.wake_phrase)
        audio.listen_for_wake(wake, cfg.sample_rate)
        bus.wake()
        gate.gate()
        _play(chimes.chime_listening())
        gate.release()
        follow_up = False
        while True:
            bus.state("recording")
            endpointer = SilenceEndpointer(cfg.vad_silence_seconds, frame_seconds=frame_seconds)
            max_s = cfg.follow_up_seconds if follow_up else cfg.max_record_seconds
            clip = audio.record_with_vad(cfg.sample_rate, vad, endpointer, max_seconds=max_s)
            if clip.size == 0:
                break  # silence -> back to listening for the wake word
            bus.state("stt")
            text = str(stt.transcribe(clip, cfg.sample_rate).text).strip()  # type: ignore[attr-defined]
            if not text:
                break
            _respond(cfg, brain, tts, gate, text)
            follow_up = True  # subsequent turns are short follow-ups (no re-wake)


def _run_browser(
    cfg: Config,
    brain: Brain,
    tts: TTSRouter,
    gate: audio.MicGate,
    stt: object | None,
    *,
    wake: bool,
    port: int = 8765,
) -> int:
    """Serve the web GUI; turns are driven from the page (and optional wake loop)."""
    import webbrowser

    from .webui import WebUI

    def on_turn(text: str) -> None:
        _respond(cfg, brain, tts, gate, text)

    def on_action(name: str, _data: dict) -> None:
        if name == "reset":
            brain.reset()
            bus.log("conversation reset")
        elif name == "list_voices":
            bus.log("voices: " + ", ".join(VOICE_PRESETS))
        elif name in {"wake_start", "wake_stop", "ptt"}:
            bus.log(
                f"'{name}' runs from the terminal in this build (use --wake / push-to-talk)",
                "error",
            )
        else:
            bus.log(f"unknown action '{name}'", "error")

    ui = WebUI(cfg, on_turn, on_action, port=port)
    bus.state("idle")
    if wake and stt is not None:
        threading.Thread(
            target=run_wake_loop, args=(cfg, brain, tts, gate, stt), daemon=True
        ).start()
    print(f"my-stt-tts web UI → {ui.url()}  (Ctrl-C to quit)")
    try:
        webbrowser.open(ui.url())
    except Exception:  # headless / no browser is fine
        pass
    try:
        ui.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


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
        if args.settings:
            print(settings_text(cfg))
            return 0
        cfg.validate()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    brain = Brain(cfg)
    tts = TTSRouter(cfg)
    gate = audio.MicGate(cfg.mic_gate_tail_seconds)

    needs_stt = not args.type_mode and args.text is None and (not args.browser or args.wake)
    stt: object | None = None
    if needs_stt:
        from .stt import ParakeetSTT  # heavy (MLX); only needed for mic modes

        stt = ParakeetSTT(cfg.stt_model)

    if not args.browser:
        print(settings_text(cfg))
    try:
        if args.browser:
            return _run_browser(cfg, brain, tts, gate, stt, wake=args.wake, port=args.port)
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
