# Changelog

Notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Pre-1.0 — the surface
may still change.

## [Unreleased]

### Added

- **Network audio transport (R2-5)**: an `AudioTransport` seam (`transport.py`) —
  `LocalTransport` (sounddevice, default) + `WebSocketTransport`; a `websockets`
  server (`ws_transport.py`, the `transport` extra) that bridges remote clients into
  the pipeline (`net_loop.run_transport_session`); a `satellite.py` mic+speaker
  client (`python -m my_stt_tts.satellite ws://HOST:PORT`); and **real browser audio**
  in the GUI (`getUserMedia` → 16 kHz PCM over a same-origin `/ws/audio` WebSocket,
  TTS PCM streamed back) via a stdlib RFC-6455 codec (`ws_frame.py`). `--transport`,
  `--transport-port`, `--transport-token`, `--browser-audio` flags.
- **In-conversation tool/function calling (R2-7)**: `tools.py` (`ToolRegistry` →
  Anthropic + OpenAI schemas; `get_time`, `calculator`, `home_control`) with the full
  tool-use round-trip wired into `Brain.stream` for both providers. Optional,
  local-first cloud STT (`CloudTranscriber`) + cloud TTS (`CloudTTS`) behind the
  seams. `TOOLS_ENABLED`, `STT_BACKEND`, `TTS_BACKEND` config.
- Core voice loop (Phases 0–3): `config` + threaded `spine` + per-turn `metrics`;
  push-to-talk and typed (`--type`/`--text`) modes; `parakeet-mlx` STT;
  provider-agnostic streaming brain; Piper / macOS `say` TTS with per-language
  routing; chimes; sentence-chunked streaming with a decimal/comma guard;
  half-duplex mic gating; graceful failure + rate limiting.
- `claude-cli` brain — uses the Claude subscription via the CLI (no API key),
  **stripped + isolated** (own prompt, no tools / CLAUDE.md / hooks; runs in a
  non-git scratch dir) and session-continued for multi-turn memory.
- `--brain` presets (`haiku|sonnet|opus`-`sub|api`, `ollama`); editable spoken
  system prompt at `prompts/system_prompt.md`; English voice menu
  (`--voice` / `--list-voices`) and a calmer default cadence.
- Phase 4: `--wake` wake-word mode (openWakeWord) + Silero-VAD capture and a
  follow-up window; `wakewords/WAKEWORD.md` training guide; `scripts/test_wakeword.py`.
- Phase 5: ECAPA-TDNN speaker matching with unknown/ambiguous rejection, plus
  threshold calibration (`calibrate_threshold`, `scripts/calibrate.py`) and `scripts/enroll.py`.
- Phase 6: say "agent, &lt;task&gt;" to delegate to a full, MCP-capable Claude agent
  in `AGENT_WORKSPACE` (`agent.py`).
- `./mstt` launcher (run without `uv run`); `--settings` / `-h` print the resolved
  config (brain, voice, prompt …) in blue; a GitHub Pages voice-sample gallery.
- Project scaffolding: `pyproject.toml` + `uv.lock`, ruff/mypy/pytest, ruff+gitleaks
  pre-commit, CI on `macos-15`, Apache-2.0 LICENSE, AGENTS.md.
