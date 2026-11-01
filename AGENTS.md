# AGENTS.md

Conventions for AI coding agents (and humans) working in this repo. The full
design, locked decisions, latency budget, and roadmap live in
**[PLAN.md](PLAN.md)** — read it first.

## What this is

A local voice assistant for macOS Apple Silicon: wake word → speech-to-text →
**an LLM** (streaming) → text-to-speech → playback, with speaker identification
and German / French / English support. The orchestrator is Python, running as one
warm long-running process. The LLM is **pluggable** — Anthropic/Claude by default,
but any OpenAI-compatible provider (OpenAI, Ollama, vLLM, local) works.

## Environment

- **Python 3.12+**, managed with [`uv`](https://docs.astral.sh/uv/) — preferred
  over Homebrew/system installs. `uv sync` installs; `uv run <cmd>` runs.
- **Native deps come from wheels where possible** (`sounddevice` bundles PortAudio;
  `imageio-ffmpeg`/`static-ffmpeg` bundle ffmpeg; `pywhispercpp` bundles
  whisper.cpp). Use Homebrew only as a fallback when no wheel exists.
- **Secrets and settings** live in `.env.sops` (dotenv SOPS, committed; tp#1386). Every
  `main()` re-execs through `sops-env-run` (vendored `src/my_stt_tts/sops_reexec.py`, a copy of
  mydotfiles `bin/sops_reexec.py` — never edit the copy), so values reach the environment only;
  no plaintext `.env` exists or is read. `ELEVENLABS_API_KEY` is written by `credctl.py derive
  elevenlabs-api` only. See `.env.example`.

## Build / test / lint

```bash
ruff format . && ruff check .     # format + lint
mypy src                          # type-check
pytest                            # tests (audio + model backends are mocked)
pre-commit run --all-files        # ruff + gitleaks secret scan
```

## Conventions

- `src/` layout, package `my_stt_tts`; CLI via `[project.scripts]`.
- Every script supports `-h/--help`.
- **The LLM provider is config, not code** — program against an OpenAI-compatible
  interface; select provider/model/base-URL via env (`LLM_PROVIDER`, `LLM_MODEL`,
  `LLM_BASE_URL`). Anthropic is the default, not the only option.
- **GPL backends (Piper, espeak-ng) are invoked as subprocesses (CLI binaries),
  never imported in-process** — this keeps the project Apache-2.0. See
  [PLAN.md](PLAN.md) §8.
- Keep STT / LLM / TTS backends behind their pluggable interfaces.
- The public README is owned by the README-designer role with its own "fanciness" checker
  loop; route README styling there (spec 2026-06-19: customer-facing "speak to your agent",
  feature list at the top, details folded, control-room link first, GitHub-safe HTML).
- Every `quickstart.sh` run writes `logs/quickstart-<ts>.log` and `logs/events-<ts>.jsonl`
  (`MSTT_EVENT_LOG`); read the newest of both before wake-word or audio work.

## Where things live

- Decisions + roadmap → `PLAN.md`
- Per-stage code → `src/my_stt_tts/{audio,wake,stt,speaker_id,brain,tts,chimes,metrics}.py`
- Conversation (Phase 7) → `turn.py` (end-of-turn analyzers; smart-turn is the
  default and auto-downloads via `ensure_smart_turn_model`), `interrupt.py`
  (false-interrupt `InterruptGate` + acoustic `InterruptPredictor`), `aec.py`
  (echo cancellation: `EchoCanceller` protocol, hardware `VoiceProcessingEchoCanceller`,
  software `NlmsEchoCanceller`). Barge-in lives in `audio.monitor_during_playback`
  (which runs the AEC + predictor) + `tts.Playback` (carries the AEC reference PCM);
  context repair in `brain.commit_spoken`. Interruption is published as bus events
  (`interrupt_start`/`interrupt_stop`/`bot_stopped_speaking`) and the captured
  barge-in audio is fed straight into `StreamingTranscriber.feed_clip`.
- Network audio transport (R2-5) → `transport.py` (`AudioTransport` seam +
  `LocalTransport` + PCM/handshake codecs), `ws_transport.py` (`WebSocketTransport`
  - `serve_websocket`/`WsSession`, the `transport` extra), `net_loop.py`
  (`run_transport_session` drives the pipeline over a transport), `satellite.py`
  (the remote mic+speaker client), `ws_frame.py` (stdlib RFC-6455 framing for the
  browser `/ws/audio` channel in `webui.py`).
- Tool / function calling (R2-7) → `tools.py` (`ToolRegistry`, example tools); the
  provider round-trip is in `brain.py` (`_stream_anthropic_tools`/`_stream_openai_tools`).
  Optional cloud backends: `stt.CloudTranscriber` + `tts.CloudTTS` (local-first, key-gated).
  Cloud voices behind the `render(text, lang)` seam → `tts_cloud.py` (ElevenLabs, Cartesia,
  `GoogleTTS` for `TTS_BACKEND=google`, tp#1286; `python -m my_stt_tts.tts_cloud` writes a test WAV).
- ElevenLabs daemon (`./mac-voice`) → `eleven_voice.py` (`VoiceSession`, CLI, client tools
  on a real `ClientTools`, `[system notice]` sending), `voice_control.py` (`VoiceDaemon`,
  wake listener, control socket, ready sound, `daemon_main` builds the bridge, `WakeKit`),
  `voice_gate.py` (ECAPA wake + call-domain profiles, `score_against`), `follow_up.py`
  (tp#1234: `FollowUpGate` — the agent hears the mic only while it speaks + `MAC_VOICE_FOLLOWUP_S`
  after, then the wake word reopens it; "pause"/"wait"), `wait_hint.py` (spoken "One moment."
  when a client tool is slow).
- Voice recognises only Albert (tp#1235; README section "Making voice recognise only you") →
  `voice_profiles.py` (kept enrollment takes under `enroll/<who>/<stamp>/`, rebuild of the
  call and wake profiles, 0600 files, settings reads/writes (`.env.sops` via `sops set`)), `interference.py` (`CallCapture`: refused /
  agent-skipped / sampled in-call utterances → `debug/recordings/interference/YYYY-MM/`, 90-day
  expiry, labels `albert` / `other`), `voice_calibration.py` (pure threshold + margin choice);
  the margin rule itself is `Authoriser.margin` + `VoiceGate.score_detail` against
  `enroll/negatives/call.npy`. Scripts: `scripts/enroll_voice.py` (guided 25-sentence
  enrollment), `scripts/label_voice_clips.py`, `scripts/calibrate_voice.py` (dry run; `-a`
  writes `.env.sops` with `sops set`).
- mac-voice LLM backend (`-B elevenlabs|claude`, tp#1097) → `claude_llm.py`: the Claude
  twin agent (ElevenLabs custom LLM → Anthropic's OpenAI-compatible endpoint), `-C` setup
  (key via `secret-run` item `anthropic-api-private-elevenlabs`), re-sync on every `-B claude` start.
- Claude bridge (tp#855; `bridge*.py`, `turns.py`, `claude_sessions.py`, `mac_*.py`,
  `attention.py`): actions pass `Authoriser` + `before_mutation`, ≤ 3 per sentence, "yes, do
  it" confirms; `mac_risk.py` classes SAFE / CONFIRM / REFUSE. Before you touch it, read
  `docs/AGENTS_REFERENCE.md` § Where things live: Claude bridge.
- Decide hub by voice (tp#1171; README section "Decide hub by voice") → `decide.py`
  (`HubClient` through Cloudflare with the `decide-cli` service token, secret only in ONE
  `secret-run` child per call with a keep-alive connection (`SecretRunTransport`, `-S`); spoken plate / briefs; `resolve_choice`; `DecideDesk` conversation
  state; read-only CLI `python -m my_stt_tts.decide -p`), `decide_tools.py` (mac-voice
  `decide_*` tools, answers behind "yes, do it", `MAC_VOICE_DECIDE`), `decide_intents.py`
  (local intent router, `DECIDE_VOICE`); the brain's plate context is `decide.PlateContext`.
  Tests use a fake hub only — never answer the real hub from a test.
- Private/local notes → `CLAUDE.local.md` (gitignored); `CLAUDE.md` is a gitignored
  shim that imports this file.
