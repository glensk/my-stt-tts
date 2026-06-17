# AGENTS.md

Conventions for AI coding agents (and humans) working in this repo. The full
design, locked decisions, latency budget, and roadmap live in
**[PLAN.md](PLAN.md)** — read it first.

## What this is

A local voice assistant for macOS Apple Silicon: wake word → speech-to-text →
Claude (streaming) → text-to-speech → playback, with speaker identification and
German / French / English support. The orchestrator is Python, running as one
warm long-running process.

## Environment

- **Python 3.12+**, managed with [`uv`](https://docs.astral.sh/uv/). `uv sync`
  installs; `uv run <cmd>` runs inside the venv.
- **System deps** (Homebrew): `brew install portaudio espeak-ng ffmpeg whisper-cpp`
- **Secrets** live in `.env` (never commit) — `ANTHROPIC_API_KEY`. See `.env.example`.

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
- **GPL backends (Piper, espeak-ng) are invoked as subprocesses (their CLI
  binaries), never imported in-process** — this keeps the project Apache-2.0.
  See [PLAN.md](PLAN.md) §8 (Third-party licenses & distribution).
- Keep STT / LLM / TTS backends behind their pluggable interfaces; selection is
  config, not code.
- Tune by measurement: per-stage latency telemetry exists before optimizing.

## Where things live

- Decisions + roadmap → `PLAN.md`
- Per-stage code → `src/my_stt_tts/{audio,wake,stt,speaker_id,brain,tts,chimes,metrics}.py`
- Private/local notes → `CLAUDE.local.md` (gitignored); `CLAUDE.md` is a gitignored
  shim that imports this file.
