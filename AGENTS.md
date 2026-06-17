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
- **Secrets** live in `.env` (never commit). See `.env.example`.

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

## Where things live

- Decisions + roadmap → `PLAN.md`
- Per-stage code → `src/my_stt_tts/{audio,wake,stt,speaker_id,brain,tts,chimes,metrics}.py`
- Private/local notes → `CLAUDE.local.md` (gitignored); `CLAUDE.md` is a gitignored
  shim that imports this file.
