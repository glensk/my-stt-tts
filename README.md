# my-stt-tts

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)
![Status](https://img.shields.io/badge/status-planning-orange)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A hand-wired, low-latency **voice assistant that runs on a MacBook (Apple
Silicon)**: **wake word → speech-to-text → an LLM (streaming) → text-to-speech →
playback**, with **speaker identification** and **German / French / English**
support. On-device STT/TTS; only transcribed text ever leaves the machine.

> **Status: planning.** No code yet — the full design, locked decisions, latency
> budget, phased roadmap, and risk register live in **[`PLAN.md`](PLAN.md)**.

## Why this exists

Off-the-shelf assistants are cloud-tethered, single-voice, and can't tell who's
speaking. This is a local, swappable pipeline where the "brain" is a pluggable LLM — Claude by default (Haiku
for speed, Opus for depth), the voices are yours to choose per language, and the
mic audio stays on your machine.

## Pipeline

```mermaid
flowchart LR
  Mic --> WW["Wake word<br/>(openWakeWord: maziko)"]
  WW --> VAD["Two-stage VAD<br/>+ smart-turn endpointing"]
  VAD --> STT["STT<br/>(parakeet-mlx)"]
  VAD --> SID["Speaker ID<br/>(ECAPA)"]
  STT --> Brain["LLM<br/>(Claude/OpenAI/local)"]
  SID --> Brain
  Brain --> TTS["TTS router<br/>(Piper / Kokoro / say)"]
  TTS --> Spk["Speakers"]
```

| Stage | Choice (v1) | Why |
|:------|:------------|:----|
| Orchestrator   | **Python**, one warm async process            | Latency is model/network-bound; another language buys ~nothing |
| Wake word      | openWakeWord, custom phrase **"maziko"**      | Free, no vendor lock, on-device |
| Speech-to-text | `parakeet-mlx` (multilingual)                 | MLX-native, sub-second, DE/FR/EN auto-detect |
| Speaker ID     | SpeechBrain ECAPA-TDNN, enrollment + cosine   | Runs in parallel with STT → ~0 added latency |
| LLM            | Any provider — Anthropic (default), OpenAI, Ollama, local; streaming | Pluggable via OpenAI-compatible API; Haiku→Opus deep path; MCP-ready |
| Text-to-speech | Piper (DE/FR/EN) · Kokoro (EN) · `say` fallback | Only local engine strong in German *and* fast on M1 |
| Confirmations  | short **chimes**, not spoken phrases          | Spoken stage cues add ~6 s/query; chimes are language-neutral |
| Turn-taking    | push-to-talk → Silero VAD → smart-turn        | Deterministic first, then voice-activated, then prosody-aware |

## LLM provider

The "brain" is **provider-agnostic**. Anthropic/Claude is the default and the
recommendation, but any OpenAI-compatible endpoint works — OpenAI, Ollama, vLLM,
LM Studio, or a local server. Select it via `.env` (see `.env.example`):

| Variable | Example | Meaning |
|:---------|:--------|:--------|
| `LLM_PROVIDER`   | `anthropic`                    | `anthropic` / `openai` / `openai-compatible` / `ollama` |
| `LLM_MODEL`      | `claude-haiku-4-5`             | fast-path model id |
| `LLM_MODEL_DEEP` | `claude-opus-4-8`              | optional "deep" model |
| `LLM_BASE_URL`   | `http://localhost:11434/v1`    | for OpenAI-compatible / local servers |

## Install

> Not implemented yet — the methods below land as the build progresses
> (tracked in [`PLAN.md`](PLAN.md), Phase 9). Planned, in order of preference:

```bash
# 1) Homebrew tap (planned, recommended) — installs the app AND native deps
brew install glensk/tap/my-stt-tts

# 2) uv tool (planned) — Python tool install; bring your own native deps
brew install portaudio espeak-ng ffmpeg          # prerequisites
uv tool install my-stt-tts

# 3) From source (for contributors)
git clone https://github.com/glensk/my-stt-tts && cd my-stt-tts
brew install portaudio espeak-ng ffmpeg whisper-cpp
uv sync && uv run my-stt-tts
```

**Docker is not supported on macOS** for this app: containers there run in a
Linux VM with **no microphone/speaker access and no Apple-Silicon GPU (Metal/MLX)**
— i.e. no audio and no acceleration. Run it natively.

## Third-party licenses

This project is **Apache-2.0**. Optional backends carry their own licenses and are
invoked as **separate processes** (not linked in), so they don't change this
project's license:

| Backend | License | Note |
|:--------|:--------|:-----|
| Piper, espeak-ng        | **GPL-3.0**            | invoked as a subprocess (CLI), never imported |
| XTTS-v2 (Coqui)         | **CPML, non-commercial** | optional; personal use only |
| openWakeWord (bundled models) | **CC-BY-NC-SA-4.0** | self-trained models avoid this |
| Kokoro, SpeechBrain, Silero-VAD | Apache-2.0 / MIT | permissive (Kokoro run with espeak-ng disabled) |

## Privacy

Local STT/TTS keep audio **on-device**; only transcribed text reaches your chosen LLM provider
(Anthropic by default, as with ordinary Claude usage). Voice-enrollment profiles stay local and
gitignored. Don't dictate confidential content.

## Development

Conventions for humans and AI agents are in **[AGENTS.md](AGENTS.md)**; the design
rationale is in **[PLAN.md](PLAN.md)**.
