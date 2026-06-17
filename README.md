# my-stt-tts

A hand-wired, low-latency **voice assistant loop that runs entirely on a MacBook
M1** (Apple Silicon): **wake word → record → speech-to-text → Claude (streaming)
→ text-to-speech → playback**, with speaker identification and German / French /
English support.

The Mac is the prototype target. The design stays portable so the "brain" can
later move to a server and the microphones/speakers to whole-house satellites
(sibling project: `infra/home-assistant-sandbox`).

> **Status: planning.** No code yet — see **[`PLAN.md`](PLAN.md)** for the full
> design, locked decisions, latency budget, phased roadmap, and risk register.

## Pipeline at a glance

| Stage | Choice (v1) | Why |
|:------|:------------|:----|
| Orchestrator | **Python**, one warm async process | Latency is model/network-bound; a different language buys ~nothing |
| Wake word | openWakeWord (custom phrase) | Free, no vendor lock, on-device |
| Speech-to-text | `parakeet-mlx` v3 (multilingual) | MLX-native, sub-second, DE/FR/EN auto-detect |
| Speaker ID | SpeechBrain ECAPA-TDNN, enrollment + cosine | Runs in parallel with STT → ~0 added latency |
| LLM | Anthropic SDK, streaming; Haiku default → Opus deep path | Fast/cheap default, pluggable, MCP-ready for multi-agent |
| Text-to-speech | Piper (DE `thorsten`, FR `tom`, EN `lessac`), `say` fallback | Only local engine strong in German *and* fast on M1 |
| Confirmations | short **chimes**, not spoken phrases | Spoken stage cues add ~6 s/query; chimes are language-neutral |
| Turn-taking | push-to-talk → Silero VAD | Deterministic first, then voice-activated |

## Quickstart

Not implemented yet. Setup, enrollment, and run instructions land as Phase 0–1
are built (tracked in [`PLAN.md`](PLAN.md)).

## Privacy

Local STT/TTS keep audio **on-device**; only transcribed text reaches Anthropic
(as with ordinary Claude usage). Voice-enrollment profiles stay local and
gitignored. Don't dictate confidential content.
