# PLAN — Checker Loop

## Session

Resume: `c --resume e7dfe88f-9001-4138-8cfc-1f8789653cc6`

## Goal

For each repo in the ranked voice-LLM survey, a **fresh, indifferent checker model**
is forced to pick — with no ties — which repo is better suited **"to have a
conversation with an LLM via voice"**: the reference repo vs **this** repo
(`my-stt-tts`). If the checker picks the reference repo, capture its concrete gap
list, implement **all** of it, then re-check with a *new* checker. Loop until a
checker picks `my-stt-tts`; then advance to the next repo. May run for days.

**Mechanics:** checkers are read-only background agents (safe). Implementers are
**worktree-isolated** (so their edits never get swept into the main session's
`Stop` auto-commit); their commits are merged into `main`, linted, and tested by
the orchestrator before re-checking.

## Repo queue (ranked, voice↔LLM suitability)

- [x] 1. **pipecat** — https://github.com/pipecat-ai/pipecat — ✅ **WON** (round 5, reframed: multi-user-on-Mac, maturity excluded)
- [x] 2. **livekit/agents** — https://github.com/livekit/agents — ✅ **WON** (round 1, first pass)
- [x] 3. **huggingface/speech-to-speech** — https://github.com/huggingface/speech-to-speech — ✅ **WON** (first pass)
- [x] 4. **dnhkng/GLaDOS** — https://github.com/dnhkng/GLaDOS — ✅ **WON** (first pass)
- [x] 5. **KoljaB/RealtimeSTT** (+ RealtimeTTS) — https://github.com/KoljaB/RealtimeSTT — ✅ **WON** (first pass)
- [ ] … remaining survey repos appended as reached

## Round log

### Repo 1 — pipecat · Round 1 → WINNER: pipecat

Reason: `my-stt-tts` is **half-duplex / non-interruptible** (mic gated shut during
playback; barge-in, smart-turn, AEC all deferred to Phase 7). Gaps to close:

- [x] **G1 — True barge-in** (interruptible playback): `tts.Playback` (killable
  subprocess), `audio.monitor_during_playback()` (live mic + VAD during playback),
  `__main__` aborts TTS + cancels the LLM stream on confirmed speech. `barge_in`
  mode (`off`/`headphones`/`always`) + RMS floor for open-speaker bleed.
- [x] **G2 — Smart-turn / prosodic endpointing** (`turn.py`): `TurnAnalyzer` protocol,
  `SilenceTurnAnalyzer` fallback, `SmartTurnAnalyzer` (Smart Turn v3 ONNX) with
  graceful fallback when the model/onnxruntime is absent.
- [x] **G4 — False-interrupt suppression** (`interrupt.py` `InterruptGate`): opens
  only after `min_speech_ms` and/or `min_words`; ignores backchannels/coughs.
- [x] **G5 — Post-interruption context repair** (`brain.py`): `commit_spoken()`
  truncates the assistant turn to the voiced prefix (drops it if nothing voiced).
- [x] **G6 — Streaming STT (partials)** (`stt.py` `StreamingTranscriber`): emits
  partials during speech via `bus.transcript(text, partial=True)`.
- [ ] **G3 — AEC + noise suppression**: macOS `VoiceProcessingIO` / RNNoise so open
  speakers don't self-trigger (the unlock that makes G1 safe in a room). *(round 2)*
- [ ] **G7 — Network audio transport**: WebRTC/WebSocket so a remote mic/speaker or
  browser client can carry audio, not just the local mic. *(round 2)*

Merged at `e591b26` (4 commits, +1434 lines, new `interrupt.py`/`turn.py`/
`tests/test_conversation.py`); **66 tests pass**, lint clean. Caveats: Smart Turn
model not downloaded → silence fallback active; no AEC yet → barge-in best with
headphones. Round-2 checker result below.

### Repo 1 — pipecat · Round 2 → WINNER: pipecat (narrowed)

Now matched on barge-in, min-words gating, and the Smart Turn model; **ahead** on
post-interruption context repair (`commit_spoken` — pipecat has open bugs #2791/#4111).
Remaining gaps:

- [x] **R2-1 — Acoustic echo cancellation (AEC)** *(highest value)*: software AEC
  (macOS `VoiceProcessingIO`, or WebRTC APM / speexdsp echo canceller referencing the
  played signal) so barge-in works on open speakers, not just headphones.
- [x] **R2-2 — True streaming STT**: replace whole-buffer re-transcription with a
  bounded sliding-window re-decode (last N s) or a streaming engine, so partial latency
  and CPU don't grow with utterance length.
- [x] **R2-3 — Acoustic interruption prediction**: a 3rd `InterruptGate` guard scoring
  barge-in audio for intent-to-take-floor (talk through "mhm", yield to a real interrupt).
- [x] **R2-4 — Smart-turn by default**: auto-download the Smart Turn ONNX on first run
  (like Piper voices) and make it the default `turn_analyzer`; silence = explicit fallback.
- [x] **R2-5 — Network transport (G7)**: a WebSocket (and/or WebRTC) audio transport so
  remote satellites / the browser carry mic+TTS audio, not just the local mic.
- [x] **R2-6 — Robust interrupt plumbing**: feed the captured barge-in audio straight
  into the streaming transcriber (no from-scratch re-transcribe); interrupt as bus events.
- [x] **R2-7 — Backend breadth + in-conversation tool calling**: optional cloud STT/TTS
  behind the existing seams (esp. better German TTS); real function calling in `Brain.stream`.

Merged R2-1/2/3/4/6 at `878bbc7` (7 commits, +1712 lines, new `aec.py`; **101 tests
pass**, lint clean). AEC: macOS hardware `VoiceProcessingIO` (PyObjC, available) + numpy
NLMS fallback (~19 dB ERLE); the HW-cancelled PCM isn't yet routed end-to-end through the
`sounddevice` capture path (residual G3). Smart-turn download guard tested with mocked
network only.

Merged R2-5/R2-7 at `9e0e115` (6 commits, +~2900 lines, new `transport.py`/`ws_transport.py`/
`ws_frame.py`/`net_loop.py`/`satellite.py`/`tools.py`; **146 tests pass**, lint clean). The
browser and satellite client now carry real int16-PCM audio over WebSocket into the pipeline;
`Brain.stream` does full provider tool-use round-trips (Anthropic+OpenAI) with example tools;
optional key-gated cloud STT/TTS behind the existing seams. Caveats: full WebRTC not done (PCM
WS is real + sufficient); cloud adapters not exercised against a live key.

**All 7 round-2 gaps closed.** Round-3 checker result below.

### Repo 1 — pipecat · Round 3 → WINNER: pipecat (very close)

Judge calls it "close"; we're now **ahead** on AEC (NLMS proven >6 dB ERLE), the 3-stage
barge-in (duration + word + acoustic-intent), and context repair. pipecat wins on
production-readiness / transport robustness / breadth. Remaining gaps — **Wave C** (core
transport/audio robustness, highest leverage):

- [x] **R3-1 — True WebRTC transport** (Opus + jitter buffer + NAT): `aiortc` as a 3rd
  `AudioTransport` (`transport=webrtc`); browser `getUserMedia({echoCancellation:true})`.
- [x] **R3-2 — Full-duplex barge-in over the network transport**: `net_loop.respond_over_transport`
  is half-duplex — satellite/browser users can't interrupt. Port `_BargeInCtx` (VAD + gate +
  AEC + predictor) into the transport loop; cancel outbound TTS + LLM on confirmed interrupt.
- [x] **R3-3 — Streamed low-latency TTS playout**: replace whole-sentence WAV→`afplay` with
  PCM-frame streaming (clause-chunked) so first audio plays in ~200–300 ms.
- [x] **R3-4 — Wire macOS hardware-AEC into capture (close G3)**: capture via `AVAudioEngine`
  `VoiceProcessingIO` (PyObjC) so HW-cancelled PCM reaches Python, replacing `sounddevice`
  when `aec_mode=voiceprocessing`.
- [x] **R3-6 — Drop-in noise suppression**: optional pre-VAD denoiser (RNNoise / ONNX speech
  enhancement) on mic frames, behind a config flag.

**Wave D** (breadth / ops):

- [x] **R3-5 — Speech-to-speech / realtime LLM option** (OpenAI Realtime / Gemini Live):
  stream mic audio to a realtime endpoint and play its audio back, bypassing the cascade.
- [x] **R3-7 — Per-stage latency telemetry**: emit STT/LLM/TTS/first-audio latencies keyed by
  `speech_id` to the bus + structured log (optional OpenTelemetry).
- [x] **R3-8 — Verified first-run bootstrap**: a preflight that fetches+checksums the Smart-Turn
  model (and Piper voices), with a surfaced warning when it falls back to the silence timer.
- [x] **R3-9 — Telephony reach**: a Twilio-Media-Streams serializer over the WS transport.

Wave C merged at `1f44a29` (8 commits, +~2900 lines, new `webrtc_transport.py`/`denoise.py`;
**170 tests pass**, lint clean). aiortc WebRTC verified end-to-end (two real peers, Opus); macOS
HW-AEC capture verified live (OS-cancelled audio into numpy); network-duplex barge-in, streamed
clause-level TTS, and a spectral-gate denoiser all wired + tested. Caveats: `pyrnnoise` is
arm64-runtime-broken → numpy spectral denoiser is the working default; concurrent wake-listen +
VoiceProcessingIO has a documented device-contention edge.

Wave D merged at `93f3227..841bc2f` (7 commits, new `realtime.py`/`metrics.py`/`preflight.py`/
`telephony.py`; **202 tests pass**, lint clean): OpenAI-Realtime speech-to-speech brain, per-stage
latency telemetry (+ live-verified OpenTelemetry), checksum-verified `--preflight` bootstrap (real
Smart-Turn SHA pinned), and a Twilio Media-Streams telephony transport (G.711 μ-law verified vs stdlib).

**All round-3 gaps (R3-1…R3-9) closed.**

### Repo 1 — pipecat · Round 4 → WINNER: pipecat ("not close")

With the conversational core matched/beaten, the checker now decides on **breadth + ecosystem +
production maturity**: pipecat is a Daily.co framework (~13k stars, ~130 contributors, 20+ STT /
30+ TTS / 23 LLM integrations, JS/React/iOS/Android/**ESP32** client SDKs, its own 14-lang Smart
Turn model, Krisp-grade AEC, mem0 memory, production deployments). 8 gaps, split by closeability:

- **CODE-achievable** (a Wave E could do): typed prioritized interruption/event model (#2);
  persistent memory + provider-agnostic context aggregation (#7); pluggable STT/TTS service
  registry + adapters (#1, the interface); cross-platform whisper.cpp + Linux playback/AEC (#8);
  Smart-Turn latency bench + language matrix (#4).
- **Account/hardware/commercial-SDK-gated** (human): live-verify cloud/realtime/Twilio adapters
  (#6); Krisp/Koala AEC measured on open speakers (#3); an ESP32/satellite hardware client (#5).
- **Structural / uncloseable by code**: ecosystem scale, contributor count, stars, production
  track record — a fair indifferent judge weights these for "real home use" and a solo repo
  cannot manufacture them.

Honest assessment: rounds 1–3 won the conversational core; the residual gap is largely
maturity/breadth/live-verification that code alone won't flip on a fair judge.

**Decision (Albert): Wave E + reframe the judge, and continue.** The reframed checker must judge the
REAL use case — **"different people talking to a Mac"** (multi-user, on-device, household) — and is
**explicitly told to IGNORE ecosystem-maturity metrics** (stars, contributors, company backing,
production track record). STT/TTS *integration breadth* stays a fair capability axis. **ESP32 is now
in scope** — Albert has an **M5Stack Atom (ESP32)** to target.

Plan:

- **Wave E** (core code): pluggable STT/TTS service registry + real Deepgram/ElevenLabs/Cartesia
  adapters; whisper.cpp (non-MLX) STT + Linux playback/AEC (cross-platform — brain off-Mac); typed
  prioritized interruption/event model (#2); persistent memory + provider-agnostic context
  aggregation (#7); Smart-Turn latency bench + language matrix (#4). **Commit per gap** (survive cutoffs).
- **Wave F** (clients): an **ESP32 firmware client for M5Stack Atom Echo** speaking the WS audio
  protocol, a formalized web/JS client + a mobile example, and `clients/PROTOCOL.md`. Builds against
  the EXISTING transport — no core `src/` changes.
- Then **round-5 reframed checker** (multi-user-on-Mac; maturity metrics excluded). Loop until it picks my-stt-tts.

**Wave E + Wave F merged.** Wave E (`fe4702f..51a0ab2`, +105 tests → **307 total**, lint clean):
pluggable registry + Deepgram/ElevenLabs/Cartesia adapters, whisper.cpp + Linux/WebRTC-APM AEC,
typed non-droppable event model, per-speaker SQLite memory + provider-agnostic context aggregator,
Smart-Turn latency bench. Wave F (`a383d86`): M5Stack-Atom ESP32 firmware (PlatformIO build passes),
web/JS client, mobile guide, `clients/PROTOCOL.md`. (Fixed one host-dependent bench test → hermetic.)

### Repo 1 — pipecat · Round 5 (reframed) → 🏆 WINNER: my-stt-tts — REPO #1 WON

Reframed judge (*"different people in a household talking to a Mac"*; maturity metrics ignored) picked
**my-stt-tts**, on three capability axes: (1) **multi-user is first-class** — ECAPA speaker-ID wired
into per-speaker persistent memory + guest bucketing + provider-agnostic context (pipecat only exposes
per-call diarization, no built-in per-speaker memory); (2) **out-of-the-box on-device open-speaker AEC**
(pipecat delegates echo cancellation to paid Krisp cloud / the client); (3) **local-first/private,
Mac-centered topology** with real ESP32/web/mobile satellites shipped, vs pipecat's cloud/WebRTC center
of gravity. Honest caveats noted (narrower premium STT/TTS breadth; prototype/not-all-verified-live;
Atom Echo half-duplex).

**Standing criteria for ALL remaining repos:** this same reframed basis — household/multi-user on a
Mac; capability only; **ignore ecosystem-maturity/popularity** (stars, contributors, company, track
record); STT/TTS *integration breadth* stays a fair axis.

### Repo 2 — livekit/agents · Round 1 (reframed) → 🏆 WINNER: my-stt-tts — REPO #2 WON (first pass)

Judge: my-stt-tts wins on voice-biometric speaker ID (ECAPA enrollment + rejection margin) wired into
**cross-session per-speaker memory** (LiveKit has only single-session cloud diarization — no enrollment,
no per-person memory), full on-device privacy + no-API-key, on-device open-speaker AEC, and a buildable
in-home satellite path (ESP32). LiveKit's only edge is cloud-integration breadth — the axis least aligned
with a private household-on-a-Mac assistant, and my-stt-tts covers it opt-in via the registry anyway. No gaps.

### Repos 3–5 · Round 1 (reframed) → 🏆 ALL WON (first pass)

All three fresh indifferent judges picked **my-stt-tts** on the standing criteria — each decided by the
same axis: per-speaker voice ID wired to per-person memory + on-device AEC/barge-in + the home-hardware
satellite path, none of which the references have. **#3 HF speech-to-speech**, **#4 GLaDOS**, **#5
RealtimeSTT/TTS** — all won, no gaps.

**🏁 Every ranked repo (1–5) is won.** But repo #3's judge flagged — and I confirmed — a real
correctness gap: the multi-user pieces (ECAPA `EcapaEmbedder`, `SpeakerIdentifier.identify`,
`Brain.set_speaker`) are implemented + unit-tested but **never invoked in the live turn/wake/transport
loop** (`set_speaker` has zero callers; the embedder is never run per-utterance). The wins lean on
multi-user being real, so this must work end-to-end, not just in tests.

### Post-win correctness fix — wire speaker-ID into the live loop (IN PROGRESS)

Embed the captured utterance → `identify` → `brain.set_speaker(name)` before generation, in `run_turn`
(PTT), `run_wake_loop`, the barge-in re-capture, and `net_loop` (transport) — gated + graceful when no
enrolled profiles / speechbrain unavailable. Then a test that the live path actually calls
identify→set_speaker. After that, the multi-user claim is genuinely runtime-true.
