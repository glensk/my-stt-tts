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

- [ ] 1. **pipecat** — https://github.com/pipecat-ai/pipecat — *in progress (round 1)*
- [ ] 2. livekit/agents — https://github.com/livekit/agents
- [ ] 3. huggingface/speech-to-speech — https://github.com/huggingface/speech-to-speech
- [ ] 4. dnhkng/GLaDOS — https://github.com/dnhkng/GLaDOS
- [ ] 5. KoljaB/RealtimeSTT (+ RealtimeTTS) — https://github.com/KoljaB/RealtimeSTT
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
- [ ] **R2-5 — Network transport (G7)**: a WebSocket (and/or WebRTC) audio transport so
  remote satellites / the browser carry mic+TTS audio, not just the local mic.
- [x] **R2-6 — Robust interrupt plumbing**: feed the captured barge-in audio straight
  into the streaming transcriber (no from-scratch re-transcribe); interrupt as bus events.
- [ ] **R2-7 — Backend breadth + in-conversation tool calling**: optional cloud STT/TTS
  behind the existing seams (esp. better German TTS); real function calling in `Brain.stream`.

Merged R2-1/2/3/4/6 at `878bbc7` (7 commits, +1712 lines, new `aec.py`; **101 tests
pass**, lint clean). AEC: macOS hardware `VoiceProcessingIO` (PyObjC, available) + numpy
NLMS fallback (~19 dB ERLE); the HW-cancelled PCM isn't yet routed end-to-end through the
`sounddevice` capture path (residual G3). Smart-turn download guard tested with mocked
network only.

Current action: Wave B implementer (R2-5 WebSocket/WebRTC transport, R2-7 in-conversation
tool calling + optional cloud STT/TTS), then a fresh round-3 checker.
