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

- [ ] **G1 — True barge-in** (interruptible playback): keep mic live during TTS,
  run VAD on it, abort the in-flight TTS subprocess + cancel the LLM stream on
  confirmed user speech. (Playback must become a cancellable async task.)
- [ ] **G2 — Smart-turn / prosodic endpointing**: a `TurnAnalyzer` (Smart Turn v3)
  layered on VAD so end-of-turn is decided from intonation, not a fixed silence timer.
- [ ] **G4 — False-interrupt suppression**: require a min word count / speech
  duration before honoring an interruption (backchannels, coughs, TV).
- [ ] **G5 — Post-interruption context repair**: store only the *spoken prefix* of
  an interrupted reply in history (today `brain.py` appends the full reply).
- [ ] **G6 — Streaming STT (partials)**: emit partial transcripts during speech
  instead of transcribing the whole clip after the turn ends.
- [ ] **G3 — AEC + noise suppression**: macOS `VoiceProcessingIO` / RNNoise so open
  speakers don't self-trigger (the unlock that makes G1 safe in a room). *(round 2)*
- [ ] **G7 — Network audio transport**: WebRTC/WebSocket so a remote mic/speaker or
  browser client can carry audio, not just the local mic. *(round 2)*

Current action: worktree-isolated implementer building the conversational core
(G1, G2, G4, G5, G6). G3/G7 follow in a later round. Then a fresh round-2 checker.
