# Wake-word & brain overhaul — PLAN

## Session

Resume: `c --resume e7dfe88f-9001-4138-8cfc-1f8789653cc6`

## Context

Wake word unreliable — **proven a RECALL problem, not level/gain**: two same-level maziko clips
scored 0.67 vs 0.001; gain-sweeping the flat one did nothing; nexus is dead on Albert's voice;
the browser path clips (AGC + 2× gain). Fix = retrain on his voice + ship better models + UX.
Plus: brain → Opus, music pause fix, exact model label, and a wake-detection checker loop vs the
best public repos. Mic-mode Voice Isolation is NOT advised (mangles wake features; no app API).

## Plan

### Phase 1 — Brain + quick fixes  (backend wave A + GUI wave B)

- [ ] Default brain = **Opus** (`opus-sub`); update config default + quickstart.sh
- [ ] Show the EXACT model + reasoning level in the response label AND transcript (`claude-cli / opus-4.8 · <thinking>`)
- [ ] Fix the music **PAUSE** button (mpv IPC / GUI wiring) — pause does nothing while a clip plays

### Phase 2 — Wake-phrase UX & models  (waves A + B)

- [ ] **Color-code** each wake word green/orange/red by training-data amount / measured reliability in the WAKE PHRASE config
- [ ] **Ship extensively-trained official openWakeWord models** (e.g. hey_jarvis, alexa, hey_mycroft, hey_rhasspy) — marked green/reliable
- [ ] Mic & audio checks: replace the fixed maziko/nexus recall rows with a **selectable wake-phrase** test (browser + server)

### Phase 3 — Retrain maziko  (VPN ON — critical fix, in progress)

- [ ] Retrain maziko on CSCS GPU with `debug/recordings/*.wav` as positives + augmentation; retrieve `maziko.onnx`; validate it scores the saved clips higher than the current model

### Phase 4 — Wake-detection checker loop  (after Phase-2 research)

- [ ] Research the **5 best wake-word GitHub repos**
- [ ] Per repo: independent checker — "for wake-word DETECTION specifically, which repo is better?" If the other is better, identify the superior feature + implement it; repeat until ours wins; then next repo

### Phase 5 — Mic mode

- [x] Voice Isolation advice: use **Standard** (Voice Isolation hurts wake recall; no app API to set Mic Mode). Browser AGC/NS/EC already toggleable; server inherits iTerm's Standard mode.

## Key decisions

- Wake failure root cause = **recall on Albert's voice**, NOT input level (proven from saved clips).
- Use the **server** path for wake testing (browser AGC/clipping is unrepresentative; the live loop is server-side).
- Custom-verifier would only add strictness (hurts recall) — not used. Retrain is the cure.
