# PLAN — backend functional fixes (`backend-fixes`)

## Session

Resume: `c --resume <session-id>` (orchestrated; worktree `.worktree-fixes`)

## Root causes found ("nothing recorded")

1. **Server-side push-to-talk used a terminal-only recorder.** `_capture_ptt` →
   `audio.record_push_to_talk`, which blocks on `input()` (Enter to start/stop).
   In `--browser` server mode the worker thread has **no interactive stdin** →
   `input()` returns on EOF immediately → a zero-length clip → empty transcript.
   Fix: a VAD-driven, no-stdin server capture (`record_until_silence`) reused for
   the GUI PTT path.

2. **Silero-VAD frame-size fragility.** Silero VAD accepts **exactly 512 samples
   at 16 kHz**; any other size raises `Input audio chunk is too short` inside the
   TorchScript model. A raised exception in the capture loop (`record_turn`,
   `capture_turn_clip`) kills the whole turn → empty clip. Fix: `SileroVad.is_speech`
   reframes/pads to 512 and never raises (returns `False` on failure), and capture
   frames are normalised to 512.

3. **Capture sample-rate not pinned to 16 kHz.** Device native rate is commonly
   48 kHz (the GUI scope shows it). PortAudio resamples on this Mac, but not on
   every backend; the macOS HW-AEC path already hardcodes 48k→resample. Fix: a
   reusable `resample_to(arr, src, dst)` + `reframe` and capture that opens at the
   device rate when 16 kHz isn't honoured and resamples to `cfg.sample_rate`.

4. **VAD threshold dropped quiet speech.** Default raised to a configurable
   `vad_threshold` (default 0.3, was hard 0.5) so a ~10% level utterance is kept.

## Plan

- [x] (1) `resample_to` + `reframe` helpers in `audio.py` (+ tests)
- [x] (1) Harden `SileroVad.is_speech` (reframe to 512, never raise) + `vad_threshold` config
- [x] (1) `record_until_silence` (no-stdin VAD capture) + route GUI/terminal PTT through it
- [x] (1) Pin capture to 16 kHz (`_supported_capture_rate` + resample) in wake + PTT
- [x] (2) Debug instrument: `bus.debug(...)`, `cfg.debug_audio`, capture/VAD/wake/STT logging
- [x] (3) Inject current local date+time into `_system_prompt()` via `zoneinfo`
- [x] (4) Cross-platform `mic_permission_status()` (Linux/Windows branches via platform.py)
- [x] (5) `voice_test` on_action handler (TTS the selected voice, worker thread)
- [x] (6) `model` on the response event (`events.py`) + llm_request detail
- [x] Tests: resample, VAD-threshold, time injection, cross-platform mic, voice_test (43 new)
- [x] Fix pre-existing flaky `test_mic_test_reports_silent_on_zero_capture` (mock permission)
- [x] Lint clean (core-only) + full pytest (457) + core-only pytest (455 + 2 skipped)
- [x] Update `.env.example` + this PLAN; commit to `backend-fixes`

## Result

- Core-only: `ruff format --check` / `ruff check` / `mypy src` clean; `pytest` 455
  passed, 2 skipped (torch-only VAD tests guarded with `importorskip`).
- All extras: `pytest` 457 passed.
- Baseline was 413 passed + 1 pre-existing failure; now +43 new tests and the
  flaky one fixed.
