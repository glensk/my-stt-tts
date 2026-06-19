# PLAN — wake-detection fix + backend features (`wake-detection-fix`)

## Session

Resume: `c --resume <session-id>` (orchestrated; worktree `.worktree-wakefix`)

## Root cause: the wake word NEVER fired (`score=0.001` forever)

**dtype/scale mismatch at the openWakeWord model boundary.** `WakeWord.detect`
fed `predict()` a **float32 array in [-1, 1]** (`np.asarray(frame, dtype=np.float32)`),
but `openwakeword==0.4.0` requires **int16 PCM** (±32768). Its `AudioFeatures`
buffers the raw audio as a Python list and re-casts it with
`np.array(...).astype(np.int16)` (utils.py:88) — so a float `[-1, 1]` signal is
**silently truncated to all zeros**. The model sees near-silence and the score is
pinned at ≈0.001 for every input — exactly the reported symptom. (No exception: the
list→int16 path bypasses the `dtype != int16` ValueError guard.)

Framing was NOT the bug — `listen_for_wake` already buffers leftover samples across
device blocks and feeds **contiguous exact-1280-sample frames** (no mid-stream
zero-padding). Resampling was also already correct. The single fault was the float
feed.

### Fix

`wake.py`: new `to_int16_pcm(frame)` (clip to [-1, 1], ×32767, cast int16; passes
int16 through unchanged) applied in `detect` before `predict`. Conversion lives at
the model boundary; the rest of the pipeline stays float32.

### REAL maziko verification (manual, against `wakewords/maziko.onnx`)

Generated "maziko" utterances with macOS `say` → 16 kHz mono, fed through the real
`WakeWord.detect()` with float32 capture-style frames + 1.5 s lead silence:

| input              | OLD (float feed) | NEW (int16 feed) |
| :----------------- | :--------------- | :--------------- |
| "ma zee ko"        | 0.0009           | **0.81** (FIRES) |
| silence            | 0.0009           | 0.0009           |
| "good morning…"    | 0.0009           | 0.0009           |

OLD path is pinned at 0.0009 for ALL inputs (the bug). NEW path crosses 0.5 on
"maziko" and stays at the floor for silence/other words. Confirmed.

## Plan

- [x] (1) Diagnose against real openwakeword 0.4.0 + maziko.onnx (float→int16 truncation)
- [x] (1) `to_int16_pcm` + feed int16 in `WakeWord.detect`; real before/after maziko proof
- [x] (2) `chime_wake` earcon; play it + fire `bus.wake()` once per detection (no repeat)
- [x] (3) `source` field on `bus.transcript`; thread `turn_source` through `_respond` /
      `run_turn` / `run_wake_loop` / `_ptt_target` / `on_turn` / transport (typed /
      push_to_talk / wake / live_audio)
- [x] (4) `mic_record_replay` on_action handler + `audio.record_fixed`; records ~3 s,
      plays it back, reports level/duration/rate via bus; worker thread, never crashes
- [x] (5) `_signal_mic_confirmed` → `mic_result(ok=True)` on any confirmed wake/PTT
      capture so the GUI can hide the macOS permission hint; record-replay emits it too
- [x] Tests: int16 feed, contiguous framing buffer, source tagging, record-replay,
      mic-confirmed signal, wake chime (14 new in `tests/test_wake_detection_fix.py`)
- [x] Lint clean (ruff/mypy/pylint) + core-only pytest + all-extras pytest
- [x] Update `.env.example` (wake-word section) + this PLAN

## Result

- Baseline: 476 passed. Now **490 passed** (all extras) / **488 passed + 2 skipped**
  (core-only). ruff + mypy clean; pylint 9.33/10 (only pre-existing duplicate-code).
- The fix is a one-line-of-intent change (float→int16 at the model boundary) with a
  real-model proof; the related backend features are additive and back-compatible
  (`source` defaults to "", new chime/handler/signal).
