# PLAN — wake fires offline but NEVER on live real voice (`wake-debug`)

## Session

Resume: `c --resume <session-id>` (orchestrated; worktree `.worktree-wakedebug`)

## Symptom

The int16 fix (see `PLAN_wake_detection_fix.md`) made a `say`-synthesized "maziko"
score **0.81 offline**, but the wake word STILL never fires on the user's REAL
voice through the mic in the always-listening loop. Push-to-talk + STT work (good
transcripts) on the SAME mic — so capture is fundamentally sound; only the live
WAKE path scores ~0 on real speech.

## Root cause — frame-PHASE sensitivity (NOT capture, NOT resample, NOT int16)

openWakeWord scores **once per 1280-sample (80 ms) frame, locked to ONE phase**
relative to the spoken word. Measured against the real `wakewords/maziko.onnx`, the
score swings **~25x (≈0.03 .. 0.85) purely with where that 1280-frame boundary
lands** relative to the word (frame-phase sweep, 80-sample steps):

```
0.28 0.75 0.71 0.45 0.29 0.08 0.05 0.06 0.53 0.73 0.85 0.79 0.72 0.39 0.48 0.49
```

In an always-listening loop the frame grid is **fixed by capture timing**, so a
single live utterance gets exactly ONE phase. If it lands in a bad window every
frame covering the word scores ~0.03 and it never fires — even though the SAME
audio at a better offset scores ~0.7. Offline you re-run / naturally hit a good
phase (0.81); live you get one fixed-phase pass and miss.

Ruled OUT by measurement (all match the clean-16k baseline at fixed phase):

- capture / int16 conversion: the live path already feeds contiguous int16 frames;
- 48k→16k resample: linear-interp `np.interp` == soxr high-quality (no degradation);
- contiguous 1280 framing: verified no mid-stream zero-padding;
- stale streaming state: `Model.reset()` only clears the prediction deque, NOT the
  AudioFeatures raw/melspec/feature buffers — but a deep reset made **no
  difference**, and a polluted continuous stream actually scored HIGHER, so state
  pollution is not the cause.

Genuine model recall ceiling: ~5/8 synthesized `say` voices (Fred/Rishi/Tessa never
clear threshold even at the best phase — TTS voices the model wasn't trained on).

## Fix — phase-diverse detection (`WakeWord.phases`, default 8)

Run K openWakeWord detectors fed the same audio but each offset by `1280/K`
samples, fire on the MAX score. Covers the phase space so a phase-unlucky utterance
still fires. Cheap (predict ≈ 2.2 ms/frame → K=8 = 0.22 real-time factor) and adds
**zero false-positives** (max score over 36 s of unrelated speech = 0.001).

End-to-end through the real model + `WakeWord.detect`, fixed awkward phase:

| voice    | phases=1 | phases=8 |
| :------- | :------- | :------- |
| Samantha | 0.012 ✗  | 0.675 ✓  |
| Karen    | 0.017 ✗  | 0.549 ✓  |
| Alex     | 0.012 ✗  | 0.675 ✓  |
| Daniel   | 0.448 ✓  | 0.556 ✓  |
| Moira    | 0.566 ✓  | 0.782 ✓  |
| Tessa    | 0.003 ✗  | 0.024 ✗  |
| **recall @0.4** | **2/6** | **5/6** |

## Wake-debug recorder (`WakeDebugRecorder`)

On Start Wake, capture the first ~5 s of the EXACT post-resample 16 kHz int16
frames fed to `predict` (tapped in `listen_for_wake`, not a separate capture), save
a 16 kHz mono WAV to `~/.cache/my-stt-tts/wake-debug.wav` (configurable), and log to
stderr + the bus: sample rate, #samples, duration, RMS, peak (0-100% level), and the
MAX + MEAN wake score over the window. So capture-problem (wrong rate / near-silent /
clipped) vs model-recognition (good audio, low score) is one glance. Gated by
`WAKE_DEBUG_CAPTURE` (auto-ON whenever the audio debug instrument is on / `--browser`).

## Plan

- [x] Worktree `.worktree-wakedebug` on `wake-debug`; `uv sync --extra all`; 509 baseline green
- [x] Read live wake path vs PTT path; build empirical probes (real maziko.onnx + `say`)
- [x] Diagnose: frame-PHASE sensitivity is the live-recall bug (ruled out capture/resample/int16/state)
- [x] Fix: `WakeWord.phases` (K offset detectors, fire on max); `WAKE_PHASES` env, default 8
- [x] `WakeDebugRecorder`: tap exact 16 kHz frames → WAV + log stats/scores; `WAKE_DEBUG_CAPTURE`
- [x] Wire recorder into `run_wake_loop`; `wake_debug_capture_enabled` auto-gate
- [x] Config fields + validation + `.env.example` + `settings_text` (wake phases line)
- [x] pytest: phase-diversity recall, all-phase reset, recorder WAV+stats, exact-frame tap
- [x] Verify CORE-ONLY (uv sync; ruff/mypy/pytest 513✓) then `--extra all` (515✓); lint clean
- [x] Prove real-voice scores rise: phases=1 → 2/6, phases=8 → 5/6 @ threshold 0.4
