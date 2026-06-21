# Wake-DETECTION checker loop — PLAN

> Distinct from `PLAN_checker_loop.md` (the voice↔LLM suitability loop, all 5 repos already won).
> This loop targets wake-word **detection quality** + gathering every useful wake/debug tool.

## Session

Resume: `c --resume e7dfe88f-9001-4138-8cfc-1f8789653cc6`

## Goal (user: "run the full checker loop", token usage no constraint)

Per repo: an independent checker panel (3 judges, distinct lenses) decides "for wake-word
DETECTION, is &lt;repo&gt; better than ours, and what specifically?" If yes → implement the winning
feature(s) (worktree-isolated, tests, CI-green, merge) → re-judge → repeat until OUR repo wins
(majority) or a 3-round/repo cap. Then next repo. Also add "all the debugging + detection tools."

## Repos (research-ranked)

- [ ] 1. **sherpa-onnx KWS** — token-sequence decoding, multi-spelling keywords, per-keyword
      boost/threshold (no retrain); we already pin `sherpa-onnx==1.10.46` for diarization. JUDGING.
- [ ] 2. **EfficientWord-Net** — few-shot siamese on user's own samples → verifier on weak oWW hits.
- [ ] 3. **microWakeWord** — PCEN/PCAN front-end + multi-window gate (PCEN NOT a drop-in for our
      oWW models — they're trained without it; only safe with a PCEN-trained model).
- [ ] 4. **Picovoice Porcupine** — sensitivity calibration; accent+noise benchmark as a test harness.
- [ ] 5. **Mycroft Precise** — retrain-on-failures active learning; per-frame probability smoothing.

## Our detection baseline

openWakeWord ONNX · phase-diverse (8 staggered, max) · int16 · threshold + `wake_gain` · official
models (hey_jarvis/alexa/hey_mycroft fire 99-100% on Albert) · `score_wake_clip`+`score_trace` ·
diagnostics (level/SNR/true-peak/crest/LUFS, gain-sweep, gauge, per-frame trace) · per-word
recordings + data-driven reliability · Silero VAD · live sherpa diarization.

## Round log

### Repo 1 — sherpa-onnx · Round 1 — JUDGING (3-judge panel: accuracy / accent-robustness / debug-tooling)

## maziko retrain (CSCS) — ATTEMPTED, FAILED (2026-06-21)

The detached GH200 run FATAL'd on a `torchmetrics`/`torchvision::nms` import before real training;
the model it left on the PVC is a **regression** (clip 2517da05 0.67→0.056 lost the fire; d03f2ad3
0.0015→0.0107, still ≪0.4). NOT committed — original maziko.onnx kept. Root causes: only 3 source
clips (thin even with 214 augmentations), oversubscribed shared node, and a brittle import chain.
**Not retrying now** — `hey_jarvis`/`alexa`/`hey_mycroft` already fire 99-100% on Albert (the real
fix). The training pod was deleted to free the GH200. A future attempt needs more of Albert's real
clips (now auto-saved under `debug/recordings/wake/maziko/`) + a fixed-deps container.
