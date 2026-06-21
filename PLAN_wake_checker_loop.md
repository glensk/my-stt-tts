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

### Repo 1 — sherpa-onnx · Round 1 — IMPLEMENTED (3-judge panel unanimous: port KWS)

**Shipped** (branch `kws-detector`, worktree `.worktree-kws`): sherpa-onnx `KeywordSpotter`
as a SECOND, OR'd wake detector for **custom / self-trained words only**. Official words
(hey_jarvis/alexa/hey_mycroft) stay openWakeWord-only — byte-identical (guardrail
`is_official_wake_word`). Zero new dependency (reuses the `diarize` pin
`sherpa-onnx==1.10.46`).

- **`kws.py`** — `SherpaKws` wraps `KeywordSpotter` with a `WakeWord`-identical surface
  (`detect`/`reset`/`last_score` + `flush`). Lazy, checksum-verified download of the
  GigaSpeech English KWS model (int8 encoder/decoder/joiner ONNX + tokens + bpe) into the
  gitignored `models/`, mirroring `diarize.py`. Keywords built via **sentencepiece BPE**
  on UPPERCASED text (NOT `text2token` — it eagerly imports `pypinyin`, not a dep), with
  `:boost #threshold @label` and multi-spelling → one label. Native float32 16 kHz frames.
  Degrades to a no-op on any failure; never raises.
- **OR-combine** — `wake.OrCombinedWake` + `make_wake_detector` (live loop) and
  `score_wake_clip_combined` (GUI/clip path): official → oWW-only; custom + enabled +
  available → BOTH run, fire if EITHER, `detector` ("oww"|"kws") names the winner.
- **Contract** — `settings_dict` adds `kws_available` / `kws_enabled` +
  `wake_word_info[w]["detector"]` ("oww"|"oww+kws"); `wake` + `wake_test_result` events
  gain `detector`. Config: `kws_enabled` (env KWS_ENABLED, default true), per-word
  `kws_boost` (1.5) / `kws_threshold` (0.25), `kws_spellings` (env KWS_SPELLINGS).
- **Coexistence PROVEN** in one process: standalone onnxruntime (oWW) + sherpa-bundled
  onnxruntime 1.17.1 (KWS) load + run, no dlopen clash (1.10.46 self-bundles a distinct
  leaf dylib). Real-model test `test_real_kws_coexists_with_openwakeword`.
- **A/B on Albert's 6 real maziko clips (HONEST)** — oWW (thr 0.4 / phases 8) fires 1/6
  (2517da05 @0.67; the other 5 are ~0.0009–0.0022, dead). KWS @ boost 4.0 / thr 0.1 with
  accent variants recovers **d03f2ad3** (oWW 0.0015 — one of the two flagged-dead clips)
  but NOT e79574a0 or the other 3. So KWS adds **zero-train custom words** + recovers one
  otherwise-missed activation; it does **NOT** fully fix maziko (GigaSpeech is English; a
  non-native accent on a non-English word stays hard). Per-clip table:

  | clip      | oWW conf | oWW fires | KWS (boost4/thr0.1) | combined detector |
  | --------- | -------- | --------- | ------------------- | ----------------- |
  | cb95ddba  | 0.0009   | no        | no                  | —                 |
  | 2517da05  | 0.6696   | YES       | yes                 | oww               |
  | 03da107b  | 0.0009   | no        | no                  | —                 |
  | d03f2ad3  | 0.0015   | no        | **YES**             | **kws** (recovered)|
  | 97ff5c8d  | 0.0016   | no        | no                  | —                 |
  | e79574a0  | 0.0022   | no        | no                  | —                 |

- **Tests/lint** — +45 tests in `test_kws_detector.py` (keyword build, OR-routing,
  detector contract, graceful-unavailable, config; 2 real-model tests gated on the model).
  Core 848 → 891; full 852 → 897. ruff/mypy clean.

### Repo 1 — sherpa-onnx · Round 1 — JUDGING (3-judge panel: accuracy / accent-robustness / debug-tooling)

## maziko retrain (CSCS) — ATTEMPTED, FAILED (2026-06-21)

The detached GH200 run FATAL'd on a `torchmetrics`/`torchvision::nms` import before real training;
the model it left on the PVC is a **regression** (clip 2517da05 0.67→0.056 lost the fire; d03f2ad3
0.0015→0.0107, still ≪0.4). NOT committed — original maziko.onnx kept. Root causes: only 3 source
clips (thin even with 214 augmentations), oversubscribed shared node, and a brittle import chain.
**Not retrying now** — `hey_jarvis`/`alexa`/`hey_mycroft` already fire 99-100% on Albert (the real
fix). The training pod was deleted to free the GH200. A future attempt needs more of Albert's real
clips (now auto-saved under `debug/recordings/wake/maziko/`) + a fixed-deps container.
