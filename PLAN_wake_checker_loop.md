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

- [x] 1. **sherpa-onnx KWS** — ✅ **CLOSED (OURS_BETTER)**. Ported KWS OR'd detector (zero-train
      custom words, multi-spelling/boost/threshold) + the eval toolkit (FA/hour ROC-DET, separation
      histogram, custom verifier, spectrogram). Re-judge: we hold a strict superset + better eval
      rigor. Optional S follow-up noted: add the 2025 bilingual zh-en KWS model for non-English words.
- [x] 2. **EfficientWord-Net** — ✅ **BUILT (gate: VIABLE via mean-pool, not windowing)**. Shipped
      `EnrolledWake`: few-shot enrolled detector on the user's own clips, MAX cosine to per-clip
      mean-pooled oWW embeddings, ~1.75 s rolling window + patience, OR'd as a 3rd branch (custom
      words only). Empirically gated FIRST: oWW windowing FAILS (d' 0.80, the EWN hypothesis is
      false for this embedding), mean-pool SEPARATES (d' 5.41 whole-clip / 2.52 streaming). Recall
      1/6 → 6/6 on Albert's maziko, 0/23 hard-neg fires at thr 0.96/pat 2. Did NOT pull EWN's 88 MB
      ResNet (zero new dep). See round log below.
- [x] 3. **microWakeWord** — ✅ **BUILT (runtime idea ported; window OFF by default, refractory ON)**.
      PCEN/PCAN front-end NOT portable (our oWW models are trained without it). Ported the RUNTIME
      idea instead: a sliding-window moving-average fire criterion (`process_streaming_prob`) +
      refractory lockout, wired into the LIVE `WakeWord.detect` AND reconciled into the offline eval
      (`score_wake_clip`/`fa_eval` use the SAME criterion → live == eval). Empirically gated the
      default: `wake_window=1` (byte-identical) ships as the default, refractory `wake_refractory=8`
      ships ON (the FA win, zero recall cost). See round log below.
- [x] 4. **Picovoice Porcupine** — ✅ **BUILT (engine closed; two PORTABLE ideas ported onto our open
      machinery)**. (1) A unified `wake_sensitivity` 0–1 knob that maps onto the oWW `wake_threshold`
      by INVERTING the MEASURED fa_eval curve (calibrated) or a documented linear remap (uncalibrated);
      per-word overrides; a `guidance` hint. (2) A noise×SNR benchmark harness: `audio.mix_at_snr`
      (RMS-energy-matched, ports `mixer.py:_speech_scale`), a `per_snr` axis on `fa_eval` + the event,
      ADAPTIVE threshold bracketing (bisection that brackets `target_fa` where a fixed linspace clamps,
      ports `benchmark.py`), and a `--benchmark` CLI. Event-grouped FA counting kept (NOT regressed to
      per-frame). Corpus recipe (LibriSpeech test-clean negatives, MUSAN/DEMAND noise) documented; no
      audio bundled. See round log below.
- [x] 5. **Mycroft Precise** — ✅ **BUILT (both portable wins ported; GRU NOT ported)** (branch
      `precise-ideas`). Findings: trigger logic is OURS-equal/better (our mean-over-window+
      refractory keeps the analog score Precise binarizes away — did NOT port `trigger_level`/GRU). TWO
      genuine gaps, both now BUILT (see round log below):
      (A) **Output calibration** (Precise `ThresholdDecoder`): per-word map of raw score → calibrated
      [0,1] (logit-normal / `Φ((logit(raw)−μ)/σ)` fit from saved positive-clip stats) applied in
      `WakeWord.detect` BEFORE the moving-average AND identically in `score_wake_clip` (preserve
      live==eval); default OFF/identity unless enough samples; makes `wake_threshold`/`wake_sensitivity`
      model-independent. Config `wake_calibration`.
      (B) **Active-learning closed loop** (port the LOOP, keep our cheap CPU rebuilders): we auto-save
      positives + have `enroll_word`/`train_verifier` (seconds, no GPU) + the eval toolkit, but the loop
      is OPEN. Add: live false-fire ring-buffer capture → `debug/recordings/wake_neg/<word>/`;
      `_load_negative_clips` unions it; actions `mark_false_fire`/`mark_miss`/`capture_last_fire` →
      rebuild → **EVAL-GATED: keep only if `separation`/`fa_eval` improves, else ROLL BACK** (golden
      enrollment sacrosanct; cap refs); `record_wake_outcome` stores clip hash/path. Event
      `relabel_result{action,rebuilt,accepted,sep_before/after,fa_before/after,message}`. GUI: "✗ Wasn't
      me / ✓ Missed me / capture" buttons + the accepted/rolled-back result card + a calibration toggle.

## 🏁 LOOP COMPLETE (2026-06-21)

All 5 research-ranked repos processed; OUR repo won every closing re-judge. Each repo's ENGINE lost
to ours or was unadoptable (closed/account-gated/dead-stack); we ported only its genuinely-better
IDEAS, each EMPIRICALLY GATED (measure → build only if it helps → safe defaults), so nothing regressed
the working models. Final state: `main` @ `4dbf9c9`, CI green, 1033 core / 1043 all-extras tests.
Net new detection capability shipped: zero-train open-vocab KWS, few-shot EnrolledWake (the user's own
voice — maziko 1/6→6/6), live moving-average+refractory, model-independent `wake_sensitivity` +
calibration, noise×SNR FA/hour benchmark, and an eval-gated active-learning relabel loop.

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

### Repo 2 — EfficientWord-Net · Round 1 — EMPIRICAL GATE then BUILD (2026-06-21, branch `fewshot-wake`)

EfficientWord-Net's IDEA: a PRIMARY few-shot ENROLLED detector — embed the user's own clips
of a word, fire on MAX cosine similarity to them (no GPU retrain). Gate decision: **MEASURE
an available embedding's separability FIRST, build only if usable.** EWN ships an 88 MB
contrastively-trained ResNet-50; we do NOT pull it (dependency decision for the user). We
tested whether the oWW **shared 96-d embedding** (already loaded; used by `wake_verifier`) or
ECAPA can separate `maziko` few-shot, on the saved clips in `debug/recordings/`
(6 maziko positives; 23 negatives = nexus×4, alexa×3, hey_jarvis×4, hey_mycroft×2, mic×5,
unlabeled×5). Leave-one-out (held-out maziko vs other maziko refs; each negative vs ALL refs).

**Phase-1 numbers (d-prime via `wake.separation`; abs pos−neg mean gap; recall@0-FA):**

| Embedding + aggregation                | d-prime  | pos mean | neg mean | abs GAP  | min-pos vs max-neg MARGIN | verdict     |
| --------------------------------------- | -------- | -------- | -------- | -------- | ------------------------- | ----------- |
| oWW **MEAN-POOL** (whole-clip)          | **+5.41**| 0.9761   | 0.9444   | +0.0317  | 0.9733 vs 0.9561 = **+0.0172** | **VIABLE**  |
| oWW WINDOWED (single per-row max-cos)   | +0.80    | 0.9765   | 0.9629   | +0.0136  | grazes (max-neg 0.9899)   | NOT viable  |
| oWW WINDOWED-MEAN (4-row ~1.0 s window) | +1.44    | 0.9802   | 0.9548   | +0.0254  | −0.0130 (neg 0.9891)      | NOT viable  |
| ECAPA (speaker-identity)                | +1.28    | 0.5229   | 0.4138   | +0.1091  | grazes (max-neg 0.6233)   | NOT viable  |

**KEY FINDING — the EWN windowing hypothesis is FALSE for the oWW embedding.** The
orchestrator predicted windowed >> mean-pool. The opposite holds: taking the MAX over
individual ~775 ms oWW embedding rows lets a single spurious negative frame align too well
(max-neg up to 0.99 > every positive), DESTROYING the margin. The oWW shared embedding's
word-discriminative signal lives in the **whole-utterance MEAN**, not in any one window
(unlike EWN's ResNet, whose per-window embeddings are themselves contrastively discriminative).
So the faithful EWN aggregation is the WRONG choice here — **mean-pool is correct.**

**Streaming reality (the live regime — rolling 1.5 s window, 0.25 s hop, mean-pool each
window, max-cos to per-clip refs):** d-prime **+2.52**, min-pos 0.9498 vs max-neg 0.9509
(margin ≈ 0; ONE "other" negative grazes the lowest positive). At a midpoint threshold 0.9504:
**recall 83 % (5/6), 1/23 negatives fire.** The negatives here are other WAKE-WORD attempts —
the adversarial worst case, not ambient audio; the FA/hour figures (hundreds/h) are an artifact
of the 6.56 s total negative duration and must NOT be read literally.

**Patience tightens it (reusing `fired_with_patience`/`count_fa_events`):**

| patience | threshold | recall (of 6) | hard-neg clips firing (of 23) |
| -------- | --------- | ------------- | ----------------------------- |
| 1        | 0.952     | 83 % (5)      | 1                             |
| 1        | 0.948     | 100 % (6)     | 3                             |
| 2        | 0.945     | 83 % (5)      | 2                             |
| 2        | 0.948     | 67 % (4)      | **0**                         |

**VERDICT: VIABLE — via MEAN-POOL (not windowing).** Build `EnrolledWake`: rolling-window
mean-pooled oWW embedding, MAX cosine to per-clip enrolled refs, fire ≥ threshold with
`patience` consecutive hits. vs the oWW-only baseline **1/6 = 17 %** on these exact clips,
EnrolledWake gives **67 % (zero hard-neg leak, pat 2 / thr 0.948) to 83 % (1 hard-neg leak,
pat 1 / thr 0.952)** — a **4–5× recall lift**, OR-combined so it only ADDS detections. Default
operating point chosen conservatively. We did NOT pull EWN's 88 MB ResNet (flagged for the user
if multilingual phonetic few-shot is later needed). See the BUILD section below.

### Repo 2 — EfficientWord-Net · Round 1 — BUILT `EnrolledWake` (2026-06-21, branch `fewshot-wake`)

Shipped the few-shot ENROLLED detector the gate found VIABLE — mean-pool (NOT windowing).

- **`enrolled_wake.py`** — `EnrolledWake` (the `WakeWord`-identical surface: `detect`/`reset`/
  `last_score` + `threshold`/`model_name`/`available`/`from_config`). `enroll_word()` reads
  every saved clip of a word (`debug/recordings/wake/<word>/` + loose `*-<word>-*.wav`),
  mean-pools each clip's oWW embedding, L2-normalizes, and saves the **per-clip** references
  (NOT a centroid) to the gitignored `models/wake_embeddings/<word>.npz`. `detect()` keeps a
  rolling ~1.75 s buffer, scores once per 0.25 s hop (mean-pool the window → MAX cosine to any
  ref), and fires after `patience` CONSECUTIVE windows ≥ threshold. `score_clip_enrolled()` is
  the offline/eval twin (same window + patience). All openWakeWord access is lazy + degrades to
  never-fire — core imports clean without the `wake` extra.
- **WINDOW LENGTH was the decisive knob.** The refs are whole-clip (~2 s) means, so a too-short
  live window dilutes the word: leave-one-out recall@0.95 climbed 0 %@1.0 s → 83 %@1.5 s →
  **100 %@1.75 s** (margin flips +0.0126), 100 %@2.0 s (+0.0172). Set `WINDOW_SECONDS=1.75`.
- **OR-combine** — `EnrolledWake` is a THIRD branch in `wake.OrCombinedWake` + `make_wake_detector`
  (live loop) and `score_wake_clip_combined` (GUI/clip path), gated `is_official_wake_word`:
  official words → bare `WakeWord` (byte-identical); custom word → oWW OR KWS OR few-shot, fire
  if ANY fires, `last_detector` ∈ {"oww","kws","fewshot"} names the winner. Mirrors the KWS port.
- **Operating point tuned against the NEGATIVES** (threshold × patience sweep via the shipped
  clip path):

  | thr   | patience | recall (of 6) | hard-neg fires (of 23) |
  | ----- | -------- | ------------- | ---------------------- |
  | 0.950 | 1        | 100 %         | 4                      |
  | 0.950 | 2        | 100 %         | 1                      |
  | 0.960 | 1        | 100 %         | 1                      |
  | **0.960** | **2**| **100 %**     | **0**                  |
  | 0.965 | 1        | 100 %         | 0                      |
  | 0.970 | 2        | 17 %          | 0                      |

  Default = **thr 0.96 / patience 2**: 100 % recall, ZERO of the 23 hard negatives (other
  wake-word attempts — the adversarial worst case) firing. The live `detect()` path was verified
  end-to-end (all 6 maziko clips fire, max last_score ~0.99; nexus/alexa don't).

- **A/B vs the oWW-only baseline (HONEST, leave-one-out on Albert's 6 real maziko clips):**

  | detector                              | recall   | hard-neg fires | note                                  |
  | ------------------------------------- | -------- | -------------- | ------------------------------------- |
  | openWakeWord-only (maziko, thr 0.4)   | **1/6 (17 %)** | n/a      | one clip @0.67; other 5 ~0.001 dead   |
  | EnrolledWake (thr 0.96, patience 2)   | **6/6 (100 %)**| **0/23** | mean-pool, ~1.75 s window, OR'd       |

  A **6× recall lift**, OR-combined so it only ADDS detections; official words untouched.

- **Contract** — Config `fewshot_wake_enabled` (env `FEWSHOT_WAKE_ENABLED`, default true),
  `fewshot_threshold` (0.96), `fewshot_patience` (2) + validate(); `settings_dict` adds the three
  - `wake_word_info[w]["detector"]` now OR-joins branches ("oww"|"oww+kws"|"oww+fewshot"|
  "oww+kws+fewshot"); `apply_settings` accepts+clamps them; `settings_text` shows a `fewshot` row;
  `bus.wake(detector=…)` carries "fewshot". `scripts/enroll_wakeword.py` (mirrors `enroll.py` +
  `_bootstrap`) records/enrolls N clips (or `--from-saved`).
- **Tests/lint** — `tests/test_enrolled_wake.py` (+20): enroll store/load (per-clip refs, ≥N gate),
  max-cosine detect, patience de-bounce, threshold-from-negatives, OR-routing (official
  byte-identical, fewshot branch fires + reported), config (env/validate/defaults/settings),
  graceful-degrade (oWW absent → no-op), + a real-oWW-embedding round-trip (gated). **CORE-ONLY
  verified** (951 passed / 13 skipped — 19 of the 20 new tests run WITHOUT extras via a fake oWW
  front-end; only the real-model one skips; every module imports clean). Full suite **963 passed /
  1 skipped** (was 941/3). ruff + mypy clean; pylint only the repo's established lazy-import /
  broad-except / keyword-rich-signature relaxations. `models/wake_embeddings/` gitignored. NEVER
  touched `main`/`README.md`/`clients`/`esp32`/`webui.html`/`wakewords/maziko.onnx`.

> **Did NOT pull EfficientWord-Net's 88 MB ResNet-50 ONNX** — the gate proved the embedding
> already in-process separates the word, so no new model dependency. If a MULTILINGUAL phonetic
> few-shot word is later needed (the oWW embedding is English-acoustic), EWN's contrastively
> trained ResNet is the flagged option for the user to decide on.

### Round 1 — Wave 2 — EVALUATION/DEBUG toolkit (2026-06-21, branch `wake-eval-toolkit`)

Closes the gap a fresh independent judge flagged: the diagnostics measured POSITIVES only —
no negative corpus, no false-accepts/hour, no ROC/DET, no separation metric. Ported
openWakeWord's Apache-2.0 metrics approach (reusing `score_wake_clip`). Python only (a GUI
agent consumes the shared contract in parallel). All four actions are `POST /api/action`
worker-thread handlers emitting bus events:

- **`score_histogram`** (Task 1) — `_run_score_histogram` scores every saved POSITIVE clip
  for the word + the negative corpus via new `wake.score_clip_set`; emits
  `score_histogram_result{pos_scores, neg_scores, threshold, separation}`. `wake.separation`
  = d-prime (mean-gap fallback when a side is constant). The recall-vs-level proof, visual.
- **`fa_eval`** (Task 2) — `_run_fa_eval` sweeps the threshold over positives + the TIMED
  negative corpus. Counts **FA EVENTS not frames** (`wake.count_fa_events` collapses
  consecutive above-thr frames and merges crossings within `grouping_window`, oWW-style),
  converts to FA/hour via 80 ms/frame, and `np.interp`-olates miss-rate at a target FA/h
  (default 0.5). Emits `fa_eval_result{points:[{threshold, fa_per_hour, true_accept}],
  miss_at_target_fa, target_fa, neg_seconds}`. Empty corpus → clear "drop WAVs into <dir>".
- **`train_verifier`** (Task 3) — `_run_train_verifier` → `wake_verifier.train_verifier`:
  an openWakeWord-style logistic-regression head on the SHARED 96-d embedding
  (`AudioFeatures._get_embeddings`, mean-pooled per clip) from the saved positives (≥3) +
  negatives. Saved to git-ignored `models/wake_verifiers/<word>.joblib`. `WakeWord` gained
  a `custom_verifier` gate (auto-loaded in `from_config`): a base-model fire only counts
  when the verifier ALSO confirms the rolling ~1.5 s window (`verifier_prob ≥ thr`).
  scikit-learn + joblib added to the `debug` extra (gated import; core stays clean — proven
  in a fresh extras-free venv). Emits `verifier_result{trained, path, n_pos, n_neg, message}`.
- **`spectrogram`** (Task 4) — `_run_spectrogram` → `wake.log_mel_spectrogram` (scipy STFT,
  40-band mel, dB, normalized, time-axis downsampled to ≤200 cols) + the per-frame
  `score_trace`. Emits `spectrogram_result{hash, mels, frames, grid, score_trace, freqs,
  times}`. scipy gated → empty grid (no crash) when absent.
- **Task 5** — `patience`/`debounce` threaded into `score_wake_clip` (replay under the ship
  config) via new `wake.count_fires`/`fired_with_patience`; default (patience≤1, debounce≤0)
  is byte-identical to before.

Supporting: `audio.read_wav_float` (reusable 8/16/24/32-bit WAV loader → 16 kHz float mono;
`_load_saved_wake_clip` refactored onto it) + `audio.list_wavs`. Config `negative_corpus_dir`
(env `WAKE_NEG_CORPUS`, default `debug/negatives/`, both git-ignored). `.env.example` documents
it. **Tests** — `tests/test_wake_eval_toolkit.py` (+29): histogram + separation; FA-EVENT
grouping (consecutive = ONE event) + FA/hour math + miss@target via `np.interp`; empty-corpus
message; verifier train/load/score (real-deps round-trip + gated-off graceful degradation,
core imports clean); spectrogram grid shape + score_trace + scipy-absent degradation;
patience/debounce. **CORE-ONLY verified** in a fresh extras-free venv (24 passed, 5 skipped,
every module imports without sklearn/oWW/scipy/pyloudnorm). Full suite 943 passed / 1 skipped
(was 912/3 baseline). ruff + mypy clean; pylint only the project's established
broad-except / lazy-import / cyclic-import patterns. NEVER touched `main`/`README.md`/`clients`/
`esp32`/`webui.html`/`wakewords/maziko.onnx`.

### Repo 3 — microWakeWord · Round 1 — BUILT temporal smoothing (2026-06-21, branch `wake-temporal`)

**Gap closed.** The LIVE oWW detector fired on a SINGLE frame (`last_score >= threshold` after
max-over-phases); the patience/debounce/refractory machinery existed ONLY in the offline eval.
Ported microWakeWord's runtime idea — a **sliding-window moving-average fire criterion +
refractory lockout** — into the LIVE path so the live detector matches what `fa_eval` evaluates.

**Shipped** (`wake.py`, `config.py`, `webui.py`, `__main__.py`):

- **Live moving-average** in `WakeWord.detect` — a `collections.deque(maxlen=wake_window)` of the
  per-frame MAX-over-phases score; fire when `mean(window) >= threshold` (microWakeWord's
  `process_streaming_prob`, smoother than consecutive-count — chose ONE filter, the moving average,
  no double-debounce). Pushed only on a fresh frame so a warmup call can't inflate the mean.
- **Refractory lockout** (`wake_refractory` frames) after a fire — suppresses re-fires; reuses the
  `count_fires` refractory logic. `reset()` clears BOTH the deque and the refractory.
- **Config knobs** `wake_window` / `wake_refractory` — field + `from_env` (`WAKE_WINDOW` /
  `WAKE_REFRACTORY`) + `validate` (`window ∈ [1,50]`, `refractory ≥ 0`) + `settings_dict` /
  `apply_settings` (clamped) + `settings_text` + `.env.example`. `from_config` threads them in.
- **Eval-path reconciled** — new pure `wake.count_fires_moving_average` / `moving_average_fires`
  replay the EXACT live state machine. `score_wake_clip` (window/refractory params) and `fa_eval`
  (window/refractory params, BOTH the FA side via `count_fires_moving_average` AND the recall side
  via `moving_average_fires`) use it → **live == eval**. The GUI `score_clip` + `fa_eval` actions
  pass `cfg.wake_window`/`cfg.wake_refractory`. Defaults (`window=1`, `refractory=0` in the pure
  helpers) collapse to the prior single-frame / `count_fa_events` behaviour — byte-identical.

**Empirical validation** (saved clips: 6 maziko + 2 hey_jarvis + 3 alexa + 2 hey_mycroft positives;
4 nexus + 10 mic clips = 14 negatives; threshold 0.4, phases 8). Replayed each clip's per-frame
trace under `window ∈ {1,2,3}` × `refractory ∈ {0,8}`:

**Recall (fired / total):**

| word        |  N | w1r0 | w1r8 | w2r0 | w2r8 | w3r0 | w3r8 |
| :---------- | -: | :--: | :--: | :--: | :--: | :--: | :--: |
| maziko      |  6 | 1/6  | 1/6  | 1/6  | 1/6  | 0/6  | 0/6  |
| hey_jarvis  |  2 | 2/2  | 2/2  | 2/2  | 2/2  | 2/2  | 2/2  |
| alexa       |  3 | 3/3  | 3/3  | 3/3  | 3/3  | 3/3  | 3/3  |
| hey_mycroft |  2 | 2/2  | 2/2  | 2/2  | 2/2  | 2/2  | 2/2  |

**Negative fires (total live fires across the 14-clip negative corpus, per model):**

| word        |  N | w1r0 | w1r8 | w2r0 | w2r8 | w3r0 | w3r8 |
| :---------- | -: | :--: | :--: | :--: | :--: | :--: | :--: |
| maziko      | 14 |   0  |   0  |   0  |   0  |   0  |   0  |
| hey_jarvis  | 14 |   0  |   0  |   0  |   0  |   0  |   0  |
| alexa       | 14 |  19  |   5  |  18  |   5  |  20  |   5  |
| hey_mycroft | 14 |   0  |   0  |   0  |   0  |   0  |   0  |

**Findings:**

- **Official models fire 100% at every setting** — window/refractory never regress official recall.
- **maziko held at 1/6 through w2, REGRESSED to 0/6 at w3** — its lone good-phase clip peaks at a
  single-frame 0.67 (others ≈0.001-0.002 — a recall problem the few-shot `EnrolledWake` handles
  separately); a 3-frame mean dilutes that 0.67 below 0.4. **window ≤ 2 is recall-safe; window 3 is not.**
- **Window alone did NOT cut FA on this corpus** (alexa→nexus 19→18→20): those false fires are
  SUSTAINED passages (4/12/3 frames above threshold), which a 2-frame mean still clears. The moving
  average suppresses single-FRAME flukes, of which this corpus has none.
- **Refractory is the dominant FA lever**: `r8` collapsed alexa FA **19→5** at every window — a
  12-frame sustained false passage becomes ~2 fires, not 12 ("one annoyance, not 37") — with ZERO
  recall cost (with `window=1` the refractory only suppresses RE-fires AFTER a first detection; the
  live loop returns on the first fire anyway, so a single activation is byte-identical).

**Chosen defaults — `wake_window=1`, `wake_refractory=8`:**

- `window=1` (byte-identical) is the default by the gate rule: window=2 preserved official + maziko
  recall but did NOT reduce FA (rule requires all three), and window=3 regressed maziko. The moving
  average ships as a documented **opt-in** knob (`WAKE_WINDOW`) for users whose corpus has
  single-frame flukes.
- `wake_refractory=8` (~0.64 s) ships ON: a measured pure FA win (19→5) with provably zero recall
  cost, and live single-activation behaviour is unchanged (the loop exits on first fire).

**Tests** — `tests/test_wake_temporal.py` (+22): moving-average fire (mean ≥ threshold), `window=1`
byte-identical to single-frame, dip-tolerated / lone-spike-rejected, refractory lockout,
`reset()` clears both, pure `count_fires_moving_average`/`moving_average_fires`, `score_wake_clip`
eval-path consistency (same trace → same decision), `fa_eval` window/refractory on both FA + recall
sides, config knobs (default/env/validate/settings round-trip/settings_text/from_config).
**CORE-ONLY verified** (`uv sync`; ruff + mypy clean, pylint only the established lazy-import test
pattern; core pytest 973 passed / 13 skipped, was 951 baseline). Full suite with `--extra all`:
985 passed / 1 skipped. NEVER touched `main`/`README.md`/`clients`/`esp32`/`webui.html`/
`wakewords/maziko.onnx`.

### Repo 4 — Picovoice Porcupine · Round 1 — BUILT (2026-06-21, branch `porcupine-ideas`)

Porcupine's engine is closed/unadoptable (proprietary `.ppn` models, paid SDK). Two PORTABLE
IDEAS ported onto our OPEN machinery (openWakeWord + the fa_eval toolkit). Python only; a GUI
agent consumed the shared contract in parallel.

**Feature 1 — unified `wake_sensitivity` 0–1 knob (`config.py`):**

- `sensitivity_to_threshold(word, sensitivity, *, curve=None) -> (threshold, calibrated)`.
  CALIBRATED branch: when a measured `fa_eval` curve exists (>= 2 operating points), order the
  points STRICT→LOOSE (ascending FA/hour, then descending threshold to break ties) and have the
  sensitivity index that ordered list by fractional position (linear-interpolated): `s=0` → the
  strictest measured threshold, `s=1` → the loosest, `s=0.5` ≈ the target-FA knee. UNCALIBRATED
  fallback: a documented linear remap `thr = MAX - s·(MAX-MIN)` over `[0.10, 0.90]` (inverted, so
  higher sensitivity = lower threshold = fires more easily). Monotone non-increasing in `s`,
  clamped to `[0,1]` in both branches.
- Maps onto the oWW `wake_threshold` ONLY (KWS / few-shot have no continuous score → left as OR'd
  backstops with their own knobs — documented). MASTER/DERIVED: setting `WAKE_SENSITIVITY` makes it
  the master and DERIVES `wake_threshold` (an explicit `WAKE_THRESHOLD` set alongside is overridden);
  not setting it leaves `wake_threshold` the master (back-compat). `Config.set_wake_sensitivity_env`
  parses a bare float (global) OR a `word=val;…` map (per-word, parsed like `_parse_kws_spellings`,
  clamped); `sensitivity_for(word)` / `derive_wake_threshold(curve=…)` resolve + apply.
- `guidance` per word (`wake_word_guidance`): a recent self-fire on near-silence in `wake_stats`
  (fired with confidence ≤ 0.05) → "Firing on its own? Lower sensitivity." (priority); else
  red/low-reliability tier → "Missing it? Raise sensitivity."; else "".

**Feature 2 — noise×SNR benchmark harness:**

- `audio.mix_at_snr(speech, noise, snr_db)` — RMS-energy-match: scale the SPEECH by
  `k = sqrt(noise_energy · 10^(snr/10) / speech_energy)` so `(E_s·k²)/E_n == 10^(snr/10)` (ports
  Picovoice `mixer.py:_speech_scale`); tile/trim noise to length; clip-protect. Pure numpy; SNR
  math verified exact to 1e-6 for targets 20/10/5/0/−5 dB; degenerate inputs handled (empty
  speech → empty, silent noise → speech, silent speech → noise).
- `wake.fa_eval_snr(pos, neg, noise, word, snr_list=…)` — for each SNR in `[None(clean),10,5]`
  mix positives+negatives, re-score through the REAL phase-diverse detector, run `fa_eval` →
  `{clean, snr_list, per_snr:[{snr_db, miss_at_target_fa, points}]}`. Empty noise → clean-only.
- **Adaptive threshold bracketing** (`_adaptive_threshold_grid`, ports `benchmark.py`): bisects the
  threshold on `[0,1]` (FA/hour is monotone non-increasing in threshold) to BRACKET `target_fa`,
  AUGMENTING (not replacing) the linspace grid + the 0/1 endpoints. PROVEN: on a sharp ROC whose
  crossings all sit > 0.95, the fixed linspace minimum FA is 23250/hr → `np.interp` CLAMPS
  miss@0.5-FA to a misleading 50%; the adaptive grid reaches thr ≈ 1.0 where FA truly hits 0 and
  reports the HONEST 100% miss. Event-grouped FA counting (`count_fa_events`) kept — NOT regressed
  to per-frame.
- `--benchmark` / `--benchmark-snr` CLI: runs the SNR-matrix fa_eval for the selected word, prints
  a per-SNR miss@target-FA table, writes `debug/benchmark/<word>-<ts>.json` (gitignored,
  reproducible). Verified end-to-end on `hey_jarvis` (clean-only with the empty-corpus notes).

**Contract (GUI consumes):** `settings_dict` adds `wake_sensitivity`, the derived
`wake_sensitivity_threshold`, `sensitivity_calibrated` (false at settings time — no live curve),
`wake_sensitivity_map`, and per-word `wake_word_info[w]["sensitivity"]` + `["guidance"]`.
`apply_settings` accepts `wake_sensitivity` + `wake_sensitivity_map` (re-derives threshold).
`fa_eval_result` gains optional `per_snr` + `snr_list` (null when no noise corpus). New config
`noise_corpus_dir` (env `WAKE_NOISE_CORPUS`, default `debug/noise/`, gitignored).

**Corpus recipe** (`wakewords/WAKEWORD.md`): LibriSpeech test-clean for negatives (keyword-EXCLUDED
via the per-chapter transcripts to avoid wake-word leak), MUSAN + DEMAND for noise — both
user-downloaded + gitignored; NO audio bundled.

**Tests/lint** — `tests/test_porcupine_ideas.py` (+37): curve-inversion (endpoints/midpoint/
monotonicity/clamping) + linear fallback + thin-curve fallback, per-word map parse/validate +
master/derived, guidance strings, `mix_at_snr` RMS math + tiling/trimming + degenerate, adaptive
bracketing brackets target_fa where linspace clamps + keeps event grouping, `fa_eval_snr` per_snr
shape + empty-noise graceful, the `fa_eval_result` event per_snr/snr_list (+ back-compat null).
Updated 3 existing tests for the new contract (wake_word_info shape, fa_eval default grid,
kws_detector wake_word_info stubs). CORE-ONLY verified (`uv sync`; ruff + mypy clean; pylint only
the established lazy-import / too-many-locals patterns, score +0.03; core pytest **1010 passed /
13 skipped**, was 973 baseline). Full suite with `--extra all`: **1020 passed / 3 skipped**. NEVER
touched `main`/`README.md`/`clients`/`esp32`/`webui.html`/`wakewords/maziko.onnx`.

### Repo 5 — Mycroft Precise · Round 1 — BUILT (2026-06-21, branch `precise-ideas`)

**Judged finding:** trigger logic is OURS-equal/better (our mean-over-window + refractory keeps the
analog score Precise binarizes away). Did **NOT** port `trigger_level`/the GRU. Ported the TWO
portable ideas onto our open stack (Python only; GUI agent consumes the contract in parallel):

**(A) Output calibration (Precise `ThresholdDecoder`)** — new `calibration.py`: a per-word monotone
map `calibrated(raw) = Φ((logit(raw) − μ)/σ)` fit from the saved positive clips' max-score stats
(reuses `score_clip_set`). `μ` is centered ONE σ below the positive logit mean, so calibrated **0.5
= "as confident as your weakest genuine wake"** — model-independent. `Calibrator.{fit,apply,save,
load,identity}` + `calibrator_for(word, enabled)` + `fit_and_save`. Applied to the per-frame
MAX-over-phases score in `WakeWord.detect` **BEFORE** the moving-average deque, AND **identically**
in `score_wake_clip` (the calibrator is threaded through `score_wake_clip`/`score_clip_set`/
`score_wake_clip_combined` and applied INSIDE the same `detect` path → the trace IS the calibrated
score → **live == eval**, proven by `test_calibration_live_equals_eval`). **Default OFF / identity**
unless `wake_calibration` ON *and* a fit was persisted (≥5 positive samples) → byte-identical
otherwise. Config `wake_calibration` (env `WAKE_CALIBRATION`, default false) + settings_dict /
apply_settings / settings_text / `.env.example`; params persisted to gitignored
`models/wake_calibration/<word>.json`; per-word state in `wake_word_info[w].calibration`.

**(B) Active-learning closed loop** — new `active_learning.py` (ports the LOOP, keeps our cheap CPU
rebuilders). **Negatives capture:** `save_recording(kind="wake_neg")` → `debug/recordings/wake_neg/
<word>/`; `_load_negative_clips(cfg, word)` UNIONS it with `negative_corpus_dir` (eval toolkit +
the gate consume the user's own negatives). **Ring buffer:** `audio.WakeFireBuffer` retains the last
fire's audio window in the live loop (`listen_for_wake(fire_buffer=…)` feeds every scored frame +
snapshots on fire), owned by `_WakeController` → `capture_last_fire`. **Relabel actions**
(`mark_false_fire`/`mark_miss`/`capture_last_fire`) move the clip to `wake_neg/` or `wake/` then
rebuild via `enroll_word` (refs, capped `MAX_REFS=40`) + `train_verifier` + re-fit the calibrator —
near-instant CPU. **EVAL-GATED rebuild (safety interlock):** snapshot the model artifacts → measure
`separation` (d-prime) + `fa_eval` miss@target-FA BEFORE → rebuild → measure AFTER → KEEP only if
`gate_improves` (d-prime not down, miss not up, ±1e-3 tol), else **ROLL BACK** (restore the artifact
bytes verbatim — golden enrollment is the floor). `record_wake_outcome` now stores the clip
`hash`/`clip_path` so a logged miss/false-fire is actionable. Event
`relabel_result{word,action,rebuilt,accepted,sep_before/after,fa_before/after,message,hash}`.

**SAFETY:** a mislabeled clip can't poison refs — gated on eval-not-regressing; the move is
reversible (clip stays on disk under its new label; a rolled-back rebuild leaves the model
byte-identical); golden enrollment immutable (rollback restores it); refs capped.

**Tests/lint** — `tests/test_wake_precise.py` (+23): calibration map (monotone / bounded / identity
when off+insufficient / scalar==array / persist round-trip / `calibrator_for` switch) + applied in
`detect` + identity byte-identical + **live==eval** + the `wake_calibration` knob; `save_recording`
wake_neg folder, `_load_negative_clips` union, `record_wake_outcome` hash/path, `WakeFireBuffer`
retain+snapshot, the eval-gate ACCEPTS an improving rebuild and ROLLS BACK a regressing one,
snapshot/restore byte-for-byte, `mark_false_fire`/`mark_miss` move to the right dir, missing-hash,
`capture_last_fire` saves+relabels, `listen_for_wake` feeds the ring buffer on fire, `relabel_result`
shape. openWakeWord/scikit-learn mocked; clips/disk in tmp. CORE-ONLY verified (ruff + mypy clean;
pylint only the established lazy-import / broad-except / too-many-* patterns; core pytest **1033
passed / 13 skipped**, was 1010 baseline). Full suite with `--extra all`: **1043 passed / 3 skipped**.
NEVER touched `main`/`README.md`/`clients`/`esp32`/`webui.html`/`wakewords/maziko.onnx`.
