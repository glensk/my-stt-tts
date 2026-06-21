# PLAN — Wake-detection EVALUATION/DEBUG toolkit

## Session

Branch: `wake-eval-toolkit` (worktree `/Users/albert/obsidian/42-Git/infra/.worktree-evaltk`).

Closes the gap an independent judge flagged: positives-only measurement, no negative
corpus, no FA/hour, no ROC/DET, no separation metric.

## Plan

- [x] Config: `negative_corpus_dir` (env `WAKE_NEG_CORPUS`, default `debug/negatives/`).
- [x] `audio.read_wav_float()` + `audio.list_wavs()` — reusable WAV loaders.
- [x] `wake.score_clip_set()` — score a list of clips → max-score arrays + traces.
- [x] `wake.score_wake_clip(patience=, debounce=)` + `count_fires`/`fired_with_patience` (Task 5).
- [x] `wake.count_fa_events()` — FA-EVENT grouping (Task 2 math, pure).
- [x] `wake.separation()` + `wake.fa_eval()` (+ `_miss_at_target_fa` via np.interp).
- [x] `wake_verifier.py`: `train_verifier` / `CustomVerifier` + `WakeWord.custom_verifier` gate (Task 3).
- [x] `wake.log_mel_spectrogram()` — scipy log-mel + downsampled grid (Task 4).
- [x] events: `score_histogram_result`, `fa_eval_result`, `verifier_result`, `spectrogram_result`.
- [x] `__main__`: `_run_score_histogram` / `_run_fa_eval` / `_run_train_verifier` / `_run_spectrogram`
      + action dispatch + `_ACTION_LABELS` + `_load_positive_clips`/`_load_negative_clips`.
- [x] pyproject: `scikit-learn` + `joblib` in `debug` extra.
- [x] gitignore `debug/negatives/` + verifier artifacts (covered by `debug/` + `models/`; documented).
- [x] `.env.example`: `WAKE_NEG_CORPUS`.
- [x] tests `tests/test_wake_eval_toolkit.py` (+29): histogram+separation; FA-EVENT grouping;
      FA/hour + miss@target np.interp; empty-corpus message; verifier gated on sklearn;
      spectrogram grid+trace + scipy-absent degradation; patience/debounce.
- [x] Verify CORE-ONLY (fresh extras-free venv: 24 passed / 5 skipped, all imports clean)
      then `--extra all` (943 passed / 1 skipped); ruff + mypy clean.
- [x] PLAN_wake_checker_loop.md Round 1 Wave 2 log.
