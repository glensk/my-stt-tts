# Porcupine ideas — repo #4 of the wake-detection checker loop (PLAN)

> Picovoice Porcupine's engine is closed/unadoptable. Port the two PORTABLE IDEAS
> onto our open machinery: (1) a unified 0–1 `wake_sensitivity` knob calibrated off
> the measured fa_eval curve, and (2) a noise×SNR benchmark harness extending fa_eval.
> Python only — a GUI agent consumes the SHARED CONTRACT in parallel.

## Session

Resume: `c --resume e7dfe88f-9001-4138-8cfc-1f8789653cc6`

Worktree: `/Users/albert/obsidian/42-Git/infra/.worktree-porc`  ·  branch `porcupine-ideas`

## Plan

- [x] Read wake.py / config.py / **main**.py / audio.py / events.py · baseline 973 core
- [x] Feature 1 — `wake_sensitivity` config field + per-word override map (env `WAKE_SENSITIVITY`)
- [x] Feature 1 — `sensitivity_to_threshold(word, sensitivity)` curve-inversion + linear fallback
- [x] Feature 1 — `wake_sensitivity` DERIVES `wake_threshold` (explicit threshold stays back-compat)
- [x] Feature 1 — `guidance` hint per word (raise/lower sensitivity from tier + wake_stats)
- [x] Feature 1 — `settings_dict` contract fields (sensitivity, derived threshold, calibrated, per-word)
- [x] Feature 2 — `mix_at_snr(speech, noise, snr_db)` in audio.py (RMS-match, pure numpy)
- [x] Feature 2 — adaptive threshold bracketing in `fa_eval` (bisection, replaces fixed linspace)
- [x] Feature 2 — SNR axis in `fa_eval` / `_run_fa_eval` → `per_snr` + `snr_list` on the event
- [x] Feature 2 — `--benchmark` CLI flag (SNR-matrix fa_eval, per-SNR table + artifacts)
- [x] Feature 2 — corpus recipe in WAKEWORD.md (LibriSpeech test-clean negatives, MUSAN/DEMAND noise)
- [x] config `noise_corpus_dir` (env `WAKE_NOISE_CORPUS`, default `debug/noise/`)
- [x] pytest — all the specified cases, mocked clips/noise
- [x] Verify core-only (uv sync; ruff/mypy/pytest), then `uv sync --extra all`
- [x] Update `.env.example` + this PLAN + repo #4 round log in PLAN_wake_checker_loop.md
- [x] Commit on `porcupine-ideas` (no author/tool attribution)
</content>
