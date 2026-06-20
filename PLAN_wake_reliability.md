# Wake-word data-driven reliability + per-word clip storage

## Session

Resume: `c --resume <session-id>`

## Goal

Make per-word wake reliability **data-driven from Albert's real tests** (not the static
training metric), save every wake-test clip per-word for future training, and render a
reliability **bar** in the WAKE PHRASE config UI. Self-trained models (maziko, …) must read
RED for a non-native accent; official models GREEN. Plus: enroll.py cycles a language hint so
one profile spans DE/EN/FR.

Worktree: `/Users/albert/obsidian/42-Git/infra/.worktree-reliab`, branch `wake-reliability`.
NEVER touch main / README.md / clients/ / esp32/ / wakewords/maziko.onnx.

## Plan

- [x] Worktree + core sync + baseline (784 passed, 5 skipped)
- [x] Read audio.py / main / config.py / webui.py / webui.html / enroll.py + tests
- [x] **1. Per-word wake clip folders** — `save_recording(kind="wake")` → `wake/<word>/<file>`
      (sanitized, traversal-safe). `audio.resolve_recording` (basename + subfolder search) and
      `find_recordings` (flat + recursive glob); `_serve_recording`/`_play_recording`/
      `_load_saved_wake_clip` updated. Filename is now `<ts>-<source>-<hash>.wav`.
- [x] **2a. wake_stats.json** — config.py: `wake_stats_path`, `load_wake_stats`,
      `record_wake_outcome`, `measured_reliability`, `wake_word_reliability`, `_tier_from_reliability`.
- [x] **2b. Data-driven reliability** — rewrote `wake_word_tier`/`wake_word_info`: priors
      (official 0.9, self-trained 0.3), measured override (server-biased, recent 10 mean conf),
      `tier` from scalar (>=0.7 green / 0.4-0.7 orange / <0.4 red). Added reliability/tested/measured.
- [x] **2c. Append outcome** on every server/browser wake_test (`record_wake_outcome`).
- [x] **3. UI reliability bar** — webui.html `#wakeReliab` bar by the dropdown for the selected
      phrase (width = reliability, tier colour, pct+note tooltip); kept option dots; legend now
      "measured chance it works for you"; reseeded demo (maziko 0.05 red, hey_jarvis 0.99 green,
      computer 0.55 orange, nexus 0.2 red).
- [x] **4. enroll.py** — `--languages de,en,fr`, cycles a language hint across clips, single .npy.
- [x] **Tests** — test_wake_reliability.py, test_enroll.py, conftest.py (isolates wake_stats.json);
      updated test_config.py / test_mic_debug_backend.py / test_wake_gain_diag.py.
- [x] **Lint** ruff/mypy/pylint clean; `node --check` OK; core 810 passed / 5 skipped, all-extras 814.
- [x] **.env.example / PLAN** updated; commit clean `feat:`/`fix:` to wake-reliability.
