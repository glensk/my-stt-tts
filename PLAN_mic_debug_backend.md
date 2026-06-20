# PLAN — mic/wake diagnostics backend overhaul + 2 music fixes (`mic-debug-backend`)

## Session

Resume: `c --resume <session-id>` (orchestrated; worktree `.worktree-micbe`)

A GUI agent builds the front-end (`webui.html`) in parallel against the SHARED
CONTRACT below; this branch is backend-only — no `README.md` / `webui.html` /
`clients/` / `esp32/`. Core baseline 662 → 716 tests (+54); `--extra all` 666 → 720.

## Shared contract (backend implements; GUI consumes)

- **Recordings**: every mic/wake clip (server OR browser) saved as 16 kHz mono WAV
  under repo-local `debug/recordings/` (gitignored `debug/`), named
  `<YYYYmmdd-HHMMSS>-<kind>-<source>[-<word>]-<hash8>.wav`. Helper
  `audio.save_recording(clip, sr, *, kind, source, word=None) -> (path, hash8, wav_url)`,
  `wav_url="/recordings/<file>"`, `hash8 = sha256(pcm_int16)[:8]`.
- **HTTP**: `GET /recordings/<file>.wav` in `webui.py` serves `audio/wav`,
  path-traversal-safe (basename only, `.wav` only).
- **Actions** (POST `/api/action`): `mic_check` (server 2.0 s / browser-pcm),
  `wake_test` (extended), `play_recording` (by hash).
- **Events**: new `mic_check_result`; `wake_test_result` extended with
  `peak/level/levels/processing/hash/wav_url`. `levels` = ~48 per-window peaks.
- **Setting**: `mic_gain` (default 2.0, env `MIC_GAIN`, `0 < g ≤ 10`) — software
  gain on SERVER captures, clip-protected to ±1.0, reported as `processing.gain`.

## Plan

- [x] (1) Recordings infra: `audio.save_recording` / `compute_levels` / `apply_gain`
      / `recordings_dir`; `GET /recordings/` route (traversal-safe); gitignore `debug/`.
- [x] (1) `bus.mic_check_result(...)`; extend `bus.wake_test_result(...)` with the new fields.
- [x] (2) Server auto-gain: `cfg.mic_gain` everywhere (field/from_env/validate/
      settings_dict/apply_settings/settings_text/.env.example); applied clip-protected
      in `mic_check` + `wake_test` server captures, reported as `processing.gain`.
- [x] (3) Unified 2.0 s `mic_check` action (server + browser) → full `mic_check_result`
      (peak/level/rms/duration/sample_rate/levels[48]/processing/hash/wav_url). Legacy
      `mic_test` action kept working.
- [x] (4) Extend `wake_test` (server + browser) to 2.0 s; real phase-diverse
      `score_wake_clip`; same peak/level/levels; save+hash under `debug/recordings/`;
      emit extended `wake_test_result` (keeps word/confidence/fired + legacy `wav_path`).
- [x] (5) `play_recording` action — server plays a saved WAV by hash (best-effort).
- [x] (6) Music intent robustness (`music.py`): first token STARTS WITH "play"
      (play/plays/playing/playform/"play for"/"play from"), DE `spiel*`/`mach … an`,
      FR `joue*`/`mets` → play with the remainder as query; verb-fusion filler stripped
      ("play from Thriller" → "Thriller"); `play*` NOUN false-friends (playground/
      playlist/player/…) guarded. STT-garble: "Playform Tool lateralis" → "Tool lateralis".
- [x] (7) LLM honesty: `prompts/system_prompt.md` + `config._DEFAULT_SYSTEM_PROMPT` —
      the assistant CANNOT play/stream media itself; never say "I'll play …/Playing …";
      music plays only when the user literally says "play <song>".
- [x] Tests: `tests/test_mic_debug_backend.py` (recordings/route/gain/mic_check/
      wake_test/play_recording/prompt-honesty); STT-variant cases in `tests/test_music.py`.
- [x] Verify CORE-ONLY (ruff/mypy/pytest) then `--extra all`. Lint clean; no regression.
