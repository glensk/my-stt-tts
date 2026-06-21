# Queued UX/bug batch — launch AFTER Wave 1 (kws-detector + gui-kws) merges

> These all touch files the in-flight Wave 1 owns (`__main__.py`, `config.py`, `wake.py`,
> `webui.html`), so they're queued to avoid merge conflicts. Launch as a backend+GUI pair once
> Wave 1 is merged + CI-green.

## Backend (after `kws-detector` merges) — branch `ptt-defaults`

- [x] **CRITICAL — PTT MLX-thread bug.** PTT errored "There is no Stream(gpu, 0) in current thread":
      `_ptt_target` ran STT in a FRESH daemon thread per click, but parakeet-mlx's GPU stream has
      thread affinity (model loaded on another thread). FIXED: all MLX/STT work now marshals onto ONE
      long-lived `_STTWorker` thread (`stt.stt_worker()`, serial queue) so the model is loaded + used
      on the same thread — `ParakeetSTT.transcribe` submits its load+decode to the worker and blocks
      for the result (public surface unchanged). Covers PTT, mic-test, wake, barge-in (all reach
      `ParakeetSTT.transcribe`). Test: two caller threads transcribe, MLX only ever touched from the
      one worker thread; PTT capture on a fresh thread no longer raises.
- [x] **Default wake phrase = `hey_jarvis`** — `config.py` field + `from_env` + `quickstart.sh` now
      default to `hey_jarvis` (fires 99-100% on Albert; maziko/nexus red). All words stay selectable.
- [x] **Exact model label `opus-4.8 xlarge`** — `CLAUDE_CLI_REASONING="xlarge"`, `model_label` joins
      it space-separated (`claude-cli / opus-4.8 xlarge`, dropping `· think`). Surfaced on
      `bus.response(model=…)`, `settings_text`, and a new `settings_dict["model_label"]` field.
- [x] **mstt-side mpv preflight** — `_mpv_preflight_gate` halts `main()` before opening `--browser`
      when mpv is missing (clear msg + install hint), skippable with `MSTT_SKIP_MPV_CHECK=1`,
      no-op when `music_enabled=false`. Mirrors quickstart.sh for the direct `./mstt --browser` path.
- [x] **Always `score_clip`/`wake_test` with a DETAILED reason** — `_classify_wake_outcome` classifies
      every outcome into `reason` ∈ {fired, level_too_low, not_detected, unavailable, no_clip} + a
      human `detail` (names the word + why: "level too low (pk N) — move closer" vs "wake word not
      detected (level OK)"). Emitted on BOTH `score_clip_result` and `wake_test_result`.

## GUI (after `gui-kws` merges) — branch e.g. `gui-unify-checks`

- [ ] **Unify ALL test buttons into ONE component** — TEST SERVER MIC, TEST BROWSER MIC,
      RECORD & PLAY · SERVER/BROWSER all show the SAME fields + the level sparkline + a Play button +
      Score-this-clip. One shared render function. **Delete RECORD & PLAY** buttons if redundant (Play
      is now in every test).
- [ ] **ⓘ info hover on EVERY field** — clipping, peak, sample-rate, AGC, NS, EC, gain, hash
      (=filename), true-peak, crest, DC offset, SNR, LUFS. Each explains what it is, its relevance,
      and which values are good/orange/bad (state-aware: explain red/orange/green, or all states).
- [ ] **Fix control-row ⓘ tooltip clipping** — the PUSH-TO-TALK ⓘ tooltip text is still cut off
      (earlier fix didn't fully cover the controls row); ensure it renders fully, anchored, un-clipped.
- [ ] **Always show 'Score this clip' result with detail** on every test panel.
- [ ] **Music player: ONE Pause/Play toggle button** — merge the separate Pause + Resume buttons
      into a single toggle (⏸ Pause while playing, ▶ Play while paused), reflecting the `music`
      event's paused/resumed state.

## Done already (this batch)

- [x] quickstart.sh mpv preflight (auto-install/halt) — `8383b53`.
