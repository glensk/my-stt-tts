# Queued UX/bug batch — launch AFTER Wave 1 (kws-detector + gui-kws) merges

> These all touch files the in-flight Wave 1 owns (`__main__.py`, `config.py`, `wake.py`,
> `webui.html`), so they're queued to avoid merge conflicts. Launch as a backend+GUI pair once
> Wave 1 is merged + CI-green.

## Backend (after `kws-detector` merges) — branch e.g. `ptt-defaults-fix`

- [ ] **CRITICAL — PTT MLX-thread bug.** PTT errors "There is no Stream(gpu, 0) in current thread":
      `_ptt_target` runs STT in a FRESH daemon thread per click, but parakeet-mlx's GPU stream has
      thread affinity (model loaded on another thread). FIX: route ALL MLX/STT work through ONE
      long-lived worker thread (serial executor) so the model is loaded+used on the same thread
      (covers PTT, mic-test, wake, barge-in). Verify PTT shows transcript + acts again.
- [ ] **Default wake phrase = `hey_jarvis`** (fires 99-100% on Albert; maziko/nexus red). Config default.
- [ ] **Exact model label `opus-4.8 xlarge`** (not just "opus"/"opus-4.8 · think") in settings + the
      ASSISTANT transcript label — show model + the effort/size tier the CLI runs at.
- [ ] **mstt-side mpv preflight** mirroring quickstart.sh (halt early / clear msg if mpv missing).
- [ ] **Always `score_clip` with a DETAILED reason** — when a clip wouldn't fire, say which word it
      didn't detect + why (e.g. "level too low", "wake word not detected", peak/SNR context).

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

## Done already (this batch)

- [x] quickstart.sh mpv preflight (auto-install/halt) — `8383b53`.
