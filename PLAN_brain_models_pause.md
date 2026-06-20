# PLAN — brain default = Opus, exact model label, music pause fix, wake reliability

Branch: `brain-models-pause`. Backend changes consumed in parallel by a GUI agent
via the SHARED CONTRACT (`settings_dict.wake_word_info` + the exact `model` label).

## Session

Resume: `c --resume <session-id>`  <!-- fill in from `claude --resume` list -->

## Shared contract (backend implements, GUI consumes)

- `settings_dict` carries `wake_word_info`:
  `{"<word>": {"tier": "green"|"orange"|"red", "note": str, "recall": float|null}}`.
- The reply `model` label is the EXACT model + reasoning level, e.g.
  `claude-cli / opus-4.8 · think` (`claude-opus-4-8` → `opus-4.8`). In `settings_text` too.

## Plan

- [x] **Default brain = Opus.** `DEFAULT_BRAIN_PRESET = "opus-sub"` in `config.py`;
  missing-anthropic-key hint points at `opus-sub`; `quickstart.sh` `ARGS=(--brain opus-sub)`
  (+ help/comments). All presets stay selectable. Bare `Config()` provider stays
  `anthropic` (provider-agnostic default; requires a key OR a key-free `--brain`).
- [x] **Exact model + reasoning label.** `config.model_label(provider, model)` resolves
  bare CLI aliases / pinned API ids to a marketing version (`opus-4.8`) and appends the
  claude-cli reasoning level (`· think`, `CLAUDE_CLI_REASONING`). Wired into
  `__main__._model_label` (the `bus.response(model=…)` source) and `settings_text`.
- [x] **Music PAUSE fix.** Root cause: `MusicPlayer._mpv_command` only waited for the
  socket FILE to exist then made ONE `connect()`; a cold mpv launch (or one resolving a
  YouTube URL) was not accepting yet, so pause silently no-oped. Fix: poll a real
  `connect()` within a raised budget (`_IPC_WAIT_S` 3→8 s) and READ mpv's `{"error":…}`
  ack so the return reflects acceptance. Verified live against a real mpv (pause property
  toggles). Tests mock the socket (retry, error-ack, quiet-mpv, give-up, no-path).
- [x] **Wake reliability metadata.** `config.wake_word_tier` / `wake_word_info`
  (official → green; recall ≥0.70 green; [0.50,0.70) orange; <0.50 / unknown-self-trained
  red). Added to `settings_dict`. Recall sourced from `WAKEWORD.md`.
- [x] **Ship official models.** `scripts/fetch_official_wakewords.py` places openWakeWord's
  official `alexa`, `hey_jarvis`, `hey_mycroft` into `wakewords/` (prefers
  `download_models()`, else bundled weights). `.gitignore` excepts the new names; official
  `alexa` replaces the self-trained one. `WAKEWORD.md` updated (groups + tier note + table).
- [x] Tests: opus default, model-label mapping, pause IPC, `wake_word_info` shape/tiers,
  official models discoverable + loadable. `.env.example` updated.
- [x] Lint clean (ruff/mypy/pylint/shellcheck); core + full pytest green.

## Notes / decisions

- The official melspectrogram/embedding feature models are NOT copied into `wakewords/` —
  openWakeWord loads them from its installed package resources even for a model loaded
  from an arbitrary path (verified).
- `hey_marvin` ships as a file in some wheels but is not a registered pretrained wake
  model (`get_pretrained_model_paths()` omits it), so it is intentionally not shipped.
- README.md left untouched per task scope (GUI/docs owners).
