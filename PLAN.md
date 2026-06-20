# PLAN — `my-stt-tts`: local voice assistant on a MacBook M1

> A hand-wired, low-latency voice loop running entirely on a MacBook M1 (Apple
> Silicon): **wake word → record → speech-to-text → an LLM (streaming) →
> text-to-speech → playback**, with speaker identification and German / French /
> English support. The Mac is the prototype target; the design stays portable so
> the brain can later move to a server and the mics/speakers to whole-house
> satellites (ties back to the sibling `home-assistant-sandbox` repo).

## Session

Resume: `c --resume <session-id>`  <!-- fill in from `claude --resume` list; this plan was authored 2026-06-17 -->

## Build status (2026-06-20) — Wave M4: unify the two log streams (EVENT LOG ⇄ terminal)

Make the GUI EVENT LOG (bus events → SSE + `logs/events-*.jsonl` file sink) and the
terminal output captured by `quickstart.sh` (`logs/quickstart-*.log`, a tee of stderr)
each a SUPERSET of the other. Branch `log-unify` (worktree `.worktree-logunify`),
backend only — no README / webui.html / clients / esp32. 646 → 662 tests core (+16),
green core-only (662 + 4 extras-gated skips) and `--extra all` (650 → 666). Lint clean
(ruff format + check, mypy `src`, pylint — no new warnings; `main`'s pre-existing
complexity warnings unchanged). `events.py` stays stdlib-only (core import).

Two bridges close the gaps (each stream lacked things the other had):

- **(A) Bus → stderr console sink** (`EventBus._write_console`, called from
  `publish`). Every published bus event is mirrored to stderr as a concise human
  one-liner (`[event:state] listening`, `[event:response] [final] …`,
  `[event:music] playing "<title>"`; unknown types fall back to a `key=value` dump
  via `render_console_line`). So the stderr tee (quickstart.log) now captures the
  SAME state/transcript/response/music/mic_result/wake_test_result/speaker/metrics/log
  events the EVENT LOG shows. **Skips** `type=="debug"` (already printed by
  `_AudioDebug`) and events tagged `_log_bridge` (already on stderr via the logging
  library — see B) → no double-print. **Gated**: ON when `MSTT_EVENT_CONSOLE` is set
  (1/true/on), else default ON whenever the file sink is active (`MSTT_EVENT_LOG`
  set — quickstart sets it), else silent (library/test). Thread-safe + suppressed.
- **(B) Python logging → bus bridge** (`LogBusHandler` + `_BusBridgeFilter` +
  `install_log_bridge`, installed ONCE in `main()` after `basicConfig`). Each log
  record is republished as `bus.publish({"type":"log","level":…,"message":…,
  "_log_bridge":True})`, and `logging.captureWarnings(True)` routes `warnings.warn(...)`
  (onnxruntime CUDAExecutionProvider, Hugging Face unauthenticated-request) through it
  too. So the EVENT LOG now also captures library/app logs + warnings (`httpx`
  requests, our own `logging`). **Level policy** (via `_BusBridgeFilter`): bridge
  WARNING+ from ALL loggers AND INFO+ from `my_stt_tts.*`. **Recursion guard**: a
  thread-local re-entrancy flag drops any record produced while the handler is
  publishing; all failures suppressed. The `_log_bridge` tag is what makes sink (A)
  skip these (no re-print to stderr).

**Net result:** quickstart.log ⊇ every EVENT-LOG event (via A); EVENT LOG ⊇ library/app
logs + warnings (via B). **Accepted divergences (both intentional):**

  1. **httpx per-request INFO** (`HTTP Request: POST … 200 OK`) is kept on stderr (raw
     quickstart.log) but OUT of the EVENT LOG, to keep it readable — the single
     capped-INFO divergence (`_NOISY_INFO_LOGGERS` in the filter; the loggers' own level
     is NOT lowered, so the lines still reach stderr).
  2. **Pre-app `uv sync` subprocess output** in quickstart.log: emitted before the bus
     / app exist, so it cannot be in the EVENT LOG.

No quickstart.sh edit needed — sink (A) defaults on because quickstart already sets
`MSTT_EVENT_LOG`. New env vars documented in `.env.example` (`MSTT_EVENT_LOG`,
`MSTT_EVENT_CONSOLE`). Tests in `tests/test_events.py`: console sink prints
non-debug/non-bridge events + skips `debug` + `_log_bridge` (capsys); gating by
`MSTT_EVENT_CONSOLE` / `MSTT_EVENT_LOG`; the handler publishes a `log` event +
`captureWarnings` reroute; no double-print; no recursion; idempotent install; the
level policy + httpx divergence.

## Build status (2026-06-20) — Wave M3: music_playback setting + wake-word TEST diagnostic + EVENT LOG relabel

Three backend features on branch `music-server-wake` (worktree `.worktree-srv`),
implemented against a SHARED CONTRACT a parallel GUI agent consumes. 614 → 646
tests core (+32), green core-only (646 + 2 extras-gated skips) and `--extra all`
(650, the 2 real-model `say` tests now run against the actual maziko/nexus ONNX).
Lint clean (ruff check + format, mypy `src`, pylint). `music.py`/`wake.py` still
import CORE-only (no yt-dlp / openWakeWord). No README / webui.html / clients /
esp32 changes (backend only).

- **(1) `music_playback` setting (server vs hybrid).** New `Config.music_playback`
  (`"hybrid"` default; env `MUSIC_PLAYBACK`; validated to {server,hybrid} via new
  `MUSIC_PLAYBACK_MODES`). Audio ALWAYS plays server-side via mpv — the setting only
  tells the GUI whether to ALSO show the (muted) YouTube video when the control room
  is local. Surfaced in `webui.settings_dict` (+ a `music_playback_modes` choice
  list), accepted in `apply_settings` (unknown value ignored), shown in
  `settings_text` ("playback <mode>"), documented in `.env.example`. The server keeps
  emitting `video_id` on the `music` event (unchanged).
- **(2) Wake-word TEST action + scoring.** New `wake.score_wake_clip(clip, sample_rate,
  word, *, threshold, phases, wakewords_dir) -> (confidence, fired)`: loads the
  `WakeWord` for `word` via `wake_model_for(word)` (NOT necessarily the configured
  one), resamples the clip to 16 kHz, reframes to 1280-sample frames, and feeds them
  through the REAL phase-diverse `WakeWord.detect` path frame-by-frame (exactly what
  the live loop sees); returns `(max last_score, max>=threshold)`. Missing model /
  empty clip / unavailable backend → `(0.0, False)` (no crash). Wired into `on_action`
  via the `wake_test` action: `source="server"` records ~2 s via `audio.record_fixed`;
  `source="browser"` builds an np array from the posted `pcm` + `sample_rate`. Both
  score, save the 16 kHz clip to `~/.cache/my-stt-tts/wake-test-<word>-<source>.wav`
  (kept for debugging), and emit `bus.wake_test_result(...)` →
  `{"type":"wake_test_result","word","source","confidence":0..1,"fired":bool,
  "message","wav_path"}`. The error/idle log is emitted BEFORE the (DATA) result so a
  SYSTEM error frame doesn't flush it.
- **(3) EVENT LOG relabel — human button names.** `_AudioDebug.action(name, **fields)`
  now emits a friendly message using the EXACT GUI button label
  (`ptt`→"clicked PUSH-TO-TALK", `wake_start`→"clicked START WAKE", … `turn`→"submitted
  a turn"; unknown → "clicked <NAME>") via `_AudioDebug.action_label`, while still
  carrying the machine-stable `stage="action:<name>"` field. Also drops a huge browser
  `pcm` payload from the logged fields. Updated the test asserting the old
  `[audio:action:…]` message text.

## Build status (2026-06-20) — Wave M2: music backend fixes (always-respond, stop-routing, system state)

Three real bugs from the first music play (branch `music-state`, worktree
`.worktree-music2`). 582 → 614 tests core (+32), green core-only (614 + 2
extras-gated skips) and `--extra all` (616). Lint clean (ruff check + format,
mypy `src`). `music.py` still imports CORE-only without yt-dlp. No webui.html /
clients / esp32 changes (backend only).

- **(1) ALWAYS emit an assistant response on a music action.** A music turn used
  to emit only `bus.log` → NO assistant bubble in the transcript. New
  `_music_respond(cfg, tts, gate, display, spoken)` in `__main__.py` emits a brief
  `bus.state("llm_response", model)` + a final `bus.response(display, final=True,
  model=…)` (so the page draws an "ASSISTANT · <model>" bubble with the ▶/⏹/⏸
  glyph) AND speaks the **glyph-free** `spoken` text (symbols must not be read
  aloud). Play → "▶ Playing: <title>." / stop → "⏹ Stopped the music." / pause →
  "⏸ Paused." / resume → "▶ Resumed."
- **(2) Reliable stop/pause/resume — never falls through to the LLM.** Root cause:
  `_STOP_RE` did not allow trailing politeness, so **"stop the music please"** failed
  to match → `match_music_intent` returned `None` → the turn reached the LLM (which
  hallucinated "there's no music playing"). Fix: `_strip_politeness()` strips a
  trailing politeness/filler clause (EN "please"/"now", DE "bitte"/"jetzt", FR
  "s'il te/vous plaît"/"stp"/"svp"/"maintenant") BEFORE matching control phrases.
  Any matched control intent is ALWAYS handled locally; a stop/pause with nothing
  playing is answered BY THE ROUTER ("Nothing is playing right now."), never the LLM.
- **(3) System-state awareness in the LLM context.** New `music.music_state_line()`
  reads the process-wide `get_player()` singleton (side-effect-free; no player ⇒
  "System state: no music is playing.") and is injected into `brain._system_prompt()`
  on EVERY turn, right next to `current_time_line()`. So an LLM-routed question like
  "what's playing?" is answered from LIVE state ("System state: music is currently
  playing: \"<title>\".", or "…is paused…"), not just chat history.
- **(4) Rich music events + GUI control actions + video_id.** `search`/`play` now
  extract the 11-char YouTube `video_id` (from yt-dlp's `id` or the page URL) onto
  `Track`/`PlayResult`; a new `bus.music(status, title, video_id, url)` emitter
  publishes `{"type":"music","status":"playing|stopped|paused|resumed","title",
  "video_id","url"}` on every play/stop/pause/resume (so a later GUI wave can embed
  - control the video). New `_music_action("music_stop"|"music_pause"|"music_resume")`
  server actions are wired into the `on_action` handler (GUI buttons) and drive the
  SAME singleton + emit the SAME event. `MusicPlayer.status()` exposes a live
  snapshot (status/title/video_id/url) and `_paused` is tracked. Server-side mpv
  playback unchanged.
- **Tests (+32, `tests/test_music.py`):** trailing-politeness control matching
  EN/DE/FR (router-level + turn-hook level, incl. "stop the music please" and the
  nothing-playing case); always-respond (`bus.response` final + glyph-free spoken
  text + the `music` event with video_id/url for play AND stop); `music_state_line()`
  idle/playing/paused; `search` video_id from `id` and from a watch URL;
  `status()` snapshot; the `_music_action` GUI handlers emit stopped/paused/resumed.
  Player/yt-dlp/network all MOCKED. CORE-only verified.

## Build status (2026-06-20) — Wave M: play music from YouTube (local intent router)

"Play We Will Rock You by Queen" now actually plays the song instead of "I can't
play music" (branch `music-youtube`, worktree `.worktree-music`). 518 → 584 tests
(+66), green with `--extra all` and core-only (582 + 2 extras-gated skips). Lint
clean (ruff check + format, mypy `src`). The mechanism is a **local intent router
in the turn path** (not only a tool) because the default `claude-cli` brain does
NOT do our tool-calling — so a turn is intercepted BEFORE the LLM and handled
locally, working for every brain.

- **(1) `music.py` — the player.** `search(query)` resolves the best match via
  **yt-dlp** `ytsearch1:` (extract_info, no download → webpage_url + title).
  `MusicPlayer.play(query)` plays through a **stoppable background process**:
  prefers `mpv --no-video` (streams the YouTube page URL, pause/resume/stop via
  JSON IPC socket), else `ffplay -nodisp` (streams the bestaudio URL; stop only),
  else a yt-dlp **download → temp file → project playback** fallback (daemon
  thread). `stop()` / `pause()` / `resume()` / `is_playing()` / `now_playing()`.
  yt-dlp is imported **lazily/guarded** so CORE imports `music.py` without the
  extra; missing yt-dlp/player → a clear spoken reason, never a crash. A
  process-wide `get_player()` singleton lets a later "stop" act on an earlier
  "play".
- **(2) Intent router in the turn path.** `match_music_intent(text)` recognises,
  case-insensitively, EN/DE/FR: play ("play/put on <X>", "play <X> from youtube",
  "play some music"; DE "spiel(e) <X>", "mach Musik an"; FR "joue/mets <X>", "mets
  de la musique") and stop/pause/resume (EN/DE "stopp/halt/pausiere/weiter"/FR
  "arrête/pause/reprends"). Wired into `_respond` in `__main__.py` via
  `maybe_handle_music(cfg, tts, gate, text)`: when `cfg.music_enabled` and the text
  matches, it searches+plays / stops, **speaks a confirmation via the normal TTS
  path** ("Playing …" / "Stopped the music."), emits `bus.state` / `bus.log` (GUI
  shows "▶ Playing: <title>"), and SKIPS the LLM. Covers terminal PTT, the wake
  loop, and the GUI server-side PTT/typed paths (all converge on `_respond`).
- **(3) Tool for API brains + system prompt.** `make_music_tools()` registers
  `play_music`/`stop_music` so anthropic/openai providers can also call them (the
  router stays primary); wired through `default_tools()` ← `brain.py` from config.
  System prompt (`prompts/system_prompt.md` + the `_DEFAULT_SYSTEM_PROMPT`
  fallback) now states the assistant CAN play YouTube music, so it stops saying "I
  can't play music".
- **(4) Config + deps.** `music_enabled=True`, `music_player="auto"`,
  `music_volume: int|None` with `MUSIC_ENABLED`/`MUSIC_PLAYER`/`MUSIC_VOLUME` env,
  validated (`MUSIC_PLAYERS` = auto|mpv|ffplay|download; volume ∈ [0,100]), shown
  in `--settings`, documented in `.env.example`. `yt-dlp` added to a new `music`
  extra in `pyproject.toml` and folded into `all`; mpv/ffmpeg documented as system
  tools (`brew install mpv`).
- **Tests (+66, `tests/test_music.py`):** intent matching across EN/DE/FR
  play/stop/pause/resume + non-music negatives; `search`/`play` with yt-dlp + the
  player subprocess MOCKED (no network/audio); graceful missing-deps (yt-dlp/player
  absent → spoken reason, no crash); mpv IPC pause/resume; ffplay fallback; the turn
  router skips the LLM on a music intent and calls play; the tool path + config
  validation/env loading. CORE-only verified (`music.py` imports without yt-dlp).

## Build status (2026-06-20) — Wave L: config-panel reorg + wake sensitivity as a VOICE setting

Layout/grouping pass on the GUI CONFIGURATION panel plus one new first-class
setting (branch `settings-reorg`, worktree `.worktree-reorg`). 502 → 509 tests
(+7), green with `--extra all` and core-only (507 + 2 extras-gated skips). Lint
clean (ruff check + format, mypy); `node --check` passes on the inline `webui.html`
script. CSP + demo mode intact; all existing field ids / `data-key` wiring / POST
contract preserved.

- **(1) Wake sensitivity is now a real VOICE setting.** `config.wake_threshold`
  default lowered **0.5 → 0.4** (env `WAKE_THRESHOLD`, validated to [0, 1] in
  `validate()`). `WakeWord.from_config` already reads `cfg.wake_threshold`, so the
  configured value drives detection (the old debug `threshold=0.5` was just the old
  default, not a hardcode). Exposed in `webui.settings_dict`; accepted in
  `apply_settings` as a float **clamped to [0, 1]**; added to `--settings`
  (`settings_text`) on the wake line. `.env.example` `WAKE_THRESHOLD` default +
  hint updated.
- **(2) CONFIGURATION panel split into 4 labeled sections** (each its own header +
  hairline divider, mission-control aesthetic):
  - **🧠 MODEL** — Brain preset (primary) + the existing "Advanced — manual
    override" foldable (Provider / Model / Model (deep)).
  - **🔊 VOICE** — Voice selector (+ ▶ TEST), Length scale, Wake phrase dropdown,
    and the NEW **Wake sensitivity** slider (`data-key="wake_threshold"`, 0–1,
    default 0.4, with a lower=more-sensitive / higher=stricter hint).
  - **🤖 AGENT** — Agent model (MOVED here from beside Wake phrase — it's the
    "agent, …" tool-dispatch model, not the conversation brain) + Agent workspace.
  - **⚙️ ADVANCED / GENERAL** — foldable `<details>` holding System prompt (the
    remaining rendered field).
  - JS: generalized the `[data-key]` range handler to update the value readout named
    by a new `data-val` attribute (so the wake slider writes `#wakeThresholdVal`, not
    `#lengthVal`); `populate()` seeds the slider; `DEMO_SETTINGS` carries
    `wake_threshold:0.4`.
- **Tests (+7):** `wake_threshold` default 0.4 / env override / [0,1] validation
  (`test_config.py`); `settings_dict` carries it, `apply_settings` sets + clamps it,
  `WakeWord.from_config` uses the configured value (`test_wakeword_select.py`).

## Build status (2026-06-20) — Wave K: PTT under wake, replay rate fix, host-app label

Three backend fixes (branch `backend-fixes2`, worktree `.worktree-fixes2`).
490 → 502 tests (+12), green with `--extra all` and core-only (500 + 2
extras-gated skips). Lint clean (ruff check + format, mypy).

- **(1) Push-to-talk works WHILE the wake loop is listening.** `--browser --wake`
  auto-starts the wake loop (holding the mic), so the GUI `ptt` action was REFUSED
  with "push-to-talk unavailable while the wake loop is listening." Fix: removed the
  refusal; `_WakeController.push_to_talk` now runs its capture+respond inside
  `_with_paused_wake(...)` — the SAME pause→run→restore pattern the record-replay
  diagnostic uses. The running wake loop is paused (stop event + thread join), PTT
  owns the mic for the capture+respond, then the loop is restarted so the user lands
  back listening. Re-entrancy is still guarded (`_ptt_busy`): a second PTT while one
  is in flight is refused. PTT also still works with wake off (nothing to pause).
  `_ptt_target` now delegates to a new `_capture_and_respond` helper.
- **(2) Record-and-replay no longer plays back sped-up / high-pitched.** Classic
  record-rate ≠ playback-rate bug: `audio.record_fixed` recorded at the device rate
  (commonly 48 kHz) but RESAMPLED the clip to 16 kHz, and `_run_mic_record_replay`
  then played it via `_play` (the 24 kHz chime rate). A 16 kHz clip played at 24/48
  kHz plays 1.5×/3× too fast. Fix: `record_fixed` now returns the RAW clip at
  `device_rate` (no resample — the 16 kHz resample is only for STT/wake), and the
  replay plays it with `audio.play(clip, device_rate)` at that SAME rate, so a 3 s
  recording replays as 3 s with faithful pitch. Duration/stats are computed at
  `device_rate` too (the old code reported 3× too long).
- **(3) Server-mic host-app detection.** New `platform.host_app_name(env=None) ->
  str` maps `TERM_PROGRAM` to the friendly app whose macOS mic permission governs
  the SERVER capture (`iTerm.app`→"iTerm", `Apple_Terminal`→"Terminal",
  `vscode`→"VS Code", `ghostty`→"Ghostty", `WezTerm`, `Hyper`, `Tabby`,
  `Alacritty`, `Warp`, …; unknown values title-cased, unset → "your terminal app";
  case-insensitive). Exposed as `host_app` in `webui.settings_dict` so the GUI can
  label the server mic "uses <App>'s microphone permission".
- **Tests (+12):** PTT pause→capture→resume around an active wake loop, PTT with
  wake off, PTT re-entrancy refusal (`test_gui_voice.py`); `record_fixed` raw-at-
  device-rate + same-rate replay round-trip + duration-at-device-rate
  (`test_backend_fixes.py`); `host_app_name` mapping/case/unknown/unset + `host_app`
  in `settings_dict` (`test_platform.py`, `test_gui_voice.py`). Updated the two
  pre-existing replay tests to assert `audio.play` rate == record rate. Audio /
  device / wake loop all mocked.

## Build status (2026-06-19) — Wave J: startup audio preflight HARD STOP

The user hit a control room that opened and *silently recorded nothing*: capture
ran at the device-native 48 kHz and either couldn't be made 16 kHz or the inbound
mic queue flooded (`inbound mic queue full; dropping a frame` / PortAudio
`Input overflowed`), so wake/STT got garbage with no error. Prior waves added
`resample_to` / `reframe` / a 16 kHz pipeline; this wave adds a **hard stop that
refuses to launch** when capture is broken. Branch `audio-preflight`. 457 → 476
tests (+19), green with `--extra all` and CI core-only (474 + 2 extras-gated skips).

- **`audio.audio_preflight(sample_rate=16000) -> PreflightResult`** (`audio.py`,
  pure-ish + faked in tests): opens a short (~0.5 s) REAL capture and checks (a) a
  usable input device exists; (b) the device delivers a rate resolvable to 16 kHz
  mono (records `device_rate`); (c) frames are consumed without **persistent**
  overflow — counts PortAudio `input_overflow` + bounded-queue-full drops over the
  window and computes a `drop_ratio` (fails at/above 25 %; a single warm-up glitch
  is tolerated). Returns `ok` / `reason` (`ok`/`no_device`/`rate_unresolvable`/
  `overflow`/`permission_denied`/`error`) / actionable `message` + `device_rate` /
  `drop_ratio` / `permission`. **Never raises**; reuses `mic_permission_status()`
  (a conclusively `denied` permission wins immediately).
- **HARD STOP wiring** (`__main__.main`): `_audio_preflight_gate` runs BEFORE the
  heavy STT load / GUI / capture for the mic-using modes — `--wake`, `--browser
  --wake`, `--browser --browser-audio`, and the default terminal push-to-talk — and
  on `not ok` prints the `message` to stderr + `bus.log(..., "error")` and returns
  exit **3** WITHOUT opening the GUI or starting capture. Mic-LESS modes skip it
  (`--type`, `--text`, plain `--browser`, and the network/telephony servers whose
  mic lives on a remote client). A passing-but-marginal preflight still logs its
  device-rate / drop-ratio / reason via the audio debug instrument.
- **Escape hatch**: `--skip-audio-preflight` (+ `SKIP_AUDIO_PREFLIGHT` env, +
  `Config.skip_audio_preflight`) bypasses the gate for power users; the hard-stop
  message names it.
- **Tests** (+19, `test_audio_preflight.py`): preflight OK on a fake 16 kHz device,
  a 48 kHz device that resamples, a tolerated warm-up overflow; hard-stop on a fake
  overflowing device (high drop ratio), unresolvable rate (no positive rate / no
  frames), no-device, denied permission, and a raising/missing-sounddevice error
  (never propagates). `main()` returns non-zero AND does NOT open the GUI
  (`_ExplodingUI`) / start the wake loop / push-to-talk on a failing preflight;
  mic-less modes never call it; the skip flag short-circuits. `sounddevice` is
  mocked entirely — no real mic.
- **Caveats:** verified only with a mocked `sounddevice` — no real microphone, no
  real 48 kHz device, and no real PortAudio overflow were exercised here. The
  bounded-queue drop count is a secondary backpressure signal; the authoritative
  overflow signal is PortAudio's `input_overflow` status flag.

## Build status (2026-06-19) — Wave I: wake-loop crash fix + GUI mic test

Two real bugs the user hit: (1) the wake word never fired — the loop crashed on
construction with `AudioFeatures.__init__() got an unexpected keyword argument
'wakeword_models'`; (2) the user couldn't be heard, with no way to tell why
(server mic vs. permission). Branch `fix-wake-mic`. 385 → 407 tests (+22), green
both with `--extra all` and CI core-only (no extras).

- **BUG 1 — version-tolerant openWakeWord construction** (`wake.py`): the installed
  `openwakeword==0.4.0` `Model.__init__` takes `wakeword_model_paths=[...]` and has
  **no** `inference_framework` arg; the modern `wakeword_models=`/`inference_framework=`
  kwargs leak through `**kwargs` into `AudioFeatures` and raise `TypeError`.
  `WakeWord._build_model` now **tries the modern API first, falls back to the 0.4.0
  signature on `TypeError`** (0.4.0 infers ONNX from the `.onnx` extension). Verified
  for real against the actual model: modern API raises the exact reported error;
  0.4.0 API loads; `predict()` on a silent frame returns `{'maziko': 0.0}` — the score
  key is the **model-file stem**, and `detect()` reads `.values()`, so the key naming
  is irrelevant (confirmed correct, no change needed there).
- **BUG 1 — fail-once, don't spin** (`wake.py` + `__main__.py`): a new
  `WakeUnavailable` error is raised **once** on unrecoverable construction/predict
  failure (sticky `_broken` flag — a second `detect()` re-raises without retrying).
  `run_wake_loop` catches it, logs a single clear hint, sets state idle, and returns
  `2` — instead of re-raising the same error on every audio frame forever.
- **BUG 2 — server mic test** (`audio.py`): `mic_test(sample_rate)` captures ~1.5 s
  from the input device and returns a `MicTestResult`; the pure `mic_test_verdict(...)`
  maps it to **working** ("✓ Microphone OK — level NN%"), **silent** ("✗ No audio —
  grant microphone permission … System Settings › Privacy & Security › Microphone …"),
  **no_device**, or **error** (the exact sounddevice/PortAudio reason). Never raises.
- **BUG 2 — action + event plumbing** (`__main__.py` + `events.py`): new `mic_test`
  action on `/api/action`; `bus.mic_result(...)` event. Runs **regardless of wake
  state** — with a controller it stops/joins the wake loop, captures, then restarts
  it (the test owns the mic); without one (voice off — when you most need to diagnose)
  it runs a standalone capture in a worker thread, never blocking the HTTP handler.
- **BUG 2 — web GUI** (`webui.html`): a **"🎤 Test mic"** button in CONTROLS, a
  prominent result chip that turns **green/red** with the verdict + a level meter; the
  **MIC COLD/HOT** indicator now goes HOT during the real capture and back to COLD on
  the verdict. A one-line macOS permission hint sits under the voice controls. BONUS:
  a **browser-mic** getUserMedia level meter ("Test browser mic"), clearly labelled
  distinct from the server "Test mic", with its own permission handling.
- **Tests** (+22): `test_wake_model.py` — both Model API branches via a **fake
  openwakeword module**, `detect()` reading score values, and `WakeUnavailable` raised
  once + sticky. `test_mic_test.py` — the verdict mapping (loud/silent/no-device/error,
  level clamp) and `mic_test` capture with a **faked sounddevice** (loud → ok, zero →
  silent, raising stream → error, missing sounddevice → error, never raises).
  `test_gui_voice.py` — `run_wake_loop` stops once on `WakeUnavailable`; `_run_mic_test`
  - controller `mic_test` publish the verdict and pause/restore the wake loop.
  `test_run_browser.py` — the `mic_test` action fires even with no voice controller.
- **Caveats:** the verification model `wakewords/maziko.onnx` was a shipped
  openwakeword model copied locally (uncommitted) — there is no committed maziko
  model. Could not test a **real** microphone or a real macOS permission denial here;
  the silent/permission path is asserted via mocked capture. JS `node --check`ed, not
  exercised in a live browser.

## Build status (2026-06-19) — Wave H: pre-shipped wake-word selector

Goal: ship several trained wake-word models in `wakewords/` (e.g. `maziko.onnx`,
`nexus.onnx`, `jarvis.onnx`, `computer.onnx`) and let a user **pick one by name**
(UI dropdown / CLI / env) without editing paths. Built generically — it discovers
whatever `.onnx` models are present, never depending on specific files. No
regression to the 371-test baseline (now 385).

- **Discovery + name→path convention** (`config.py`): `available_wake_words(dir)`
  lists the stems of the `*.onnx` models on disk (sorted; `[]` when the dir is
  missing/empty). `wake_model_for(phrase, dir)` is the one path convention
  (`<dir>/<phrase>.onnx`), and `WAKEWORDS_DIR` names the default folder.
- **Selecting = setting the phrase** (`config.py`): `Config.from_env` now reads
  `WAKE_PHRASE` and **auto-derives** `wake_model_path` as `wakewords/<phrase>.onnx`
  when `WAKE_MODEL_PATH` isn't set; an explicit `WAKE_MODEL_PATH` still wins.
  `Config.select_wake_word(name)` sets the phrase + re-derives the path in one call.
- **CLI** (`__main__.py`): new `--wake-word NAME` (alias that selects by name) and
  `--wake-model-path PATH` (explicit override, wins over `--wake-word`). `--settings`
  now shows the selected wake word, its model path, whether the file **exists**, and
  the `[available]` wake words discovered on disk.
- **Web UI** (`webui.py` + `webui.html`): `settings_dict` adds `wake_words: [...]`;
  `apply_settings` accepts `wake_phrase` and re-derives the path via
  `select_wake_word`. The **Wake phrase** control renders as a **dropdown** of
  `wake_words` (with a *custom…* entry that reveals a free-text box), and falls back
  to free text when the list is empty. Selecting one POSTs `{wake_phrase: NAME}`.
- **Docs** (`wakewords/WAKEWORD.md`): new "Pre-shipped wake words" section — the repo
  ships several, how to select (UI / `--wake-word` / `WAKE_PHRASE`), and that custom
  ones can still be trained. Notes "alexa"/"jarvis" are third-party trademarks
  (community models, personal use).
- **Tests** (`tests/test_wakeword_select.py`, +14): `available_wake_words` over a
  **temp dir** of fake `.onnx` (sorted, ignores non-onnx, empty/missing dir), the
  `wake_phrase → path` derivation (default + explicit-override precedence),
  `select_wake_word`, the `--wake-word` flag + `--wake-model-path` override,
  `settings_text` surfacing, the `settings_dict` `wake_words` list, and
  `apply_settings` re-derivation. Never touches the real `wakewords/`.
- **Caveats:** the `.onnx` model files are committed separately by the orchestrator
  (gitignored binaries); the JS was `node --check`ed; not exercised in a real browser.

## Build status (2026-06-19) — Wave G++: GUI voice controls actually work

Goal: fix a misleading-UI bug in the browser control room. The **Start Wake**,
**Push-to-Talk**, and **Live Audio** buttons existed even in typed-only launches
and, for wake/PTT, did nothing — `on_action` for `wake_start`/`wake_stop`/`ptt`
just logged *"runs from the terminal in this build"*. Now those buttons drive the
**server-side** pipeline when it is available, and are **honestly disabled** when
it is not. No regression to the 356-test baseline (now 371).

- **`run_wake_loop` is now cleanly stoppable** (`__main__.py`): accepts an optional
  `stop: threading.Event`. It is checked at the top of both the outer wake-listen
  loop and the inner record loop, and passed into `audio.listen_for_wake(...)` so a
  GUI-driven loop tears down promptly even while idle waiting for the phrase.
  `audio.listen_for_wake` grew a `stop` param and now returns `bool` (True = fired,
  False = stopped). `None` keeps the classic run-forever terminal behaviour.
- **GUI-controlled voice via `_WakeController`** (`__main__.py`): one lock-guarded
  controller owns the three GUI voice actions. `wake_start` → starts the wake loop
  in a daemon thread (idempotent; double-start is a no-op log); `wake_stop` → sets
  the stop event and returns to idle; `ptt` → runs one `_capture_ptt` → `_respond`
  in a worker thread (refused while the wake loop is listening, and while a prior
  PTT is still capturing). A blank capture logs a macOS mic-permission hint instead
  of failing silently. `_run_browser`'s `on_action` now dispatches to it; with no
  pipeline it logs an honest `unavailable: <reason>` error.
- **Launch flag that enables it:** `./mstt --browser --wake` (already loads STT via
  `needs_stt`) now also makes the GUI buttons live, and `--wake` on launch starts
  the loop immediately (still stoppable from the GUI). Plain `./mstt --browser
  --type` stays typed-only (voice off). `--browser-audio` still gates Live Audio.
- **Honest UI when voice is unavailable** (`webui.py` + `webui.html`): `settings_dict`
  / the `/api/settings` payload gained `voice_available: bool` + `voice_hint: str`,
  set true only when STT is loaded **and** the wake model exists **and** a mic is
  usable (`audio.mic_available()` — a new defensive sounddevice probe). When false
  the page **disables** Start Wake / Push-to-Talk (greyed, `aria-disabled`, guarded
  click/keydown so they can't POST) and shows a one-line note
  (*"Voice off — relaunch with `./mstt --browser --wake` and grant mic access"*).
  Live Audio still keys off `audio_enabled` (R2-5).
- **`quickstart.sh`** keeps typed mode as the instant, mic-free default but now
  prints a clear hint after sync — `🎙️  To talk to it (wake word / mic), run:
  ./mstt --browser --wake` — noting macOS asks for mic permission on first capture.
  `shellcheck`-clean.
- **Tests** (`tests/test_gui_voice.py`, +15): `_WakeController` wake_start/stop
  toggling the loop (mock `run_wake_loop`/thread, assert the stop event fires),
  double-start guard, PTT one-shot + blank-capture mic hint + PTT-blocked-while-wake,
  `run_wake_loop` stop-event exit, `voice_available` true/false in `settings_dict`,
  `_voice_status` capability resolution (no STT / no wake model / no mic / all
  present), and `_run_browser`'s `on_action` dispatching to the controller (voice on)
  vs logging honestly (voice off). All mic/model/network boundaries mocked.
- **Caveats:** not exercised in a real browser or against a real mic in this env —
  the JS was `node --check`ed, the settings payload smoke-tested over loopback, and
  the Python paths unit-tested with the audio/wake/STT layers mocked. macOS will
  prompt for Terminal mic permission on the first real server-side capture.

## Build status (2026-06-19) — Wave G+: seamless key-free quickstart

Goal: make `./quickstart.sh` **genuinely seamless**. Before this, it launched the
default `anthropic` provider, which needs `ANTHROPIC_API_KEY`, so a fresh clone died
with `Invalid configuration: ANTHROPIC_API_KEY is required for provider 'anthropic'`.
Fixed: quickstart now **auto-detects a key-free brain**; a new `codex-cli` provider
was added; the browser URL is shown + opened; and the no-key error self-guides.
No regression to the 348-test baseline (now 356).

- **`quickstart.sh` auto-detects a key-free brain** after `uv sync --extra all`, in
  order: (1) `claude` CLI on PATH → `--brain haiku-sub` (Claude CLI, no key);
  (2) else `ollama` on PATH **and** at least one model pulled (`ollama list`) →
  `LLM_PROVIDER=ollama` + the first installed model + `LLM_BASE_URL=
  http://localhost:11434/v1`; (3) else `codex` CLI on PATH → `--brain codex`
  (OpenAI codex CLI, no key); (4) else a friendly message (install claude / ollama
  - pull / codex, or set `ANTHROPIC_API_KEY`) and exit 1 — no stack trace. Prints
  which brain it chose. `-h`/`--help` and the 100755 git mode preserved;
  `shellcheck`-clean.
- **New `codex-cli` brain provider** (`brain.py` `_stream_codex_cli` + `config.py`):
  mirrors `claude-cli` but shells out to the OpenAI `codex` CLI in non-interactive
  `codex exec` mode (uses your logged-in codex auth, no API key). Isolated:
  `--sandbox read-only`, `--skip-git-repo-check` in a scratch cwd, and
  `--ignore-user-config` (skips `$CODEX_HOME/config.toml`). `codex exec` prints only
  the final assistant message to stdout, which we capture as the reply; stateless
  per call (no resume session id). Base command overridable via `CODEX_CLI_CMD`
  (default `codex exec`). Added `codex-cli` to `PROVIDERS`, a `codex` brain preset
  (`gpt-5-codex`), and a validate() PATH check.
  **ASSUMPTION:** the exact `codex exec` flags (`--model`, `--sandbox read-only`,
  `--skip-git-repo-check`, `--ignore-user-config`) are taken from the documented
  OpenAI Codex CLI reference (developers.openai.com/codex/cli/reference) and were
  **NOT verified against a live binary** (codex is not installed in this env).
  Override with `CODEX_CLI_CMD` if a build differs; noted in code + `.env.example`.
- **`__main__.py` `_run_browser`** now announces the URL via a small
  `_announce_browser_url(url)` helper that **prints it prominently**
  (`▶ Open in your browser:  http://127.0.0.1:8765/`) **and auto-opens** it with
  `webbrowser.open(url)`. (Extracted to a helper so `_run_browser` gains no extra
  local and pylint stays at the prior warning count.)
- **Friendlier no-key error** (`config.py`): the `ANTHROPIC_API_KEY is required`
  message now points at the easy fixes — run `./quickstart.sh`, or `--brain
  haiku-sub`, or `LLM_PROVIDER=ollama`, or `--brain codex` — so even a bare `./mstt`
  failure self-guides.
- **Tests** (`tests/test_logic.py`, `tests/test_config.py`, new
  `tests/test_run_browser.py`): codex-cli runs `codex exec` with the isolation flags
  - returns stdout, error propagation, `CODEX_CLI_CMD` override; `codex-cli` in
  `PROVIDERS`, the codex PATH-gate, the `codex` preset, and the self-guiding no-key
  message; and `_run_browser` prints + `webbrowser.open`s the URL. All subprocess /
  browser boundaries are mocked — nothing runs live.

## Build status (2026-06-19) — Wave G: locale awareness + one-command quickstart

Goal: make the assistant generally **location- and units-aware**, ship a **real
weather tool**, and get a brand-new user **talking to the LLM in one command**. All
wired + tested; no regression to the prior 326-test baseline (now 345).

- **Location + units settings** (`config.py`): new `location` (default
  `"Lausanne, Switzerland"`) and `units` (`metric` | `imperial`, default `metric`)
  fields with env overrides (`LOCATION`, `UNITS`), fail-fast validation (units in
  the set; location non-empty), CLI flags (`--location` / `--units`), a `locale`
  row in `--settings`, and the **web-UI settings** (`settings_dict` + `apply_settings`,
  plus a `units_modes` choice list).
- **System-prompt locale injection** (`config.locale_prompt_line` +
  `Brain._system_prompt`): the editable base prompt (`prompts/system_prompt.md`,
  `cfg.system_prompt`) is kept verbatim and a single line —
  "The user is in {location} and uses {units} units; answer measurements…
  accordingly." — is appended at the backend boundary for **every** provider path
  (claude-cli, anthropic, openai, both tool-call loops). The base prompt is never
  mutated, so the UI/`--settings` still show the editable text.
- **Real `get_weather` tool** (`tools.py`): uses **Open-Meteo (NO API KEY)** —
  geocodes the place via the geocoding endpoint, fetches current conditions from the
  forecast endpoint, and returns a concise units-aware summary (°C/km·h for metric,
  °F/mph for imperial; WMO code → phrase). Defaults to `cfg.location`/`cfg.units` but
  accepts an explicit `location` arg. Registered in `default_tools` alongside
  get_time/calculator/home_control (built with the config's location+units in
  `Brain.__init__`). Dependency-light `urllib`; network/parse failures return a clear
  "weather unavailable" message and never crash the turn.
- **One-command quickstart** (`quickstart.sh`): checks for `uv` (prints an install
  hint if missing), runs `uv sync --extra all`, then launches `./mstt --browser
  --type` — the web control room in typed mode (no mic), so a new user is typing to
  a real LLM in seconds. `-h`/`--help` print purpose/usage and exit 0; executable
  bit set in the git index. (Superseded by the key-free auto-detect below — the
  default provider is actually `anthropic`, so a bare launch needed a key.)
- **Tests** (`tests/test_weather.py` new, `tests/test_config.py` extended): weather
  metric vs imperial formatting + the unit params sent, default-vs-explicit location,
  graceful network/unknown-place/empty failures, WMO mapping, the full `urllib`
  wiring (only `urlopen` faked) hitting the documented endpoints, plus location/units
  config defaults + env override + validation and the system-prompt injection line.
  **Open-Meteo is never hit live** — every HTTP boundary is mocked.
- **`.env.example`**: documents `LOCATION` + `UNITS`.

## Build status (2026-06-19) — Wave E: multi-user / household maturity (G1/G2/G4/G7/G8)

Reframed goal: be the best choice for a **household** (different people) talking to a
Mac — multi-user, on-device, natural, interruptible — and close the code-achievable
maturity gaps a judge cited vs `pipecat`. All wired + tested; no regression to the
prior 265-test baseline (now 307+).

- **G1 — Pluggable backend registry + real cloud adapters** (`registry.py`,
  `stt_cloud.py`, `tts_cloud.py`): a `ServiceRegistry` (name→builder, namespaced by
  `stt`/`tts`/`llm`) formalises the existing `Transcriber`/TTS/`Brain` seams.
  **Real, key-gated** adapters speak the actual provider APIs with graceful
  local-first fallback: **Deepgram** streaming STT, **ElevenLabs** + **Cartesia** TTS.
  Each lazy-imports its SDK (optional `deepgram`/`elevenlabs`/`cartesia` extras) with
  a dependency-light `urllib` HTTP fallback. Selected via `stt_backend`/`tts_backend`
  (env, `--stt-backend`/`--tts-backend`, `--settings`); validation cross-checks names
  against the registry. Tested against **mocked** SDK/HTTP responses (no live keys).
- **G8 — Cross-platform brain (off-Mac)** (`stt.py` whisper.cpp + faster-whisper,
  `platform.py`, `aec.py` WebRTC-APM): the central brain can run on a **Linux** box
  with Mac/ESP32 satellites. Non-MLX STT backends (`whispercpp` via `pywhispercpp`,
  `faster-whisper` via CTranslate2); a `platform` module (OS detect + override)
  selecting a Linux-native WAV player (`aplay`/`paplay`) and a cross-platform
  `play_array` (sounddevice → CLI fallback); a Linux **WebRTC Audio Processing Module**
  AEC backend (`aec_mode=webrtc`/`auto`) behind the `EchoCanceller` seam with NLMS
  fallback. **macOS path unchanged** when auto-detected. `platform`/`playback_backend`/
  `whispercpp_model`/`faster_whisper_compute` config + flags. Selection/fallback faked.
- **G2 — Typed, prioritized, non-droppable events** (`events.py`): a typed `Frame`
  model with two priority classes. **SYSTEM** frames (interruption / error /
  end-of-turn) **never drop**, **bypass** queued data, and **flush** stale data ahead
  of them; **DATA** frames ride a bounded queue and may drop under back-pressure.
  Per-subscriber delivery drains the system lane first, so ordering is **consistent
  across every transport** (local, ws, webrtc, telephony). The ad-hoc interruption
  emitters are now backed by this; the public API + wire types are unchanged
  (back-compat). Tested: classification, bypass, flush, non-drop under load, ordering.
- **G7 — Per-speaker persistent memory + provider-agnostic context** (`memory.py`):
  a `ContextAggregator` assembles a neutral `[{role,content}]` list (per-speaker
  recall + live session, deduped at the seam, budget-bounded) **independent of the
  LLM provider**; a persistent `MemoryStore` (**SQLite** default, **JSON** for `.json`
  paths) keyed **per enrolled speaker** (ties into `speaker_id` via `SpeakerIdentifier`)
  for cross-session recall, with `unknown`/`ambiguous` bucketed to a shared guest so
  one person's history never leaks into another's; plus a `DialogueFlow` structured-
  dialogue hook. Wired into `Brain` (`set_speaker`, assembled context for every
  backend, persist on completion, `commit_spoken` amends persistent memory after a
  barge-in). `memory_store`/`memory_max_turns` config + `--memory-store`. Tested:
  persistence (both backends), per-speaker isolation, context assembly, the flow,
  Brain integration. Fixed a real aliasing bug (`reset_live` rebound instead of cleared).
  - **G7 wiring fix (2026-06-19)** — closed a real correctness gap: `set_speaker` /
    `EcapaEmbedder` / `SpeakerIdentifier.identify` were unit-tested but **never
    invoked in the live loop**, so the speaker was always `None` and per-speaker
    memory never keyed to a real person. New `speaker_pipeline.py` builds the
    embedder + identifier + enrolled centroids **once** (gated: only when
    `SPEAKER_ID=true` AND voices are enrolled under `enroll_dir` AND `speechbrain`
    is importable; fully `try/except`-wrapped so no enrollment / no model degrades
    to guest at zero latency). Wired into **every** spoken path — local PTT
    (`run_turn`, clip now returned from `_capture_ptt`), wake loop (`run_wake_loop`),
    the barge-in re-capture (the interrupter is re-identified), and the transport
    path (`respond_over_transport` / `capture_turn_clip` now keep the clip) for
    satellites + browser audio. Sequential embed right before `brain.stream` (note:
    not yet parallel with STT). Identified speaker published on the bus
    (`bus.speaker`) for the UI. New config `speaker_id_enabled` (`SPEAKER_ID`),
    `ENROLL_DIR`/`SPEAKER_THRESHOLD`/`SPEAKER_MARGIN` env overrides, `--settings`
    row. Tested in `test_speaker_wiring.py` (live `run_turn`/transport paths call
    `identify`→`set_speaker`; graceful-skip when no pipeline; gating + centroid
    loading).
- **G4 — Smart-Turn latency bench + language matrix** (`scripts/bench_smart_turn.py`):
  measures Smart Turn v3 on-device inference latency (warm + p50/p95) and **asserts**
  the p95 fits inside the silence window (`vad_silence_seconds`) with headroom — so
  smart endpointing can't clip the user. Pure assertion logic (`fits_silence_window`,
  `summarize`, `run_bench` with an injectable clock) is unit-tested with a fake clock +
  fake session; the real path reports `skipped` without the model. Language matrix below.

### Supported-language matrix + per-language fallback (G4)

The pipeline is built for **DE / FR / EN** (a Swiss household). Each stage degrades
gracefully when a language isn't natively covered — never a hard failure.

| Stage                                            | DE                                  | FR              | EN              | Other                       | Per-language fallback                                                                                                            |
| :----------------------------------------------- | :---------------------------------- | :-------------- | :-------------- | :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **STT** `parakeet-tdt-0.6b-v3` (local, default)  | ✅ native                           | ✅ native       | ✅ native       | ⚠️ ~25 langs                | parakeet v3 is multilingual + auto language-ID; off-Mac `whispercpp`/`faster-whisper` cover ~99 Whisper langs; cloud `deepgram` |
| **Smart-Turn v3** (prosodic end-of-turn)         | ✅                                  | ✅              | ✅              | ✅ language-agnostic        | falls back to the fixed silence timer if the ONNX/runtime is missing (logged, R3-8) — works for ANY language                    |
| **TTS — Piper** (local, default)                 | ✅ `de_DE-thorsten-high`            | ✅ `fr_FR-tom`  | ✅ `en_US-lessac` | ⚠️ 30+ Piper voices       | unmapped language → macOS `say` premium voice for that language → `say` default voice                                           |
| **TTS — `say`** (always-available fallback)      | ✅ Anna                             | ✅ Thomas       | ✅ Ava          | ✅ system voices            | used whenever Piper lacks/fails a voice; never blocks the turn                                                                   |
| **TTS — cloud** (opt-in, key-gated)              | ✅ ElevenLabs `multilingual_v2`     | ✅              | ✅              | ✅ 29 langs                 | cloud is the **best DE voice** (local DE is the weak spot); a missing key → local Piper/`say`                                    |
| **LLM**                                          | multilingual (Claude/GPT/Ollama)    | ✅              | ✅              | ✅                          | system prompt: "reply in the language the user spoke"; unrecognized → `default_language` (`en`)                                 |

Detection: STT returns a detected language; TTS uses `lingua` on the answer text
(`detect_language`, the `lang` extra) and falls back to `default_language` when
`lingua` is absent or unsure. So a language outside DE/FR/EN still produces audio
(via `say` / cloud) rather than erroring.

## Build status (2026-06-19) — round-3 breadth/ops gaps (R3-5/7/8/9)

Closed the final four round-3 breadth/ops gaps a fair judge still ranked `pipecat`
above us on (speech-to-speech, observability, first-run reliability, telephony).
All wired + tested; no regression to the 170 baseline.

- **R3-5 — Speech-to-speech / realtime LLM** (`realtime.py`): a `RealtimeBrain` +
  `RealtimeClient` speaking the **real OpenAI Realtime WS protocol** (`session.update`,
  `input_audio_buffer.append`/`commit`, `response.create`, `response.audio.delta`,
  `response.done`). `run_realtime_session(transport, cfg)` bypasses the STT→LLM→TTS
  cascade: mic PCM → base64 g711/pcm16 frames → realtime endpoint → decoded audio
  deltas sunk back to the transport. Key-gated (`OPENAI_API_KEY` / `REALTIME_API_KEY`):
  `make_realtime_brain` returns `None` when no key/endpoint so `__main__` falls back to
  the cascade. `RealtimeProtocol` (event encode/decode, base64 PCM ⇄ int16) is **pure**
  and unit-tested; the WS connect is isolated + lazy. `brain=realtime` config +
  `--brain realtime`. Tested against a **mocked realtime WS server** (no key/network).
- **R3-7 — Per-stage latency telemetry** (`metrics.py`): `TurnMetrics` now records
  per-turn `stt` / `llm_first_token` / `tts` / `first_audio` latencies keyed by a
  `speech_id`, emits each turn to `events.bus` (`metrics` event) **and** a structured
  JSON-lines log (`MetricsLog`), with `mark()`/`stage()` driven by an **injectable clock**
  (fake-clock tested) and a `MetricsAggregator` (count / mean / p50 / p95 per stage). An
  **OpenTelemetry span hook** is lazy-imported and OFF by default (`telemetry_otel`).
  Wired into `_respond` (`__main__`) + `respond_over_transport`/`capture_turn`
  (`net_loop`). `telemetry` / `telemetry_log_file` / `telemetry_otel` config.
- **R3-8 — Verified first-run bootstrap** (`preflight.py`, `turn.py`): `my-stt-tts
  --preflight` fetches **and SHA-256-checksums** the Smart-Turn ONNX (pinned hash) plus
  the configured Piper voices ahead of time and prints a clear ready/again report.
  `verify_checksum` + `ensure_smart_turn_model(expected_sha256=...)` reject a corrupt
  download (delete + retry). At runtime, when endpointing falls back to silence the
  `SmartTurnAnalyzer` now surfaces an **explicit warning** (log + a `bus` `endpoint_fallback`
  event + an optional one-time spoken cue) instead of silently degrading. Tested: the
  checksum verify, the missing/corrupt-download path, and the fallback warning.
- **R3-9 — Telephony reach** (`telephony.py`): a `TwilioMediaStreamSerializer` over the
  existing WebSocket transport — decodes Twilio's **base64 μ-law 8 kHz** media frames ⇄
  our int16 PCM with an **8k↔16k resample**, and handles the Twilio WS event protocol
  (`connected` / `start` / `media` / `stop`, outbound `media` frames with `streamSid`).
  `serve_twilio()` answers a phone call into the same pipeline (`run_transport_session`).
  μ-law transcode (`ulaw_encode`/`ulaw_decode`, the ITU-T G.711 algorithm) + the frame
  protocol are **pure** and unit-tested with fakes (no Twilio/network). `telephony` config
  - `--telephony`; the `transport` extra (websockets) suffices.

**202 tests passing** (170 baseline + 32 in `tests/test_round3d.py`); lint-clean
(ruff format/check + mypy clean on every touched file; pylint at parity with the
existing baseline — the only finding is the tolerated `duplicate-code` for the tiny
JSON-decode guard / `_suppress_full` idiom that `ws_transport`/`webrtc_transport`
already trigger). Optional extras added: `realtime`/`telephony` (both alias the
existing `websockets` dep — the protocols are pure-numpy here, no new package),
`otel` → opentelemetry-api/sdk (installs + the span hook emits a real per-turn span;
OFF by default and a clean no-op without the SDK). Caveats: the realtime WS server,
the Twilio call, the mic, the model download, and the clock are ALL faked in tests —
nothing opens a socket, downloads a file, calls an API, or reads a real clock. The
Smart-Turn SHA-256 pin (`07a133ab…`) was verified against the real upstream ONNX
(8.4 MB, valid protobuf) on 2026-06-19. μ-law decode is byte-exact vs stdlib
`audioop`; encode is a true nearest-quantizer (round-trip RMS < 1% of amplitude on
speech) and standard-G.711-compatible so Twilio reconstructs it exactly.

## Build status (2026-06-19) — round-3 transport/audio robustness (R3-1/2/3/4/6)

Closed the five gaps a round-3 judge ranked `pipecat` above us on (transport/audio
robustness). All wired + tested; no regression to the 146 baseline.

- **R3-2 — Full-duplex barge-in over the NETWORK transport** (`net_loop.py`):
  `respond_over_transport` is now duplex when `barge_in` is on — a shared
  `_MicSource` keeps the inbound mic live during TTS playout and a `_TransportBargeIn`
  monitor runs the same VAD + `InterruptGate` + AEC + `InterruptPredictor` chain as
  the local loop on every frame. A confirmed interruption cancels the outbound TTS
  **and** the in-flight LLM stream (`stream.close()` + `commit_spoken`) and the
  captured audio seeds the next turn (chained in `run_transport_session`). So
  satellite/browser users can interrupt, not just the local user.
- **R3-3 — Streamed, low-latency TTS playout** (`tts.py`, `text.py`): a `ClauseChunker`
  - `TTSRouter.synth_pcm_stream` (per-clause synthesis) + `StreamingPlayback` that
  pipes PCM into a `sounddevice` `OutputStream` as each clause renders, so first audio
  is the first clause (~200–300 ms), not the whole sentence. `start_speaking_stream`
  returns the same cancel surface as `Playback` so `monitor_during_playback` /
  barge-in are unchanged. The network sink streams clause PCM too. `tts_streaming`
  config + `--no-tts-streaming`.
- **R3-4 — macOS hardware-AEC capture** (`aec.py` `VoiceProcessingCapture`): captures
  THROUGH the `AVAudioEngine` VoiceProcessingIO node (PyObjC) — enables VP, installs a
  tap on the input bus, bridges the already-echo-cancelled channel-0 float32 PCM
  (48 kHz → pipeline rate) into Python. Wired into the `--wake` capture + barge-in
  path via a `source=` arg on `audio.record_turn` / `monitor_during_playback`; the SW
  NLMS is bypassed when HW capture is live. **Verified on arm64**: the tap delivers
  OS-cancelled buffers to numpy. Falls back to sounddevice + NLMS if PyObjC/VP is
  unavailable. `aec_hw_capture` config.
- **R3-1 — True WebRTC transport** (`webrtc_transport.py`): a third `AudioTransport`
  (`WebRtcTransport`) backed by **aiortc** — real `RTCPeerConnection`, **Opus**, jitter
  buffer, RTP/SRTP, ICE NAT traversal. The queue bridge is pure (numpy + queues, tested
  with fakes); the SDP signaling (`negotiate_answer`) is tested with a fake peer; the
  aiortc media plumbing (`_make_pcm_track`, `run_webrtc_offer`) is isolated + lazy.
  Verified end-to-end with two **real** aiortc peers (Opus negotiated, tone decoded to
  16 kHz frames). Browser path uses a real `RTCPeerConnection` +
  `getUserMedia({echoCancellation:true})`, signaled via `/api/webrtc/offer`; the WS PCM
  path stays as a fallback (CSP/demo intact). `transport=webrtc` + `--transport webrtc`.
- **R3-6 — Drop-in noise suppression** (`denoise.py`): a `Denoiser` seam +
  `SpectralGateDenoiser` (pure-numpy spectral gate, always available, raises SNR on
  steady noise) + `RnnoiseDenoiser` (optional wheel, graceful fallback) + null. Applied
  to mic frames AFTER AEC and BEFORE VAD/STT in both loops. `denoiser` config +
  `--denoiser`.

**170 tests passing** (146 baseline + 24 in `tests/test_round3.py`); lint-clean
(ruff/mypy/pylint at parity with the existing baseline). Optional extras added:
`webrtc` → aiortc (installs + imports on arm64), `denoiser` → pyrnnoise (resolves in
the lock but is **broken at runtime** on this arm64 setup — `audiolab`/`av.option`
conflict — so the pure-numpy `spectral` denoiser is the working default and `rnnoise`
falls back to it). Caveats: HW-AEC capture is wired into the `--wake` path only (PTT
stays on sounddevice); WebRTC ICE/STUN, the mic, models, and providers are all faked in
tests (no real network/GPU/device); WebRTC is browser-first (`--transport webrtc`
reuses the GUI signaling server).

## Build status (2026-06-19) — round-2 conversation gaps (R2-1/2/3/4/6)

**Phase 7 round 2 — closing the pipecat gaps (this session):**

- **R2-1 — Acoustic echo cancellation** (`aec.py`): `EchoCanceller` protocol +
  three backends — `VoiceProcessingEchoCanceller` (macOS **hardware** AEC via
  `AVAudioEngine`/`VoiceProcessingIO` through PyObjC; the `aec` extra installs on
  arm64 and the API is live), a pure-numpy **NLMS adaptive filter** (`NlmsEchoCanceller`,
  ~19 dB ERLE in tests, no native deps, the cross-platform fallback), and a null
  pass-through. `Playback` now carries its synthesized PCM as the AEC **reference**;
  `audio.monitor_during_playback` feeds it to the canceller, processes every mic
  frame before VAD, and **relaxes the energy floor when AEC is active**. `aec_mode`
  config (`off`/`nlms`/`voiceprocessing`/`auto`) + `--aec` flag + web UI.
- **R2-2 — Bounded sliding-window streaming STT** (`stt.py`): replaced whole-buffer
  re-decode with a `window_s`-bounded trailing re-decode stitched onto a committed
  prefix (`stitch_partial` de-dupes word overlap). Per-partial decode is bounded
  (≤ ~1.5× window) regardless of utterance length; `final()` still decodes the full
  clip for accuracy. `stt_window_s` config + `--stt-window` flag + web UI.
- **R2-3 — Acoustic interruption prediction** (`interrupt.py` `InterruptPredictor`):
  a 3rd barge-in guard scoring sustained voiced energy + spectral flux + ZCR for
  intent-to-take-the-floor; composes with the duration/word gate in the monitor
  loop (either may fire), so it talks through backchannels but yields to a sustained
  interruption before two words transcribe. `interrupt_predict*` config + flag + UI.
- **R2-4 — Smart-turn by default** (`turn.py`): `turn_analyzer` now defaults to
  **`smart`**; the Smart Turn v3 ONNX is **auto-downloaded on first run**
  (`ensure_smart_turn_model`, mirroring `_ensure_piper_voice`), with a clean
  fallback to silence when the model/runtime is genuinely unavailable.
  `smart_turn_model_url` / `smart_turn_auto_download` config.
- **R2-6 — Robust interrupt plumbing** (`events.py`, `__main__.py`): interruption is
  now formalised as **bus events** (`interrupt_start`/`interrupt_stop`/
  `bot_stopped_speaking`); on barge-in the captured audio is handed **straight into
  the streaming transcriber** (`StreamingTranscriber.feed_clip`) for the next turn
  instead of being re-transcribed from scratch.

**Phase 7 round 3 — network transport + tool calling (this session):**

- **R2-5 — Network audio transport** (`transport.py`, `ws_transport.py`,
  `net_loop.py`, `satellite.py`, `ws_frame.py`, browser audio): an `AudioTransport`
  seam (PCM frames in/out + control) with `LocalTransport` (sounddevice, default)
  and a `WebSocketTransport`. A real `websockets` server (`serve_websocket` /
  `WsSession`, the `transport` extra) accepts remote mic PCM and streams TTS PCM
  back, driving the existing pipeline via `run_transport_session` (capture →
  streaming STT → Brain → TTS-to-PCM sink). A **satellite** client
  (`python -m my_stt_tts.satellite ws://HOST:PORT`) captures mic + plays TTS over
  the link. The **browser GUI** now carries REAL audio: `getUserMedia` → 16 kHz PCM
  over a same-origin WebSocket (`/ws/audio`, CSP `connect-src 'self'`), TTS PCM
  streamed back for Web-Audio playback — implemented on the stdlib `http.server`
  with a hand-rolled RFC-6455 codec (`ws_frame.py`), so the GUI keeps zero web deps
  and the demo fallback is intact. `transport` config + `--transport`/`--browser-audio`.
- **R2-7 — In-conversation tool calling + cloud backends** (`tools.py`, `brain.py`,
  `stt.py`, `tts.py`): a `Tool`/`ToolRegistry` that serializes to **both** Anthropic
  and OpenAI wire formats, with the full tool-use round-trip wired into
  `Brain.stream` (model requests a tool → executed → result fed back → final answer
  streamed) for both providers. Example tools: `get_time`, a safe `calculator`
  (AST-guarded), and `home_control` (routes to the agent / HA dispatch). The legacy
  "agent, …" path still works. Optional **cloud STT** (`CloudTranscriber`) and **cloud
  TTS** (`CloudTTS`, e.g. a high-quality German voice) sit behind the existing
  seams — **local-first**, selected only when a key is present, graceful fallback
  otherwise. `tools_enabled` / `stt_backend` / `tts_backend` config.

**146 tests passing (101 baseline + 28 in `tests/test_transport.py` + 17 in
`tests/test_tools.py`), lint-clean** (ruff/mypy/pylint) on every touched file.
Verified live: a real `websockets` client handshakes, streams mic PCM, and receives
TTS PCM back through the server end-to-end (STT/Brain/TTS faked). Caveats: hardware
AEC enables the OS unit but capture still flows through `sounddevice`; the Smart
Turn download and all provider/network/mic boundaries are mocked in tests; the
WebSocket lib installed is `websockets` 16.0 (the `transport` extra). Pending: full
WebRTC for the browser (the PCM channel is real and sufficient), and the broader
Phase 8 whole-house / Home Assistant integration.

### Round-1 (prior session)

Barge-in (G1, cancellable playback + live-mic VAD + LLM-stream cancel), Smart Turn
v3 prosodic end-of-turn with silence fallback (G2), false-interrupt suppression
(G4, min-words/min-duration gate), post-interruption context repair (G5,
spoken-prefix history), and streaming STT partial transcripts (G6). New config
knobs (`barge_in`, `interrupt_min_*`, `turn_analyzer`, `smart_turn_*`,
`stt_streaming`) with env + CLI overrides, surfaced in `--settings` and the web UI.

## Build status (2026-06-17)

**Implemented + unit-tested (31 tests passing, lint-clean):** project scaffold
(`pyproject.toml` / `uv.lock`, `src/` layout, ruff/mypy/pytest, ruff+gitleaks
pre-commit, CI on `macos-15`); **Phase 0** (`config` + fail-fast validate,
`metrics` with shared `speech_id`, threaded `spine`); the pure logic of
**Phases 1–2** (`text` sentence-chunker with decimal/comma guard + non-spoken
stripping, `RateLimiter`, `PreRollBuffer`, half-duplex `MicGate`); the
provider-agnostic streaming `Brain` (Anthropic / OpenAI-compatible); the
`TTSRouter` (Piper-subprocess / `say` + language routing); `chimes`; and the
testable cores of **Phases 4–5** (`SilenceEndpointer`, `match_speaker`).
Backends (`stt` parakeet-mlx, `vad` Silero, `wake` openWakeWord, `speaker_id`
ECAPA) are coded with lazy imports; the push-to-talk loop is wired in
`__main__.py` with chimes, mic-gating, streaming, and graceful failure.

**Needs your machine (cannot run here):** live end-to-end test (mic + speakers +
`ANTHROPIC_API_KEY`), installing the heavy extras + the `piper-tts` CLI + Piper
voices, verifying the exact `parakeet-mlx` result API, training the "maziko"
wake-word model, and enrolling family voices (`uv run scripts/enroll.py <name>`).

**Update (this session):** `claude-cli` provider (subscription, no API key,
session-continued), now **stripped + isolated** (own prompt, no tools / CLAUDE.md /
hooks → ~8x faster, ~280x cheaper); `--brain` presets (haiku/sonnet/opus × sub/api,
ollama); editable spoken prompt at `prompts/system_prompt.md`; voice menu
(`--list-voices` / `--voice`) + calmer cadence; `./mstt` launcher (run without
`uv run`). **Phase 6** — agent dispatch: say "agent, &lt;task&gt;" to delegate to a
full MCP-capable Claude agent in `AGENT_WORKSPACE` (`agent.py`). **Phase 4** —
`--wake` mode: wake word → VAD capture → respond → follow-up (`wake.py`, `audio.py`
VAD helpers); train "maziko" per `wakewords/WAKEWORD.md`. 36 tests, lint-clean.
Still needs the M1 for: live mic/STT, the trained "maziko" model, enrollment.

---

## 1. Goal (restatement)

Build a single, always-on Python process on the M1 that listens for a wake word,
records one utterance, transcribes it, sends the text to an LLM (streaming),
speaks the answer back through the Mac speakers, and — as it goes — identifies
*who* spoke. It must feel responsive (target perceived first-audio ≈ 1–1.5 s
excluding model thinking time), work in **Hochdeutsch (standard German), French,
and English**, and keep the large language model (LLM) layer pluggable so we can
start on a fast cheap model and later default to a stronger one and orchestrate
other agents. The repository is **public and meant to be polished for external
users** (§8, §9).

Abbreviations: **STT** = Speech-to-Text, **TTS** = Text-to-Speech, **LLM** =
Large Language Model, **VAD** = Voice Activity Detection, **AEC** = Acoustic Echo
Cancellation, **MCP** = Model Context Protocol, **TTFA** = time-to-first-audio,
**TTFT** = time-to-first-token, **RTF** = Real-Time Factor, **EER** = Equal Error
Rate, **EOU** = End-Of-Utterance, **G2P** = Grapheme-to-Phoneme, **CI** =
Continuous Integration, **TCC** = macOS Transparency/Consent/Control (privacy).

---

## 2. Locked decisions (with rationale)

| # | Decision | Choice | Why |
|:--|:---------|:-------|:----|
| D1 | **Implementation language** | **Python** orchestrator; optional thin **Swift** audio front-end deferred to Phase 7 | Latency is dominated by native model inference (Metal/MLX/C++) and the Claude network round-trip. Glue is <0.2 % of a ~1–2 s turn; the GIL is released inside native calls and I/O. Rust/Swift win single-digit ms while costing weeks against immature M1 ML bindings. Python's Apple-Silicon ML ecosystem (MLX, `mlx-audio`, `parakeet-mlx`, PyTorch MPS, ONNX) is the most mature. |
| D2 | **STT engine** | **`parakeet-mlx`** (`parakeet-tdt-0.6b-v3`, multilingual) primary; `whisper.cpp` large-v3-turbo alternate | v3 is multilingual (DE/FR/EN + auto language-ID), MLX-native, sub-second, beats Whisper-large on WER. **`faster-whisper` is CPU-only on Mac — do not use it.** |
| D3 | **TTS engine** | **Piper** (DE `thorsten-high`, FR `tom-medium`, EN `lessac`) primary, **invoked as a subprocess** (see D10); **macOS `say` premium** instant fallback; optional **Kokoro via `mlx-audio`** (English, espeak disabled) | Piper is the only local engine with strong **German**, correct French, good English, **and** sub-300 ms TTFA on M1 CPU. Kokoro has **no German**. XTTS-v2 (non-commercial, MPS hangs) and Qwen3-TTS (GPU-oriented) deferred behind the Router. |
| D4 | **Speaker identification** | **SpeechBrain ECAPA-TDNN** embeddings + enrollment + cosine to per-person centroids, with unknown/ambiguous rejection | Best accuracy (~0.80 % EER), text-independent + cross-lingual-robust (DE/FR/EN), runs **in parallel with STT** → ~0 added latency. Resemblyzer rejected (English-biased). No surveyed repo ships this — bespoke. |
| D5 | **LLM layer** | **Provider-agnostic** via an OpenAI-compatible interface (Anthropic default; OpenAI / Ollama / vLLM / local also work). Streaming; default **`claude-haiku-4-5`** → **`claude-opus-4-8`** (deep path) on trigger; chosen by `LLM_PROVIDER`/`LLM_MODEL`/`LLM_BASE_URL`; tool-use / MCP-ready for multi-agent dispatch | Voice turns want a fast cheap default; Opus is a latency/cost tax. Anthropic exposes an OpenAI-compatible endpoint (as do most providers), so one client targets all. A `Brain` interface keeps provider + model as config. |
| D6 | **Stage confirmations** | **Earcons (chimes)**, not spoken phrases. Wake chime + optional end-of-record chime. Spoken narration behind `--debug` only | The four spoken phrases in the original sketch add **~6–7 s dead air/query**. Chimes are ~150 ms, language-neutral, don't re-trigger the wake word. |
| D7 | **End-of-turn detection** | **Push-to-talk** (v1) → **two-stage VAD** (WebRTC gate → Silero confirm) → **smart-turn** model-based endpointing; hard max-recording cap | Endpointing is the hardest part of voice UX. PTT removes it so we validate the core loop; VAD then smart-turn (prosody-aware) follow. |
| D8 | **Process model** | **One warm long-running process**; all models pre-loaded at startup; **threaded producer-consumer spine** (one queue per stage, generator stages stream), `SESSION_END` vs `PIPELINE_END` signals | Model load + Metal warm-up is hundreds of ms–seconds; pay once. This (not language) is the biggest latency lever. Spine pattern from HF `speech-to-speech`. |
| D9 | **Echo / self-trigger** | **Half-duplex mic gating** (suspend wake + capture during playback + ~200 ms tail), built **barge-in-ready**. Full AEC + barge-in in Phase 7 (Swift `VoiceProcessingIO`) | Speaker+mic share the enclosure; gating kills ~95 % of self-trigger at ~zero cost. Design the gate so interruption can be switched on later (GLaDOS mute-event pattern). |
| D10 | **Licensing & distribution** | Project is **Apache-2.0**. **GPL backends (Piper, espeak-ng) invoked as subprocesses (CLI), never imported.** Non-permissive backends (XTTS CPML non-commercial, openWakeWord bundled models CC-BY-NC-SA) are **opt-in extras**; shipped default leans permissive (Kokoro espeak-disabled / `say`). | `pip install piper-tts` now pulls GPL-3.0 `OHF-Voice/piper1-gpl` (embeds espeak-ng); the old MIT `rhasspy/piper` was archived Oct 2025. Subprocess use = "mere aggregation" (FSF) → project stays Apache-2.0. Apache > MIT here for the explicit ML patent grant. |
| D11 | **AI/contributor docs** | Commit a **public `AGENTS.md`** (build/lint/run conventions); **gitignore `CLAUDE.md`** (a shim that `@AGENTS.md`-imports) and `CLAUDE.local.md` (private notes). README links AGENTS.md. | Don't link a gitignored file from a public README. AGENTS.md is the tool-agnostic standard; Claude Code still reads CLAUDE.md, hence the gitignored import shim. Avoids leaking infra the way the sibling repo's CLAUDE.md would. |

---

## 3. Architecture

```text
                         ┌─────────────────────────────────────────────┐
                         │  one warm Python process — threaded spine    │
                         │  (one queue per arrow; generator stages;     │
                         │   SESSION_END = per-turn reset, PIPELINE_END │
                         │   = shutdown; shared speech_id for telemetry)│
                         │                                              │
  mic ──► ring buffer ──►│  Wake word         Endpointing               │
        (pre-roll deque) │  (openWakeWord     PTT → 2-stage VAD         │
                         │   "maziko")         → smart-turn (prosody)   │
                         │        │                │                    │
                         │   [chime: live]   ┌──────▼──────┐            │
                         │                   │ utterance    │           │
                         │                   │ PCM clip      │          │
                         │                   └──────┬───────┘           │
                         │            ┌─────────────┴───────────┐       │
                         │            ▼ (parallel)             ▼        │
                         │      STT (parakeet-mlx)      Speaker-ID       │
                         │      → text + lang           (ECAPA centroid  │
                         │            │                  cosine match)   │
                         │            ▼                      │           │
                         │      Brain (LLM/Claude, streaming)│           │
                         │      Haiku / Opus + memory        │           │
                         │      strip non-spoken text        │           │
                         │            │ tokens → sentence/fragment       │
                         │            ▼  (decimal/comma guard)           │
                         │      TTS Router (per-language, subprocess)    │
                         │      DE→Piper-thorsten  FR→Piper-tom          │
                         │      EN→Kokoro/Piper    fallback→say          │
                         │            │ stream first fragment early      │
   speakers ◄────────────│            ▼  (mic gated; barge-in-ready)     │
                         └─────────────────────────────────────────────┘
```

### Latency budget (target per stage, M1, short command)

| Stage | Target | Notes |
|:------|:-------|:------|
| Wake-word detection lag | 80–150 ms | continuous, low-power |
| Endpointing | PTT ≈0 / VAD 300–700 ms / smart-turn ~10 ms decision | smart-turn catches "let me think…" pauses VAD would cut |
| STT (parakeet-mlx) | 80–400 ms | native MLX; concurrent with speaker-ID |
| Speaker-ID (ECAPA) | hidden under STT | ~80–150 ms in parallel → ~0 added |
| LLM TTFT (Haiku, streaming) | 400–800 ms | network-bound; Opus higher (deep-path tradeoff) |
| TTS first fragment (Piper/Kokoro) | 40–200 ms | fragment-streamed; playback before full answer |
| Playback start | 10–30 ms | CoreAudio buffer |
| **Perceived first audio** | **~1.0–1.5 s** | with streaming + overlap; physics floor for a cloud LLM |

---

## 4. Repository layout (target)

```text
my-stt-tts/
├── README.md                # overview, install methods, license note  [done]
├── PLAN.md                  # this file                                 [done]
├── AGENTS.md                # AI/contributor conventions (public)       [done]
├── LICENSE                  # Apache-2.0                                [done]
├── CLAUDE.md                # gitignored shim → @AGENTS.md              [done]
├── pyproject.toml           # PEP 621; deps + ruff/mypy/pytest config; uv-managed
├── uv.lock                  # committed lockfile
├── .python-version          # pin interpreter
├── .env.example             # ANTHROPIC_API_KEY=…                       [done]
├── config.toml              # voices, models, thresholds, wake phrase
├── .pre-commit-config.yaml  # gitleaks (+ ruff hooks)                   [seeded]
├── .github/
│   ├── workflows/ci.yml     # macos-15 runner: ruff + mypy + pytest
│   ├── dependabot.yml       # uv + github-actions, weekly
│   └── ISSUE_TEMPLATE/      # YAML forms (OS/chip/backend fields)
├── SECURITY.md  CHANGELOG.md  CONTRIBUTING.md
├── src/my_stt_tts/
│   ├── __main__.py          # entrypoint: warm models, run spine
│   ├── spine.py             # threaded producer-consumer; signals; speech_id
│   ├── config.py            # central Config + fail-fast validate
│   ├── audio.py             # capture, pre-roll ring buffer, playback, mic-gating
│   ├── wake.py              # openWakeWord ("maziko")
│   ├── vad.py               # 2-stage VAD (WebRTC→Silero) + smart-turn endpointing
│   ├── stt.py               # parakeet-mlx / whisper.cpp
│   ├── speaker_id.py        # ECAPA enrollment + match + reject
│   ├── brain.py             # Claude streaming + routing + memory + text-strip
│   ├── tts.py               # TTS Router (Piper subprocess / Kokoro / say) + lang detect
│   ├── chimes.py            # earcons; pre-synth error clips
│   └── metrics.py           # per-stage latency + transcript logging (speech_id)
├── scripts/
│   ├── enroll.py            # record ~30s/person → ECAPA centroid
│   └── bench.py             # measure per-stage latency on this Mac
├── tests/                   # smoke tests; audio + backends mocked
├── samples/                 # audio demo clips for README/Pages gallery
└── enroll/                  # gitignored: per-person voice profiles
```

Every script gets `-h/--help`, is made executable + git-exec-bit set, and follows
`$mygit/README_SETUP_PYTHON_ENVIRONMENT.md` (read before the first Python file).
Lint gate before every commit: `ruff format && ruff check && mypy && pylint`
(Python), `shellcheck` (shell).

---

## 5. Phased plan (checkboxes)

### Phase 0 — Scaffold, spine & environment ✅ done

- [ ] Read `$mygit/README_SETUP_PYTHON_ENVIRONMENT.md`; `uv init --package`; `pyproject.toml` (PEP 621, `license = "Apache-2.0"`); commit `uv.lock`
- [ ] `config.py`: central Config (string-dispatch backends) + fail-fast `validate()`; `.env.example`; `config.toml`
- [ ] `spine.py`: threaded producer-consumer (queue per stage, generator stages, `SESSION_END`/`PIPELINE_END`) — HF `speech-to-speech` pattern
- [ ] `metrics.py` first: per-stage timing keyed by shared **`speech_id`** (we tune by numbers) — LiveKit pattern
- [ ] `scripts/bench.py`: measure real STT/TTS/LLM latency on *this* M1

### Phase 1 — Core loop (push-to-talk, English, batch) ✅ done (code; live mic test pending)

- [ ] `audio.py`: `sounddevice` capture, explicit device, **pre-roll ring buffer** (no clipped onset), push-to-talk hotkey, max-recording cap
- [ ] `stt.py`: `parakeet-mlx` warm-loaded
- [ ] `brain.py`: Claude streaming (Haiku); **strip non-spoken text** before TTS (markdown, `(parentheticals)`, reasoning blocks) — GLaDOS pattern
- [ ] `tts.py`: Piper English **via subprocess** → playback
- [ ] `chimes.py`: wake chime; `--debug` spoken cues (the original "yes/recorded/analyzing" narration, off by default)
- [ ] End-to-end: press key → speak → hear Claude; log per-stage latency

### Phase 2 — Responsiveness (streaming + safety) ✅ done (barge-in → Phase 7)

- [ ] **Prosody-preserving fragment streaming**: Claude tokens → sentence/fragment chunker (first-fragment-fast, full prosody after) with **decimal/comma guard** (keep `3.14` / German `3,14`) — RealtimeTTS + GLaDOS patterns; BufferStream bridge (Linguflex)
- [ ] Overlap stages on the spine; confirm pre-roll + streaming feel
- [ ] Half-duplex **mic gating** during playback + 200 ms tail, **barge-in-ready** (D9)
- [ ] Graceful failure: catch every stage; play **pre-synthesized** error clips even if TTS is what failed
- [ ] Runaway guard: per-minute request cap + cooldown (self-trigger / cost protection)

### Phase 3 — Multilingual (DE / FR / EN) ✅ done

- [ ] STT multilingual: Parakeet v3 language-ID (or Whisper auto-detect); expose detected language
- [ ] `tts.py` **Router**: `lingua-py` detection on the answer → voice map (`de→thorsten-high`, `fr→tom-medium`, `en→Kokoro/lessac`), `say` premium + low-confidence fallback
- [ ] Test Hochdeutsch + French end-to-end; verify pronunciation
- [ ] (Optional) Kokoro-via-`mlx-audio` for higher-quality English

### Phase 4 — Wake word & always-listening ◑ wired — needs the trained "maziko" model

- [ ] Train + integrate **openWakeWord** for **"maziko"** (custom model, ~1 h via the training notebook; no vendor lock)
- [ ] Replace PTT with wake-word + **two-stage VAD** (WebRTC gate → Silero confirm) — RealtimeSTT pattern; tune `silero_sensitivity`, silence durations
- [ ] **smart-turn** model-based endpointing (vendor pipecat smart-turn, CoreML variant for the Neural Engine) to augment the silence timeout
- [ ] Wake-word debounce; conversation **follow-up window** (~8 s open mic, no re-wake); multi-turn **memory** (rolling `messages`, capped, idle reset)

### Phase 5 — Speaker identification (bespoke) ◑ logic + calibration done — needs enrollment recordings

- [ ] `scripts/enroll.py`: ~30 s/person across 5–10 clips per language → L2-normalized ECAPA **centroid** (gitignored)
- [ ] `speaker_id.py`: extract embedding **in parallel** with STT; cosine `argmax` over centroids
- [ ] Rejection: absolute threshold (~0.40–0.50, **calibrated on our family + guest clips**) + margin gate (~0.06) → `unknown` / `ambiguous`
- [ ] Bias to `unknown` over misattribution; **never gate safety-critical actions on child ID**; re-enroll children quarterly
- [ ] Pass identified speaker into the Brain prompt for personalization

### Phase 6 — LLM flexibility & agent orchestration ✅ agent dispatch + presets done

- [ ] Model routing: Haiku fast / Opus deep via trigger or per-speaker default
- [ ] Prompt caching for the stable system prompt
- [ ] **Layered context** assembly (system + prefs + tools + compacted history) — GLaDOS `context.py`
- [ ] Tool-use / **MCP** wiring to dispatch to other home/work agents; tool pre-filtering (Linguflex)
- [ ] Per-speaker + per-language context (Swiss defaults: metric, ISO-8601)

### Phase 7 — Barge-in & native audio ◑ round-3 closed network transport (R2-5) + tool calling / cloud backends (R2-7); only the full HW-AEC HAL path + menubar packaging remain

- [x] **Barge-in** (G1): cancellable TTS playback (`tts.Playback` kills the `afplay`/`say` subprocess mid-utterance; `TTSRouter.start_speaking`), mic kept LIVE during playback (`audio.monitor_during_playback`), in-flight LLM stream cancelled (generator `.close()`), `bus.interrupted(...)` event for the UI. Configurable `barge_in` mode (`off`/`headphones`/`always`) + energy gate (`barge_in_energy`) for open-speaker bleed.
- [x] **Smart-turn / prosodic end-of-turn** (G2 + R2-4): `turn.TurnAnalyzer` protocol + `SilenceTurnAnalyzer` (always-available fallback) + `SmartTurnAnalyzer` (loads `pipecat-ai/smart-turn-v3` ONNX via Whisper feature extractor; silence-gated inference; **graceful fallback** to silence when the model/deps are missing). **Now the DEFAULT** `turn_analyzer`, with the ONNX **auto-downloaded on first run** (`ensure_smart_turn_model`).
- [x] **False-interrupt suppression** (G4): `interrupt.InterruptGate` — min speech duration AND/OR min word count (pipecat `MinWords` equivalent) so backchannels/coughs/TV don't abort the assistant. Thresholds in config (`interrupt_min_speech_ms`, `interrupt_min_words`).
- [x] **Post-interruption context repair** (G5): track voiced prefix; `Brain.commit_spoken()` stores only what was actually spoken (dropping the assistant turn if nothing was voiced) — fixed the `finally`-block full-append.
- [x] **Streaming STT** (G6 + R2-2): `stt.StreamingTranscriber` emits `bus.transcript(text, partial=True)` during the turn; finalises on end-of-turn. Now uses a **bounded sliding-window** re-decode (`stt_window_s`) stitched onto a committed prefix (`stitch_partial`) so latency/CPU don't grow with utterance length. Toggle via `stt_streaming`.
- [x] **R2-1 — Acoustic echo cancellation** (`aec.py`): `EchoCanceller` seam + macOS hardware `VoiceProcessingEchoCanceller` (PyObjC `aec` extra) + pure-numpy `NlmsEchoCanceller` (~19 dB ERLE) + null. `Playback` carries the synthesized PCM reference; the monitor loop cancels per-frame and relaxes the energy floor when AEC is active. `aec_mode` config + `--aec`.
- [x] **R2-3 — Acoustic interruption prediction** (`interrupt.InterruptPredictor`): a 3rd, purely-acoustic barge-in guard (sustained voiced energy + spectral flux + ZCR) composed with the gate so a real interruption wins before two words transcribe while backchannels are talked through. `interrupt_predict*` config + `--no-interrupt-predict`.
- [x] **R2-6 — Robust interrupt plumbing**: interruption formalised as bus events (`interrupt_start`/`interrupt_stop`/`bot_stopped_speaking`); captured barge-in audio fed straight into the streaming transcriber (`feed_clip`) — no from-scratch re-transcribe.
- [x] **G3 / R3-4 — full hardware-AEC path end-to-end**: `aec.VoiceProcessingCapture` captures THROUGH the `AVAudioEngine` VoiceProcessingIO node (PyObjC tap) so already-OS-cancelled PCM reaches Python (48 kHz → pipeline rate); wired into the `--wake` capture + barge-in path (`source=` on `record_turn`/`monitor_during_playback`), SW NLMS bypassed when HW capture is live. Verified on arm64; falls back to sounddevice+NLMS otherwise. `aec_hw_capture` config.
- [x] **R3-1 — true WebRTC transport**: `webrtc_transport.WebRtcTransport` (aiortc) — real `RTCPeerConnection`, Opus, jitter buffer, ICE NAT traversal; browser uses a real `RTCPeerConnection` + `getUserMedia({echoCancellation:true})` signaled via `/api/webrtc/offer`, WS PCM fallback intact. `transport=webrtc` + the `webrtc` extra.
- [x] **R3-2 — full-duplex barge-in over the network transport**: `net_loop.respond_over_transport` keeps the mic live during TTS playout (`_MicSource` + `_TransportBargeIn`) and cancels TTS + the LLM stream on a confirmed interruption, chaining the captured audio to the next turn.
- [x] **R3-3 — streamed low-latency TTS**: clause-chunked synthesis (`ClauseChunker`/`synth_pcm_stream`) piped into a `sounddevice` `OutputStream` (`StreamingPlayback`) / the transport sink; first audio in ~200–300 ms, cancel semantics preserved. `tts_streaming` config.
- [x] **R3-6 — pre-VAD noise suppression**: `denoise.SpectralGateDenoiser` (pure-numpy, default) + optional RNNoise (graceful fallback), applied after AEC and before VAD/STT in both loops. `denoiser` config.
- [x] **G7 / R2-5 — network audio transport**: `AudioTransport` seam (`transport.py`) with `LocalTransport` (sounddevice, default) + `WebSocketTransport`; a real `websockets` server (`ws_transport.serve_websocket`/`WsSession`, the `transport` extra) bridges remote clients into the pipeline via `net_loop.run_transport_session`; a `satellite.py` client streams mic up + plays TTS back; the **browser GUI carries real audio** (`getUserMedia` → 16 kHz PCM over a same-origin `/ws/audio` WebSocket, TTS PCM streamed back), implemented on the stdlib `http.server` with a hand-rolled RFC-6455 codec (`ws_frame.py`). `transport`/`transport_*` config + `--transport`/`--browser-audio`.
- [x] **R2-7 — In-conversation tool calling + cloud backends**: `tools.ToolRegistry` (Anthropic + OpenAI schemas) + the full tool-use round-trip in `Brain.stream` for both providers (request → execute → feed result back → stream the answer); example tools `get_time`/`calculator`/`home_control` (→ agent/HA dispatch); legacy "agent, …" still works. Optional **local-first** cloud STT (`CloudTranscriber`) + cloud TTS (`CloudTTS`) behind the seams, key-gated with graceful fallback. `tools_enabled`/`stt_backend`/`tts_backend` config.
- [ ] Multi-agent floor-control ("conch" lock — voicemode) so two agents don't talk at once
- [ ] Package as menubar app (`rumps`) / `launchd` with a **stable bundle id** (TCC keyed to it); idle model unload

### Phase 8 — Whole-house / Home Assistant (future) ⬜ future

- [ ] Move brain to a server, mics/speakers to satellites; integrate with `home-assistant-sandbox` Assist + Wyoming; revisit Sonos vs satellite-local playback latency

### Phase 9 — External polish / OSS readiness (parallel track) ◑ most done — Homebrew tap & hero MP4 pending

- [x] LICENSE (Apache-2.0), public AGENTS.md, gitignored CLAUDE.md shim, README with install methods + license note
- [x] Repo description + topics + (todo) social-preview image
- [ ] `pyproject.toml` (PEP 621, SPDX license string, extras for opt-in backends), `uv.lock`, src layout
- [ ] pre-commit: add **ruff** hooks beside gitleaks; `pytest` smoke suite
- [ ] GitHub Actions CI on **`macos-15`** (arm64): `brew install` native deps → `uv sync --locked` → ruff/mypy/pytest (audio mocked); Dependabot (uv + actions)
- [ ] SECURITY.md, CHANGELOG.md (Keep a Changelog), CONTRIBUTING.md, YAML issue forms
- [ ] README hero **demo with audio** (MP4 — a voice app must be heard; VHS GIF secondary); GitHub Pages **voice-sample gallery** (`<audio>` can't play inline in README); comparison table
- [ ] Homebrew tap (`glensk/tap/my-stt-tts`) — primary install; PyPI + `uv tool install` secondary; Docker documented as unsupported on macOS

---

## 6. Dependencies (initial)

```commands
uv add anthropic openai parakeet-mlx mlx-audio speechbrain torchaudio \
       sounddevice silero-vad webrtcvad-wheels openwakeword onnxruntime \
       lingua-language-detector
brew install whisper-cpp espeak-ng portaudio ffmpeg piper        # piper = CLI binary (subprocess)
# vendor pipecat smart-turn (CoreML) model for endpointing (Phase 4)
python -m piper.download_voices de_DE-thorsten-high fr_FR-tom-medium en_US-lessac-medium
# macOS premium voices: System Settings → Accessibility → Spoken Content → Manage Voices
#   (Anna (Premium) [de], Thomas [fr], Ava (Premium) [en])
```

Notes: **Piper is used via its CLI binary (subprocess), not `import piper`** (D10).
Kokoro (via `mlx-audio`) is run with `misaki` espeak-ng **disabled** to stay
permissive. `mlx-audio` can also expose an OpenAI-compatible local server
(`python -m mlx_audio.server`) if we later want to process-isolate the engine.

---

## 7. Borrowed building blocks (vendor / study)

| Feature | Source repo | Verdict |
|:--------|:------------|:--------|
| **smart-turn** model-based endpointing (8 MB, CoreML, DE/FR/EN) | `pipecat-ai/smart-turn` | **vendor** the model |
| **Two-stage VAD** (WebRTC gate → Silero confirm) + endpointing knobs | `KoljaB/RealtimeSTT` `core/voice_activity.py` | **vendor/copy** |
| **Pre-roll ring buffer** (no clipped onset) | RealtimeSTT (`pre_recording_buffer_duration`), GLaDOS (`BUFFER_SIZE=800ms`) | **copy** |
| **Prosody-preserving fragment streaming** TTS (first-fragment-fast; decimal guard) | `KoljaB/RealtimeTTS` `text_to_stream.py`, GLaDOS `llm_processor.py` | **copy** |
| **Threaded producer-consumer spine** (generator stages; SESSION/PIPELINE end) | `huggingface/speech-to-speech` `baseHandler.py` | **adopt as skeleton** |
| **Per-turn latency telemetry** with shared `speech_id` | `livekit/agents` `metrics/base.py` | **copy hooks** |
| **Strip non-spoken text** before TTS (markdown / parentheticals / reasoning) | GLaDOS `llm_processor.py` | **copy** |
| **Barge-in + false-interrupt suppression** | GLaDOS (cancel), pipecat `MinWordsUserTurnStartStrategy` | **study → Phase 7** |
| **BufferStream bridge** (Claude stream → TTS without blocking) | `KoljaB/Linguflex` `modules/speech/logic.py` | **copy** |
| **Streaming engine** (Kokoro/Parakeet on M1; per-segment `sample_rate`/RTF) | `Blaizzy/mlx-audio` `tts/generate.py` | **primary engine** |
| **MCP tools + multi-agent handoff / floor-control** | `livekit/agents`, `mbailey/voicemode` (conch) | **study → Phase 7** |
| **Config seam** (string-dispatch providers + fail-fast validate) | `PromtEngineer/Verbi` `config.py` | **copy (lightweight)** |

`RealtimeSTT` and `RealtimeTTS` are pip-installable (MIT) — consider using them
directly in Phases 1–2 rather than reimplementing, then specialize.

---

## 8. Third-party licenses & distribution

Project license: **Apache-2.0**. Backends are invoked as **separate processes**
(subprocess / local HTTP), which is "mere aggregation" under the FSF GPL FAQ — so
they do **not** make this project a derivative work.

| Backend | License | Handling |
|:--------|:--------|:---------|
| Piper, espeak-ng | **GPL-3.0** | subprocess (CLI) only; never `import` |
| XTTS-v2 (Coqui) | **CPML — non-commercial** | optional extra; personal use only |
| openWakeWord (pretrained models) | **CC-BY-NC-SA-4.0** | self-trained "maziko" model avoids this |
| Kokoro, SpeechBrain, Silero-VAD, parakeet-mlx, mlx-audio, PortAudio | Apache-2.0 / MIT | permissive; Kokoro run espeak-disabled |
| ffmpeg | LGPL-2.1+ | subprocess |

A `Third-party licenses` section in the README mirrors this so external users
aren't misled. Default shipped TTS leans permissive (Kokoro/`say`); Piper/XTTS are
opt-in.

---

## 9. External-readiness checklist (condensed)

**Tier 0 (done / quick):** Apache-2.0 LICENSE ✅ · third-party-license note ✅ ·
repo description + topics ✅ · social-preview image (todo).
**Tier 1 (hygiene):** `pyproject.toml` (PEP 621) + `uv.lock` · src layout · ruff +
mypy · pre-commit (ruff + gitleaks) · pytest smoke · CI on `macos-15` · Dependabot ·
SECURITY.md.
**Tier 2 (attract):** README hero **demo with audio** (MP4) · Pages voice gallery ·
mermaid diagram ✅ · badges ✅ · comparison table.
**Tier 3 (docs):** AGENTS.md ✅ (commit) / CLAUDE.md gitignored shim ✅ · CONTRIBUTING ·
CODE_OF_CONDUCT · CHANGELOG.
**Install:** Homebrew tap (primary) · `uv tool` / PyPI (secondary) · from-source ·
Docker **documented as unsupported on macOS** (no mic/speaker/Metal in container).
**Skip (over-engineering for one dev):** Renovate, semantic-release, multi-OS CI
matrix, codecov, Astral `ty` in CI.

---

## 10. Risk register

| Risk | Severity | Confidence | Mitigation |
|:-----|:---------|:-----------|:-----------|
| **Children's voices** misidentified (esp. youngest, 2-word commands) | High for kids | High | Buffer full utterance, bias to `unknown`, margin gate, never gate safety actions on child ID |
| **Piper/espeak GPL-3.0** contaminating an Apache-2.0 project if imported | High (legal) | High | Invoke as subprocess only (D10); default to permissive engines; third-party-license table |
| **German TTS quality ceiling** (Piper-Thorsten best local but below ElevenLabs) | Medium | High | Accept v1; `say -v "Anna (Premium)"` fallback; revisit Qwen3-TTS-MLX / Chatterbox when stable |
| **No usable Metal TTS acceleration** on M1 (XTTS MPS hangs; Qwen3 GPU-oriented) | Medium | Moderate | Stay on Piper (CPU) + `say`/Kokoro; heavy models deferred behind Router |
| **Echo / self-trigger** on a single-box laptop | High | High | Half-duplex mic gating (Phase 2); AEC only in Phase 7 |
| **Cost / runaway loop** (self-trigger firing Claude) | Medium | High | Per-minute cap + cooldown; default Haiku |
| **M1 latency numbers indicative**, not lab-measured on base M1 | Low | Moderate | `scripts/bench.py` measures the real budget first |
| **CI can't exercise audio** (no mic on runners) | Low | High | Mock `sounddevice`/backends; CI tests glue/config, `macos-15` only for the MLX path |

---

## 11. Open items (defaulting as noted unless overridden)

1. **Wake word** — **LOCKED:** openWakeWord, phrase **"maziko"** (custom model, ~1 h; PTT until Phase 4).
2. **STT** — default **`parakeet-mlx` v3**; `whisper.cpp` fallback if multilingual punctuation/accuracy disappoints.
3. **TTS** — default **Piper (subprocess) for all three languages** v1; Kokoro-for-English optional.
4. **License** — **LOCKED: Apache-2.0** (MIT is a one-file swap if preferred).
5. **Primary install** — **Homebrew tap**; PyPI/`uv tool` secondary; Docker unsupported on macOS.
6. **AI docs** — **LOCKED:** commit AGENTS.md; gitignore CLAUDE.md/CLAUDE.local.md.
7. **Speaker-ID roster** — confirm who to enroll at Phase 5.

---

## 12. Data / privacy note (SDSC context)

Local STT + TTS keep voice audio **on-device**; only the transcribed *text* leaves
the machine (to your chosen LLM provider — Anthropic by default). Do not dictate
Confidential / Strictly-Confidential content. Enrollment voice profiles stay local
and gitignored.
