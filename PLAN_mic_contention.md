# PLAN — server mic-check CoreAudio device-contention fix (`mic-contention`)

## Session

Resume: `c --resume <session-id>` (worktree `.worktree-miccontend`, branch `mic-contention`)

Backend-only. NEVER touch `main` / `README.md` / `clients/` / `esp32/` / `webui.html`
/ `wakewords/maziko.onnx`. Core baseline ~1033 tests (1043 with `--extra all`).

## Diagnosis (from `logs/`)

Under `--browser --wake` the always-listen wake loop holds the CoreAudio input device
continuously. The server mic-check is supposed to pause it (`_with_paused_wake`), but
the wake loop's `sounddevice` InputStream isn't fully released before the mic-check
opens its own → they collide → `||PaMacCore (AUHAL)|| Error on line 2523: err='-50'`
(paramErr) → the check records near-silence (int16_peak ~96–125, lufs −64).
INTERMITTENT: in the same run an earlier server capture got int16_peak 8463 +
hey_jarvis fired 0.997 — so the mic + iTerm permission are FINE; it's a device-handoff
race. The "✗ No audio captured — check the microphone permission and input device"
copy misfired even though permission was fine.

## Fixes

- [x] (1) Fully release + settle before the capture opens. `_with_paused_wake`
      (`__main__.py`): snapshot the wake thread under the lock, `stop_wake()` (its
      InputStream context-exit stops+closes), `thread.join(timeout=3.0)`, THEN sleep
      `cfg.mic_check_settle_s` (default 0.15 s) so CoreAudio frees the device before
      `fn()` opens its capture stream. No settle/join when the loop wasn't running.
- [x] (2) Retry the capture on a CoreAudio contention error. `audio.record_fixed`
      gained `retries` / `settle_s`; the whole open+record+close attempt is wrapped in
      `_capture_with_retry`, which retries (default 3) with a backoff on a
      `is_device_contention_error` (a `sounddevice.PortAudioError`, or any exception
      carrying `-50` / `paramErr` / `PaMacCore` / `device unavailable`). A
      non-contention error (genuine no-device) is re-raised immediately. An
      opened-but-silent capture that ALSO flagged a PortAudio input-error status is
      raised as contention (so it retries / reports busy, not a bogus silent OK).
      `_run_mic_check_server` / `_run_mic_record_replay` pass `cfg.mic_check_retries`
      / `cfg.mic_check_settle_s`.
- [x] (3) Fix the misleading message. On a contention error the server mic-check /
      record-replay verdict says "Microphone was busy (device contention) — … Try …
      again." (record-replay verdict tag `busy`) instead of "check the microphone
      permission and input device". A genuinely silent (no-error) capture, a
      `denied` / `restricted` / `notDetermined` permission, and a no-device error all
      keep their real permission/no-device messages.

## Knobs (new)

- `MIC_CHECK_SETTLE_S` (`cfg.mic_check_settle_s`, default 0.15, range [0, 2]).
- `MIC_CHECK_RETRIES` (`cfg.mic_check_retries`, default 3, range [1, 5]).
- Documented in `.env.example`; field/env/validate in `config.py`.

## Tests (`tests/test_mic_contention.py`, +25)

- [x] `_with_paused_wake` closes + joins the wake stream and settles BEFORE the
      paused capture (event-order assertion via a mock loop + patched `time.sleep`).
- [x] `record_fixed` retry: open raises -50 → retry succeeds; all retries fail →
      contention error surfaced (not a crash); non-contention error not retried;
      silent+status capture treated as contention; healthy+stray-status returned OK.
- [x] `is_device_contention_error` classification (markers + PortAudioError instance).
- [x] Message distinguishes contention ("busy") vs permission vs no-device, for both
      `_run_mic_check_server` and `_run_mic_record_replay`.
- [x] Config defaults / validation / env parsing for the two knobs.

## Verify

- [x] Core-only (`uv sync`): ruff/mypy clean; pytest green (no regression).
- [x] `--extra all`: pytest green.
