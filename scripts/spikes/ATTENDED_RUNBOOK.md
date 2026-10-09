# Attended spikes S-SEND and S-PICKER — runbook

Phase 0 of `PLAN_claude-bridge.md` (tp#855), §6 `send` / `answer`, §9 thresholds. The
orchestrator runs every command below from its own Claude tab while Albert watches. Nothing
here touches a real session: all typing goes into the scratch sessions `scratch-idle` and
`scratch-busy` in `/tmp/bridge-scratch` (cpriv account, model sonnet).

```commands
PY=/Users/albert/obsidian/42-Git/llms/claude-command-center/.venv/bin/python
SP=/Users/albert/obsidian/42-Git/infra/my-stt-tts/scripts/spikes
```

Every script checks before it types that the iTerm tab `-s` belongs to the named session
(the tab's tty equals the tty of the session's pid) and refuses otherwise. Results are
written to `tests/fixtures/spikes/` (`s_send_idle.json`, `s_send_busy.json`,
`s_picker.json`).

## 0. Setup (≈ 1 min)

```commands
$SP/scratch_setup.sh -n                 # dry run: shows the three actions
$SP/scratch_setup.sh    # new-session-ok   -> ITERM_SESSION=<uuid A>
$SP/scratch_setup.sh -b # new-session-ok   -> ITERM_SESSION=<uuid B>
IDLE=<uuid A>; BUSY=<uuid B>
```

What Albert sees: two new iTerm tabs in the current window, each starting Claude Code in
`/tmp/bridge-scratch` with no trust dialog (`/rename`-free: the names come from `-n`).
Each new tab takes focus when it opens — **after setup, click back to the orchestrator's
tab and keep it in front for the rest of the run.** If a tab shows a trust or other
dialog, Albert answers it once; the spike continues after that.

Check: `claude agents --json` lists `scratch-idle` and `scratch-busy` as `idle`.

## 1. S-SEND idle

```commands
$PY $SP/s_send.py -m idle -N scratch-idle -s $IDLE -n     # resolve + validate only
$PY $SP/s_send.py -m idle -N scratch-idle -s $IDLE
```

Albert sees (in the background tab, visible when he switches to it afterwards — he
should NOT switch during the run): `s-send idle <timestamp> - reply with just: ok` as a
submitted prompt, and the reply `ok`.

**Pass**: `✅ test text sent via python-api` and `✅ accepted: user after <t> s` with
t ≤ 8 s. A channel of `applescript` is a partial pass (it works, but the Python-API rung
did not; note why).

## 2. S-SEND busy

```commands
$PY $SP/s_send.py -m busy -N scratch-busy -s $BUSY
```

The script first sends ``Run `sleep 40` in bash, then say done.``, waits until the
session reports `busy` (the project settings allow `Bash(sleep:*)`, so no permission
prompt), then sends the test text while the `sleep` runs.

Albert sees in scratch-busy: the sleep running, the test text shown as a queued prompt
under the spinner, and after ~40 s `done` followed by the queued prompt being answered.

**Pass**: `✅ accepted: queue-operation/enqueue after <t> s`, t ≤ 8 s
(`attachment/queued_command` within 8 s also passes but is unexpected: it is normally
written only when the queue drains, i.e. after the sleep). **Fail** → per the plan, busy
sends stay refused in `ccc send`.

## 3. S-PICKER — discover, then answer

The picker protocol below is the hypothesis from reading Claude Code 2.1.295's bundled UI
code; step 3a confirms it on screen before anything is answered. Keys go through
`Session.async_send_text` on the iTerm2 Python API — no AppleScript, no focus change —
so the orchestrator's tab stays in front the whole time. That is the pass criterion
"without switching tabs".

Hypothesis (from the bundled source):

| Key in picker                    | Single-select question                         | Multi-select question                                              |
| :------------------------------- | :--------------------------------------------- | :----------------------------------------------------------------- |
| `up` / `down`                    | move focus (options, then "Type something")    | move focus; `down` from the last item ("Type something") focuses the **Submit/Next** button; one more `down` focuses "Chat about this" (avoid) |
| `enter`                          | pick the focused option, advance               | toggle the focused option; on the Submit/Next button: submit        |
| `space`                          | —                                              | toggle the focused option                                           |
| digit `1`–`9`                    | pick option N (unverified)                     | toggle option N **without moving focus**                           |
| digit `N+2` (N = option count)   | "Chat about this" (leaves the picker — avoid)   | same                                                                |
| `left` / `right` / `tab` / `btab` | switch question tab (multi-question calls)      | `tab` is ALSO the multi-select's own "next item" key — ambiguous, do not use |
| `esc`                            | cancel the whole picker (rejected tool_result) | same                                                                |
| after the last question          | "Review your answers" tab: `enter` = **Submit answers** | same                                                       |

Deterministic multi-select submit (the plan's condition for supporting multi-select):
toggle with digits (focus stays on option 1), then exactly `down` × (option count + 1),
then `enter`. Overshooting by one lands on "Chat about this", so the count must be exact.

### 3a. Single question, single-select

```commands
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -p -q single -c
```

Prints the pending question (`Q0 [single] … 1=red, 2=green, 3=blue`) and the screen.
Then answer "green":

```commands
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -k "down,enter" -e '{"0": "green"}' -c
```

If no tool_result arrives, the screen dump shows what is open (e.g. a review tab) —
send the next key alone: `-k enter -e '{"0": "green"}' -c`.

**Pass**: `✅ answers match -e`, `toolUseResult.answers = {"<question>": "green"}`.

### 3b. Single question, multi-select

```commands
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -p -q multi -c
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -k "1,3" -c                  # toggles only, no submit
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -k "down,down,down,down,enter" -e '{"0": ["apple", "plum"]}' -c
```

The second command ends with `❌ no tool_result` — expected; its screen dump must show
apple and plum ticked. **Pass**: answers `"apple, plum"` (order = toggle order; the check
ignores order). Record which key submitted: this is the "deterministic submit key" the
plan asks for. If `down`×4 + `enter` misses, step one `down` at a time with `-c` and note
the count.

### 3c. Two questions in one call (the plan's "3 options + one multi-select")

```commands
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -p -q both -c
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -k "down,enter,wait:0.5,1,3,down,down,down,down,enter,wait:0.5,enter" \
    -e '{"0": "green", "1": ["apple", "plum"]}' -c
```

Keys: Q0 `down,enter` = green and advance; Q1 toggles 1+3, `down`×4 → "Submit", `enter`
→ review tab, `enter` → "Submit answers". If it stalls, re-run in pieces with `-c`
between them (the script finds the still-pending call each time).

**Pass**: answers `{"<colour q>": "green", "<fruit q>": "apple, plum"}` with
`✅ answers match -e`, and Albert confirms the orchestrator's tab never lost focus.

### Reset when a picker gets stuck

```commands
$PY $SP/s_picker.py -s $IDLE -N scratch-idle -k esc -c     # cancels the pending picker
```

## 4. Cleanup

No picker may be pending (else `/exit` lands in the picker — send `esc` first as above).

```commands
$SP/scratch_setup.sh -C $IDLE      # types /exit, waits 3 s, closes the tab
$SP/scratch_setup.sh -C $BUSY
```

Optional: `rm -rf /tmp/bridge-scratch` (the trust entries in `~/.claude.json` are harmless;
the backup is `~/.claude.json.bak-bridge-<timestamp>`).

## 5. Record

Tick S-SEND / S-PICKER in `PLAN_claude-bridge.md` §9 with: channel, latencies, record
types, the exact key recipes that worked (single-select, multi-select submit, review tab),
and whether digits pick in single-select. These recipes become `ccc answer`'s protocol;
without a deterministic multi-select submit, `answer` returns `unsupported_shape` for it.
