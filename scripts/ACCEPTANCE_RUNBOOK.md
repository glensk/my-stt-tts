# Phase-8 acceptance run — runbook (A1–A12)

Attended acceptance of the Claude bridge, `PLAN_claude-bridge.md` §9 Phase 8 (tp#855).
Albert speaks; the orchestrator (a Claude tab) runs every command below from the repo root
and records ✅/❌ per line in the plan. All Claude-side typing goes into the scratch
sessions `scratch-idle`, `scratch-busy` and `scratch-dead` in `/tmp/bridge-scratch`
(cpriv account, model sonnet). None of them is a real work session.

Order: **setup → doctor → daemon → A1 … A12 → cleanup.** Run A9 before A10 (A10
replaces scratch-idle's last reply), and A12 last before cleanup.

Every check is one call of `scripts/acceptance_setup.sh -q NAME`. With `-x REGEX` it
prints a ✅/❌ verdict; with `-w SECONDS` it also polls until the value matches. The
polling clock starts when the command starts, so the orchestrator starts it and then
tells Albert "go". Time limits below count from the end of Albert's sentence: when the
✅ line's elapsed time minus the length of the sentence is above the limit, the line
fails.

```commands
cd /Users/albert/obsidian/42-Git/infra/my-stt-tts
A=scripts/acceptance_setup.sh   # tables below write $A; each Bash call is a fresh shell, so set it per call
```

## 0. Setup (≈ 3 min)

```commands
scripts/acceptance_setup.sh -n                 # dry run: lists every step, reads volume + brightness
scripts/acceptance_setup.sh   # new-session-ok
eval "$(scripts/acceptance_setup.sh -E)"       # IDLE/BUSY/DEAD (+ _SID, _TX), VOL_BEFORE, BRIGHT_BEFORE
```

What it does (see `-h`): creates `/tmp/bridge-scratch/delete-me.txt`; opens three iTerm
tabs; seeds `scratch-idle` so its last reply ends with

```text
## To-do list
1. You [decision]: a) keep the scratch file or b) delete it (Recommended) — it is a leftover test file nobody needs
```

and checks that `ccc inspect` parses it as a two-option `todo_line` decision with a
recommendation (on ❌: `scripts/acceptance_setup.sh -R`); leaves `scratch-busy` idle; kills
`scratch-dead`'s claude process (SIGTERM, SIGKILL after 5 s); records output volume, mute and
built-in brightness in `~/.local/state/mac-voice/acceptance-state.json` (0600); sets the
output volume to 40; re-focuses the orchestrator's tab; prints the `.env` flag check.

Albert sees three new tabs open and take focus one after the other; at the end the
orchestrator's tab comes back to the front. If any tab shows a dialog, Albert answers it
once.

Check right after setup:

```commands
scripts/acceptance_setup.sh -q sessions          # scratch-idle + scratch-busy interactive idle
scripts/acceptance_setup.sh -q dead              # "pid N dead" + ccc's row for scratch-dead (if any)
scripts/acceptance_setup.sh -q before            # the recorded volume / brightness
```

Note what `-q dead` shows for ccc's row of scratch-dead (listed with a dead pid, or gone):
A12 needs it to stay addressable by name. If ccc no longer lists it, A12 tests the
"unknown session" refusal instead of a failed send — write that down in the plan.

## 1. Doctor and daemon

The `.env` in the repo root must hold (the setup prints ✅/❌ per flag):

```text
MAC_VOICE_CLAUDE_BRIDGE=1
MAC_VOICE_MAC_CONTROL=1
MAC_VOICE_OPERATOR=1
MAC_VOICE_AUTHORIZED=albert
```

Albert opens a new iTerm tab himself (the daemon owns the mic there) and runs:

```commands
cd /Users/albert/obsidian/42-Git/infra/my-stt-tts
./mac-voice -D      # doctor: every line ✅ — fix the ❌ ones (hints printed) before going on
./mac-voice -d      # the daemon
```

Pass: the doctor ends with `✅ Mac control ready`; the daemon starts without ❌ lines.
Voice on = "voice on" / "hey jarvis", or `./mac-voice -n` (the orchestrator may run it);
voice off = "voice off", or `./mac-voice -f`.

Safari must be open with at least one window for A1–A4 and A7 (the `-q url/host/…`
checks read the current tab of the front window).

## 2. Acceptance lines

"Silence" in a pass criterion means: successes are not spoken — the agent says nothing,
or at most a brief acknowledgement, never a confirmation code. Refusals and failures are
always spoken.

### A1 — open YouTube

Albert says: **"open YouTube"**

```commands
scripts/acceptance_setup.sh -q host -w 15 -x '^www\.youtube\.com$'
```

Pass: ✅ within 5 s of the end of the sentence; silence.

### A2 — play the first video

Albert says: **"play the first video"**

```commands
scripts/acceptance_setup.sh -q paused -w 20 -x '^false$'
scripts/acceptance_setup.sh -q vid               # note the id: VID1
```

Pass: `paused = false` within 10 s; silence.

### A3 — media controls (one sentence at a time, check after each)

| #  | Albert says            | Check                                               | Pass                                             |
| :- | :--------------------- | :-------------------------------------------------- | :----------------------------------------------- |
| a  | **"pause"**            | `$A -q paused -w 10 -x '^true$'`                    | `true`                                           |
| b  | **"skip ten seconds"** | before: `$A -q time` (T0); after: `$A -q time` (T1) | T1 − T0 = 10 ± 2 (paused, so the clock is still) |
| c  | **"next"**             | `$A -q vid` (VID2)                                  | VID2 ≠ VID1 (and not empty)                      |
| d  | **"previous"**         | `$A -q vid -w 10 -x "^${VID1}\$"`                   | back on VID1                                     |
| e  | **"fullscreen"**       | `$A -q fullscreen -w 10 -x '^true$'`                | `true` (`document.fullscreenElement` set)        |

All silent. After e, Albert presses Esc to leave fullscreen (or says "exit fullscreen").

### A4 — open jellyfin

Albert says: **"open jellyfin"**

```commands
scripts/acceptance_setup.sh -q host -w 15 -x '^jellyfin\.dom42\.space$'
```

Pass: front tab host `jellyfin.dom42.space`; silence.

### A5 — volume and brightness

The setup left the volume at 40.

| Albert says     | Check                            | Pass                                                                                               |
| :-------------- | :------------------------------- | :------------------------------------------------------------------------------------------------- |
| **"volume 30"** | first command below              | 30 (macOS quantises, ± 3 tolerated)                                                                |
| **"louder"**    | `$A -q volume`                   | a value > the one just read                                                                        |
| **"brighter"**  | `$A -q brighter -w 10 -x '^yes'` | brightness above the recorded value, or the recorded value already 1.0 (`-q brighter` checks both) |

```commands
scripts/acceptance_setup.sh -q volume -w 10 -x '^(2[7-9]|3[0-3])$'
scripts/acceptance_setup.sh -q volume
scripts/acceptance_setup.sh -q brighter -w 10 -x '^yes'
```

All silent.

### A6 — open Calculator

Albert says: **"open Calculator"**

```commands
scripts/acceptance_setup.sh -q front -w 10 -x '^Calculator$'
```

Pass: Calculator frontmost; silence.

### A7 — Focus, closing tabs, deleting the file

1. Albert says: **"turn on Do Not Disturb"**

   ```commands
   scripts/acceptance_setup.sh -q focus -w 10 -x '^on$'
   ```

   Pass: Focus on, silence. When `-q focus` prints `unknown` (the orchestrator's terminal has
   no Full Disk Access), Albert confirms the Focus icon in the menu bar instead. Albert turns
   it off again afterwards.

2. Precondition: the front Safari window has at least 2 tabs (`$A -q tabs`; if not, the
   orchestrator opens one more: `osascript -e 'tell application "Safari" to tell front window to make new tab with properties {URL:"https://example.org"}'`).
   Albert says: **"close the other Safari tabs"** → the agent speaks a summary and a
   two-digit code; Albert says **"confirm <code>"** (the code he heard).

   ```commands
   scripts/acceptance_setup.sh -q tabs -w 15 -x '^1 in front window'
   ```

   Pass: spoken summary + code before anything closes (check `$A -q tabs` once before the
   confirm: unchanged), then exactly one tab left in the front window.

3. Albert says: **"delete the file delete-me"** → the agent asks for a code; Albert says
   **"confirm <code>"**.

   ```commands
   scripts/acceptance_setup.sh -q file                # before the confirm: exists
   scripts/acceptance_setup.sh -q file -w 15 -x '^gone$'
   ```

   Pass: code requested, the file still exists until the confirm, gone after it.

### A8 — which sessions are running

Albert says: **"which sessions are running"**

```commands
scripts/acceptance_setup.sh -q sessions
```

Pass: the spoken list contains scratch-idle and scratch-busy, and every session that
`-q sessions` shows as `background` is marked as background in the speech (D4: those are
list/read only). The orchestrator compares names one by one.

### A9 — what does scratch-idle need

Albert says: **"what does scratch-idle need"**

```commands
scripts/acceptance_setup.sh -q decision
```

Pass: a spoken briefing in the D5 shape — "The session scratch-idle, which works on …,
needs a decision: keep the scratch file or delete it? Options: a) keep the scratch file,
b) delete it. It recommends b) because …" — whose question, both options and the
recommendation match `.decision` in the `-q decision` output (`source: todo_line`). No
code is asked for (reading is not a mutation). Albert does not answer it; A10 moves on.

### A10 — sending and answering by voice

1. Idle send. Albert says: **"tell scratch-idle: print hello"** → the agent repeats the
   proposal and gives a code; Albert says **"confirm <code>"**.

   ```commands
   scripts/acceptance_setup.sh -q prompts -w 20 -x 'print hello'
   ```

   Pass: the prompt "print hello" is in scratch-idle's transcript (a `user` record); the
   send itself is silent.

2. Busy send. Right before it, the orchestrator makes scratch-busy busy:

   ```commands
   scripts/acceptance_setup.sh -b        # sends "Run `sleep 60` in bash, then say done.", waits for busy
   ```

   Then within 60 s Albert says: **"tell scratch-busy: print hello"** → code →
   **"confirm <code>"**.

   ```commands
   scripts/acceptance_setup.sh -q enqueued -w 20 -x 'print hello'
   ```

   Pass: a `queue-operation/enqueue` record with "print hello" in scratch-busy's transcript
   (it is answered after the sleep).

3. Picker. Wait until scratch-idle has answered "print hello" (`$A -q sessions` shows it
   idle), then the orchestrator makes it call AskUserQuestion:

   ```commands
   scripts/acceptance_setup.sh -p        # waits until the picker is pending, prints the question
   ```

   The prompt it sends:

   ```text
   Use the AskUserQuestion tool to ask me exactly one single-select question: which colour the test banner should be, with the options red, green, blue. Ask nothing else and do not call any other tool; after my answer reply with one short sentence.
   ```

   Albert says: **"what does scratch-idle need"** (briefing: the colour question, options
   red, green, blue) → **"answer green"** → code → **"confirm <code>"**.

   ```commands
   scripts/acceptance_setup.sh -q answers -w 20 -x '"green"'
   ```

   Pass: `toolUseResult.answers` of the call maps the question to `green` (the label Albert
   chose); the orchestrator's tab stayed in front the whole time. If the picker gets stuck,
   Albert presses Esc in scratch-idle's tab (cancels it) and the step is repeated from `-p`.

### A11 — other voices are refused

Needs a second person, or a TV / phone playing someone else saying "louder" (not a
recording of Albert — replaying Albert is the documented residual risk, not tested).

```commands
scripts/acceptance_setup.sh -q volume            # V0
```

1. The other person says: **"louder"**. Then the TV/phone plays **"louder"**.

```commands
scripts/acceptance_setup.sh -q volume            # still V0 after each
```

Pass: both are refused — spoken refusal, or ignored with no volume change — and the volume
reads V0 after each attempt. The conversation has to be on for this (Albert starts it,
then stays silent).

### A12 — failure on a dead session, attention inbox

1. In a call. Albert says: **"tell scratch-dead: hi"** → code → **"confirm <code>"**.

   ```commands
   scripts/acceptance_setup.sh -q problems          # a row with subject scratch-dead
   scripts/acceptance_setup.sh -q events            # ccc events of scratch-dead (delivery_failed / stop_failure), if any
   ```

   Pass: the failure is spoken in the call (a `[system notice]` or the tool result), and
   `attention.db` has an open problem for scratch-dead.

2. Voice off. Albert says **"tell scratch-dead: hi again"** → code → **"confirm <code>"**,
   then at once **"voice off"** (or the orchestrator runs `./mac-voice -f`). He does NOT say
   "got it".

   ```commands
   scripts/acceptance_setup.sh -q problems          # scratch-dead open: briefed=- acked=-
   ```

3. Next voice on. Albert says **"voice on"** (or `./mac-voice -n`).

   Pass: the agent's FIRST message is a briefing that names scratch-dead and the failure;
   afterwards `-q problems` shows `briefed=<time>` and still `acked=-`.

4. Albert says: **"got it, scratch-dead"**.

   ```commands
   scripts/acceptance_setup.sh -q problems -w 15 -x 'scratch-dead.*acked=[0-9]'
   ```

   Pass: the scratch-dead item shows `acked=<time>`; the next "voice on" no longer briefs it.

## 3. Cleanup

No picker may be pending in scratch-idle (the cleanup sends Esc first anyway).

```commands
scripts/acceptance_setup.sh -C -n     # what it will undo
scripts/acceptance_setup.sh -C        # closes the three tabs (Esc + /exit, then close), restores
                                      # volume, mute and brightness, removes /tmp/bridge-scratch
                                      # and the state file
```

Albert stops the daemon (Ctrl-C in its tab) and turns Focus off if A7 left it on. The
trust entries for `/tmp/bridge-scratch` in `~/.claude.json` stay (harmless). The attention
inbox is not touched — the acknowledged scratch-dead items expire after 7 days.

## 4. Record

Tick A1–A12 in `PLAN_claude-bridge.md` §9 Phase 8 with the measured values (latencies,
volumes, brightness before/after, video ids, the answers JSON, the problem rows) and any
deviation; a ❌ line gets the observed behaviour and the fix it needs.
