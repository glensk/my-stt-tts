#!/usr/bin/env bash
# Phase-8 acceptance fixtures for the Claude bridge (PLAN_claude-bridge.md §9 Phase 8,
# A1–A12). Runbook: scripts/ACCEPTANCE_RUNBOOK.md. Opens iTerm2 tabs with scratch
# Claude Code sessions on purpose (attended run with Albert) — new-session-ok.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPIKES="$REPO/scripts/spikes"
SCRATCH_SETUP="$SPIKES/scratch_setup.sh"
SCRATCH=/tmp/bridge-scratch
DELETE_ME="$SCRATCH/delete-me.txt"
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
CCC_REPO="${CCC_REPO:-$HOME/obsidian/42-Git/llms/claude-command-center}"
CCC_PY="$CCC_REPO/.venv/bin/python"
CCC_BIN="${CCC_BIN:-ccc}"
STATE_DIR="$HOME/.local/state/mac-voice"
STATE="$STATE_DIR/acceptance-state.json"
ATTENTION_DB="$STATE_DIR/attention.db"
DND_DB="$HOME/Library/DoNotDisturb/DB/Assertions.json"
NAMES=(scratch-idle scratch-busy scratch-dead)
MODEL=sonnet
VOLUME=40
TIMEOUT=180
ACTION=setup
DRY=0
QUERY=""
WAIT=0
EXPECT=""

# read -d '' (not $(cat <<EOF)): bash 3.2 cannot parse unbalanced ')' in a heredoc inside $( ).
IFS= read -r -d '' SEED_PROMPT <<'EOF' || true
This is a scratch session for a voice-assistant acceptance test. Do not call any tool.
Reply with one short sentence saying that /tmp/bridge-scratch/delete-me.txt is a leftover test file.
Then end your reply with exactly the following two lines, copied verbatim as plain text (no code block, no quotes), with nothing after them:

## To-do list
1. You [decision]: a) keep the scratch file or b) delete it (Recommended) — it is a leftover test file nobody needs
EOF
SEED_PROMPT=${SEED_PROMPT%$'\n'}
# shellcheck disable=SC2016  # literal backticks for the model
BUSY_PROMPT='Run `sleep 60` in bash, then say done.'
PICKER_PROMPT='Use the AskUserQuestion tool to ask me exactly one single-select question: which colour the test banner should be, with the options red, green, blue. Ask nothing else and do not call any other tool; after my answer reply with one short sentence.'

usage() {
	cat <<'EOF'
Usage: acceptance_setup.sh [-n] [-m MODEL] [-v VOLUME] [-T SECONDS]   set up (default)
       acceptance_setup.sh -C [-n]                                     undo everything
       acceptance_setup.sh -b | -p | -R                                in-run helpers
       acceptance_setup.sh -E                                          print exports
       acceptance_setup.sh -q NAME [-w SECONDS] [-x REGEX]             one check

Fixtures for the attended Phase-8 acceptance run (PLAN_claude-bridge.md §9, runbook
scripts/ACCEPTANCE_RUNBOOK.md). Setup:
  1. /tmp/bridge-scratch + delete-me.txt (trust pre-seeded via spikes/scratch_setup.sh)
  2. iTerm tab scratch-idle (claude -n scratch-idle --model MODEL), seeded so its last
     reply ends with a "## To-do list" holding one "You [decision]: a) … or b) …" line
  3. iTerm tab scratch-busy (idle; -b makes it busy right before A10's busy step)
  4. iTerm tab scratch-dead: registered, then its claude process is killed (SIGTERM,
     then SIGKILL) so a later send fails (A12)
  5. output volume + muted + built-in brightness recorded in
     ~/.local/state/mac-voice/acceptance-state.json (0600), then volume set to VOLUME
  6. prints the daemon's .env flags and how to start it
Refuses while the state file exists (run -C first) or a scratch-* session is live.

Options:
  -n, --dry-run          print what would be done; read-only checks only
  -C, --cleanup          close every tab/session from the state file, restore volume,
                         mute and brightness, remove /tmp/bridge-scratch and the state file
  -b, --make-busy        send scratch-busy `sleep 60` and wait until it is busy (A10)
  -p, --picker           make scratch-idle call AskUserQuestion (red/green/blue) and wait
                         until the picker is pending (A10)
  -R, --reseed           re-send the to-do-line seed prompt to scratch-idle (A9 retry)
  -E, --exports          print `export IDLE=… IDLE_SID=… …` from the state file
  -q, --query NAME       print one live value (see below)
  -w, --wait SECONDS     with -q: poll until the value matches -x (default: read once)
  -x, --expect REGEX     with -q: ✅/❌ verdict against this extended regex
  -m, --model MODEL      model of the scratch sessions (default sonnet)
  -v, --volume N         output volume set at setup (default 40)
  -T, --timeout SECONDS  per-session wait for registration / the seed reply (default 180)
  -h, --help             this help

Query names (-q):
  url host paused time vid fullscreen   Safari, current tab of the front window
  volume muted bright brighter front     Mac state (brighter: yes/no vs the recorded value)
  tabs focus file                        Safari tab count, Focus on/off, delete-me exists/gone
  sessions decision answers prompts      ccc sessions; scratch-idle inspect / picker answers /
  enqueued dead events problems          user prompts; scratch-busy enqueues; scratch-dead
                                         pid + row; its ccc events; attention.db problems
  before                                 the recorded volume / brightness

Examples:
  scripts/acceptance_setup.sh -n
  scripts/acceptance_setup.sh            # new-session-ok
  eval "$(scripts/acceptance_setup.sh -E)"
  scripts/acceptance_setup.sh -q host -w 15 -x '^www\.youtube\.com$'
  scripts/acceptance_setup.sh -q brighter -x '^yes'
  scripts/acceptance_setup.sh -b
  scripts/acceptance_setup.sh -C
EOF
}

ok() { printf '\xe2\x9c\x85 %s\n' "$*"; }
fail() { printf '\xe2\x9d\x8c %s\n' "$*" >&2; }
die() {
	fail "$*"
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	-n | --dry-run) DRY=1 ;;
	-C | --cleanup) ACTION=cleanup ;;
	-b | --make-busy) ACTION=busy ;;
	-p | --picker) ACTION=picker ;;
	-R | --reseed) ACTION=reseed ;;
	-E | --exports) ACTION=exports ;;
	-q | --query)
		ACTION=query
		QUERY="${2:?-q needs a query name}"
		shift
		;;
	-w | --wait)
		WAIT="${2:?-w needs seconds}"
		shift
		;;
	-x | --expect)
		EXPECT="${2:?-x needs a regex}"
		shift
		;;
	-m | --model)
		MODEL="${2:?-m needs a model}"
		shift
		;;
	-v | --volume)
		VOLUME="${2:?-v needs a volume 0-100}"
		shift
		;;
	-T | --timeout)
		TIMEOUT="${2:?-T needs seconds}"
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		fail "unknown option: $1"
		usage >&2
		exit 2
		;;
	esac
	shift
done

[[ "$VOLUME" =~ ^[0-9]+$ && "$VOLUME" -le 100 ]] || {
	fail "-v must be 0-100"
	exit 2
}
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || {
	fail "-T must be whole seconds"
	exit 2
}
QUERIES=" url host paused time vid fullscreen volume muted bright brighter front tabs focus file sessions decision answers prompts enqueued dead events problems before "
if [[ "$ACTION" == query && "$QUERIES" != *" $QUERY "* ]]; then
	fail "unknown query: $QUERY (one of:$QUERIES)"
	exit 2
fi
[[ "$WAIT" =~ ^[0-9]+$ ]] || {
	fail "-w must be whole seconds"
	exit 2
}

# --------------------------------------------------------------------------- helpers

# Python helper under ccc's venv (iterm2 + command_center); status lines on stderr,
# results as JSON on stdout. Reuses scripts/spikes/bridge_spike_lib.py and s_bright.py.
helper() {
	CCC_REPO="$CCC_REPO" SPIKES="$SPIKES" "$CCC_PY" - "$@" <<'PY'
import json
import os
import re
import signal
import sys
import time

sys.path.insert(0, os.environ["SPIKES"])
sys.path.insert(0, os.environ["CCC_REPO"])
import bridge_spike_lib as lib  # noqa: E402


def say(good, msg):
    print(("✅ " if good else "❌ ") + msg, file=sys.stderr, flush=True)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve(sid):
    sess = lib.find_session(session_id=sid)
    if sess is None:
        say(False, f"session {sid} not found in claude agents")
    return sess


def validated(iterm, sess):
    good, why = lib.validate_tab(iterm, sess)
    say(good, f"{sess.name}: tab validation ({why})")
    return good


def names_free(names):
    live = [s for s in lib.list_sessions() if s.kind == "interactive" and s.name in names]
    for s in live:
        say(False, f"live session named {s.name} ({s.session_id}, pid {s.pid}) — close it first")
    if not live:
        say(True, f"no live session named {', '.join(names)}")
    return 0 if not live else 1


def ccc_name(sid, name):
    # ccc keeps `claude -n` as an observation, never as its own name, so the voice
    # bridge would not find the session by NAME without this
    import subprocess
    ccc = os.environ.get("CCC_BIN", "ccc")
    for _ in range(10):  # the SessionStart hook may not have written ccc's row yet
        res = subprocess.run([ccc, "name", "-s", sid, name], capture_output=True, text=True)
        if res.returncode == 0:
            say(True, f"{name}: named in ccc")
            return True
        holder = re.search(r"already used by session (\w+)", res.stdout + res.stderr)
        if holder:  # a scratch session of an earlier run still holds it: move it aside
            old = holder.group(1)
            moved = subprocess.run(
                [ccc, "name", "-s", old, f"{name}-{old}"], capture_output=True, text=True
            )
            say(moved.returncode == 0, f"{name}: freed the name from earlier session {old}")
            continue
        time.sleep(1.0)
    say(False, f"{name}: ccc name failed: {(res.stderr or res.stdout).strip()[-160:]}")
    return False


def wait_session(name, iterm, timeout):
    end = time.monotonic() + timeout
    sess = None
    while time.monotonic() < end:
        try:
            sess = lib.find_session(name=name)
        except Exception:  # noqa: BLE001  # claude agents hiccup while the tab starts
            sess = None
        if sess is not None and sess.pid:
            break
        time.sleep(1.0)
    if sess is None or not sess.pid:
        say(False, f"{name}: not registered within {timeout:.0f} s")
        return 1
    if not validated(iterm, sess):
        return 1
    idle = lib.wait_until(
        lambda: lib.session_status(sess.session_id) == "idle",
        max(10.0, end - time.monotonic()),
        1.0,
    )
    status = "idle" if idle else lib.session_status(sess.session_id)
    say(bool(idle), f"{name}: registered as {sess.session_id} (pid {sess.pid}), {status}")
    if not idle:
        return 1
    if not ccc_name(sess.session_id, name):
        return 1
    transcript = lib.transcript_for(sess.session_id) or lib.expected_transcript(sess)
    print(json.dumps({
        "session_id": sess.session_id,
        "pid": sess.pid,
        "tty": lib.pid_tty(sess.pid),
        "cwd": sess.cwd,
        "transcript": str(transcript),
    }))
    return 0


def _end_turn(rec):
    msg = rec.get("message")
    return (
        rec.get("type") == "assistant"
        and isinstance(msg, dict)
        and msg.get("stop_reason") == "end_turn"
        and not rec.get("isSidechain")
    )


def _ask_call(rec):
    msg = rec.get("message")
    if rec.get("type") != "assistant" or not isinstance(msg, dict):
        return None
    for block in msg.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "AskUserQuestion":
            return block
    return None


def prompt(mode, iterm, sid, text, timeout):
    sess = resolve(sid)
    if sess is None or not validated(iterm, sess):
        return 1
    status = lib.session_status(sid)
    if status != "idle":
        say(False, f"{sess.name} is {status}; this step needs it idle")
        return 1
    transcript = lib.transcript_for(sid) or lib.expected_transcript(sess)
    anchor = lib.Anchor.take(transcript)
    channel = lib.send_text(iterm, text)
    say(bool(channel), f"{sess.name}: prompt sent via {channel or 'nothing'}")
    if not channel:
        return 1
    if mode == "busy":
        busy = lib.wait_until(lambda: lib.session_status(sid) == "busy", timeout, 0.5)
        say(bool(busy), f"{sess.name}: {'busy' if busy else 'never became busy'}")
        return 0 if busy else 1
    if mode == "ask":
        def pending():
            for rec in anchor.appended():
                call = _ask_call(rec)
                if call is not None:
                    return call
            return None
        call = lib.wait_until(pending, timeout, 0.5)
        if not call:
            say(False, f"{sess.name}: no AskUserQuestion call within {timeout:.0f} s")
            return 1
        say(True, f"{sess.name}: AskUserQuestion pending (tool_use {call.get('id')})")
        print(json.dumps((call.get("input") or {}).get("questions"), ensure_ascii=False))
        return 0
    end = time.monotonic() + timeout
    if not lib.wait_until(lambda: any(_end_turn(r) for r in anchor.appended()), timeout, 0.5):
        say(False, f"{sess.name}: no end of turn within {timeout:.0f} s")
        return 1
    stable = None  # idle 3 s in a row: a Stop hook may continue the turn
    while time.monotonic() < end:
        if lib.session_status(sid) == "idle":
            stable = stable or time.monotonic()
            if time.monotonic() - stable >= 3.0:
                say(True, f"{sess.name}: reply finished, idle")
                print(json.dumps({"transcript": str(transcript)}))
                return 0
        else:
            stable = None
        time.sleep(0.5)
    say(False, f"{sess.name}: not idle within {timeout:.0f} s")
    return 1


def kill(iterm, sid):
    sess = resolve(sid)
    if sess is None or not validated(iterm, sess):
        return 1
    pid = int(sess.pid)
    sig = "SIGTERM"
    os.kill(pid, signal.SIGTERM)
    dead = lib.wait_until(lambda: not alive(pid), 5.0, 0.2)
    if not dead:
        sig = "SIGKILL"
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        dead = lib.wait_until(lambda: not alive(pid), 3.0, 0.2)
    say(bool(dead), f"{sess.name}: claude pid {pid} {'gone' if dead else 'STILL ALIVE'} after {sig}")
    print(json.dumps({"killed_pid": pid, "signal": sig, "killed_at": lib.now_iso()}))
    return 0 if dead else 1


def close(name, iterm, pid, tty):
    pid = int(pid or 0)
    ours = alive(pid) and bool(tty) and lib.pid_tty(pid) == tty
    if ours:
        try:
            lib.send_keys(iterm, [("esc", lib.KEYS["esc"])])  # cancels a pending picker
            time.sleep(1.0)
            channel = lib.send_text(iterm, "/exit")
            say(bool(channel), f"{name}: /exit sent via {channel or 'nothing'}")
        except Exception as exc:  # noqa: BLE001  # still close the tab below
            say(False, f"{name}: could not type /exit ({exc})")
        lib.wait_until(lambda: not alive(pid), 8.0, 0.3)
    from command_center import terminal  # ccc venv only

    closed = terminal.close_iterm_session(iterm)
    say(bool(closed), f"{name}: tab {'closed' if closed else 'not found (already closed?)'}")
    if ours and alive(pid):
        os.kill(pid, signal.SIGTERM)
        if not lib.wait_until(lambda: not alive(pid), 3.0, 0.2):
            os.kill(pid, signal.SIGKILL)
    gone = not (ours and alive(pid))
    say(gone, f"{name}: claude pid {pid or '-'} {'gone' if gone else 'STILL ALIVE'}")
    return 0 if gone else 1


def bright_get():
    import s_bright

    disp = s_bright.builtin_display()
    print(json.dumps({"display": disp, "brightness": s_bright.read(disp) if disp else None}))
    return 0


def bright_set(value):
    import ctypes

    import s_bright

    disp = s_bright.builtin_display()
    if disp is None:
        say(False, "no built-in display")
        return 1
    s_bright.DS.DisplayServicesSetBrightness(disp, ctypes.c_float(float(value)))
    time.sleep(0.5)
    now = s_bright.read(disp)
    good = now is not None and abs(now - float(value)) < 0.02
    say(good, f"brightness restored to {now} (wanted {float(value):.4f})")
    return 0 if good else 1


def main(argv):
    cmd, args = argv[0], argv[1:]
    if cmd == "names-free":
        return names_free(args)
    if cmd == "wait-session":
        return wait_session(args[0], args[1], float(args[2]))
    if cmd == "prompt":
        return prompt(args[0], args[1], args[2], args[3], float(args[4]))
    if cmd == "kill":
        return kill(args[0], args[1])
    if cmd == "close":
        return close(args[0], args[1], args[2], args[3])
    if cmd == "bright-get":
        return bright_get()
    if cmd == "bright-set":
        return bright_set(args[0])
    say(False, f"unknown helper command {cmd}")
    return 2


sys.exit(main(sys.argv[1:]))
PY
}

state_get() { # $1 = jq path; prints "" when absent
	[[ -f "$STATE" ]] || return 0
	jq -r "($1) | select(. != null)" "$STATE"
}

state_merge() { # $1 = JSON object, deep-merged into the state file (0600, atomic)
	local tmp
	if [[ ! -d "$STATE_DIR" ]]; then
		mkdir -p "$STATE_DIR"
		chmod 700 "$STATE_DIR"
	fi
	[[ -f "$STATE" ]] || (
		umask 077
		printf '{}\n' >"$STATE"
	)
	tmp=$(
		umask 077
		mktemp "$STATE_DIR/.acceptance-state.XXXXXX"
	)
	jq --argjson p "$1" '. * $p' "$STATE" >"$tmp"
	chmod 600 "$tmp"
	mv -f "$tmp" "$STATE"
}

now() { perl -MTime::HiRes=time -e 'printf "%.2f\n", time'; }

read_volume() { osascript -e 'output volume of (get volume settings)'; }
read_muted() { osascript -e 'output muted of (get volume settings)'; }
read_bright() { helper bright-get | jq -r '.brightness // "null"'; }

safari_js() {
	osascript -e 'on run argv' \
		-e 'tell application "Safari" to do JavaScript (item 1 of argv) in current tab of front window' \
		-e 'end run' "$1"
}

transcript_of() { # $1 = session name; the live transcript path (falls back to the recorded one)
	local sid f
	sid=$(state_get ".sessions[\"$1\"].session_id")
	[[ -n "$sid" ]] || return 0
	for f in "$HOME"/.claude/projects/*/"$sid".jsonl; do
		[[ -f "$f" ]] && {
			printf '%s\n' "$f"
			return 0
		}
	done
	state_get ".sessions[\"$1\"].transcript"
}

need_state() {
	[[ -f "$STATE" ]] || die "no state file $STATE — run scripts/acceptance_setup.sh first"
}

decision_ok() { # scratch-idle's last reply parses as a todo-line decision with 2 options
	local sid out
	sid=$(state_get '.sessions["scratch-idle"].session_id')
	out=$("$CCC_BIN" inspect -s "$sid" -N -j 2>/dev/null) || true
	if jq -e '.ok and .data.decision.source == "todo_line"
		and ((.data.decision.questions[0].options // []) | length) == 2
		and ((.data.decision.recommendation // "") | length) > 0' >/dev/null 2>&1 <<<"$out"; then
		ok "scratch-idle: ccc inspect sees the todo-line decision: $(jq -c '{q: .data.decision.questions[0].text, options: [.data.decision.questions[0].options[].label], rec: .data.decision.recommendation}' <<<"$out")"
		return 0
	fi
	fail "scratch-idle: ccc inspect shows no 2-option todo-line decision ($(jq -c '{ok, error, decision: .data.decision}' <<<"$out" 2>/dev/null || echo "no JSON")) — retry with -R"
	return 1
}

focus_iterm_session() { # best effort: bring the orchestrator's own tab back to the front
	local uuid="${ITERM_SESSION_ID##*:}"
	[[ -n "$uuid" ]] || return 0
	osascript - "$uuid" <<'APPLESCRIPT' >/dev/null 2>&1 || true
on run argv
	set target to item 1 of argv
	tell application "iTerm2"
		repeat with w in windows
			repeat with t in tabs of w
				repeat with s in sessions of t
					if (id of s) is target then
						select t
						select w
						return "ok"
					end if
				end repeat
			end repeat
		end repeat
	end tell
	return "missing"
end run
APPLESCRIPT
}

print_exports() {
	need_state
	local n var
	printf 'export ACC_STATE=%q\n' "$STATE"
	for n in "${NAMES[@]}"; do
		case "$n" in
		scratch-idle) var=IDLE ;;
		scratch-busy) var=BUSY ;;
		*) var=DEAD ;;
		esac
		printf 'export %s=%q %s_SID=%q %s_TX=%q\n' \
			"$var" "$(state_get ".sessions[\"$n\"].iterm_session")" \
			"$var" "$(state_get ".sessions[\"$n\"].session_id")" \
			"$var" "$(transcript_of "$n")"
	done
	printf 'export VOL_BEFORE=%q BRIGHT_BEFORE=%q\n' \
		"$(state_get .volume_before)" "$(state_get .brightness_before)"
}

print_daemon_help() {
	local key want line
	echo
	echo "Daemon flags (repo .env, $REPO/.env):"
	for key in MAC_VOICE_CLAUDE_BRIDGE MAC_VOICE_MAC_CONTROL MAC_VOICE_OPERATOR MAC_VOICE_AUTHORIZED; do
		case "$key" in
		MAC_VOICE_AUTHORIZED) want=albert ;;
		*) want=1 ;;
		esac
		line=$(grep -E "^${key}=" "$REPO/.env" 2>/dev/null | tail -1 || true)
		if [[ "$line" == "$key=$want" ]]; then
			ok "$key=$want set"
		else
			fail "$key=$want missing (now: ${line:-unset}) — add it to .env"
		fi
	done
	cat <<EOF

Start order (in a NEW iTerm tab you open yourself, so the daemon owns the mic):
  cd $REPO
  ./mac-voice -D          # doctor: every line must be ✅ (fix hints are printed)
  ./mac-voice -d          # the daemon, with the flags above in .env
Then follow scripts/ACCEPTANCE_RUNBOOK.md A1…A12; afterwards: scripts/acceptance_setup.sh -C
EOF
}

# --------------------------------------------------------------------------- queries

query_value() {
	local tx sid out
	case "$QUERY" in
	url) osascript -e 'tell application "Safari" to get URL of current tab of front window' ;;
	host)
		osascript -e 'tell application "Safari" to get URL of current tab of front window' |
			python3 -c 'import sys, urllib.parse; print(urllib.parse.urlsplit(sys.stdin.read().strip()).hostname or "")'
		;;
	paused) safari_js 'var v=document.querySelector("video"); v ? String(v.paused) : "no-video"' ;;
	time) safari_js 'var v=document.querySelector("video"); v ? v.currentTime.toFixed(1) : "no-video"' ;;
	vid) safari_js 'new URLSearchParams(location.search).get("v") || ""' ;;
	fullscreen) safari_js 'String(!!(document.fullscreenElement || document.webkitFullscreenElement))' ;;
	volume) read_volume ;;
	muted) read_muted ;;
	bright) read_bright ;;
	brighter)
		local before cur
		before=$(state_get .brightness_before)
		cur=$(read_bright)
		if [[ -z "$before" || "$before" == null || "$cur" == null ]]; then
			echo "unknown (before ${before:-unrecorded}, now $cur)"
		elif awk -v b="$before" -v c="$cur" 'BEGIN { exit !((c > b + 0.001) || (b >= 0.999)) }'; then
			echo "yes (now $cur, before $before)"
		else
			echo "no (now $cur, before $before)"
		fi
		;;
	front) osascript -e 'tell application "System Events" to get name of first process whose frontmost is true' ;;
	tabs)
		osascript -e 'tell application "Safari"' \
			-e 'set total to 0' \
			-e 'repeat with w in windows' -e 'set total to total + (count of tabs of w)' -e 'end repeat' \
			-e 'return ((count of tabs of front window) as text) & " in front window, " & (total as text) & " total"' \
			-e 'end tell'
		;;
	focus)
		if out=$(jq '[.data[]?.storeAssertionRecords[]?] | length' "$DND_DB" 2>/dev/null); then
			[[ "$out" -gt 0 ]] && echo on || echo off
		else
			echo "unknown (Assertions.json unreadable: no Full Disk Access — look at the menu-bar Focus icon)"
		fi
		;;
	file) [[ -e "$DELETE_ME" ]] && echo exists || echo gone ;;
	sessions) "$CCC_BIN" sessions -j | jq -r '.data.sessions[] | [.name, .kind, .status, .account] | @tsv' ;;
	decision)
		need_state
		sid=$(state_get '.sessions["scratch-idle"].session_id')
		"$CCC_BIN" inspect -s "$sid" -j | jq -c '{ok, state: .data.state, decision: .data.decision, error}'
		;;
	answers)
		need_state
		tx=$(transcript_of scratch-idle)
		jq -c 'select((.toolUseResult | type) == "object" and .toolUseResult.answers != null) | .toolUseResult.answers' "$tx" | tail -1
		;;
	prompts)
		need_state
		tx=$(transcript_of scratch-idle)
		jq -r 'select(.type == "user" and (.isMeta | not)) | .message.content
			| if type == "string" then . else ([.[]? | select(.type == "text") | .text] | join(" ")) end
			| select(length > 0) | .[0:200]' "$tx" | tail -3
		;;
	enqueued)
		need_state
		tx=$(transcript_of scratch-busy)
		jq -r 'select(.type == "queue-operation" and .operation == "enqueue") | .content | tostring | .[0:200]' "$tx" | tail -3
		;;
	dead)
		need_state
		local pid
		pid=$(state_get '.sessions["scratch-dead"].killed_pid')
		if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then echo "pid $pid ALIVE"; else echo "pid ${pid:-?} dead"; fi
		"$CCC_BIN" sessions -j | jq -c '.data.sessions[] | select(.name == "scratch-dead") | {session_id, pid, status, tab}'
		;;
	events)
		need_state
		sid=$(state_get '.sessions["scratch-dead"].session_id')
		"$CCC_BIN" events -a 0 -l 10000 -j | jq -c --arg s "$sid" '.data.events[] | select(.session_id == $s)'
		;;
	problems)
		[[ -f "$ATTENTION_DB" ]] || {
			echo "no $ATTENTION_DB yet"
			return 0
		}
		sqlite3 -readonly -separator ' | ' "$ATTENTION_DB" \
			"SELECT kind, subject, reason, severity,
				datetime(opened_at, 'unixepoch', 'localtime') AS opened,
				'briefed=' || COALESCE(datetime(briefed_at, 'unixepoch', 'localtime'), '-'),
				'acked=' || COALESCE(datetime(acknowledged_at, 'unixepoch', 'localtime'), '-'),
				'resolved=' || COALESCE(datetime(resolved_at, 'unixepoch', 'localtime'), '-')
			FROM problems ORDER BY last_seen DESC LIMIT 8"
		;;
	before)
		need_state
		jq -c '{volume_before, muted_before, brightness_before, created_at}' "$STATE"
		;;
	*)
		fail "unknown query: $QUERY (see -h)"
		exit 2
		;;
	esac
}

run_query() {
	local t0 out el
	t0=$(now)
	while :; do
		out=$(query_value 2>&1) || true
		if [[ -z "$EXPECT" ]]; then
			printf '%s\n' "$out"
			return 0
		fi
		el=$(awk -v a="$t0" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }')
		if [[ "$out" =~ $EXPECT ]]; then
			ok "$QUERY = $out (after $el s)"
			return 0
		fi
		if awk -v e="$el" -v w="$WAIT" 'BEGIN { exit !(e >= w) }'; then
			fail "$QUERY = $out — does not match /$EXPECT/ (waited $el s)"
			return 1
		fi
		sleep 0.5
	done
}

# --------------------------------------------------------------------------- actions

preflight() {
	local bad=0 c
	for c in osascript jq perl sqlite3 python3; do
		command -v "$c" >/dev/null || {
			fail "$c not on PATH"
			bad=1
		}
	done
	command -v "$CCC_BIN" >/dev/null || {
		fail "ccc not on PATH (set CCC_BIN)"
		bad=1
	}
	[[ -x "$CCC_PY" ]] || {
		fail "ccc venv python missing: $CCC_PY"
		bad=1
	}
	[[ -x "$CLAUDE_BIN" ]] || {
		fail "claude binary missing: $CLAUDE_BIN"
		bad=1
	}
	[[ -x "$SCRATCH_SETUP" ]] || {
		fail "missing $SCRATCH_SETUP"
		bad=1
	}
	[[ $bad -eq 0 ]] || die "prerequisites missing"
	ok "prerequisites present (ccc, ccc venv, claude, jq, osascript, scratch_setup.sh)"
	if [[ -f "$STATE" ]]; then
		if [[ $DRY -eq 1 ]]; then
			fail "state file $STATE exists — a real run would refuse until -C"
		else
			die "state file $STATE exists from an earlier run — run -C first"
		fi
	fi
	if ! helper names-free "${NAMES[@]}"; then
		[[ $DRY -eq 1 ]] || die "close the scratch sessions listed above first (or run -C)"
	fi
}

open_session() { # $1 = name; opens the tab, waits for registration, records it
	local name="$1" out iterm info
	# new-session-ok — scratch session for the attended acceptance run
	out=$("$SCRATCH_SETUP" -N "$name" -m "$MODEL") || {
		printf '%s\n' "$out"
		die "$name: scratch_setup.sh failed"
	}
	printf '%s\n' "$out" | grep -v '^ITERM_SESSION=' || true
	iterm=$(sed -n 's/^ITERM_SESSION=//p' <<<"$out")
	[[ -n "$iterm" ]] || die "$name: no iTerm session id from scratch_setup.sh"
	state_merge "$(jq -n --arg n "$name" --arg i "$iterm" '{sessions: {($n): {iterm_session: $i}}}')"
	info=$(helper wait-session "$name" "$iterm" "$TIMEOUT") || die "$name: did not come up — run -C to undo"
	state_merge "$(jq -n --arg n "$name" --argjson i "$info" '{sessions: {($n): $i}}')"
}

seed_idle() {
	local iterm sid
	iterm=$(state_get '.sessions["scratch-idle"].iterm_session')
	sid=$(state_get '.sessions["scratch-idle"].session_id')
	[[ -n "$iterm" && -n "$sid" ]] || die "scratch-idle not in the state file"
	helper prompt turn "$iterm" "$sid" "$SEED_PROMPT" "$TIMEOUT" >/dev/null ||
		die "scratch-idle: seed reply did not finish — retry with -R"
	decision_ok || return 1
}

do_setup() {
	preflight
	if [[ $DRY -eq 1 ]]; then
		local vol mut br
		ok "dry run: would create $SCRATCH and $DELETE_ME"
		"$SCRATCH_SETUP" -n -N scratch-idle -m "$MODEL" | sed 's/^/   /'
		ok "dry run: would wait (≤ ${TIMEOUT} s) until scratch-idle is registered, tab-validated and idle, then send the seed prompt:"
		printf '%s\n' "$SEED_PROMPT" | sed 's/^/   | /'
		ok "dry run: would check \`ccc inspect -s <id> -N -j\` shows a 2-option todo-line decision with a recommendation"
		ok "dry run: would open scratch-busy the same way and leave it idle"
		ok "dry run: would open scratch-dead, wait until registered, then SIGTERM (SIGKILL after 5 s) its claude pid"
		vol=$(read_volume 2>/dev/null || echo "?")
		mut=$(read_muted 2>/dev/null || echo "?")
		br=$(read_bright 2>/dev/null || echo "?")
		ok "now: output volume $vol, muted $mut, built-in brightness $br"
		ok "dry run: would record those in $STATE (0600) and set the output volume to $VOLUME"
		ok "dry run: would re-focus this tab (${ITERM_SESSION_ID:-not in iTerm}) and print the daemon flags:"
		print_daemon_help
		return 0
	fi

	mkdir -p "$SCRATCH"
	printf 'Scratch file of the mac-voice acceptance run (A7) — the assistant deletes it.\n' >"$DELETE_ME"
	ok "created $DELETE_ME"
	state_merge "$(jq -n --arg t "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg d "$SCRATCH" '{created_at: $t, scratch_dir: $d}')"

	open_session scratch-idle
	seed_idle || die "scratch-idle seed check failed — fix with -R, or -C to undo"

	open_session scratch-busy

	open_session scratch-dead
	local dead_iterm dead_sid info
	dead_iterm=$(state_get '.sessions["scratch-dead"].iterm_session')
	dead_sid=$(state_get '.sessions["scratch-dead"].session_id')
	info=$(helper kill "$dead_iterm" "$dead_sid") || die "scratch-dead: kill failed — run -C"
	state_merge "$(jq -n --argjson i "$info" '{sessions: {"scratch-dead": $i}}')"

	local vol mut br
	vol=$(read_volume)
	mut=$(read_muted)
	br=$(read_bright)
	state_merge "$(jq -n --arg v "$vol" --arg m "$mut" --arg b "$br" \
		'{volume_before: ($v | tonumber? // null), muted_before: ($m == "true"),
		  brightness_before: ($b | tonumber? // null)}')"
	chmod 600 "$STATE"
	ok "recorded volume $vol, muted $mut, brightness $br in $STATE"
	osascript -e "set volume output volume $VOLUME" -e 'set volume output muted false'
	vol=$(read_volume)
	if [[ "$vol" =~ ^[0-9]+$ ]] && ((vol >= VOLUME - 3 && vol <= VOLUME + 3)); then
		ok "output volume set to $vol (unmuted)"
	else
		fail "output volume reads $vol after setting $VOLUME"
	fi

	focus_iterm_session
	echo
	ok "setup done — exports for the checks:"
	print_exports
	print_daemon_help
}

do_cleanup() {
	if [[ ! -f "$STATE" ]]; then
		ok "no state file $STATE — nothing recorded to undo"
		[[ -d "$SCRATCH" ]] && fail "$SCRATCH still exists (not created by a recorded run?) — remove it by hand if unwanted"
		return 0
	fi
	local bad=0 n iterm pid tty vol mut br
	for n in "${NAMES[@]}"; do
		iterm=$(state_get ".sessions[\"$n\"].iterm_session")
		pid=$(state_get ".sessions[\"$n\"].pid")
		tty=$(state_get ".sessions[\"$n\"].tty")
		[[ -n "$iterm" ]] || continue
		if [[ $DRY -eq 1 ]]; then
			ok "dry run: would send esc + /exit to $n (pid ${pid:-?}) and close iTerm session $iterm"
			continue
		fi
		helper close "$n" "$iterm" "${pid:-0}" "${tty:-}" || bad=1
	done
	vol=$(state_get .volume_before)
	mut=$(state_get .muted_before)
	br=$(state_get .brightness_before)
	if [[ $DRY -eq 1 ]]; then
		ok "dry run: would restore volume ${vol:-unrecorded}, muted ${mut:-unrecorded}, brightness ${br:-unrecorded}"
		ok "dry run: would remove $SCRATCH and $STATE"
		return 0
	fi
	if [[ "$vol" =~ ^[0-9]+$ ]]; then
		osascript -e "set volume output volume $vol"
		[[ "$mut" == true || "$mut" == false ]] && osascript -e "set volume output muted $mut"
		ok "volume restored to $(read_volume), muted $(read_muted)"
	fi
	if [[ -n "$br" && "$br" != null ]]; then
		helper bright-set "$br" || bad=1
	fi
	if [[ -d /tmp/bridge-scratch ]]; then
		rm -rf /tmp/bridge-scratch && ok "removed /tmp/bridge-scratch"
	fi
	if [[ $bad -eq 0 ]]; then
		rm -f "$STATE"
		ok "removed $STATE — cleanup complete"
	else
		fail "cleanup incomplete — $STATE kept; fix the ❌ lines and re-run -C"
		return 1
	fi
}

do_session_prompt() { # $1 = mode (busy|ask|turn), $2 = name, $3 = prompt, $4 = timeout
	need_state
	local iterm sid
	iterm=$(state_get ".sessions[\"$2\"].iterm_session")
	sid=$(state_get ".sessions[\"$2\"].session_id")
	[[ -n "$iterm" && -n "$sid" ]] || die "$2 not in the state file"
	if [[ $DRY -eq 1 ]]; then
		ok "dry run: would send to $2 ($sid): $3"
		return 0
	fi
	helper prompt "$1" "$iterm" "$sid" "$3" "$4"
}

case "$ACTION" in
setup) do_setup ;;
cleanup) do_cleanup ;;
busy) do_session_prompt busy scratch-busy "$BUSY_PROMPT" 30 ;;
picker) do_session_prompt ask scratch-idle "$PICKER_PROMPT" 120 ;;
reseed)
	if [[ $DRY -eq 1 ]]; then
		do_session_prompt turn scratch-idle "$SEED_PROMPT" "$TIMEOUT"
	else
		need_state
		seed_idle
	fi
	;;
exports) print_exports ;;
query) run_query ;;
esac
