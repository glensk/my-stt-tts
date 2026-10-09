#!/usr/bin/env bash
# Scratch Claude Code sessions for the attended spikes S-SEND / S-PICKER
# (PLAN_claude-bridge.md §9 Phase 0). See ATTENDED_RUNBOOK.md next to this file.
set -euo pipefail

SCRATCH=/tmp/bridge-scratch
CLAUDE_BIN="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
CCC_REPO="${CCC_REPO:-$HOME/obsidian/42-Git/llms/claude-command-center}"
CCC_PY="$CCC_REPO/.venv/bin/python"
CLAUDE_JSON="$HOME/.claude.json"
NAME=scratch-idle
MODEL=sonnet
DRY=0
CLEANUP=""

usage() {
	cat <<'EOF'
Usage: scratch_setup.sh [-n] [-b] [-N NAME] [-m MODEL] | -C ITERM_SESSION_ID

Creates /tmp/bridge-scratch (with a project settings file allowing `Bash(sleep:*)`, so
the busy test never stops on a permission prompt), pre-trusts it for the cpriv account
(~/.claude.json, both /tmp/... and /private/tmp/...; backup first, atomic write), then
opens ONE new iTerm2 tab in the current window running
  cd /tmp/bridge-scratch && claude -n scratch-idle --model sonnet
under cpriv, and prints the new tab's iTerm session id.

Options:
  -n, --dry-run        print what would be done, change nothing
  -b, --busy           name the session scratch-busy (the second scratch session)
  -N, --name NAME      session name (default scratch-idle)
  -m, --model MODEL    model for the scratch session (default sonnet)
  -C, --cleanup ID     send /exit to the Claude session in iTerm session ID, then close
                       that tab
  -h, --help           this help

Examples:
  scripts/spikes/scratch_setup.sh -n
  scripts/spikes/scratch_setup.sh            # -> prints ITERM_SESSION=<uuid>
  scripts/spikes/scratch_setup.sh -b
  scripts/spikes/scratch_setup.sh -C <uuid>
EOF
}

ok() { printf '\xe2\x9c\x85 %s\n' "$*"; }
fail() { printf '\xe2\x9d\x8c %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
	case "$1" in
	-n | --dry-run) DRY=1 ;;
	-b | --busy) NAME=scratch-busy ;;
	-N | --name)
		NAME="${2:?-N needs a name}"
		shift
		;;
	-m | --model)
		MODEL="${2:?-m needs a model}"
		shift
		;;
	-C | --cleanup)
		CLEANUP="${2:?-C needs an iTerm session id}"
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

if [[ -n "$CLEANUP" ]]; then
	if [[ $DRY -eq 1 ]]; then
		ok "dry run: would send /exit to $CLEANUP, wait for the shell, then close its tab"
		exit 0
	fi
	"$CCC_PY" - "$CLEANUP" "$CCC_REPO" <<'PY'
import sys, time
sys.path.insert(0, sys.argv[2])
from command_center import terminal
sid = sys.argv[1]
ch = terminal.send_text_via(sid, "/exit")
print(("✅" if ch else "❌") + f" /exit sent via {ch or 'nothing'}")
time.sleep(3)
closed = terminal.close_iterm_session(sid)
print(("✅" if closed else "❌") + f" closed: {closed or 'not found'}")
sys.exit(0 if closed else 1)
PY
	exit $?
fi

LAUNCH="cd $SCRATCH && env -u CLAUDE_CONFIG_DIR -u CLAUDE_SECURESTORAGE_CONFIG_DIR $CLAUDE_BIN -n $NAME --model $MODEL"

if [[ $DRY -eq 1 ]]; then
	ok "dry run: would mkdir -p $SCRATCH and write $SCRATCH/.claude/settings.local.json (allow Bash(sleep:*))"
	ok "dry run: would back up $CLAUDE_JSON and set projects[\"$SCRATCH\"] + [\"/private$SCRATCH\"].hasTrustDialogAccepted = true"
	ok "dry run: would open one iTerm2 tab in the current window running: $LAUNCH"
	exit 0
fi

mkdir -p "$SCRATCH/.claude"
cat >"$SCRATCH/.claude/settings.local.json" <<'JSON'
{
  "permissions": {
    "allow": ["Bash(sleep:*)", "Bash(sleep 40)"]
  }
}
JSON
ok "scratch dir $SCRATCH ready (Bash(sleep:*) allowed)"

python3 - "$CLAUDE_JSON" "$SCRATCH" <<'PY'
import json, os, shutil, sys, tempfile, time
path = os.path.realpath(sys.argv[1])
scratch = sys.argv[2]
keys = [scratch, os.path.realpath(scratch)]
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
projects = data.setdefault("projects", {})
if all(projects.get(k, {}).get("hasTrustDialogAccepted") is True for k in keys):
    print(f"✅ already trusted: {', '.join(keys)}")
    sys.exit(0)
backup = f"{path}.bak-bridge-{time.strftime('%Y%m%d-%H%M%S')}"
shutil.copy2(path, backup)
os.chmod(backup, 0o600)
# Re-read right before writing: Claude Code processes rewrite this file often.
with open(path, encoding="utf-8") as fh:
    data = json.load(fh)
projects = data.setdefault("projects", {})
for k in keys:
    projects.setdefault(k, {})["hasTrustDialogAccepted"] = True
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".claude.json.")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
os.chmod(tmp, 0o600)
os.replace(tmp, path)
print(f"✅ trusted {', '.join(keys)} (backup {backup})")
PY

# new-session-ok — an attended spike scratch session the orchestrator opens on purpose.
ITERM_SESSION=$(
	osascript - "$LAUNCH" <<'APPLESCRIPT'
on run argv
	set cmd to item 1 of argv
	tell application "iTerm2"
		if (count of windows) is 0 then
			set w to (create window with default profile)
			set newTab to current tab of w
		else
			tell current window
				set newTab to (create tab with default profile)
			end tell
		end if
		set s to current session of newTab
		delay 0.5
		set n to 0
		repeat while (is processing of s) and n < 100
			delay 0.1
			set n to n + 1
		end repeat
		delay 0.5
		tell s to write text cmd
		return id of s
	end tell
end run
APPLESCRIPT
) || {
	fail "could not open the iTerm2 tab"
	exit 1
}
ok "opened iTerm2 tab running '$NAME' in $SCRATCH"
echo "ITERM_SESSION=$ITERM_SESSION"
