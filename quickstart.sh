#!/usr/bin/env bash
# quickstart.sh — one command from a fresh clone to typing at a real LLM.
#
# Purpose : bootstrap my-stt-tts and open the browser control room in typed mode,
#           so a brand-new user is chatting with the LLM in seconds — no mic, no
#           API key (the default `claude-cli` brain uses your logged-in Claude CLI).
# Usage   : ./quickstart.sh [-h|--help]
# What it does:
#   1. checks for `uv` (prints an install hint and exits if missing),
#   2. runs `uv sync --extra all` to install the project + optional extras,
#   3. launches `./mstt --browser --type` (web GUI, typed input, no microphone).
set -euo pipefail

usage() {
	cat <<'EOF'
quickstart.sh — bootstrap my-stt-tts and start typing to the LLM in seconds.

Usage:
  ./quickstart.sh            Set up the environment and open the browser control room.
  ./quickstart.sh -h|--help  Show this help and exit.

What it does:
  1. Checks that `uv` is installed (prints an install hint if it is not).
  2. Runs `uv sync --extra all` to create .venv and install all extras.
  3. Launches `./mstt --browser --type` — the web GUI in typed mode (no mic).

No API key is needed: the default `claude-cli` brain uses your logged-in
Claude Code CLI. Type a message in the browser and you are talking to the LLM.
EOF
}

case "${1:-}" in
	-h | --help)
		usage
		exit 0
		;;
	"") ;;
	*)
		echo "quickstart.sh: unknown argument '$1'" >&2
		echo "Try './quickstart.sh --help'." >&2
		exit 2
		;;
esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${here}"

if ! command -v uv >/dev/null 2>&1; then
	echo "quickstart: 'uv' is not installed (it manages the Python env)." >&2
	echo "Install it, then re-run ./quickstart.sh :" >&2
	echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
	echo "  # or: brew install uv   (macOS, Homebrew)" >&2
	exit 1
fi

echo "quickstart: syncing the environment (uv sync --extra all)…"
uv sync --extra all

echo "quickstart: opening the browser control room (typed mode, no mic needed)…"
echo "            Type a message in the browser to talk to the LLM. Ctrl-C to quit."
exec ./mstt --browser --type
