#!/usr/bin/env bash
# quickstart.sh — one command from a fresh clone to typing at a real LLM.
#
# Purpose : bootstrap my-stt-tts and open the browser control room in typed mode,
#           so a brand-new user is chatting with the LLM in seconds — no mic, and
#           NO API key. It auto-detects a key-free "brain" you already have:
#           the Claude CLI, a local Ollama model, or the OpenAI codex CLI.
# Usage   : ./quickstart.sh [-h|--help]
# What it does:
#   1. checks for `uv` (prints an install hint and exits if missing),
#   2. runs `uv sync --extra all` to install the project + optional extras,
#   3. auto-detects a key-free brain (claude CLI -> ollama -> codex CLI),
#   4. launches `./mstt --browser --type` wired to that brain (web GUI, typed
#      input, no microphone). If none is found, prints how to enable one.
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
  3. Auto-detects a key-free brain, in this order:
       - the `claude` CLI (claude.ai/code)         -> --brain haiku-sub
       - `ollama` + an installed model             -> LLM_PROVIDER=ollama
       - the OpenAI `codex` CLI                     -> --brain codex
  4. Launches `./mstt --browser --type` wired to that brain — the web GUI in
     typed mode (no mic). Type a message in the browser and you are talking to
     the LLM. No API key needed for any of the three brains above.
     To talk to it (wake word / mic) instead, run `./mstt --browser --wake`
     (macOS will ask for microphone permission on first capture).

If none is found it prints how to enable one (install claude/ollama/codex, or
set ANTHROPIC_API_KEY) and exits 1.
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

# --- Auto-detect a key-free brain --------------------------------------------
# Each branch sets ARGS (extra ./mstt flags) and, for ollama, exports the env the
# provider needs. The first available brain wins. None requires an API key.
ARGS=()

# First installed Ollama model name (column 1 of `ollama list`, skipping the
# header row). Empty if ollama has no models pulled yet.
ollama_first_model() {
	ollama list 2>/dev/null | awk 'NR>1 && $1!="" {print $1; exit}'
}

if command -v claude >/dev/null 2>&1; then
	echo "quickstart: using the Claude CLI brain (no API key needed)."
	ARGS=(--brain haiku-sub)
elif command -v ollama >/dev/null 2>&1 && [ -n "$(ollama_first_model)" ]; then
	model="$(ollama_first_model)"
	echo "quickstart: using the local Ollama brain '${model}' (no API key needed)."
	export LLM_PROVIDER="ollama"
	export LLM_MODEL="${model}"
	export LLM_BASE_URL="http://localhost:11434/v1"
elif command -v codex >/dev/null 2>&1; then
	echo "quickstart: using the OpenAI codex CLI brain (no API key needed)."
	ARGS=(--brain codex)
else
	echo "quickstart: no key-free brain found." >&2
	echo "Enable ONE of these, then re-run ./quickstart.sh :" >&2
	echo "  - Claude CLI : install from https://claude.ai/code  (then 'claude' is on PATH)" >&2
	echo "  - Ollama     : install from https://ollama.com, then 'ollama pull llama3.2'" >&2
	echo "  - codex CLI  : install the OpenAI codex CLI            (then 'codex' is on PATH)" >&2
	echo "  - or set ANTHROPIC_API_KEY in .env and run ./mstt --browser --type" >&2
	exit 1
fi

echo "quickstart: opening the browser control room (typed mode, no mic needed)…"
echo "            Type a message in the browser to talk to the LLM. Ctrl-C to quit."
echo
echo "🎙️  To talk to it (wake word / mic), run:  ./mstt --browser --wake"
echo "    (loads on-device STT + the wake model so the GUI voice buttons go live;"
echo "     macOS asks for microphone permission on the first capture — grant it to"
echo "     the Terminal/app.)"
echo
exec ./mstt --browser --type "${ARGS[@]}"
