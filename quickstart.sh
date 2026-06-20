#!/usr/bin/env bash
# quickstart.sh — one command from a fresh clone to talking with a real LLM.
#
# Purpose : bootstrap my-stt-tts and open the browser control room. If the wake
#           model is present it launches with the wake word LIVE (--browser
#           --wake) so you can just say it; otherwise it falls back to typed mode
#           (--browser --type). Either way it auto-detects a key-free "brain" you
#           already have: the Claude CLI, a local Ollama model, or the codex CLI.
#           No API key needed.
# Usage   : ./quickstart.sh [-h|--help]
# What it does:
#   1. checks for `uv` (prints an install hint and exits if missing),
#   2. runs `uv sync --extra all` to install the project + optional extras,
#   3. auto-detects a key-free brain (claude CLI -> ollama -> codex CLI),
#   4. launches `./mstt --browser --wake` if the wake model exists (voice live),
#      else `./mstt --browser --type` (typed, no mic). If no brain is found,
#      prints how to enable one.
set -euo pipefail

usage() {
	cat <<'EOF'
quickstart.sh — bootstrap my-stt-tts and start talking to the LLM in seconds.

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
  4. Opens the web control room:
       - wake model present  -> `./mstt --browser --wake`  (say the wake word, or
         click Start wake / Push-to-talk; macOS asks for mic permission once)
       - no wake model yet   -> `./mstt --browser --type`  (typed, no mic). Train
         the wake word (see wakewords/WAKEWORD.md) to enable voice.
     No API key needed for any of the three brains above.

If no brain is found it prints how to enable one (install claude/ollama/codex, or
set ANTHROPIC_API_KEY) and exits 1.

Every run is logged (stdout+stderr, incl. warnings/errors and live wake scores)
to logs/quickstart-<timestamp>.log (gitignored) while still streaming to the
terminal. Override the location with QUICKSTART_LOG_DIR=/path.
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

# --- Capture this run's output to a timestamped logfile -----------------------
# Tee EVERYTHING from here on (uv sync, brain detection, and the app's own
# stdout/stderr — warnings, errors, and live wake scores) to a logfile, so a run
# can be investigated after the fact (e.g. "why didn't the wake word fire?"). The
# output still streams live to the terminal. Logs are gitignored (*.log). The
# final `exec ./mstt` inherits these redirected fds, so the app's output is
# captured too. Override the directory with QUICKSTART_LOG_DIR.
log_dir="${QUICKSTART_LOG_DIR:-logs}"
mkdir -p "${log_dir}"
stamp="$(date +%Y%m%d-%H%M%S)"
logfile="${log_dir}/quickstart-${stamp}.log"
exec > >(tee -a "${logfile}") 2>&1
# Capture the full in-app EVENT LOG (state/transcript/response/wake/debug events)
# to disk too — the bus auto-attaches this sink from MSTT_EVENT_LOG.
export MSTT_EVENT_LOG="${log_dir}/events-${stamp}.jsonl"
echo "quickstart: logging this run to ${logfile}"
echo "            EVENT LOG -> ${MSTT_EVENT_LOG} (both also shown live below)."

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

# --- Launch: wake word live if the model is present, else typed --------------
wake_phrase="${WAKE_PHRASE:-maziko}"
wake_model="${WAKE_MODEL_PATH:-wakewords/${wake_phrase}.onnx}"
if [ -f "${wake_model}" ]; then
	echo "quickstart: opening the browser control room with the WAKE WORD live…"
	echo "            Say \"${wake_phrase}\" (or click Start wake / Push-to-talk), or just type."
	echo "            macOS will ask for microphone permission on the first capture — grant it."
	echo "            Ctrl-C to quit."
	echo
	exec ./mstt --browser --wake "${ARGS[@]}"
else
	echo "quickstart: opening the browser control room (typed mode — no wake model found)…"
	echo "            Type a message to talk to the LLM. Ctrl-C to quit."
	echo
	echo "🎙️  To enable the wake word: train it (see wakewords/WAKEWORD.md) so"
	echo "    ${wake_model} exists, then re-run ./quickstart.sh — it will launch with voice."
	echo
	exec ./mstt --browser --type "${ARGS[@]}"
fi
