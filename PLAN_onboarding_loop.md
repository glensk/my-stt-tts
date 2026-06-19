# PLAN — Onboarding Seamlessness Loop

## Session

Resume: `c --resume e7dfe88f-9001-4138-8cfc-1f8789653cc6`

## Part 1 — Settings: location + units (Albert: "for sure")

Add to settings (config + `--settings` + web-UI + CLI), so the assistant answers
location/units-aware questions (e.g. weather):

- [ ] **`location`** — free-text place, default **`Lausanne, Switzerland`** (Albert's).
- [ ] **`units`** — `metric` | `imperial`, default **`metric`** (Albert's).
- [ ] Wire into a real **weather tool** (`tools.py`) using a **no-API-key** provider
  (Open-Meteo: geocode the location → forecast; respect `units`), and inject
  `location` + `units` into the system-prompt/context so the LLM is location/units-aware
  generally. (Done in **Wave G**, after the in-flight speaker-ID fix merges — both touch
  `config.py`/`brain.py`, so sequence to avoid conflicts.)

## Part 2 — Seamless onboarding (README-first)

Make it dead-simple for a new user to test the LLM. Right at the top of `README.md`:

- [ ] A prominent **"▶ Try it live →"** leading to the hosted demo (GitHub Pages control
  room `gui.html` — works with zero setup, shows the experience) AND
- [ ] a **one-command real run** (a `quickstart.sh` / a single copy-paste block →
  `uv sync` → `./mstt --browser` so they're talking to a real LLM in the browser; typed
  mode needs no mic). Make the very first thing a reader sees the get-started path.

(README work is owned by the dedicated designer — see `PLAN_readme_loop.md`. The `quickstart.sh`
script and the demo URL come from Wave G; the designer references them.)

## Part 3 — The loop (single question)

A fresh **indifferent checker** judges ONE thing only: **"How seamless is it for a NEW user
to get onboarded and actually test this — from landing on the repo to talking to the LLM?"**
my-stt-tts vs each previously-checked repo **and other award-winning onboarding exemplars**:

- Prior repos: pipecat · livekit/agents · huggingface/speech-to-speech · dnhkng/GLaDOS · KoljaB/RealtimeSTT
- Award-winning onboarding/"try it live" exemplars (checker researches + picks the best, e.g.
  one-command/`uvx` installs, hosted live demos, top READMEs): include the fanciest/easiest it finds.

Force a single winner each round on seamless-onboarding only. If a reference wins, implement the
cited onboarding improvements (quickstart, demo link, install simplicity, first-run UX, clarity)
until a fresh checker picks my-stt-tts; iterate over the set. Maturity/popularity NOT a factor —
judge the actual onboarding/test experience.

## Status

Queued behind the in-flight speaker-ID wiring fix (`config.py`/`brain.py` overlap). Order: merge
speaker-fix → **Wave G** (location/units + weather + `quickstart.sh`) → designer onboarding README
round → **onboarding seamlessness checker** (round 1) → iterate.
