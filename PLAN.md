# PLAN — `my-stt-tts`: local voice assistant on a MacBook M1

> A hand-wired, low-latency voice loop running entirely on a MacBook M1 (Apple
> Silicon): **wake word → record → speech-to-text → an LLM (streaming) →
> text-to-speech → playback**, with speaker identification and German / French /
> English support. The Mac is the prototype target; the design stays portable so
> the brain can later move to a server and the mics/speakers to whole-house
> satellites (ties back to the sibling `home-assistant-sandbox` repo).

## Session

Resume: `c --resume <session-id>`  <!-- fill in from `claude --resume` list; this plan was authored 2026-06-17 -->

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

### Phase 7 — Barge-in & native audio (deferred) ⬜ deferred

- [ ] **Barge-in**: keep mic live during playback, abort TTS on confirmed speech (mute-event empty-array cancel — GLaDOS); **false-interrupt suppression** (min-words/min-duration — pipecat `MinWordsUserTurnStartStrategy`); decide history truncation (spoken-prefix vs keep-full)
- [ ] Swift `AVAudioEngine` + `VoiceProcessingIO` front-end (hardware AEC) feeding PCM to Python
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
