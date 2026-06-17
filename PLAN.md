# PLAN — `my-stt-tts`: local voice assistant on a MacBook M1

> A hand-wired, low-latency voice loop running entirely on a MacBook M1 (Apple
> Silicon): **wake word → record → speech-to-text → Claude (streaming) →
> text-to-speech → playback**, with speaker identification and German / French /
> English support. The Mac is the prototype target; the design stays portable so
> the brain can later move to a server and the mics/speakers to whole-house
> satellites (ties back to the sibling `home-assistant-sandbox` repo).

## Session

Resume: `c --resume <session-id>`  <!-- fill in from `claude --resume` list; this plan was authored 2026-06-17 -->

---

## 1. Goal (restatement)

Build a single, always-on Python process on the M1 that listens for a wake word,
records one utterance, transcribes it, sends the text to Claude (streaming),
speaks the answer back through the Mac speakers, and — as it goes — identifies
*who* spoke. It must feel responsive (target perceived first-audio ≈ 1–1.5 s
excluding model thinking time), work in **Hochdeutsch (standard German), French,
and English**, and keep the large language model (LLM) layer pluggable so we can
start on a fast cheap model and later default to a stronger one and orchestrate
other agents.

Abbreviations used below: **STT** = Speech-to-Text, **TTS** = Text-to-Speech,
**LLM** = Large Language Model, **VAD** = Voice Activity Detection, **AEC** =
Acoustic Echo Cancellation, **MCP** = Model Context Protocol, **TTFA** =
time-to-first-audio, **TTFT** = time-to-first-token, **RTF** = Real-Time Factor,
**EER** = Equal Error Rate, **G2P** = Grapheme-to-Phoneme, **TCC** = macOS
Transparency/Consent/Control (privacy permissions).

---

## 2. Locked decisions (with rationale)

| # | Decision | Choice | Why |
|:--|:---------|:-------|:----|
| D1 | **Implementation language** | **Python** for the orchestrator; optional thin **Swift** audio front-end deferred to Phase 7 | End-to-end latency is dominated by model inference (native Metal/MLX/C++) and the Claude network round-trip. The orchestration glue is <0.2 % of a ~1–2 s turn; the GIL is released inside every native call and during I/O. Rust/Swift/C++ would win single-digit milliseconds while costing weeks and fighting immature M1 ML bindings. Python's Apple-Silicon ML ecosystem (MLX, `mlx-audio`, `parakeet-mlx`, PyTorch MPS, ONNX) is the most mature by far. |
| D2 | **STT engine** | **`parakeet-mlx`** (`parakeet-tdt-0.6b-v3`, multilingual) primary; `whisper.cpp` large-v3-turbo as alternate | v3 is multilingual (25 EU languages incl. DE/FR/EN with auto language-ID), MLX-native on M1, sub-second on short commands, and beats Whisper-large on word error rate. `whisper.cpp` (Metal/CoreML) is the fallback if Parakeet's language-ID or punctuation disappoints. **Do not use `faster-whisper` on Mac — it is CPU-only there (no Metal).** |
| D3 | **TTS engine** | **Piper** for all three languages in v1 (`de_DE-thorsten-high`, `fr_FR-tom-medium`, `en_US-lessac-medium`); **macOS `say` premium** as instant fallback; optional **Kokoro via `mlx-audio`** for nicer English later | Piper is the *only* local engine that has strong **German** (Thorsten, audiobook-grade), correct French, good English, **and** actually hits sub-300 ms TTFA on M1 CPU without fighting broken MPS. Kokoro has **no German** (disqualified as primary). XTTS-v2 (non-commercial, MPS hangs) and Qwen3-TTS (GPU-oriented; only immature MLX ports on M1) are deferred behind the same interface. |
| D4 | **Speaker identification** | **SpeechBrain ECAPA-TDNN** embeddings + enrollment + cosine-similarity to per-person centroids, with unknown/ambiguous rejection | Best accuracy (≈0.80 % EER), text-independent and cross-lingual-robust (matters for DE/FR/EN), trivial Python API, ~80–150 ms on M1 CPU — and it runs **in parallel with STT on the same clip**, so ≈0 added wall-clock. Resemblyzer rejected (English-biased, weakest). |
| D5 | **LLM layer** | **Anthropic SDK**, streaming, model **pluggable**: default **`claude-haiku-4-5`** (fast path) now → **`claude-opus-4-8`** (deep path) on a trigger word; designed for tool-use / MCP so it can dispatch to other home/work agents | Voice turns want a fast, cheap default; Opus is a latency + cost tax for routine queries. Keep a `Brain` interface so the model and (later) multi-agent routing are config, not code rewrites. |
| D6 | **Stage confirmations** | **Earcons (short chimes)**, not spoken phrases. One chime on wake (mic live), optional chime on end-of-record. Spoken stage narration ("analyzing using opus 4.8", …) kept **behind a `--debug` flag** only | The four spoken phrases the original sketch proposed add **~6–7 s of dead air per query** — 5× the entire acceptable turn budget. Chimes are ~150 ms, language-neutral (no DE/FR/EN translation needed), and don't risk re-triggering the wake word. |
| D7 | **End-of-turn detection** | **Push-to-talk** in v1 (deterministic), then **Silero VAD** silence-timeout; always a hard max-recording cap | Endpointing is the hardest part of voice UX. Push-to-talk removes it entirely so we can validate STT→Claude→TTS first; VAD comes once the core loop is proven. |
| D8 | **Process model** | **One warm long-running process**, all models pre-loaded at startup; stages overlapped with `asyncio` + a worker thread/queue for blocking native calls | Model load + Metal kernel warm-up is hundreds of ms–seconds; pay it **once**. This (not the language) is the single biggest latency lever. |
| D9 | **Echo / self-trigger** | **Half-duplex mic gating**: suspend wake-word + capture during all playback + a ~200 ms tail. Full AEC + barge-in deferred to Phase 7 (Swift `VoiceProcessingIO`) | Speaker and mic share the laptop enclosure; without gating the assistant records and re-triggers on its own voice. Gating kills ~95 % of the problem at ~zero cost. |

---

## 3. Architecture

```text
                         ┌─────────────────────────────────────────────┐
                         │   one warm Python process (asyncio loop)     │
                         │                                              │
  mic ──► ring buffer ──►│  Wake word        Endpointing                │
        (pre-roll ~300ms)│  (openWakeWord/   (push-to-talk → Silero VAD)│
                         │   Porcupine)            │                    │
                         │        │                ▼                    │
                         │        ▼          ┌───────────┐              │
                         │   [chime: live]   │  utterance │             │
                         │                   │  PCM clip  │             │
                         │                   └─────┬─────┘              │
                         │            ┌────────────┴───────────┐        │
                         │            ▼ (parallel)             ▼        │
                         │     STT (parakeet-mlx)      Speaker-ID        │
                         │     → text + lang           (ECAPA centroid   │
                         │            │                 cosine match)    │
                         │            ▼                     │            │
                         │     Brain (Claude SDK, streaming)│            │
                         │     Haiku default / Opus deep    │            │
                         │     + conversation memory        │            │
                         │            │ tokens                           │
                         │            ▼ sentence-chunked                 │
                         │     TTS Router (per-language)                 │
                         │     DE→Piper-thorsten  FR→Piper-tom           │
                         │     EN→Piper/Kokoro    fallback→say           │
                         │            │ stream first sentence early      │
   speakers ◄────────────│            ▼  (mic gated during playback)     │
                         └─────────────────────────────────────────────┘
```

### Latency budget (target per stage, M1, short command)

| Stage | Target | Notes |
|:------|:-------|:------|
| Wake-word detection lag | 80–150 ms | continuous, low-power on efficiency cores |
| Endpointing | push-to-talk ≈0 / VAD 300–700 ms | VAD silence wait is a *timer*, not compute — tune aggressively |
| STT (parakeet-mlx) | 80–400 ms | native MLX/Metal; runs concurrently with speaker-ID |
| Speaker-ID (ECAPA) | hidden under STT | ~80–150 ms in parallel → ~0 added |
| LLM TTFT (Haiku, streaming) | 400–800 ms | network-bound; Opus higher — that's the deep-path tradeoff |
| TTS first sentence (Piper) | 40–200 ms | sentence-chunked; start playback before full answer |
| Playback start | 10–30 ms | CoreAudio buffer |
| **Perceived first audio** | **~1.0–1.5 s** | with streaming + overlap; physics floor for a cloud LLM |

---

## 4. Repository layout (planned)

```text
my-stt-tts/
├── PLAN.md                  # this file
├── README.md                # project overview + quickstart
├── pyproject.toml           # deps + tool config (uv-managed)
├── .env.example             # ANTHROPIC_API_KEY=..., config knobs
├── config.toml              # voices, models, thresholds, wake phrase
├── src/voiceloop/
│   ├── __main__.py          # entrypoint: warm models, run async loop
│   ├── audio.py             # capture, ring buffer, playback, mic-gating
│   ├── wake.py              # wake-word backend (openWakeWord/Porcupine)
│   ├── stt.py               # parakeet-mlx / whisper.cpp backend
│   ├── speaker_id.py        # ECAPA enrollment + match + reject
│   ├── brain.py             # Claude SDK streaming + model routing + memory
│   ├── tts.py               # TTS Router (Piper/Kokoro/say) + lang detect
│   ├── chimes.py            # earcons; pre-synth error clips
│   └── metrics.py           # per-stage latency + transcript logging
├── scripts/
│   ├── enroll.py            # record ~30s/person → store ECAPA centroid
│   └── bench.py             # measure per-stage latency on this Mac
└── enroll/                  # gitignored: per-person voice profiles
```

Every script gets `-h/--help`, is made executable + git-exec-bit set, and follows
`$mygit/README_SETUP_PYTHON_ENVIRONMENT.md` (read it before writing the first
Python file). Lint gate before every commit: `ruff format && ruff check && mypy &&
pylint` (Python), `shellcheck` (shell).

---

## 5. Phased plan (checkboxes)

### Phase 0 — Scaffold & environment

- [ ] Read `$mygit/README_SETUP_PYTHON_ENVIRONMENT.md`; create `uv` venv + `pyproject.toml`
- [ ] `.env.example` (`ANTHROPIC_API_KEY`) and `config.toml` (wake phrase, voices, thresholds)
- [ ] Package skeleton `src/voiceloop/`; `metrics.py` logging first (we tune by numbers)
- [ ] `scripts/bench.py` to measure STT/TTS/LLM latency on *this* M1 (validate budget above)

### Phase 1 — Core loop (push-to-talk, English, batch)

- [ ] `audio.py`: capture via `sounddevice`, explicit input/output device, push-to-talk hotkey, max-recording cap
- [ ] `stt.py`: `parakeet-mlx` warm-loaded; transcribe captured clip
- [ ] `brain.py`: Claude streaming call (Haiku), accumulate answer
- [ ] `tts.py`: Piper English voice → playback via CoreAudio
- [ ] `chimes.py`: wake chime; wire `--debug` spoken stage cues (the original "yes/recorded/analyzing" narration lives here, off by default)
- [ ] End-to-end: press key → speak → hear Claude. Log per-stage latency.

### Phase 2 — Responsiveness (streaming + safety)

- [ ] Stream Claude tokens → sentence-boundary chunker → TTS starts on first sentence
- [ ] `asyncio` + worker thread so STT/LLM/TTS/playback overlap; mic pre-roll ring buffer
- [ ] Half-duplex **mic gating** during playback + 200 ms tail (D9)
- [ ] Graceful failure: catch every stage; play **pre-synthesized** error clips ("sorry, network problem") even if TTS is what failed
- [ ] Runaway guard: per-minute request cap + cooldown (protects against self-trigger loops and cost)

### Phase 3 — Multilingual (DE / FR / EN)

- [ ] STT multilingual: Parakeet v3 language-ID (or Whisper auto-detect); expose detected language
- [ ] `tts.py` **Router**: `lingua-py` language detection on the answer → voice map (`de→thorsten-high`, `fr→tom-medium`, `en→lessac/Kokoro`), `say` premium fallback + low-confidence fallback
- [ ] Test Hochdeutsch and French answers end-to-end; verify pronunciation, not just EN
- [ ] (Optional) add Kokoro-via-`mlx-audio` for higher-quality English

### Phase 4 — Wake word & always-listening

- [ ] Train + integrate **openWakeWord** for the wake phrase **"maziko"** (custom model, ~1 h via the openWakeWord training notebook; no vendor lock; Porcupine only as a zero-training fallback)
- [ ] Replace push-to-talk with wake-word + **Silero VAD** endpointing (tune silence timeout)
- [ ] Conversation **follow-up window** (~8 s open mic after a reply, no re-wake needed)
- [ ] Multi-turn **memory** (rolling `messages`, capped length, idle reset)

### Phase 5 — Speaker identification

- [ ] `scripts/enroll.py`: record ~30 s/person across 5–10 clips in each language they use → store L2-normalized ECAPA **centroid** per person (gitignored)
- [ ] `speaker_id.py`: extract embedding **in parallel** with STT; cosine `argmax` over centroids
- [ ] Rejection: absolute threshold (~0.40–0.50, **calibrated on our own family + guest clips**) + margin gate (~0.06) → `unknown` / `ambiguous`
- [ ] Bias toward `unknown` over misattribution; **never gate safety-critical actions on child ID**; re-enroll children quarterly
- [ ] Pass identified speaker into the Brain prompt for per-person personalization

### Phase 6 — LLM flexibility & agent orchestration

- [ ] Model routing: Haiku fast path / Opus deep path via trigger ("think hard…") or per-speaker default
- [ ] Prompt caching for the stable system prompt (cut input cost/latency)
- [ ] Tool-use / **MCP** wiring so the assistant can dispatch to other home/work agents (the longer-term goal); start with one or two local tools
- [ ] Per-speaker + per-language system-prompt context (Swiss defaults: metric, ISO-8601)

### Phase 7 — Barge-in & native audio (deferred, only if needed)

- [ ] Swift `AVAudioEngine` + `VoiceProcessingIO` front-end (hardware AEC) feeding PCM to Python over a socket
- [ ] True barge-in: keep mic live during playback, abort TTS on detected intentional speech
- [ ] Package as menubar app (`rumps`) / `launchd` agent with a **stable bundle id** (TCC permissions are keyed to it)

### Phase 8 — Whole-house / Home Assistant (future)

- [ ] Move brain to a server, mics/speakers to satellites; integrate with `home-assistant-sandbox` Assist + Wyoming; revisit Sonos vs satellite-local playback latency

---

## 6. Dependencies (initial)

```commands
uv add anthropic parakeet-mlx mlx-audio piper-tts speechbrain torchaudio \
       sounddevice silero-vad openwakeword onnxruntime lingua-language-detector
brew install whisper-cpp espeak-ng portaudio ffmpeg
python -m piper.download_voices de_DE-thorsten-high fr_FR-tom-medium en_US-lessac-medium
# macOS premium voices: System Settings → Accessibility → Spoken Content → Manage Voices
#   (download Anna (Premium) [de], Thomas [fr], Ava (Premium) [en])
```

---

## 7. Risk register

| Risk | Severity | Confidence | Mitigation |
|:-----|:---------|:-----------|:-----------|
| **Children's voices** misidentified / confused (esp. youngest, 2-word commands) | High for kids | High | Buffer full utterance, bias to `unknown`, margin gate for son/daughter pair, never gate safety actions on child ID, accept lower accuracy |
| **German TTS quality ceiling** — Piper-Thorsten is best local but below ElevenLabs/XTTS | Medium | High | Accept for v1; `say -v "Anna (Premium)"` fallback; re-evaluate Qwen3-TTS-MLX / Chatterbox-multilingual when their Apple-Silicon paths stabilize |
| **No usable Metal TTS acceleration** on M1 today (XTTS MPS hangs; Qwen3 GPU-oriented) | Medium | Moderate | Stay on Piper (CPU, predictable) + `say`; keep heavy models behind the Router interface, deferred |
| **Echo / self-trigger** on a single-box laptop | High | High | Half-duplex mic gating (Phase 2); real AEC only in Phase 7 if barge-in is wanted |
| **Porcupine free-tier license** (non-commercial, platform-locked custom keyword) | Low | Moderate | Prefer openWakeWord (Apache, self-trained, no lock); Porcupine only if zero-training matters |
| **Cost / runaway loop** (self-trigger firing Claude repeatedly) | Medium | High | Per-minute request cap + cooldown; default Haiku not Opus (~1¢/query on Opus, less on Haiku) |
| **M1 latency numbers** are indicative, not lab-measured on base M1 | Low | Moderate | `scripts/bench.py` in Phase 0 measures the real budget on this exact machine before we optimize |
| **Swiss German dialect** degrades STT | Low (household speaks Hochdeutsch) | High | N/A per user — household uses standard German/French/English; note only if guests speak dialect |

---

## 8. Open items to confirm (defaulting as noted unless you object)

1. **Wake-word engine + phrase** — **LOCKED: openWakeWord** (no vendor lock) with the custom wake phrase **"maziko"** (distinctive, multi-syllabic → low false-accept). Needs a custom-trained openWakeWord model (~1 h via the openWakeWord training notebook); Phases 1–3 run on push-to-talk, so training is not a blocker until Phase 4.
2. **STT** — defaulting to **`parakeet-mlx` v3**; will fall back to `whisper.cpp` large-v3-turbo if multilingual punctuation/accuracy disappoints in Phase 3.
3. **TTS** — defaulting to **Piper for all three languages** (one runtime) in v1; Kokoro-for-English is optional polish.
4. **Speaker-ID scope** — phased to Phase 5 (core loop first). Confirm the roster (who to enroll) when we get there.

---

## 9. Data / privacy note (SDSC context)

Local STT + TTS keep voice audio **on-device**; only the transcribed *text* leaves
the machine (to Anthropic, same as ordinary Claude Code use). Do not dictate
Confidential / Strictly-Confidential content into the assistant. Enrollment voice
profiles stay local and gitignored.
