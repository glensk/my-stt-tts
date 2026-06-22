# Relevant Git Repos

Catalogue of the external projects that matter to `my-stt-tts`, in two parts:

1. **Checker-loop repos** — every repo the project's two adversarial "checker
   loops" ran against. These are factual, taken from the `PLAN_*` files: what each
   repo is, the specific capability it offered, and how it fared / what we ported.
2. **Fluent-conversation agent projects** — fresh web research (2026-06) into the
   most prominent stacks for full-duplex, interruptible voice conversation with an
   agent across hardware.

Abbreviations used: STT = speech-to-text, TTS = text-to-speech, VAD = voice
activity detection, KWS = keyword spotting, AEC = acoustic echo cancellation,
FA = false accept, oWW = openWakeWord, SDK = software development kit.

---

## Part 1 — Checker-loop repos

Two independent loops were run. In each, a fresh indifferent judge was forced to
pick whether a reference repo or `my-stt-tts` was better for a stated use case; on
a loss we captured the gap, implemented it (worktree-isolated, tested, merged),
and re-judged until `my-stt-tts` won. We adopted only genuinely-better **ideas**
from each reference, each empirically gated — no reference engine was vendored
unless it strictly beat ours.

### Loop A — Voice↔LLM conversation suitability

Source: `PLAN_checker_loop.md`. Use case (final, reframed): *different people in a
household talking to a Mac* — multi-user, on-device, capability-only (ecosystem
maturity/popularity explicitly excluded). All 5 repos were ultimately won by
`my-stt-tts`; the conversational core (barge-in, smart-turn, AEC, streaming STT,
context repair) was built out across rounds 1–5 to get there.

| # | Repo (URL)                                                                   | License    | Purpose / what it's good at                                                                 | Loop outcome / what we ported                                                                                  |
| - | ---------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 1 | **pipecat** ([pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat))    | BSD-2      | Mature real-time voice/multimodal AI framework (Daily.co); huge STT/TTS/LLM integration breadth, smart-turn, transports, client SDKs. | Hardest opponent (5 rounds). Drove our barge-in, Smart-Turn endpointing, false-interrupt gate, streaming STT, AEC, WebRTC/WS transport, telemetry, ESP32 client. Won round 5 on multi-user + on-device AEC + local-first topology. |
| 2 | **livekit/agents** ([livekit/agents](https://github.com/livekit/agents))     | Apache-2.0 | Framework for realtime voice AI agents over LiveKit WebRTC; strong cloud integration breadth. | Won first pass on voice-biometric speaker ID wired to cross-session per-speaker memory + on-device privacy/AEC + ESP32 satellite path. No gaps implemented. |
| 3 | **huggingface/speech-to-speech** ([huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech)) | Apache-2.0 | Modular local speech-to-speech pipeline built from open models (STT→LLM→TTS).               | Won first pass on per-speaker ID + memory + on-device AEC + home satellite path. Its judge flagged the real gap that speaker-ID was tested but not wired into the live loop — fixed afterward. |
| 4 | **dnhkng/GLaDOS** ([dnhkng/GLaDOS](https://github.com/dnhkng/GLaDOS))         | MIT        | Low-latency local voice "personality core" (Portal GLaDOS); interruptible local voice assistant. | Won first pass on the same multi-user + on-device + satellite axes. No gaps implemented.                       |
| 5 | **KoljaB/RealtimeSTT** ([KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)) (+ [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)) | MIT | Low-latency streaming STT with VAD + wake-word activation (paired with RealtimeTTS for streaming TTS). | Won first pass on per-speaker ID + memory + on-device AEC + satellite path. No gaps implemented.               |

### Loop B — Wake-word detection

Source: `PLAN_wake_checker_loop.md`. Use case: wake-word **detection quality** plus
gathering every useful detection/debug tool. Baseline = **openWakeWord** (8-phase
diverse, int16, threshold + gain). For each reference: a 3-judge panel decided if
it beat ours and on what; we ported only the portable, empirically-gated ideas and
re-judged until ours won. Loop completed 2026-06-21; ours won every closing
re-judge.

| # | Repo (URL)                                                                          | License    | Purpose / what it's good at                                                              | Loop outcome / what we ported                                                                                       |
| - | ----------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 0 | **openWakeWord** ([dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)) | Apache-2.0 | Open audio wake-word/phrase detection framework, performance + simplicity focus.         | **Our detection baseline** (not a contender). Official words hey_jarvis/alexa/hey_mycroft fire 99–100% on Albert.   |
| 1 | **sherpa-onnx KWS** ([k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx))   | Apache-2.0 | Next-gen-Kaldi on-device STT/TTS/VAD/KWS via ONNX; zero-train open-vocabulary keyword spotting. | Closed OURS_BETTER. Ported the KWS engine OR'd in for custom/self-trained words (multi-spelling/boost/threshold), zero new dep; recovered one otherwise-missed maziko activation. |
| 2 | **EfficientWord-Net** ([Ant-Brain/EfficientWord-Net](https://github.com/Ant-Brain/EfficientWord-Net)) | Apache-2.0 | Few-shot ("one-shot") hotword detection — enroll a word from the user's own clips.       | Built `EnrolledWake` (the idea, not the 88 MB ResNet). Mean-pool oWW embedding, max-cosine to enrolled refs; maziko recall 1/6 → 6/6, 0/23 hard-neg fires. Gated: windowing FALSE, mean-pool VIABLE. |
| 3 | **microWakeWord** ([OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)) | Apache-2.0 | TensorFlow wake-word training for microcontrollers (synthetic samples); streaming runtime. | Built temporal smoothing — ported the runtime idea (sliding moving-average fire criterion + refractory lockout) into the live detector, reconciled live==eval. PCEN front-end not portable. Defaults: window=1, refractory=8. |
| 4 | **Picovoice Porcupine** ([Picovoice/porcupine](https://github.com/Picovoice/porcupine)) | Apache-2.0 (SDK; models/engine proprietary, paid) | On-device wake-word detection powered by deep learning; commercial accuracy + tooling.   | Engine unadoptable (proprietary `.ppn`, paid). Ported two portable ideas: a unified `wake_sensitivity` 0–1 knob (calibrated off our FA curve) and a noise×SNR FA/hour benchmark harness with adaptive threshold bracketing. |
| 5 | **Mycroft Precise** ([MycroftAI/mycroft-precise](https://github.com/MycroftAI/mycroft-precise)) | Apache-2.0 | Lightweight RNN/GRU wake-word listener (Mycroft assistant).                              | Built two ideas (not the GRU): output calibration (per-word raw→[0,1] map, model-independent thresholds) and an eval-gated active-learning relabel loop (capture false-fire/miss → rebuild → keep only if separation/FA improves, else roll back). |

---

## Part 2 — Fluent-conversation agent projects (research 2026-06)

Web research into the most prominent stacks for **fluent spoken conversation with
an agent**: wake-word OR push-to-talk to start, the user speaks, the agent replies
in voice, and the user can **barge in / interrupt** the agent mid-speech to ask a
follow-up — i.e. full-duplex, low-latency, interruptible, streaming STT/TTS. Not
just wake-word detection. All URLs below were verified to resolve (2026-06-22).
Where a capability could not be pinned down, it is flagged.

In the capability columns: **WW** = has built-in wake-word; **Barge-in** = user can
interrupt the agent's speech; **Duplex** = listens while speaking (true full-duplex);
**Stream** = streaming STT and/or TTS.

### Open-source frameworks (pipeline / orchestration)

| Project (URL)                                                              | OSS / license                  | Local / cloud  | WW  | Barge-in | Duplex | Stream | Hardware                          | Best at                                                                 |
| -------------------------------------------------------------------------- | ------------------------------ | -------------- | --- | -------- | ------ | ------ | --------------------------------- | ----------------------------------------------------------------------- |
| **pipecat** ([pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat))  | OSS · BSD-2                     | local or cloud | via plugin | yes | yes | yes | Mac/server, browser, **ESP32** SDK | Production-grade, provider-agnostic real-time voice/multimodal pipelines with first-class barge-in + turn detection. |
| **LiveKit Agents** ([livekit/agents](https://github.com/livekit/agents))   | OSS · Apache-2.0 (turn model separate license) | cloud-leaning, open | via plugin | yes | yes | yes | WebRTC clients all platforms, **ESP32** SDK, telephony | Scalable WebRTC voice agents with robust semantic turn detection + broad client reach. |
| **Vocode** ([vocodedev/vocode-core](https://github.com/vocodedev/vocode-core)) | OSS · MIT                   | self-host or cloud | no | yes (cancels mid-utterance) | yes | yes | standard compute (system audio) | Phone-call LLM voice agents, modular Python. Activity has slowed (last release 2024-06). |
| **TEN Framework / TEN Agent** ([TEN-framework/ten-framework](https://github.com/TEN-framework/ten-framework)) | OSS · Apache-2.0 (added restrictions in some dirs) | cloud / hybrid | via VAD | yes | yes | yes | server, RTC clients, **ESP32** | Real-time multimodal agents with strong native VAD + turn detection; OpenAI Realtime / Gemini Live integration. |

### Open-source local / household assistants

| Project (URL)                                                              | OSS / license     | Local / cloud  | WW  | Barge-in | Duplex | Stream | Hardware                          | Best at                                                                 |
| -------------------------------------------------------------------------- | ----------------- | -------------- | --- | -------- | ------ | ------ | --------------------------------- | ----------------------------------------------------------------------- |
| **Home Assistant Voice** (Assist + [Wyoming](https://github.com/rhasspy/wyoming-satellite) + [Voice PE](https://www.home-assistant.io/voice-pe/)) | OSS + open HW/firmware | local-capable | **yes** (microWakeWord on-device) | **no** (turn-based) | no | yes (Wyoming) | **ESP32-S3** + XMOS satellites, Pi/server | Privacy-first multi-room **smart-home control** with cheap ESP32 satellites. No documented mid-speech interrupt. |
| **Rhasspy** ([rhasspy/rhasspy](https://github.com/rhasspy/rhasspy))        | OSS · MIT         | fully offline  | yes | no       | no     | partial | Raspberry Pi, Linux               | Offline intent/command control on modest hardware. Largely superseded by the HA/Wyoming stack (same author). |
| **Willow** ([HeyWillow/willow](https://github.com/HeyWillow/willow))       | OSS · Apache-2.0  | local / self-host | **yes** | no (not headline) | no | yes | **ESP32-S3-BOX** + self-hosted inference server | Fast on-device wake + offloaded ASR/TTS/LLM to a self-hosted GPU server. Quiet development recently. |
| **dnhkng/GLaDOS** ([dnhkng/GLaDOS](https://github.com/dnhkng/GLaDOS))       | OSS · MIT         | fully local    | **no** (always-listening) | **yes** (VAD clips playback) | yes | yes | Linux (primary), **Mac (experimental)**, 8 GB SBCs | Low-latency (<600 ms), interruptible, personality-driven local companion. Parakeet + Kokoro + Ollama. |
| **RealtimeSTT + RealtimeTTS** ([KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT), [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)) | OSS · MIT (both) | local-first | yes (STT) | yes (via [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat)) | yes | yes | Mac/Linux, browser front-end | DIY building blocks for a local, interruptible voice loop; <100 ms first-audio TTS. |

### Open-source speech-native models (Kyutai)

| Project (URL)                                                              | OSS / license            | Local / cloud  | WW  | Barge-in | Duplex | Stream | Hardware                          | Best at                                                                 |
| -------------------------------------------------------------------------- | ------------------------ | -------------- | --- | -------- | ------ | ------ | --------------------------------- | ----------------------------------------------------------------------- |
| **Unmute** ([kyutai-labs/unmute](https://github.com/kyutai-labs/unmute))   | OSS · MIT                | local (GPU)    | no  | yes (semantic VAD) | yes | yes | GPU-class server                  | Giving any reasoning LLM a fast, low-latency, interruptible voice + tool use. |
| **Moshi** ([kyutai-labs/moshi](https://github.com/kyutai-labs/moshi))      | OSS · code Apache-2.0/MIT, weights CC-BY-4.0 | local | no | implied by duplex | **yes (true full-duplex S2S)** | yes | **Apple Silicon via MLX**, else ~24 GB GPU | Most natural, lowest-latency end-to-end spoken dialogue, runnable on a Mac. No tools/multi-user/smart-home. |

### Commercial / cloud APIs (not local)

| Project (URL)                                                              | OSS / commercial | Local / cloud | WW  | Barge-in | Duplex | Stream | Hardware           | Best at                                                                 |
| -------------------------------------------------------------------------- | ---------------- | ------------- | --- | -------- | ------ | ------ | ------------------ | ----------------------------------------------------------------------- |
| **OpenAI Realtime API / Agents SDK** ([docs](https://developers.openai.com/api/docs/guides/realtime)) | commercial | cloud-only | no | **yes** (server VAD cancels response) | yes | yes | WebRTC/WS/SIP, any client | Highest-quality production speech-to-speech with realtime tool use. No local/privacy mode. |
| **Google Gemini Live API** ([docs](https://ai.google.dev/gemini-api/docs/live-api)) | commercial | cloud-only | no | **yes** (halts audio on VAD interrupt) | yes | yes | WebSocket, any client | Low-latency multimodal (voice + vision) cloud agents. |
| **Vapi** ([vapi.ai](https://vapi.ai/pricing))                              | commercial       | cloud         | no  | yes (interrupt detection) | yes | yes | telephony/SIP, web | Turnkey phone voice agents with natural turn-taking (~600 ms). |
| **Retell AI** ([retellai.com](https://www.retellai.com/))                  | commercial       | cloud         | no  | yes (clean recovery w/ context) | yes | yes | telephony, web     | Production phone voice automation at scale (~580–620 ms). |

### Top picks for a local, interruptible, Mac / household voice agent

Ranked for the specific brief — local-first, privacy-respecting, runs on a Mac,
multi-user household, ESP32 satellites, true mid-speech barge-in:

1. **pipecat** — Best all-rounder for this brief. BSD-2 license, runs **fully local
   on a Mac** (Whisper + Kokoro/Piper + Ollama), native **barge-in + turn detection
   + streaming**, and an official **ESP32 client SDK** for household satellites. The
   most active and production-proven of the local-capable frameworks — and the
   reference `my-stt-tts` worked hardest to match (5 rounds in Loop A).
2. **dnhkng/GLaDOS** — The strongest ready-made **local, interruptible** assistant:
   MIT, real barge-in (VAD clips playback), <600 ms latency, Parakeet + Kokoro +
   Ollama, runs on Mac (experimental) and SBCs. Best if you want a working
   interruptible agent now rather than assembling a framework; weaker on
   multi-user / ESP32-fleet and smart-home control.
3. **Home Assistant Voice (Assist + Wyoming + Voice PE / ESP32)** — The pick for
   **multi-user household + smart-home control with cheap ESP32 satellites**, fully
   local and open hardware. Caveat: it is turn-based — **true barge-in during the
   agent's speech is not a documented capability** — so pair it with pipecat (or
   GLaDOS for the conversational layer) if mid-speech interruption is a hard
   requirement.

Honorable mention: **Moshi (MLX)** if you specifically want the most natural
full-duplex speech-to-speech *on Apple Silicon* — but it is a chat model, not a
household controller (no tools / multi-user / Home Assistant integration).

### Caveats on the research

+ **Barge-in**: Home Assistant Assist, Rhasspy, and Willow are turn-based; treat
  "no mid-speech interruption" as the safe assumption unless confirmed otherwise.
+ **Activity**: Vocode (last release 2024-06) and Willow appear less actively
  developed than pipecat / LiveKit / TEN.
+ URLs verified 2026-06-22. Willow's repo moved from `toverainc/willow` to
  `HeyWillow/willow`; Rhasspy's `rhasspy3` preview was archived 2025-10 (the
  maintained line is the Home Assistant / Wyoming stack).
