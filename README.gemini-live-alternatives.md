# Gemini Live and its open-source alternatives

A research note answering two questions about Google's "talk to Gemini" voice
mode (Gemini **Live API**):

1. **Are there open-source projects with similar capability/quality?** — Yes, a
   handful get close on *latency* and *full-duplex/barge-in*; none yet match the
   *combined* quality + multilingual breadth + reliability of Gemini Live as a
   product. The closest open analog is **Kyutai Moshi** (open-weights, native
   full-duplex speech-to-speech) and the **Unmute** stack (open low-latency
   STT + TTS that turns any text LLM into a voice agent).
2. **What is Gemini Live's tech stack, and is there a Google repo?** — Gemini
   Live is a **closed model** offered as a **stateful WebSocket API** with two
   selectable architectures (a **native audio-to-audio** model and a
   **half-cascade** model). The model **weights are proprietary**; what Google
   open-sources is the **API surface, SDKs, and reference/demo code**
   (`googleapis/python-genai`, `google-gemini/cookbook`,
   `google-gemini/live-api-web-console`, `GoogleCloudPlatform/generative-ai`).

> **How to read this note.** Lines tagged **FACT** are sourced to public
> Google/vendor docs or repos (linked inline). Lines tagged **INFERENCE** are
> my reasoning from those facts. I do **not** claim knowledge of Gemini's
> closed internals beyond what Google has published.

All URLs were checked during research (June 2026). Model previews move fast —
exact preview suffixes (dates) will drift; the architecture split and repos are
stable.

---

## 1. What "talk to Gemini" / Gemini Live actually is

"Talk to Gemini" in the phone app, and the developer-facing **Gemini Live API**
(formerly "Multimodal Live API") in Google AI Studio / Vertex AI, are the same
capability: a **real-time, low-latency, interruptible spoken conversation** with
a Gemini model.

**FACT — it is a stateful WebSocket session, not request/response.** Google's
docs describe a "Stateful WebSocket connection (WSS)" carrying bidirectional
streams: audio in continuously, audio out continuously, so there is no
per-turn round-trip delay
([ai.google.dev/gemini-api/docs/live-api](https://ai.google.dev/gemini-api/docs/live-api)).

**FACT — published capabilities:**

- **Native audio in/out.** Input is "raw 16-bit PCM audio, 16 kHz"; output is
  "raw 16-bit PCM audio, 24 kHz" — i.e. the API returns real audio, not text it
  expects you to synthesize
  ([live-api overview](https://ai.google.dev/gemini-api/docs/live-api)).
- **Interruption / barge-in.** "Users can interrupt the model at any time."
  Voice Activity Detection (VAD) drives turn-taking; on a detected interruption
  "the ongoing generation is canceled and discarded" and the server emits an
  interruption signal. VAD sensitivity is tunable
  (`startOfSpeechSensitivity` / `endOfSpeechSensitivity`), or you can disable
  automatic VAD and send manual `activityStart` / `activityEnd`
  ([capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities)).
- **Multimodal input.** Audio + images (JPEG, ≤ 1 FPS) + text in the same
  session, so the model can talk about a live camera/screen stream
  ([live-api overview](https://ai.google.dev/gemini-api/docs/live-api)).
- **Function calling / tool use**, including Google Search grounding and code
  execution, mid-conversation
  ([capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities)).
- **Voices and languages.** Native-audio voices share the Gemini text-to-speech
  voice set (e.g. "Kore"); Google's own Vertex blog cites "30 HD voices in 24
  languages" for native audio, while the developer capabilities guide lists
  "97 languages" overall for the Live API
  ([Vertex blog](https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai),
  [capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities)).
- **Affective dialog + proactive audio** (native-audio only, `v1alpha`): the
  model adapts tone to the user's emotion, and can "proactively decide not to
  respond" instead of relying on naive VAD
  ([capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities)).

**FACT — latency is "real-time / sub-second" by design** (continuous streaming,
no STT→LLM→TTS turn boundary). Google markets it as eliminating "the awkward,
turn-taking delays"; it does not publish a single headline millisecond number,
which varies by model, region, and modality
([Vertex blog](https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai)).

---

## 2. The tech stack — cascade or native audio-to-audio?

**Both — Gemini Live ships two architectures and you pick one per session.**
This is the single most important technical fact about the stack.

### 2a. Native audio (audio-to-audio)

**FACT.** The native-audio model "processes raw audio natively through a single,
low-latency model" and "generate[s] speech directly from the model's internal
state," which Google calls "the core technical innovation that dramatically
reduces latency" and yields richer prosody/emotion
([Vertex blog](https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai)).
Current native-audio model IDs include
`gemini-2.5-flash-native-audio-preview-12-2025`,
`gemini-live-2.5-flash-preview-native-audio-09-2025`, and the newer
`gemini-3.1-flash-live-preview`
([capabilities guide](https://ai.google.dev/gemini-api/docs/live-api/capabilities),
[python-genai issue #1725](https://github.com/googleapis/python-genai/issues/1725)).

### 2b. Half-cascade

**FACT.** The **half-cascade** model takes **audio in**, reasons in **text
internally**, then renders the reply through a **separate top-tier TTS** layer.
Google positions it as the production workhorse for **tool-heavy** flows
because the internal text step "plays nicely with function calls and logs," at a
small cost in vocal warmth vs. native audio
([Gemini Live API guide, binaryverseai](https://binaryverseai.com/gemini-live-api-guide/)).
LiveKit's plugin docs describe the same pattern: pair the Live API's real-time
speech *comprehension* with a separate TTS plugin to keep "complete control over
the speech output"
([LiveKit Gemini plugin](https://docs.livekit.io/agents/models/realtime/plugins/gemini/)).
The half-cascade Live model is exposed as `gemini-live-2.5-flash` (vs. the
`...native-audio...` variants above)
([Vertex 2.5 Flash Live API docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api)).

> **INFERENCE.** "Native audio" is genuine end-to-end speech-to-speech;
> "half-cascade" is a cascade where only the *output* is cascaded (audio-in is
> still native, text is the bridge, TTS is the final stage). So Gemini Live is
> *not* a single architecture — Google sells the audio-to-audio model for
> naturalness and the half-cascade model for tool reliability.

### 2c. Transport

**FACT.** The wire protocol is **WebSocket (WSS)**, stateful and bidirectional
([live-api overview](https://ai.google.dev/gemini-api/docs/live-api)). Unlike
OpenAI's Realtime API, Google does **not** document a first-class WebRTC entry
point to the model itself; WebRTC typically appears one layer out, in
orchestrators like LiveKit/Pipecat that bridge a browser's WebRTC to the Gemini
WebSocket
([LiveKit Gemini plugin](https://docs.livekit.io/agents/models/realtime/plugins/gemini/)).

### 2d. Is there a Google repo? What is open vs. proprietary?

**Proprietary:** the **model weights** and serving stack. There is no
open-weights Gemini.

**Open (Apache-2.0), and officially Google's:**

| Repo                                                                                                  | What it is                                                                                                                 |
| :---------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| [`googleapis/python-genai`](https://github.com/googleapis/python-genai)                               | Official Google Gen AI Python SDK. **Has** Live support via `client.aio.live.connect()` (handles the WebSocket handshake). |
| [`google-gemini/cookbook`](https://github.com/google-gemini/cookbook)                                 | Official Gemini API cookbook; includes "Get started with Live API" and Live-translation notebooks.                         |
| [`google-gemini/live-api-web-console`](https://github.com/google-gemini/live-api-web-console)         | React starter for the Live API over WebSockets. Explicitly an **experiment, "not an official Google product."**            |
| [`google-gemini/gemini-live-api-examples`](https://github.com/google-gemini/gemini-live-api-examples) | Voice-agent examples (vision + text in real time).                                                                         |
| [`GoogleCloudPlatform/generative-ai`](https://github.com/GoogleCloudPlatform/generative-ai)           | Vertex AI samples incl. `multimodal-live-api/` native-audio WebSocket demo apps (React frontend + Python WebSocket proxy). |

Sources for the table:
[python-genai live.py](https://github.com/googleapis/python-genai/blob/main/google/genai/live.py),
[Live API SDK quickstart](https://ai.google.dev/gemini-api/docs/live-api/get-started-sdk),
[cookbook](https://github.com/google-gemini/cookbook),
[live-api-web-console](https://github.com/google-gemini/live-api-web-console),
[GCP generative-ai live demo](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/multimodal-live-api/native-audio-websocket-demo-apps/react-demo-app).

**INFERENCE.** "Reference code" for Gemini Live exists and is permissively
licensed, but it is *client/glue* code against a hosted closed model — you can
clone the demos, you cannot run Gemini Live offline.

---

## 3. Open-source projects with similar capability

Two families approximate Gemini Live:

- **Native / near-native speech-to-speech models** (one model does audio→audio,
  open weights) — the true architectural analog. Examples: Moshi, Qwen-Omni,
  GLM-4-Voice, Step-Audio 2, mini-omni, Freeze-Omni, Ultravox (audio-in only).
- **Cascade orchestrators** (open frameworks that wire STT + LLM + TTS with
  barge-in) — the *product* analog you'd actually deploy. Examples: Pipecat,
  LiveKit Agents, TEN, Unmute, HF speech-to-speech. These can *call* a hosted
  S2S brain (incl. Gemini Live or OpenAI Realtime) too.

### Comparison table

| Project                 | Open weights?                  | On-device (Mac)?              | Full-duplex / barge-in             | Quality vs. Gemini Live                         | URL                                                                             |
| :---------------------- | :----------------------------- | :---------------------------- | :--------------------------------- | :---------------------------------------------- | :------------------------------------------------------------------------------ |
| **Kyutai Moshi**        | Yes (CC-BY-4.0 weights)        | Yes — MLX/Apple Silicon       | **Native full-duplex** (2 streams) | Closest open analog; lower text smarts/breadth  | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)                       |
| **Kyutai Unmute**       | Yes (open STT+TTS stack)       | Server (GPU); STT/TTS modular | Streaming, semantic VAD barge-in   | Quality follows the LLM you plug in             | [kyutai.org/unmute](https://kyutai.org/unmute)                                  |
| **OpenAI Realtime**     | **No** (closed `gpt-realtime`) | No (hosted)                   | Yes (interruptible)                | Comparable product tier; both closed            | [Realtime docs](https://developers.openai.com/api/docs/guides/realtime)         |
| **Qwen2.5-Omni**        | Yes (Apache-2.0, 3B/7B)        | Possible (7B, quantized)      | Streaming S2S; not true duplex     | Strong open S2S; behind on latency/duplex       | [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)                   |
| **GLM-4-Voice**         | Yes (9B; code Apache-2.0)      | Heavy (9B)                    | Streaming S2S (ZH/EN)              | Good ZH/EN; narrower than Gemini                | [THUDM/GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice)                       |
| **Step-Audio 2 (mini)** | Yes (Apache-2.0, 8B)           | Possible (8B mini)            | Streaming S2S + tools/RAG          | Claims > GPT-4o-Audio on benchmarks; new (2025) | [stepfun-ai/Step-Audio2](https://github.com/stepfun-ai/Step-Audio2)             |
| **mini-omni / -omni2**  | Yes (research weights)         | Yes (small)                   | Streaming; -omni2 adds duplex      | Research-grade; clearly below Gemini            | [gpt-omni/mini-omni2](https://github.com/gpt-omni/mini-omni2)                   |
| **Freeze-Omni**         | Yes (frozen-LLM S2S)           | GPU-leaning                   | Low-latency S2S                    | Research-grade; preserves base LLM smarts       | [arXiv 2411.00774](https://arxiv.org/html/2411.00774v1)                         |
| **Ultravox (Fixie)**    | Yes (weights on HF)            | Possible                      | Audio-**in** only (needs TTS)      | ~150 ms TTFT understanding; not audio-out       | [fixie-ai/ultravox](https://github.com/fixie-ai/ultravox)                       |
| **Pipecat**             | Framework (BSD-2)              | Yes (orchestrator)            | Yes (endpointing + interrupts)     | = quality of chosen STT/LLM/TTS                 | [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat)                     |
| **LiveKit Agents**      | Framework (Apache-2.0)         | Yes (orchestrator)            | Yes (turn detection, WebRTC)       | = quality of chosen components                  | [livekit/agents](https://github.com/livekit/agents)                             |
| **TEN-framework**       | Framework (Apache-2.0)         | Yes (orchestrator)            | Yes (VAD, turn detection)          | = quality of chosen components                  | [TEN-framework/ten-framework](https://github.com/TEN-framework/ten-framework)   |
| **HF speech-to-speech** | Framework (modular)            | Yes — MLX path                | Cascade; community barge-in forks  | = quality of chosen components                  | [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) |

### Notes on the strongest entries

**Kyutai Moshi — the truest open analog.**
**FACT.** Full-duplex speech-text foundation model: it models **two audio
streams at once** (its own + the user's) plus a text "inner monologue,"
built on the streaming **Mimi** neural codec. **160 ms** theoretical latency,
~**200 ms** on an L4 GPU. **7B** temporal transformer. Code is MIT (Python) /
Apache-2.0 (Rust); **weights CC-BY-4.0**. Runs on **PyTorch, Rust/Candle, and
MLX (Apple Silicon)**
([kyutai-labs/moshi](https://github.com/kyutai-labs/moshi),
[Moshi paper](https://arxiv.org/html/2410.00037v2)).
**INFERENCE.** Moshi is the only widely-available open model that is *natively*
full-duplex like Gemini's native-audio mode; the gap is in raw LLM intelligence,
language breadth, and tool-use polish, not in the duplex mechanism.

**Kyutai Unmute — the modular open analog.**
**FACT.** Open low-latency streaming STT + TTS with semantic VAD that wraps "any
LLM" into a voice agent at **sub-second** total latency; every component is
open-source ([kyutai.org/unmute](https://kyutai.org/unmute)).
**INFERENCE.** This is the *practical* way to get Gemini-Live-like UX on open
weights today: Unmute (or Pipecat/LiveKit/TEN) for the duplex plumbing + a good
open LLM for the brain.

**OpenAI Realtime API — the closest *product*, also closed.**
**FACT.** Speech-to-speech `gpt-realtime` / `gpt-4o-realtime-preview` over
**WebRTC or WebSocket**, with interruption and function calling; the model is
**proprietary** (no open weights). What's "open" is client patterns and
SDKs/examples
([Realtime guide](https://developers.openai.com/api/docs/guides/realtime),
[gbaeke/realtime-webrtc](https://github.com/gbaeke/realtime-webrtc)).

**Cascade orchestrators (Pipecat, LiveKit Agents, TEN).**
**FACT.** Open frameworks (BSD-2 / Apache-2.0) that assemble STT→LLM→TTS with
phrase endpointing, VAD/turn detection, barge-in, and WebRTC/WebSocket transport
— and can also drive native S2S brains (OpenAI Realtime, Gemini Live)
([pipecat](https://github.com/pipecat-ai/pipecat),
[livekit/agents](https://github.com/livekit/agents),
[TEN](https://github.com/TEN-framework/ten-framework)).
**INFERENCE.** These don't *match* Gemini Live's model — they *orchestrate*
toward the same UX, and their quality is exactly the quality of the components
you plug in.

---

## 4. Bottom line

**FACT + INFERENCE.** As of mid-2026:

- **Closest open analog by architecture:** **Kyutai Moshi** — open-weights,
  natively full-duplex, runs locally on Apple Silicon via MLX.
- **Closest open analog by deliverable UX:** the **Unmute** stack (or
  **Pipecat / LiveKit / TEN**) gluing open STT + open LLM + open TTS, with
  barge-in, at sub-second latency.
- **Runs on a Mac?** Yes — Moshi (MLX), `huggingface/speech-to-speech` (MLX
  path), and the smaller omni models (mini-omni, Qwen2.5-Omni-3B) can run on
  Apple Silicon, though large omni models (GLM-4-Voice 9B, Step-Audio 2 8B) want
  more memory/GPU than a typical laptop.
- **The honest gap:** open S2S models still trail Gemini Live on the
  *combination* of conversational quality, multilingual breadth (Gemini cites
  ~24–97 languages depending on metric), tool-calling reliability, and
  managed-service latency. They are catching up on the *mechanism* (duplex,
  barge-in, streaming) faster than on the *model quality*.

### How this relates to `my-stt-tts`

`my-stt-tts` is squarely in the **cascade orchestrator** family — wake word →
STT → pluggable LLM brain → TTS, with full-duplex **barge-in** (mic stays live
during playback, false-interrupt gate + acoustic interruption predictor + AEC
for open speakers). That's the same UX target as Gemini Live's **half-cascade**
mode, run locally with audio staying on-device.

It also has an **optional native S2S "brain"**: `BRAIN_MODE=realtime` streams
mic audio to the **OpenAI Realtime API** over WebSocket and plays the returned
audio back, collapsing the cascade's per-turn latency (key-gated; falls back to
the cascade without a key). So `my-stt-tts` already spans both Gemini-Live
shapes — a local cascade by default, an optional hosted audio-to-audio brain
when you want minimal latency. Swapping that brain to **Gemini Live** (over its
WebSocket) or to **Moshi** (fully local) is the natural extension to close the
remaining quality gap.

---

## Sources

- Gemini Live API overview — <https://ai.google.dev/gemini-api/docs/live-api>
- Gemini Live API capabilities — <https://ai.google.dev/gemini-api/docs/live-api/capabilities>
- Gemini Live API SDK quickstart — <https://ai.google.dev/gemini-api/docs/live-api/get-started-sdk>
- Native audio on Vertex AI (Google Cloud blog) — <https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai>
- Vertex 2.5 Flash Live API model docs — <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api>
- Gemini Live API guide (half-cascade vs native, binaryverseai) — <https://binaryverseai.com/gemini-live-api-guide/>
- LiveKit Gemini plugin — <https://docs.livekit.io/agents/models/realtime/plugins/gemini/>
- `googleapis/python-genai` — <https://github.com/googleapis/python-genai> · live.py — <https://github.com/googleapis/python-genai/blob/main/google/genai/live.py>
- `google-gemini/cookbook` — <https://github.com/google-gemini/cookbook>
- `google-gemini/live-api-web-console` — <https://github.com/google-gemini/live-api-web-console>
- `google-gemini/gemini-live-api-examples` — <https://github.com/google-gemini/gemini-live-api-examples>
- `GoogleCloudPlatform/generative-ai` (Live demo) — <https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/multimodal-live-api/native-audio-websocket-demo-apps/react-demo-app>
- Kyutai Moshi — <https://github.com/kyutai-labs/moshi> · paper — <https://arxiv.org/html/2410.00037v2>
- Kyutai Unmute — <https://kyutai.org/unmute>
- OpenAI Realtime API — <https://developers.openai.com/api/docs/guides/realtime>
- Qwen2.5-Omni — <https://github.com/QwenLM/Qwen2.5-Omni>
- GLM-4-Voice — <https://github.com/THUDM/GLM-4-Voice>
- Step-Audio 2 — <https://github.com/stepfun-ai/Step-Audio2>
- mini-omni2 — <https://github.com/gpt-omni/mini-omni2>
- Freeze-Omni (paper) — <https://arxiv.org/html/2411.00774v1>
- Ultravox (Fixie AI) — <https://github.com/fixie-ai/ultravox>
- Pipecat — <https://github.com/pipecat-ai/pipecat>
- LiveKit Agents — <https://github.com/livekit/agents>
- TEN-framework — <https://github.com/TEN-framework/ten-framework>
- Hugging Face speech-to-speech — <https://github.com/huggingface/speech-to-speech>
