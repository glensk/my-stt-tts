# my-stt-tts — Mobile client guide (React Native)

A **skeleton/guide**, not a finished app: it shows how to build a phone satellite
that captures the mic, streams **16 kHz mono int16-LE PCM** to a my-stt-tts
server over a WebSocket, and plays the TTS PCM that comes back — conformant to
[`../PROTOCOL.md`](../PROTOCOL.md). The same approach maps 1:1 to native Swift
(iOS) and Kotlin (Android); see [§5](#5-native-swift--kotlin-pointers).

A phone is a **satellite**, so it uses the **native transport** (PROTOCOL.md §2):
connect to `ws://HOST:8770/`, send a JSON `hello`, wait for `ready`, then stream
binary PCM both ways.

## 1. Architecture (the loop)

```text
   mic (PCM frames) ──► resample to 16 kHz ──► float→int16 LE ──► WS binary frame ──► server
                                                                                        │
   speaker ◄── int16 LE→float ◄── WS binary frame ◄──────────────────────────────── TTS PCM
```

Three concurrent concerns, exactly like the reference Python satellite
(`src/my_stt_tts/satellite.py`, which runs a mic→socket pump and a
socket→speaker pump in parallel):

1. **Handshake** — open the socket, send `hello`, wait for the `ready` text frame.
2. **Uplink** — capture mic → 16 kHz mono int16-LE → `ws.send(arrayBuffer)`.
3. **Downlink** — on each binary message → decode int16-LE → enqueue to playback.

Keep the mic **open during playback** (the server does barge-in on the live mic —
PROTOCOL.md §5). Rely on the OS echo canceller (`VoiceProcessingIO` on iOS,
`VOICE_COMMUNICATION` audio source on Android) so the speaker doesn't feed back.

## 2. Dependencies

| Need                 | React Native package                                  | Why                                                           |
| :------------------- | :---------------------------------------------------- | :----------------------------------------------------------- |
| Raw PCM mic capture  | `@dr.pogodin/react-native-audio` or `react-native-live-audio-stream` | Emits raw PCM buffers (NOT a compressed file) frame-by-frame. |
| PCM playback         | `react-native-track-player` (PCM) or a native module  | Stream raw PCM to the speaker with low latency.              |
| WebSocket            | built-in `global.WebSocket`                           | RN ships a WebSocket; set `binaryType` is not configurable but binary `send`/`onmessage` work via `ArrayBuffer`/base64 — see §4. |
| base64 (if needed)   | `base64-js`                                           | Some PCM libs hand you base64 strings, not `ArrayBuffer`.    |

`react-native-live-audio-stream` is the simplest path: configure it for
**16000 Hz, 1 channel, 16-bit** and it emits base64-encoded int16-LE chunks you
forward straight to the socket — **no client-side resampling needed**.

```bash
npm i react-native-live-audio-stream base64-js
# + a PCM player module of your choice
cd ios && pod install
```

Both platforms need mic permission: iOS `NSMicrophoneUsageDescription` in
`Info.plist`; Android `RECORD_AUDIO` in `AndroidManifest.xml`.

## 3. PCM conversion helpers

These mirror the server's `encode_frame`/`decode_frame` (PROTOCOL.md §1) and the
JS module in `clients/web/`. Scale = `32767`, little-endian, mono.

```js
// Float32 [-1,1] -> Int16 LE ArrayBuffer (server scale ×32767).
export function floatToInt16LE(f32) {
  const dv = new DataView(new ArrayBuffer(f32.length * 2));
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    dv.setInt16(i * 2, Math.round(s * 32767), true); // true = little-endian
  }
  return dv.buffer;
}

// Int16 LE ArrayBuffer -> Float32 [-1,1].
export function int16LEToFloat(buf) {
  const dv = new DataView(buf);
  const n = buf.byteLength >> 1; // drop a trailing odd byte
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt16(i * 2, true) / 32767;
  return out;
}
```

If your mic library already gives 16 kHz 16-bit PCM (recommended config), you do
NOT need `floatToInt16LE` on the uplink — forward the raw int16-LE bytes as-is.
You still need `int16LEToFloat` only if your player wants float; most PCM players
accept raw int16-LE bytes directly.

## 4. The client (the load-bearing parts)

```js
import LiveAudioStream from "react-native-live-audio-stream";
import { Buffer } from "buffer"; // RN base64 <-> bytes

const SERVER_URL = "ws://192.168.1.10:8770/"; // native transport, PROTOCOL.md §2
const TOKEN = null;                            // set iff server has TRANSPORT_TOKEN

// --- 1. Handshake (PROTOCOL.md §2.1) -------------------------------------
function makeHello() {
  const env = { type: "hello", version: 1, sample_rate: 16000, role: "satellite" };
  if (TOKEN) env.token = TOKEN;
  return JSON.stringify(env);
}

export function startSatellite() {
  const ws = new WebSocket(SERVER_URL);
  ws.binaryType = "arraybuffer"; // RN: binary frames arrive as ArrayBuffer
  let ready = false;

  ws.onopen = () => ws.send(makeHello());           // TEXT hello first

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      // TEXT control frame (PROTOCOL.md §4). Switch on `type`.
      const msg = JSON.parse(ev.data);
      if (msg.type === "ready") { ready = true; startMic(ws); }   // §2.2
      // unknown types: ignore (forward-compatible)
    } else {
      // BINARY frame = TTS PCM (int16 LE). Decode + play.
      const f32 = int16LEToFloat(ev.data);
      playPcm(f32); // hand to your PCM player (16 kHz mono)
    }
  };

  ws.onclose = (e) => {
    // Code 1008 = handshake rejected (bad token/version) — PROTOCOL.md §2.3
    if (e.code === 1008) console.warn("rejected:", e.reason);
    LiveAudioStream.stop();
  };
  return ws;
}

// --- 2. Uplink: mic -> 16 kHz int16 LE -> socket -------------------------
function startMic(ws) {
  LiveAudioStream.init({
    sampleRate: 16000,   // capture at the pipeline rate -> no resampling
    channels: 1,         // mono
    bitsPerSample: 16,   // int16
    audioSource: 6,      // Android VOICE_RECOGNITION (or 7 = VOICE_COMMUNICATION for AEC)
    bufferSize: 1024,    // ~32 ms @ 16 kHz; ~20-40 ms/frame is ideal (PROTOCOL.md §3)
  });
  LiveAudioStream.on("data", (base64Chunk) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    // The lib emits base64 int16-LE PCM; decode to bytes and send as binary.
    const bytes = Buffer.from(base64Chunk, "base64");
    ws.send(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  });
  LiveAudioStream.start();
}
```

`playPcm(f32 | bytes)` is the one piece that depends on your chosen PCM player.
The contract: enqueue **16 kHz, mono** samples and play them **back-to-back**
(gapless), like the web client's `playHead` scheduling. With
`react-native-track-player`'s PCM mode or a small native module, push each frame
to its ring buffer as it arrives.

## 5. Native (Swift / Kotlin) pointers

The protocol is identical; only the platform APIs change.

**iOS / Swift:**

- Mic: `AVAudioEngine` input node with an `AVAudioFormat` of 16 kHz, 1 channel,
  `pcmFormatInt16`. Install a tap, read `AVAudioPCMBuffer.int16ChannelData`, send
  the bytes as a binary WS frame.
- Echo: use `AVAudioSession` category `.playAndRecord` with mode `.voiceChat`
  (engages `VoiceProcessingIO` AEC), so the mic stays open during playback.
- WebSocket: `URLSessionWebSocketTask`; `.send(.string(hello))` then
  `.send(.data(pcm))`; `receive` returns `.string` (control) or `.data` (PCM).
- Playback: schedule `AVAudioPCMBuffer`s on an `AVAudioPlayerNode` back-to-back.

**Android / Kotlin:**

- Mic: `AudioRecord` with `MediaRecorder.AudioSource.VOICE_COMMUNICATION` (AEC),
  16000 Hz, `CHANNEL_IN_MONO`, `ENCODING_PCM_16BIT`; read into a `ShortArray` →
  `ByteBuffer.order(LITTLE_ENDIAN)` → send as binary.
- WebSocket: OkHttp `WebSocket`; `send(hello)` (String), then
  `send(ByteString.of(pcmBytes))`; `onMessage(text)` = control, `onMessage(bytes)`
  = PCM.
- Playback: `AudioTrack` in `MODE_STREAM`, 16000 Hz, mono, 16-bit; `write()` each
  decoded frame.

## 6. Conformance checklist

- [ ] Connect to `ws://HOST:8770/` (native), plain `ws`.
- [ ] Send TEXT `hello`: `{type:"hello",version:1,sample_rate:16000,role:"satellite"[,token]}`.
- [ ] Wait for TEXT `{type:"ready",...}` before streaming mic.
- [ ] Capture mono 16 kHz int16 (configure the mic lib; avoid resampling).
- [ ] Send mic PCM as **binary** int16-LE frames (~20–40 ms each, even byte count).
- [ ] Treat **binary** downstream frames as TTS PCM → decode int16-LE → play gapless.
- [ ] Parse **text** frames as JSON `{type,...}`; ignore unknown types.
- [ ] Keep the mic open during playback; rely on OS AEC (server does barge-in).
- [ ] Handle close code `1008` as "handshake rejected".

## 7. Reference

- Wire protocol: [`../PROTOCOL.md`](../PROTOCOL.md) (sections §1–§4, §8).
- Reference satellite (Python, same protocol): `src/my_stt_tts/satellite.py`.
- Browser client (same PCM helpers, JS): [`../web/`](../web/).
