# my-stt-tts Satellite Audio Protocol

This document specifies the wire protocol a **satellite client** (ESP32, mobile
app, browser, headless script) speaks to the `my-stt-tts` server to carry audio
to and from the central "brain". It is derived **directly** from the server
source — every field below is cross-referenced to the exact function that
produces or consumes it. Implement against this and your client will interoperate
without touching the core app.

There are **two transports** that share the *same* audio payload format but differ
in how the connection is established and authenticated:

| Transport       | Endpoint            | Default port | Handshake                       | WebSocket impl on server                  |
| :-------------- | :------------------ | :----------- | :------------------------------ | :---------------------------------------- |
| **Native WS**   | `ws://HOST:PORT/`   | `8770`       | JSON `hello` text frame + token | `websockets` lib (`ws_transport.py`)      |
| **Browser WS**  | `ws://HOST:PORT/ws/audio` | `8765`  | HTTP upgrade only (no JSON, no token) | hand-rolled RFC 6455 (`ws_frame.py`) |

**An ESP32 / mobile / headless satellite uses the Native WS transport.** The
Browser WS path exists only because the GUI is served by stdlib `http.server` on
the same origin (so its strict CSP `connect-src 'self'` permits the socket); it
has no token gate and is documented in section [§7](#7-browser-ws-transport).

Both carry the identical audio payload: **16 kHz, mono, signed 16-bit
little-endian PCM**, sent as raw **binary** WebSocket frames in both directions.

---

## 1. Audio payload format (identical on both transports)

The canonical encode/decode lives in
[`src/my_stt_tts/transport.py`](../src/my_stt_tts/transport.py)
(`encode_frame` / `decode_frame`). It is the single source of truth for the wire
sample format:

```python
PROTOCOL_VERSION = 1
_INT16_SCALE = 32767.0

def encode_frame(pcm: np.ndarray) -> bytes:
    arr = np.asarray(pcm, dtype=np.float32).ravel()
    if arr.size == 0:
        return b""
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * _INT16_SCALE).astype("<i2").tobytes()

def decode_frame(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2:
        data = data[:-1]  # drop a truncated half-sample
    ints = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return ints / _INT16_SCALE
```

Pinned facts, in order of how often clients get them wrong:

| Field            | Value                                  | Source / proof                                                            |
| :--------------- | :------------------------------------- | :------------------------------------------------------------------------ |
| Sample format    | signed 16-bit integer (`int16`)        | `.astype("<i2")` in `encode_frame`                                        |
| Endianness       | **little-endian** (`<i2`)              | the `<` in `"<i2"`; browser does `setInt16(..., true)`                    |
| Channels         | **1 (mono)**                           | server `mic_frames` take `indata[:, 0]`; `OutputStream(channels=1)`       |
| Sample rate      | **16000 Hz**                           | `Config.sample_rate: int = 16000` (`config.py`); `WebSocketTransport(sample_rate=16000)` |
| Float ↔ int scale | float `[-1.0, 1.0]` × `32767`         | `_INT16_SCALE = 32767.0`                                                  |
| Decode scale     | int / `32767`                          | `ints / _INT16_SCALE`                                                     |
| Frame size       | **not fixed by the wire** — any byte count; server decodes whatever arrives | `decode_frame` accepts any length, drops a trailing odd byte |

> **Frame size is a client choice, not a protocol constant.** The reference
> Python satellite and `LocalTransport` use `frame_samples = 512` (512 samples =
> 1024 bytes = 32 ms at 16 kHz). The browser uses a `ScriptProcessor` block of
> 2048 samples downsampled to 16 kHz. The server's `decode_frame` decodes any
> length; `WebSocketTransport.feed_mic` then enqueues it whole. **Recommended:
> send ~20–40 ms per frame** (320–640 samples → 640–1280 bytes). The ESP32 client
> in this repo sends 512-sample (1024-byte) frames to match the reference.

If you encode in a non-Python language, the equivalent of `encode_frame` for one
float sample `s ∈ [-1, 1]` is:

```text
int16 = round( clamp(s, -1, 1) * 32767 )      # then store little-endian
```

and to decode an `int16` back: `s = int16 / 32767.0`. Note the asymmetry vs. the
browser, which uses `s * 0x8000` for negatives and `s * 0x7fff` for positives
(see [§7](#7-browser-ws-transport)); both are within one LSB and interoperate
fine. **The authoritative server scale is `32767` for both signs.**

---

## 2. Native WS transport — connection lifecycle

This is the transport for satellites (ESP32, mobile, headless). The server is
[`serve_websocket`](../src/my_stt_tts/ws_transport.py) driven per-connection by
`WsSession.handle`. Start it with:

```bash
my-stt-tts --transport websocket            # or: TRANSPORT=websocket
# optional auth + non-default port:
my-stt-tts --transport websocket --transport-port 8770 --transport-token SECRET
```

Endpoint / bind defaults (`config.py`):

```python
transport_host: str = "0.0.0.0"   # binds LAN-wide so satellites can reach it
transport_port: int = 8770
transport_token: str | None = None
sample_rate: int = 16000
```

So the client connects to **`ws://<server-lan-ip>:8770/`** (path is the root `/`;
`websockets.serve` accepts any path). Plain `ws://`, **not** `wss://`, unless you
front the server with a TLS-terminating reverse proxy.

### 2.1 Handshake (client → server, first frame)

The **first frame the client sends MUST be a TEXT frame** containing a JSON
`hello` envelope, built by `make_handshake`:

```python
def make_handshake(*, sample_rate, token=None, role="satellite") -> str:
    env = {"type": "hello", "version": 1, "sample_rate": int(sample_rate), "role": role}
    if token:
        env["token"] = token
    return json.dumps(env)
```

Concrete bytes a satellite sends (no token configured):

```json
{"type": "hello", "version": 1, "sample_rate": 16000, "role": "satellite"}
```

With a shared token:

```json
{"type": "hello", "version": 1, "sample_rate": 16000, "role": "satellite", "token": "SECRET"}
```

Field semantics (from `make_handshake` / `check_handshake`):

| Field         | Required | Value                            | Validation on the server (`check_handshake`)                    |
| :------------ | :------- | :------------------------------- | :-------------------------------------------------------------- |
| `type`        | yes      | `"hello"`                        | must equal `"hello"`, else `ValueError` → socket closed         |
| `version`     | yes      | `1` (`PROTOCOL_VERSION`)         | must equal `1`, else rejected                                   |
| `sample_rate` | yes      | `16000`                          | parsed but **not** rejected if it differs (server uses its own) |
| `role`        | no       | `"satellite"` or `"browser"`     | accepted; not validated. `"satellite"` = full-duplex mic+speaker |
| `token`       | only if server has one | shared secret string | if the **server** configured a token, the client's must match exactly, else rejected |

### 2.2 Server acceptance (server → client)

On a valid handshake the server replies with **exactly one TEXT control frame**
(`WsSession.handle`):

```python
await conn.send(control_message("ready", sample_rate=self._sample_rate))
```

i.e. the client receives this JSON text frame:

```json
{"type": "ready", "sample_rate": 16000}
```

`control_message(kind, **fields)` is just `json.dumps({"type": kind, **fields})`.
A client **should wait for `ready`** before streaming mic audio (the reference
satellite does: `ready = await conn.recv()`).

### 2.3 Rejection

If the handshake is malformed, the version is wrong, or the token mismatches,
the server logs the reason and closes the socket with a **WebSocket close code
`1008` (policy violation)**, reason `"handshake rejected"`:

```python
except Exception as exc:
    log.warning("rejecting client: %s", exc)
    await conn.close(code=1008, reason="handshake rejected")
    return None
```

### 2.4 Steady state — mic up, TTS down

After `ready`:

- **Upstream (client → server):** the client streams mic audio as **binary**
  frames, each carrying int16-LE PCM per [§1](#1-audio-payload-format-identical-on-both-transports).
  The server loop is `async for message in conn: if isinstance(message, bytes): transport.feed_mic(message)`.
  **Text frames sent by the client after the handshake are currently ignored**
  (`# text frames are client control messages; ignored for now`). So there is no
  client→server control message other than the `hello` and the eventual close.
- **Downstream (server → client):** the server sends TTS audio as **binary**
  frames, same int16-LE PCM format, produced by `WebSocketTransport.send_tts` →
  `encode_frame` and pumped out by `_pump_out`:

  ```python
  async def _pump_out():
      while not transport.closed:
          data = await loop.run_in_executor(None, transport.iter_outbound, 0.1)
          if data:
              await conn.send(data)
  ```

  A client distinguishes audio from control by **frame type**: binary = PCM to
  play; text = JSON control (see [§4](#4-control--event-messages-server--client)).

### 2.5 Disconnect

When the client closes the socket, the server runs (`finally` in `handle`):

```python
transport.end_mic()    # pushes EOF so the pipeline's mic_frames() ends cleanly
transport.close()
out_task.cancel()
```

A satellite ends a session simply by closing the WebSocket. The server tears the
turn loop down gracefully (`run_transport_session` exits when `mic_frames` is
exhausted).

---

## 3. PCM framing rules a client must honour

1. **Send mono int16-LE only.** No WAV header, no length prefix — the WebSocket
   frame *is* the boundary. One binary frame = one chunk of PCM samples.
2. **Even byte count.** Each sample is 2 bytes; send a whole number of samples.
   The server tolerates a trailing odd byte (drops it) but don't rely on it.
3. **16 kHz.** If your hardware captures at a different rate, resample to 16 kHz
   **before** sending (the browser decimates from its `AudioContext.sampleRate`;
   the ESP32 configures its I2S clock to 16 kHz directly).
4. **Don't send empty frames.** `encode_frame(b"")` is `b""`; the server's
   `feed_mic` ignores zero-size decoded frames anyway, but skip the send.
5. **Backpressure:** the server's inbound queue is bounded (`max_queue=512`
   frames in `WebSocketTransport`); when full it **drops** the oldest-arriving
   frame and logs a warning. Keep your frame cadence near real time (≈1× wall
   clock); bursting faster just wastes bandwidth and risks drops.
6. **Playback:** decode each downstream binary frame to int16-LE → your DAC.
   The server may send TTS frames of varying length; play them back-to-back
   (the browser schedules them gaplessly via `playHead`).

---

## 4. Control / event messages (server → client)

All control messages are **JSON TEXT frames** of the shape
`{"type": <kind>, ...}` (built by `control_message`). The only one a satellite is
guaranteed to receive today is:

| `type`   | When                       | Fields            | Client action                              |
| :------- | :------------------------- | :---------------- | :----------------------------------------- |
| `ready`  | once, right after handshake | `sample_rate`    | begin streaming mic audio                  |
| `bye`    | (reserved; `control_message("bye")`) | —      | stop, close socket                         |

`control_message` can emit any `type` (e.g. `state`), but in the current server
only `ready` is sent on the native transport. **A robust client should:**

- parse any **text** frame as JSON and switch on `type`;
- treat an unknown `type` as a no-op (forward-compatible);
- treat **binary** frames as PCM audio unconditionally.

> Rich UI state (`recording`, `stt`, `llm_response`, `speaking`, `idle`,
> transcripts, interrupts) is published on the server's **Server-Sent Events**
> stream at `GET /events` (see `webui.py` `_sse`), *not* over the audio
> WebSocket. A satellite does not need it; an app that wants a live transcript
> can open an `EventSource("/events")` against the **WebUI** port (8765)
> separately. That SSE stream is one-way, text/event-stream, `data: <json>\n\n`.

---

## 5. Barge-in / full-duplex behaviour (informational)

The server keeps the **inbound mic stream live during TTS playout** and runs
barge-in detection (VAD + interrupt gate + AEC + predictor) on it — see
`net_loop.respond_over_transport` and `_TransportBargeIn`. For a client this
means:

- **Keep streaming mic audio even while TTS is playing back.** Do not half-duplex
  (mute the mic during playback) unless you have no echo cancellation and would
  otherwise feed the speaker back into the mic. The server expects continuous mic.
- If the user speaks over the assistant and the server confirms an interruption,
  the server **stops sending TTS frames mid-stream** and starts a new turn from
  the captured audio. The client just keeps playing whatever binary frames arrive
  and keeps sending mic — no special control message is involved.
- **Echo:** if you have no hardware/software AEC and the mic hears the speaker,
  enable it on the device. The ESP32 Atom Echo's mic and speaker are close; this
  client streams continuously and relies on the server-side AEC, but a real
  deployment benefits from a low speaker volume or half-duplex toggle.

---

## 6. Reference exchange (native WS, no token)

```text
client → server  (TEXT)   {"type":"hello","version":1,"sample_rate":16000,"role":"satellite"}
server → client  (TEXT)   {"type":"ready","sample_rate":16000}
client → server  (BINARY) <1024 bytes: 512 int16-LE samples of mic PCM>   ← repeated ~31×/s
client → server  (BINARY) <1024 bytes ...>
   ...
server → client  (BINARY) <N bytes: int16-LE TTS PCM>                     ← while assistant speaks
server → client  (BINARY) <N bytes ...>
   ...
client → server  (CLOSE)                                                  ← user ends session
```

---

## 7. Browser WS transport

This transport serves the GUI at `/ws/audio` on port 8765.

The browser GUI uses a **different connection setup** but the **same PCM payload**.
It exists so the page (served by stdlib `http.server` with CSP `connect-src
'self'`) can carry audio same-origin. Key differences vs. the native transport,
from `webui.py` `_ws_audio` / `run_audio_session` and `ws_frame.py`:

| Aspect            | Native WS (§2)                       | Browser WS (`/ws/audio`)                          |
| :---------------- | :----------------------------------- | :------------------------------------------------ |
| Endpoint          | `ws://HOST:8770/`                    | `ws://HOST:8765/ws/audio` (same origin as the page) |
| WS library        | `websockets`                         | hand-rolled RFC 6455 in `ws_frame.py`             |
| Handshake         | JSON `hello` + optional token        | **HTTP `Upgrade: websocket` only** — no JSON, no token |
| `ready` frame     | yes (`{"type":"ready",...}`)         | **no** — server starts the session immediately    |
| Auth              | optional shared token                | **none** (relies on localhost / same-origin)      |
| Client→server frames | masked (browser does this natively) | masked (RFC 6455 requires it; server unmasks)  |
| Server→client frames | unmasked binary                   | unmasked binary (`ws_frame.encode_frame`)         |
| Payload           | int16-LE 16 kHz mono PCM             | **identical**                                     |

The browser opening handshake is the standard RFC 6455 upgrade; the server
answers with `Sec-WebSocket-Accept = base64(sha1(key + GUID))`
(`ws_frame.accept_key`, GUID `258EAFA5-E914-47DA-95CA-C5AB0DC85B11`). After the
`101 Switching Protocols` response the browser immediately streams binary PCM and
receives binary PCM — there is **no JSON handshake on this path**.

The browser also *prefers WebRTC* (`POST /api/webrtc/offer`, Opus) and only falls
back to this raw-PCM WebSocket when WebRTC is unavailable — see `webui.html`
`startWebRtc` / `startLiveAudio`. WebRTC is out of scope for this document
(satellites use the native WS PCM path).

The browser's float↔int conversion (`webui.html`) uses a sign-dependent scale:

```js
out.setInt16(i*2, s < 0 ? s*0x8000 : s*0x7fff, true);   // f32 → int16, little-endian
out[i] = v / (v < 0 ? 0x8000 : 0x7fff);                  // int16 → f32
```

This differs by ≤1 LSB from the server's symmetric `×32767` and interoperates
without audible difference. New clients **should follow the server's `×32767`**
(see [§1](#1-audio-payload-format-identical-on-both-transports)).

---

## 8. Implementation checklist

A conformant satellite client:

- [ ] Connects to `ws://HOST:8770/` (native) — plain `ws`, the configured port.
- [ ] Sends a TEXT `hello` envelope first: `type:"hello"`, `version:1`,
      `sample_rate:16000`, `role:"satellite"`, and `token` iff the server has one.
- [ ] Waits for the TEXT `{"type":"ready", ...}` frame before streaming mic.
- [ ] Captures mono audio at 16 kHz (resample if needed).
- [ ] Sends mic PCM as **binary** frames of int16-LE samples (≈20–40 ms each).
- [ ] Receives **binary** frames downstream → decodes int16-LE → plays back-to-back.
- [ ] Parses any **text** frame as JSON `{type,...}` and ignores unknown types.
- [ ] Keeps the mic live during playback (server-side barge-in).
- [ ] On a `1008` close, surfaces "handshake rejected" (bad token/version).
- [ ] Closes the socket to end the session.

## 9. Source map

| Concern                       | File                                                       | Symbol(s)                                  |
| :---------------------------- | :--------------------------------------------------------- | :----------------------------------------- |
| PCM encode/decode, handshake  | [`src/my_stt_tts/transport.py`](../src/my_stt_tts/transport.py) | `encode_frame`, `decode_frame`, `make_handshake`, `check_handshake`, `control_message`, `PROTOCOL_VERSION` |
| Native WS server + session    | [`src/my_stt_tts/ws_transport.py`](../src/my_stt_tts/ws_transport.py) | `serve_websocket`, `WsSession.handle`, `WebSocketTransport` |
| Session turn loop             | [`src/my_stt_tts/net_loop.py`](../src/my_stt_tts/net_loop.py) | `run_transport_session`, `respond_over_transport`, `_TransportBargeIn` |
| Reference Python satellite    | [`src/my_stt_tts/satellite.py`](../src/my_stt_tts/satellite.py) | `run_satellite` |
| Browser RFC 6455 codec        | [`src/my_stt_tts/ws_frame.py`](../src/my_stt_tts/ws_frame.py) | `accept_key`, `encode_frame`, `decode_frame` |
| Browser WS bridge + page      | [`src/my_stt_tts/webui.py`](../src/my_stt_tts/webui.py), `webui.html` | `_ws_audio`, `run_audio_session`, `startLiveAudio` |
| Ports / token / sample rate   | [`src/my_stt_tts/config.py`](../src/my_stt_tts/config.py) | `transport_port=8770`, `transport_token`, `sample_rate=16000`; WebUI `port=8765` |
