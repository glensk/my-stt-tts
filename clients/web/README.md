# my-stt-tts — Web (browser) audio client

A reusable, dependency-free ES module that captures the browser mic, streams
**16 kHz mono int16-LE PCM** to a my-stt-tts server over a WebSocket, and plays
the TTS reply gaplessly via the Web Audio API. Extracted from the reference page
`src/my_stt_tts/webui.html` and conformant to [`../PROTOCOL.md`](../PROTOCOL.md).

## Files

| File                    | What it is                                                              |
| :---------------------- | :---------------------------------------------------------------------- |
| `my-stt-tts-client.js`  | The reusable ES module. Exports `SttTtsClient` + pure PCM helpers.      |
| `demo.html`             | A tiny standalone page that wires the module to start/stop buttons.     |

## Quick start

The demo uses an ES module `import`, so it must be **served over HTTP** (opening
`file://demo.html` is blocked by the browser's module loader and by CSP). Two
options:

1. **Serve it next to your my-stt-tts server (recommended).** Drop these two
   files where the WebUI serves them, or point the URL field at your server's
   `/ws/audio`. The default URL is `ws(s)://<page-host>/ws/audio`, which is the
   browser transport the WebUI already exposes (port 8765). No token needed.

2. **Serve the folder standalone** with any static server, then type your
   server's WebSocket URL into the page:

   ```bash
   # from clients/web/
   python3 -m http.server 8080
   # open http://127.0.0.1:8080/demo.html
   ```

   For the **native** transport (port 8770, which requires a JSON `hello` and an
   optional token), tick "Native transport" and enter
   `ws://<server-ip>:8770/` plus the token if the server set one. Note: browsers
   block cross-origin `ws://` only via CSP, not same-origin policy — the demo's
   own CSP is `connect-src 'self'`, so to hit a *different* host you must serve
   the page from that host or relax the demo's CSP for your deployment.

## Using the module in your own page

```html
<script type="module">
  import { SttTtsClient } from "./my-stt-tts-client.js";

  const client = new SttTtsClient({
    url: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/audio`,
    onStatus: (s) => console.log("status:", s),  // connecting | ready | streaming | closed
    onLog:    (m) => console.log(m),
    onError:  (e) => console.error(e),
    onTtsFrame: (f32) => {/* optional: tap each decoded TTS frame */},
  });

  document.querySelector("#start").onclick = () => client.start();  // prompts for mic
  document.querySelector("#stop").onclick  = () => client.stop();
</script>
```

### Targeting the native transport (port 8770)

```js
const client = new SttTtsClient({
  url: "ws://192.168.1.10:8770/",
  nativeHandshake: true,     // send JSON `hello`, wait for `ready`
  token: "SECRET",           // iff the server configured TRANSPORT_TOKEN
});
```

## API

`SttTtsClient(options)`:

| Option            | Type      | Default              | Meaning                                                       |
| :---------------- | :-------- | :------------------- | :------------------------------------------------------------ |
| `url`             | string    | (required)           | WebSocket URL.                                                |
| `nativeHandshake` | boolean   | `false`              | Send JSON `hello` + wait for `ready` (native transport).      |
| `token`           | string    | `null`               | Shared token for the native transport.                        |
| `blockSize`       | number    | `2048`               | ScriptProcessor block size (capture granularity).             |
| `onStatus`        | function  | no-op                | `connecting` → `ready`/`streaming` → `closed`.                |
| `onLog`           | function  | no-op                | Free-form log lines.                                          |
| `onError`         | function  | `console.error`      | Mic-permission / socket / rejection errors.                   |
| `onTtsFrame`      | function  | `null`               | Optional tap on each decoded `Float32Array` TTS frame.        |

Methods: `await client.start()` (acquire mic + open socket; returns `false` on
mic denial), `client.stop()` (tear everything down). Property `client.running`.

Exported pure helpers (unit-testable, mirror the server): `floatToInt16`,
`int16ToFloat`, `downsampleTo16k`, `makeHandshake`, plus constants
`SAMPLE_RATE` (16000) and `PROTOCOL_VERSION` (1).

## Protocol conformance

- **Browser `/ws/audio` path:** HTTP upgrade only, no JSON handshake, no token.
  The server begins the session at `101 Switching Protocols`. This module's
  default (`nativeHandshake: false`) matches it.
- **PCM:** mono, 16 kHz, signed 16-bit little-endian, raw binary frames each way.
  Capture is downsampled from `AudioContext.sampleRate` to 16 kHz before sending.
- **Echo:** `getUserMedia` requests `echoCancellation` + `noiseSuppression`; the
  server also runs AEC + barge-in on the live mic during playback, so the mic
  stays open while TTS plays (full-duplex — see PROTOCOL.md §5).

## CSP

Both files are self-contained: no external scripts/styles/fonts, no AudioWorklet
URL (a deprecated `ScriptProcessorNode` is used precisely so a strict
`script-src 'self'` page works). The demo's CSP is `connect-src 'self'`, matching
the WebUI page, so it can only open a WebSocket back to its own origin. To target
a different host, serve from that host or adjust the demo CSP for your deployment.
