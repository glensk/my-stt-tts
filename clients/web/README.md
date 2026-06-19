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

## Hosted GUI → your own server (`?backend=`)

The full mission-control page (`src/my_stt_tts/webui.html`, also published to
GitHub Pages as `gui.html`) can talk to **any** my-stt-tts server you run — not
just its own origin. Append a `backend` query param pointing at your running
`my-stt-tts --browser` instance and the page becomes a **real** conversation
instead of the scripted demo:

```text
https://glensk.github.io/my-stt-tts/gui.html?backend=https://<your-server>
```

- `backend` is used as the base for every call: `GET <base>/api/settings`,
  the SSE stream `<base>/events`, `POST <base>/api/turn` and `<base>/api/action`,
  and the live-audio socket `<base>/ws/audio` (the scheme is auto-switched to
  `ws`/`wss`). A bare host (`my-box:8765`) is normalised to `https://`.
- An optional `&token=<TOKEN>` is appended as a query param to those calls
  (EventSource/WebSocket can't send custom headers, so the token rides the URL).
  The bundled key-free server ignores it; a token only matters for a transport
  you've explicitly secured.
- The connection indicator reflects reality: it shows **`connected · <host>`**
  (green) when your server answers, and falls back to the amber **demo** badge
  with the scripted showcase only if the backend is genuinely unreachable.

Your server must be reachable from the browser — same LAN, or exposed via a
tunnel (e.g. `cloudflared`, `tailscale funnel`, `ngrok`). The bundled key-free
brain works out of the box, so no API keys are required for a real chat. This
works because the page's CSP allows cross-origin `connect-src` and the server
sends permissive CORS headers (see [CSP](#csp)).

Run your server with:

```bash
my-stt-tts --browser            # serves the WebUI + API on 127.0.0.1:8765
```

then point a tunnel at `127.0.0.1:8765` and use that public URL as `?backend=`.

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
`script-src 'self'` page works). This standalone `demo.html`'s CSP is
`connect-src 'self'`, so it can only open a WebSocket back to its own origin — to
target a different host, serve from that host or adjust the demo CSP for your
deployment.

The full mission-control page (`webui.html` / hosted `gui.html`) is different: it
relaxes **only** `connect-src` to `'self' https: http: ws: wss:` so the
[`?backend=`](#hosted-gui--your-own-server-backend) flow can reach a user-run
server cross-origin (`script-src`/`style-src`/`default-src` stay locked). That
server replies with `Access-Control-Allow-Origin: *` (plus `-Methods`/`-Headers`
and an `OPTIONS` preflight handler) on its `/api/*` and `/events` responses so the
browser permits the cross-origin calls.
