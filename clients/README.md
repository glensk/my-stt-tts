# my-stt-tts — Client SDKs

Client SDKs that let household satellites (microcontrollers, browsers, phones)
carry audio to and from the central my-stt-tts "brain". They **do not modify the
core app** — they speak its existing WebSocket audio transport.

The wire protocol is the source of truth for all of them:

> **[`PROTOCOL.md`](PROTOCOL.md)** — the exact satellite WebSocket audio protocol
> (endpoints, handshake + token auth, PCM frame format, control messages,
> TTS-downstream framing), derived field-by-field from the server source.

In one line: **16 kHz mono signed-16-bit little-endian PCM, as raw binary
WebSocket frames in both directions.** There are two transports that share that
payload — a **native** one for satellites (port 8770, JSON `hello` + optional
token) and a **browser** one for the same-origin GUI (port 8765 `/ws/audio`, no
handshake). See PROTOCOL.md for the full distinction.

## Clients in this repo

| Client                              | Target                          | Transport      | Status                                   |
| :---------------------------------- | :------------------------------ | :------------- | :--------------------------------------- |
| [`../esp32/`](../esp32/)            | M5Stack Atom Echo (ESP32)       | native (8770)  | Real, buildable PlatformIO firmware.     |
| [`web/`](web/)                      | Browser (any modern browser)    | browser (8765) | Reusable ES module + standalone demo.    |
| [`mobile/`](mobile/README.md)       | React Native / Swift / Kotlin   | native (8770)  | Implementation guide + key code snippets.|
| `src/my_stt_tts/satellite.py`       | Python (Pi / spare Mac)         | native (8770)  | Reference client shipped with the server.|

## Running the server side

```bash
# Native transport for satellites (ESP32, mobile, Python) — port 8770:
my-stt-tts --transport websocket
my-stt-tts --transport websocket --transport-port 8770 --transport-token SECRET

# Browser transport is served by the WebUI — port 8765:
my-stt-tts --browser          # GUI + /ws/audio + /events
```

(Both need the `transport` extra: `uv sync --extra transport`.)

## Picking a transport

- **Microcontroller / phone / headless box** → **native** (8770). Sends a JSON
  `hello`, supports a shared token, waits for `ready`. This is what `esp32/` and
  `mobile/` use.
- **Web page** → **browser** (8765 `/ws/audio`). Same PCM payload, no JSON
  handshake, no token (relies on same-origin / localhost). This is what `web/`
  uses by default; it can also target the native transport via a flag.

## Verifying a client speaks the protocol

Every client here was checked against the real server code:

- `PROTOCOL.md` cross-references each field to its `transport.py` / `ws_frame.py`
  / `config.py` symbol.
- The JS module + demo pass `node --check`.
- The ESP32 firmware is a real PlatformIO project (see `../esp32/README.md` for
  the build/flash steps and what was verified).
