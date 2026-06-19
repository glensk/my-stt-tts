# my-stt-tts — ESP32 satellite (M5Stack Atom Echo)

Real, buildable firmware that turns an **M5Stack Atom Echo** (ESP32-PICO, SPM1423
PDM mic, NS4168 speaker, SK6812 status LED) into a my-stt-tts voice satellite. It
streams the mic to the server and plays the TTS reply, speaking the **native
WebSocket transport** in [`../clients/PROTOCOL.md`](../clients/PROTOCOL.md)
**exactly**.

## What it does

1. Connects to WiFi (`WIFI_SSID` / `WIFI_PASS`).
2. Opens a WebSocket to `ws://SERVER_HOST:SERVER_PORT/` (default port 8770).
3. Sends the JSON `hello` handshake (`type:"hello", version:1,
   sample_rate:16000, role:"satellite"`, plus `token` if `SERVER_TOKEN` is set).
4. Waits for the `ready` control frame, then streams the SPM1423 mic as
   **16 kHz mono int16-LE PCM** in binary frames (512 samples / 1024 bytes each).
5. Plays downstream binary frames (TTS PCM) on the NS4168 speaker.
6. Shows status on the RGB LED:

   | Colour | Meaning                          |
   | :----- | :------------------------------- |
   | Yellow | connecting WiFi                  |
   | Cyan   | connecting WebSocket             |
   | Green  | connected + listening            |
   | Blue   | playing TTS                      |
   | Red    | error / disconnected / rejected  |

## Prerequisites

- An **M5Stack Atom Echo** and a USB-C cable.
- **PlatformIO Core** (`pio`). Install with `uv tool install platformio` (or
  `pipx install platformio`, or the VS Code PlatformIO extension).
- A running server: `my-stt-tts --transport websocket` (needs the `transport`
  extra: `uv sync --extra transport`). Note its LAN IP and, if you set one, the
  `--transport-token`.

## Configure

Copy the secrets template and fill it in:

```bash
cd esp32
cp src/secrets.h.example src/secrets.h
$EDITOR src/secrets.h          # WIFI_SSID, WIFI_PASS, SERVER_HOST, SERVER_PORT, SERVER_TOKEN
```

`src/secrets.h` is git-ignored. Alternatively, pass values via `build_flags` in
`platformio.ini` (e.g. `-DWIFI_SSID='"myssid"'`) without editing source.

## Build & flash

```bash
cd esp32

# Compile only (no board needed):
pio run

# Compile + flash over USB (auto-detects the port):
pio run -t upload

# Watch the serial log (115200 baud):
pio device monitor
```

If the port isn't auto-detected, list ports with `pio device list` and pass it:
`pio run -t upload --upload-port /dev/cu.usbserial-XXXX`. On macOS the Atom Echo
enumerates as a CP210x/CH9102 serial device; install the driver if no port
appears.

Expected serial output on success:

```text
[wifi] connecting to <ssid> ...
[wifi] connected, ip=192.168.x.y
[ws] connecting ws://192.168.1.10:8770/
[ws] connected; sending hello
[ws] sent hello: {"type":"hello","version":1,"sample_rate":16000,"role":"satellite"}
[ws] server ready -> streaming mic
```

Speak; the assistant's reply plays on the speaker (LED turns blue while
speaking, back to green when listening).

## Half-duplex (an Atom Echo hardware constraint — read this)

The Atom Echo's SPM1423 mic and NS4168 speaker **share the same I2S peripheral**
(`I2S_NUM_0`) and the **GPIO33 clock line**, so they cannot run simultaneously.
This firmware therefore:

- records continuously while listening, and
- **switches the I2S bus to OUTPUT** when TTS frames arrive (re-installing the
  driver), plays them, then switches back to INPUT ~150 ms after the last frame.

Consequences:

- The device is **half-duplex**: it does not capture the mic *while* TTS is
  playing. The wire protocol supports full-duplex barge-in (PROTOCOL.md §5), but
  this hardware can't both ways at once. Barge-in mid-utterance is therefore not
  available on the Atom Echo with a single shared I2S bus.
- This is a property of the Atom Echo, not of the protocol. A board with
  separate I2S mic + DAC peripherals (e.g. an ESP32 + INMP441 mic + MAX98357A
  DAC on two I2S ports) can stay full-duplex; the protocol and the rest of this
  firmware are unchanged — only `i2sInstallMic()`/`i2sInstallSpeaker()` would
  become two always-on ports.

## Pin map (M5Atom Echo)

| Signal              | GPIO | Notes                                |
| :------------------ | :--- | :----------------------------------- |
| I2S WS / PDM clock  | 33   | shared by mic + speaker              |
| I2S BCK             | 19   | speaker bit clock                    |
| I2S data out (DOUT) | 22   | → NS4168 speaker                     |
| I2S data in (DIN)   | 23   | ← SPM1423 PDM mic                    |
| RGB LED (SK6812)    | 27   | status (driven via M5Atom/FastLED)   |
| Button              | 39   | (available; not required)            |

## Protocol conformance

- **Native transport (8770):** sends the JSON `hello` first, waits for `ready`,
  then binary PCM both ways — matches PROTOCOL.md §2 and the reference
  `src/my_stt_tts/satellite.py` byte-for-byte at the handshake.
- **PCM:** 16 kHz mono signed-16-bit **little-endian**. The ESP32 is natively
  little-endian, so mic samples are sent and TTS samples are played **without
  byte-swapping** — the I2S buffer *is* the wire format (PROTOCOL.md §1).
- **Frame size:** 512 samples (1024 bytes, 32 ms), matching the reference
  satellite and PROTOCOL.md §3's 20–40 ms recommendation.
- **Token:** sent only when `SERVER_TOKEN` is non-empty; a `1008` close
  surfaces as a red LED (handshake rejected — PROTOCOL.md §2.3).

## Files

| File                    | Purpose                                                        |
| :---------------------- | :------------------------------------------------------------- |
| `platformio.ini`        | PlatformIO project: board `m5stack-atom`, pinned deps.         |
| `src/main.cpp`          | The firmware (WiFi → WS handshake → I2S mic/speaker → LED).    |
| `src/secrets.h.example` | Credentials/server template — copy to `src/secrets.h`.         |
| `.gitignore`            | Ignores `.pio/` build output and `src/secrets.h`.              |

## Troubleshooting

- **Red LED right after connect:** handshake rejected (close `1008`) — wrong
  `SERVER_TOKEN` or version. Check the server's `--transport-token`.
- **Cyan forever:** can't reach the server. Verify `SERVER_HOST`/`SERVER_PORT`,
  that the server runs `--transport websocket`, and that the ESP32 is on the
  same LAN (the server binds `0.0.0.0:8770` by default).
- **No audio captured:** confirm you're on the native transport (8770), not the
  browser `/ws/audio` (8765) — the latter has no `hello` and the device would
  never get `ready`.
- **Garbled/loud playback:** the server sends 16 kHz mono int16-LE; this matches
  the I2S config. If you changed `sample_rate` on the server, change
  `SAMPLE_RATE` in `main.cpp` to match.
