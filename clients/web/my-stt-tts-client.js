/*
 * my-stt-tts browser audio client (WebSocket PCM)
 * ===============================================
 *
 * A small, dependency-free ES module that captures the browser microphone,
 * streams 16 kHz mono int16-LE PCM to a my-stt-tts server over a WebSocket, and
 * plays the TTS PCM that comes back — gaplessly, via the Web Audio API.
 *
 * It is extracted from the reference implementation in
 * `src/my_stt_tts/webui.html` (`startLiveAudio` / `playPcm` / `f32ToInt16`) and
 * matches the wire protocol in `clients/PROTOCOL.md` exactly:
 *   - mono, 16 kHz, signed 16-bit little-endian PCM
 *   - raw BINARY WebSocket frames in both directions (no length prefix, no header)
 *   - the browser `/ws/audio` transport: HTTP upgrade only, NO JSON handshake,
 *     NO token (the server starts the session immediately after `101`).
 *
 * CSP-safe: no imports, no external URLs, no AudioWorklet module URL. Uses a
 * (deprecated but ubiquitous) ScriptProcessorNode so the strict page CSP
 * `connect-src 'self'; script-src 'self'` is satisfied.
 *
 * Usage:
 *
 *     import { SttTtsClient } from "./my-stt-tts-client.js";
 *     const client = new SttTtsClient({
 *       url: `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/audio`,
 *       onStatus: (s) => console.log("status", s),
 *       onError:  (e) => console.error(e),
 *     });
 *     await client.start();   // prompts for mic permission
 *     // ... later ...
 *     client.stop();
 *
 * NOTE on transports: this module speaks the **browser** `/ws/audio` path, which
 * has no JSON `hello` handshake. The **native** satellite transport (port 8770)
 * DOES require a JSON `hello` (see PROTOCOL.md §2); set `nativeHandshake: true`
 * (and `token`) to target that instead — e.g. when proxying directly to the
 * native server rather than the same-origin browser endpoint.
 */

/** The pipeline sample rate. Fixed by the server (`Config.sample_rate = 16000`). */
export const SAMPLE_RATE = 16000;

/** Protocol version for the native `hello` handshake (`transport.PROTOCOL_VERSION`). */
export const PROTOCOL_VERSION = 1;

/**
 * Convert a Float32 PCM block in [-1, 1] to a little-endian int16 ArrayBuffer.
 * Matches `f32ToInt16` in webui.html and the server's `encode_frame`.
 * @param {Float32Array} f32
 * @returns {ArrayBuffer}
 */
export function floatToInt16(f32) {
  const view = new DataView(new ArrayBuffer(f32.length * 2));
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    // Sign-dependent scale matches the reference browser client (within 1 LSB
    // of the server's symmetric ×32767; both interoperate).
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); // little-endian
  }
  return view.buffer;
}

/**
 * Convert a little-endian int16 ArrayBuffer back to Float32 PCM in [-1, 1].
 * Matches `int16ToF32` in webui.html and the server's `decode_frame`.
 * @param {ArrayBuffer} buf
 * @returns {Float32Array}
 */
export function int16ToFloat(buf) {
  const view = new DataView(buf);
  const n = Math.floor(buf.byteLength / 2); // drop a trailing odd byte, like the server
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = view.getInt16(i * 2, true);
    out[i] = v / (v < 0 ? 0x8000 : 0x7fff);
  }
  return out;
}

/**
 * Downsample a capture-rate Float32 block to 16 kHz by linear decimation.
 * Matches `downsample` in webui.html. Cheap (nearest-sample); good enough for
 * speech. For higher fidelity, replace with a proper resampler.
 * @param {Float32Array} block
 * @param {number} fromRate
 * @returns {Float32Array}
 */
export function downsampleTo16k(block, fromRate) {
  if (fromRate === SAMPLE_RATE) return block;
  const ratio = fromRate / SAMPLE_RATE;
  const n = Math.floor(block.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = block[Math.floor(i * ratio)];
  return out;
}

/**
 * Build the native-transport JSON `hello` envelope (PROTOCOL.md §2.1).
 * Only needed when `nativeHandshake: true` — the browser `/ws/audio` path does
 * NOT use this.
 * @param {{token?: string|null, role?: string}} [opts]
 * @returns {string}
 */
export function makeHandshake({ token = null, role = "browser" } = {}) {
  const env = { type: "hello", version: PROTOCOL_VERSION, sample_rate: SAMPLE_RATE, role };
  if (token) env.token = token;
  return JSON.stringify(env);
}

/**
 * Full-duplex my-stt-tts audio client.
 *
 * @typedef {Object} SttTtsClientOptions
 * @property {string}   url               WebSocket URL (e.g. ws://host:8765/ws/audio).
 * @property {boolean} [nativeHandshake]  Send the JSON `hello` + wait for `ready`
 *                                        (native transport, port 8770). Default false.
 * @property {string|null} [token]        Shared token for the native transport.
 * @property {number}  [blockSize]        ScriptProcessor block size (default 2048).
 * @property {(status: string) => void} [onStatus]  "connecting" | "ready" | "streaming" | "closed".
 * @property {(text: string) => void}   [onLog]     Free-form log lines.
 * @property {(err: Error) => void}     [onError]   Errors (mic denied, socket error, rejection).
 * @property {(frame: Float32Array) => void} [onTtsFrame]  Optional tap on each decoded TTS frame.
 */
export class SttTtsClient {
  /** @param {SttTtsClientOptions} opts */
  constructor(opts) {
    if (!opts || !opts.url) throw new Error("SttTtsClient: `url` is required");
    this.url = opts.url;
    this.nativeHandshake = !!opts.nativeHandshake;
    this.token = opts.token ?? null;
    this.blockSize = opts.blockSize ?? 2048;
    this.onStatus = opts.onStatus ?? (() => {});
    this.onLog = opts.onLog ?? (() => {});
    this.onError = opts.onError ?? ((e) => console.error(e));
    this.onTtsFrame = opts.onTtsFrame ?? null;

    /** @private */ this._ws = null;
    /** @private */ this._ctx = null;
    /** @private */ this._micNode = null;
    /** @private */ this._micStream = null;
    /** @private */ this._source = null;
    /** @private */ this._playHead = 0;
    /** @private */ this._ready = !this.nativeHandshake; // browser path is ready at open
    /** @private */ this.running = false;
  }

  /**
   * Acquire the mic, open the socket, and begin streaming. Resolves true once the
   * mic + socket are up (and, for the native path, after `ready`). Rejects/false
   * on mic-permission denial.
   * @returns {Promise<boolean>}
   */
  async start() {
    if (this.running) return true;
    this.onStatus("connecting");
    try {
      this._micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
    } catch (e) {
      this.onError(new Error("mic permission denied: " + e.message));
      return false;
    }

    const Ctx = window.AudioContext || window.webkitAudioContext;
    this._ctx = new Ctx();
    this._playHead = 0;

    await new Promise((resolve, reject) => {
      this._ws = new WebSocket(this.url);
      this._ws.binaryType = "arraybuffer";
      this._ws.onopen = () => {
        this.onLog("audio channel open");
        if (this.nativeHandshake) {
          this._ws.send(makeHandshake({ token: this.token, role: "browser" }));
          // `ready` (text) arrives in onmessage; we resolve there.
        } else {
          this._ready = true;
        }
        resolve();
      };
      this._ws.onmessage = (ev) => this._onMessage(ev);
      this._ws.onclose = () => {
        this.onLog("audio channel closed");
        this.onStatus("closed");
        this.stop(true);
      };
      this._ws.onerror = () => {
        const err = new Error("audio channel error");
        this.onError(err);
        reject(err);
      };
    });

    // Wire mic capture -> downsample -> int16 -> socket.
    this._source = this._ctx.createMediaStreamSource(this._micStream);
    // ScriptProcessor is deprecated but ubiquitous and CSP-clean (no worklet URL).
    this._micNode = this._ctx.createScriptProcessor(this.blockSize, 1, 1);
    this._micNode.onaudioprocess = (e) => {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN || !this._ready) return;
      const pcm = downsampleTo16k(e.inputBuffer.getChannelData(0), this._ctx.sampleRate);
      this._ws.send(floatToInt16(pcm));
    };
    this._source.connect(this._micNode);
    this._micNode.connect(this._ctx.destination);

    this.running = true;
    this.onStatus(this._ready ? "streaming" : "ready");
    return true;
  }

  /** @private */
  _onMessage(ev) {
    if (ev.data instanceof ArrayBuffer) {
      // Binary frame = TTS PCM to play back (PROTOCOL.md §2.4).
      const f32 = int16ToFloat(ev.data);
      if (this.onTtsFrame) this.onTtsFrame(f32);
      this._playPcm(f32);
      return;
    }
    // Text frame = JSON control message (PROTOCOL.md §4). Unknown types are no-ops.
    try {
      const msg = JSON.parse(ev.data);
      if (msg && msg.type === "ready") {
        this._ready = true;
        this.onStatus("streaming");
        this.onLog("server ready");
      } else if (msg && msg.type === "bye") {
        this.onLog("server said bye");
        this.stop();
      } else {
        this.onLog("control: " + ev.data);
      }
    } catch {
      this.onLog("non-JSON text frame: " + ev.data);
    }
  }

  /**
   * Schedule a decoded TTS frame for gapless back-to-back playback.
   * Matches `playPcm` in webui.html.
   * @private
   * @param {Float32Array} f32
   */
  _playPcm(f32) {
    if (!this._ctx || f32.length === 0) return;
    const buf = this._ctx.createBuffer(1, f32.length, SAMPLE_RATE);
    buf.getChannelData(0).set(f32);
    const src = this._ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this._ctx.destination);
    const now = this._ctx.currentTime;
    if (this._playHead < now) this._playHead = now; // schedule back-to-back
    src.start(this._playHead);
    this._playHead += buf.duration;
  }

  /**
   * Stop capture/playback and close the socket.
   * @param {boolean} [fromClose] internal: called from the socket's onclose.
   */
  stop(fromClose = false) {
    if (this._micNode) {
      try { this._micNode.disconnect(); } catch { /* ignore */ }
      this._micNode = null;
    }
    if (this._source) {
      try { this._source.disconnect(); } catch { /* ignore */ }
      this._source = null;
    }
    if (this._micStream) {
      this._micStream.getTracks().forEach((t) => t.stop());
      this._micStream = null;
    }
    if (this._ws && !fromClose) {
      try { this._ws.close(); } catch { /* ignore */ }
    }
    this._ws = null;
    if (this._ctx) {
      try { this._ctx.close(); } catch { /* ignore */ }
      this._ctx = null;
    }
    this.running = false;
    this._ready = !this.nativeHandshake;
  }
}

export default SttTtsClient;
