// my-stt-tts satellite firmware for the M5Stack Atom Echo (ESP32-PICO).
// =====================================================================
//
// Connects to WiFi, opens a WebSocket to a my-stt-tts server (native transport,
// PROTOCOL.md §2), streams the on-board SPM1423 PDM microphone as 16 kHz mono
// int16-LE PCM upstream, plays TTS PCM downstream on the NS4168 speaker, and
// shows status on the single SK6812 RGB LED.
//
// Wire protocol — clients/PROTOCOL.md (derived from the server source):
//   1. Connect ws://HOST:8770/  (plain ws, native transport).
//   2. Send a TEXT `hello` envelope:
//        {"type":"hello","version":1,"sample_rate":16000,"role":"satellite"[,"token":...]}
//   3. Wait for the TEXT `{"type":"ready","sample_rate":16000}` frame.
//   4. Stream mic PCM as BINARY frames (int16 LE, 16 kHz mono).
//   5. Receive BINARY frames downstream = TTS PCM -> play on the speaker.
//      TEXT frames = JSON control; unknown types ignored.
//   6. Close code 1008 = handshake rejected (bad token/version).
//
// HARDWARE NOTE (Atom Echo is half-duplex): the SPM1423 mic and the NS4168
// speaker share the SAME I2S peripheral (I2S_NUM_0) and the GPIO33 clock line,
// so they cannot run at once. This firmware records continuously and, when TTS
// audio arrives, switches the I2S bus to OUTPUT to play it, then switches back
// to INPUT. The protocol permits full-duplex barge-in; this device is
// hardware-limited to half-duplex. See README.md "Half-duplex" for details.

#include <Arduino.h>
#include <M5Atom.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>

#include "secrets.h"

// ----------------------------------------------------------------------------
// Protocol / audio constants (PROTOCOL.md §1).
// ----------------------------------------------------------------------------
static const uint32_t SAMPLE_RATE = 16000;  // Config.sample_rate
static const int PROTOCOL_VERSION = 1;       // transport.PROTOCOL_VERSION

// Mic frame: 512 samples = 1024 bytes = 32 ms @ 16 kHz (matches the reference
// Python satellite's frame_samples=512; PROTOCOL.md §3 recommends ~20-40 ms).
static const size_t MIC_FRAME_SAMPLES = 512;
static const size_t MIC_FRAME_BYTES = MIC_FRAME_SAMPLES * sizeof(int16_t);

// ----------------------------------------------------------------------------
// M5Atom Echo pin map (I2S0 shared by PDM mic + NS4168 speaker).
// These are the documented Atom Echo pins.
// ----------------------------------------------------------------------------
static const i2s_port_t I2S_PORT = I2S_NUM_0;
static const int PIN_I2S_WS = 33;    // word-select / PDM clock (shared)
static const int PIN_I2S_BCK = 19;   // bit clock (speaker path)
static const int PIN_I2S_DOUT = 22;  // data out -> NS4168 speaker
static const int PIN_I2S_DIN = 23;   // data in  <- SPM1423 PDM mic

// ----------------------------------------------------------------------------
// LED status colours (GRB order for FastLED on the SK6812; M5Atom wraps it).
// ----------------------------------------------------------------------------
static const uint32_t LED_OFF = 0x000000;
static const uint32_t LED_WIFI = 0x202000;     // dim yellow: connecting WiFi
static const uint32_t LED_CONNECTING = 0x002020; // dim cyan: connecting WS
static const uint32_t LED_READY = 0x002000;    // green: connected + listening
static const uint32_t LED_SPEAKING = 0x000040; // blue: playing TTS
static const uint32_t LED_ERROR = 0x400000;    // red: error / rejected

static WebSocketsClient ws;
static bool wsConnected = false;     // socket open
static bool serverReady = false;     // got the `ready` control frame
static bool playing = false;         // bus currently in OUTPUT (playback) mode
static uint32_t lastPlayMs = 0;      // millis() of the last TTS frame written

static void setLed(uint32_t rgb) {
  // M5Atom.dis.drawpix takes a packed 0xRRGGBB and handles the SK6812 order.
  M5.dis.drawpix(0, rgb);
}

// ----------------------------------------------------------------------------
// I2S driver management. The Atom Echo shares one I2S bus between the PDM mic
// and the speaker, so we install the right config and re-install on switch.
// ----------------------------------------------------------------------------
static void i2sInstallMic() {
  i2s_driver_uninstall(I2S_PORT);
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT;  // SPM1423 is on the right
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 6;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = false;
  cfg.fixed_mclk = 0;
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);

  i2s_pin_config_t pins = {};
  pins.bck_io_num = I2S_PIN_NO_CHANGE;     // PDM has no bit clock
  pins.ws_io_num = PIN_I2S_WS;             // PDM clock
  pins.data_out_num = I2S_PIN_NO_CHANGE;
  pins.data_in_num = PIN_I2S_DIN;
  i2s_set_pin(I2S_PORT, &pins);
  i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
  playing = false;
}

static void i2sInstallSpeaker() {
  i2s_driver_uninstall(I2S_PORT);
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate = SAMPLE_RATE;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format = I2S_CHANNEL_FMT_ALL_RIGHT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 6;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;
  cfg.tx_desc_auto_clear = true;
  cfg.fixed_mclk = 0;
  i2s_driver_install(I2S_PORT, &cfg, 0, NULL);

  i2s_pin_config_t pins = {};
  pins.bck_io_num = PIN_I2S_BCK;
  pins.ws_io_num = PIN_I2S_WS;
  pins.data_out_num = PIN_I2S_DOUT;
  pins.data_in_num = I2S_PIN_NO_CHANGE;
  i2s_set_pin(I2S_PORT, &pins);
  i2s_set_clk(I2S_PORT, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
  playing = true;
}

// ----------------------------------------------------------------------------
// Handshake (PROTOCOL.md §2.1). Built like transport.make_handshake().
// ----------------------------------------------------------------------------
static void sendHello() {
  JsonDocument doc;
  doc["type"] = "hello";
  doc["version"] = PROTOCOL_VERSION;
  doc["sample_rate"] = (int)SAMPLE_RATE;
  doc["role"] = "satellite";
  if (strlen(SERVER_TOKEN) > 0) doc["token"] = SERVER_TOKEN;
  char buf[160];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  ws.sendTXT((uint8_t *)buf, n);
  Serial.printf("[ws] sent hello: %.*s\n", (int)n, buf);
}

// ----------------------------------------------------------------------------
// Playback: a downstream BINARY frame is int16-LE PCM. Switch the bus to
// OUTPUT, write the samples, then return to INPUT so we keep capturing.
// ----------------------------------------------------------------------------
static void playPcm(const uint8_t *data, size_t len) {
  if (len < 2) return;
  if (!playing) {
    i2sInstallSpeaker();
    setLed(LED_SPEAKING);
  }
  size_t written = 0;
  // The wire is little-endian int16, which is also the ESP32's native byte
  // order, so the bytes can be handed to I2S as-is (drop a trailing odd byte).
  i2s_write(I2S_PORT, data, len & ~static_cast<size_t>(1), &written, portMAX_DELAY);
  lastPlayMs = millis();  // mark playback activity (see loop()'s drain check)
}

// ----------------------------------------------------------------------------
// WebSocket events.
// ----------------------------------------------------------------------------
static void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("[ws] connected; sending hello");
      wsConnected = true;
      serverReady = false;
      setLed(LED_CONNECTING);
      sendHello();  // TEXT hello must be the first frame (PROTOCOL.md §2.1)
      break;

    case WStype_DISCONNECTED:
      Serial.println("[ws] disconnected");
      wsConnected = false;
      serverReady = false;
      setLed(LED_ERROR);
      break;

    case WStype_TEXT: {
      // JSON control frame (PROTOCOL.md §4). Switch on `type`.
      JsonDocument doc;
      DeserializationError err = deserializeJson(doc, payload, length);
      if (err) {
        Serial.printf("[ws] bad control JSON: %s\n", err.c_str());
        break;
      }
      const char *t = doc["type"] | "";
      if (strcmp(t, "ready") == 0) {
        Serial.println("[ws] server ready -> streaming mic");
        serverReady = true;
        if (playing) i2sInstallMic();  // ensure we are in capture mode
        setLed(LED_READY);
      } else if (strcmp(t, "bye") == 0) {
        Serial.println("[ws] server bye");
        serverReady = false;
      } else {
        Serial.printf("[ws] control: %.*s\n", (int)length, (const char *)payload);
      }
      break;
    }

    case WStype_BIN:
      // Downstream TTS PCM (PROTOCOL.md §2.4). Play it.
      playPcm(payload, length);
      break;

    case WStype_ERROR:
      setLed(LED_ERROR);
      break;

    default:
      break;
  }
}

// ----------------------------------------------------------------------------
// Mic capture: read one PDM frame, send it upstream as a BINARY frame.
// ----------------------------------------------------------------------------
static void captureAndSendMic() {
  static int16_t frame[MIC_FRAME_SAMPLES];
  size_t bytesRead = 0;
  esp_err_t r = i2s_read(I2S_PORT, frame, MIC_FRAME_BYTES, &bytesRead, 20 / portTICK_PERIOD_MS);
  if (r != ESP_OK || bytesRead < sizeof(int16_t)) return;
  // int16-LE on the wire == native ESP32 byte order, so send the buffer as-is
  // (PROTOCOL.md §1). bytesRead is always an even number of bytes here.
  ws.sendBIN((uint8_t *)frame, bytesRead);
}

// ----------------------------------------------------------------------------
static void connectWifi() {
  setLed(LED_WIFI);
  Serial.printf("[wifi] connecting to %s ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print(".");
    if (millis() - start > 30000) {
      Serial.println("\n[wifi] timeout; restarting");
      ESP.restart();
    }
  }
  Serial.printf("\n[wifi] connected, ip=%s\n", WiFi.localIP().toString().c_str());
}

void setup() {
  // M5.begin(serial, i2c, display) — enable the LED ("display"); we manage I2S
  // ourselves, so leave M5's audio init out.
  M5.begin(true, false, true);
  delay(50);
  Serial.begin(115200);
  setLed(LED_OFF);

  connectWifi();

  i2sInstallMic();  // start in capture mode

  Serial.printf("[ws] connecting ws://%s:%d%s\n", SERVER_HOST, SERVER_PORT, SERVER_PATH);
  setLed(LED_CONNECTING);
  ws.begin(SERVER_HOST, SERVER_PORT, SERVER_PATH);
  ws.onEvent(onWsEvent);
  ws.setReconnectInterval(2000);  // auto-reconnect on drop
}

void loop() {
  M5.update();
  ws.loop();

  // Stream mic frames once the server is ready and we are not mid-playback.
  // (Atom Echo is half-duplex; playback briefly owns the I2S bus.)
  if (wsConnected && serverReady && !playing) {
    captureAndSendMic();
  }

  // When the playback DMA has drained, hand the bus back to the mic so we keep
  // listening. The WStype_BIN handler updates `lastPlayMs` for every TTS frame
  // it writes; if none have arrived for ~150 ms we assume the utterance ended
  // and switch the shared I2S bus back to capture.
  if (playing && lastPlayMs != 0 && (millis() - lastPlayMs > 150)) {
    i2sInstallMic();  // sets playing = false
    setLed(serverReady ? LED_READY : LED_CONNECTING);
    lastPlayMs = 0;
  }
}
