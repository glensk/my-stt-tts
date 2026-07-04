# my-stt-tts

## Purpose

A fully local voice assistant for macOS Apple Silicon that provides a hands-free, conversational experience: a custom wake word ("maziko") triggers on-device speech-to-text, streams the transcript to a pluggable LLM brain (Claude by default, but OpenAI/Ollama/local models work), and speaks the reply back with neural TTS voices. The system supports full-duplex barge-in interruption (even on open speakers via echo cancellation), speaker identification, multi-language operation (German/French/English), tool calling, and network transport for whole-house satellites and browser-based control rooms — all while keeping raw audio on-device.

## Key Capabilities

- On-device wake-word detection (openWakeWord, custom ONNX models) with hands-free activation
- Streaming speech-to-text via parakeet-mlx (MLX-native, sub-second on Apple Silicon) with partial transcripts
- Pluggable LLM brain: Anthropic Claude (default, works without API key via CLI), OpenAI, Ollama, or any OpenAI-compatible endpoint
- Neural TTS with Piper voices (DE/FR/EN), mlx-audio, optional cloud voices (ElevenLabs, Cartesia, Deepgram)
- Full-duplex barge-in with acoustic echo cancellation (hardware VoiceProcessingIO + software NLMS)
- Speaker identification and per-person greetings via speechbrain embeddings
- Browser-based control room (WebSocket/WebRTC transport) and ESP32/mobile satellite clients
- LLM tool/function calling with a pluggable ToolRegistry
- Smart end-of-turn detection (ONNX model, auto-downloaded)

## Tech Stack

Python 3.12+ on macOS Apple Silicon; MLX for STT/TTS inference; ONNX for wake-word/turn models; numpy for audio DSP; asyncio orchestrator; WebSocket/WebRTC for network transport

## Key Scripts / Files

| File                                | Purpose                                                                 |
| :---------------------------------- | :---------------------------------------------------------------------- |
| `mstt`                              | Main CLI entry point — launches the full voice pipeline                  |
| `quickstart.sh`                     | One-command bootstrap: installs deps, detects brain, opens control room  |
| `src/my_stt_tts/audio.py`          | Microphone capture, VAD, barge-in monitoring during playback             |
| `src/my_stt_tts/wake.py`           | Wake-word detection (openWakeWord ONNX models)                           |
| `src/my_stt_tts/stt.py`            | Speech-to-text (parakeet-mlx streaming transcriber)                      |
| `src/my_stt_tts/brain.py`          | LLM orchestration — streaming responses, tool-call round-trips           |
| `src/my_stt_tts/tts.py`            | Text-to-speech (Piper subprocess, mlx-audio, macOS say fallback)         |
| `src/my_stt_tts/speaker_id.py`     | Speaker identification via speechbrain embeddings                        |
| `src/my_stt_tts/aec.py`            | Echo cancellation (hardware VoiceProcessingIO + software NLMS)           |
| `src/my_stt_tts/turn.py`           | End-of-turn detection (smart-turn ONNX model)                            |
| `src/my_stt_tts/interrupt.py`      | Barge-in interrupt gate and acoustic interrupt predictor                  |
| `src/my_stt_tts/webui.py`          | Browser control room (WebSocket server, HTML GUI)                        |
| `src/my_stt_tts/tools.py`          | Tool/function-calling registry (time, math, home control)                |
| `src/my_stt_tts/transport.py`      | Network audio transport abstraction (local/WebSocket/WebRTC)             |
| `src/my_stt_tts/satellite.py`      | Remote mic+speaker satellite client                                      |
| `scripts/enroll.py`                 | Speaker enrollment — record and save voice embeddings                    |
| `scripts/calibrate.py`             | Microphone calibration for wake-word sensitivity                         |
| `wakewords/maziko.onnx`            | Custom-trained wake-word model                                           |
| `PLAN.md`                           | Full design decisions, latency budget, and roadmap                        |
