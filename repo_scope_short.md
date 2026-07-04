# my-stt-tts

A private, local-first voice assistant for macOS Apple Silicon that chains wake-word detection → on-device speech-to-text → a pluggable LLM (Claude/OpenAI/Ollama) → neural text-to-speech → playback, with speaker identification, barge-in interruption, and German/French/English support — all orchestrated as one warm Python process where audio never leaves the machine.

Key tools: `mstt` (main CLI entry point — runs the full voice pipeline), `quickstart.sh` (one-command bootstrap: installs deps, detects a key-free brain, opens the browser control room)

Stack: Python 3.12+, MLX, ONNX, macOS CoreAudio | Deps: numpy, python-dotenv, anthropic, openai, sounddevice, parakeet-mlx, mlx-audio, openwakeword, speechbrain, silero-vad, websockets
