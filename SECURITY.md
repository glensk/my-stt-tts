# Security Policy

## Supported versions

Pre-1.0 — only the latest `main` is supported.

## Reporting a vulnerability

Please report privately via **GitHub Security Advisories**
(<https://github.com/glensk/my-stt-tts/security/advisories/new>) rather than a
public issue.

## Notes specific to this project

- **Secrets** (API keys) live in `.env`, which is gitignored; never commit them.
  `gitleaks` runs in pre-commit to catch accidental leaks.
- **Privacy:** STT and TTS run on-device — only the *transcribed text* leaves the
  machine (to your chosen LLM provider).
- **`claude-cli` brain** runs stripped and in a non-git scratch dir, so the nested
  CLI cannot read your global config or touch your repositories.
- **Agent dispatch** (`agent, <task>`) runs a *full, capable* Claude agent (tools,
  MCP, commands) in `AGENT_WORKSPACE`. It is disabled until you set that variable —
  point it only at a directory you are comfortable letting an agent act in.
