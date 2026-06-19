# Contributing

This is a community project. Isues and PRs are very welcome.

## Dev setup

```bash
uv sync                 # core + dev (pure-logic + tests)
uv sync --extra all     # + ML backends (parakeet-mlx, mlx-audio, speechbrain, …)
uv tool install piper-tts   # Piper CLI for TTS (GPL; invoked as a subprocess)
./mstt --type           # run from the venv (or .venv/bin/my-stt-tts)
```

## Checks (all must pass before a PR)

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src && uv run pytest
```

## Conventions

- Build/lint/run details live in [AGENTS.md](AGENTS.md); design in [PLAN.md](PLAN.md).
- `src/` layout; every script supports `-h/--help`.
- **GPL backends (Piper, espeak-ng) are invoked as subprocesses, never imported**,
  so the project stays Apache-2.0.
- Heavy ML deps are lazy-imported behind optional extras — the package must import
  with `uv sync` alone.
- Add a note under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
