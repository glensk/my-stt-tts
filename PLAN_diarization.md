# Within-turn speaker diarization (sherpa-onnx + existing ECAPA naming)

## Session

Resume: `c --resume <session-id>` (worktree `/Users/albert/obsidian/42-Git/infra/.worktree-diar`, branch `speaker-diarization`)

## Goal

A single captured turn that contains multiple voices + TV is split into per-speaker
segments, each labelled with an enrolled name (or dropped as `unknown`), instead of
collapsing into one unattributed transcript.

- **Segmentation/diarization**: sherpa-onnx offline speaker diarization (Apache-2.0,
  pure ONNX, no torch, arm64 wheels, no HF token). Returns anonymous local speaker
  segments `[(start_s, end_s, local_id)]`.
- **Naming**: REUSE the existing SpeechBrain ECAPA path (`EcapaEmbedder` +
  `SpeakerIdentifier`/`match_speaker`) — slice each segment, embed, match to an
  enrolled `enroll/<name>.npy` centroid (or `unknown`/`ambiguous` -> `None`).
  Background TV / overlap naturally falls below `speaker_threshold` -> `unknown` -> dropped.
- **Default OFF**, opt-in via `SPEAKER_DIARIZE` (like `SPEAKER_ID`); active only when
  sherpa + models + enrolled centroids all present. Degrades to a single whole-clip
  segment (today's behaviour) when anything is missing — NEVER crashes a turn.

## Plan

- [x] Read codebase (speaker_id, speaker_pipeline, net_loop, **main**, events, config, pyproject)
- [x] Confirm baseline tests (784 passed, 5 skipped, core-only)
- [x] Research the verified sherpa-onnx OfflineSpeakerDiarization API + model URLs + SHA-256
- [x] `pyproject.toml`: add `diarize = ["sherpa-onnx>=1.10"]`, fold into `all`
- [x] `config.py`: `speaker_diarize_enabled` (env `SPEAKER_DIARIZE`), diar model paths/urls/sha256 + validation
- [x] `diarize.py`: checksum-verified lazy model fetch + `SherpaDiarizer.segments(clip)` (graceful single-segment fallback)
- [x] `speaker_pipeline.py`: `identify_segments(clip) -> list[(start_s, end_s, name|None)]`; build diarizer in `from_config`
- [x] `events.py`: `bus.transcript(..., speaker=...)` labelled-transcript field + per-segment speaker
- [x] `net_loop.py` / `__main__.py`: per-segment `bus.speaker` + labelled transcript + unknown-drop for commands; single-speaker fallback when off
- [x] Tests: segmentation (mock sherpa), per-segment ECAPA match, labelled transcript + per-seg speaker, unknown/TV drop, single-segment fallback, gating
- [x] `.env.example` + a `docs/DIARIZATION.md` enabling + per-language family enrollment note
- [x] Verify CORE-ONLY (uv sync; ruff/mypy/pytest — imports without sherpa/speechbrain), then `uv sync --extra all`
- [x] Lint clean; no regression (~784 core)
- [x] Commit clean feat:/test: commits on `speaker-diarization`

## Verified sherpa-onnx facts

- PyPI `sherpa-onnx` 1.13.3, Apache-2.0, dep only `sherpa-onnx-core` (no torch), arm64 wheels.
- Config classes: `OfflineSpeakerDiarizationConfig`, `OfflineSpeakerSegmentationModelConfig`,
  `OfflineSpeakerSegmentationPyannoteModelConfig`, `SpeakerEmbeddingExtractorConfig`,
  `FastClusteringConfig`, `OfflineSpeakerDiarization`.
- `OfflineSpeakerDiarization(config).process(audio_f32_16k).sort_by_start_time()` ->
  iterable of segments with `.start` / `.end` (float s) / `.speaker` (int).
- `FastClusteringConfig(num_clusters=-1, threshold=...)` -> auto speaker count.
- Models (GitHub Releases, no HF token):
  - segmentation: `speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`
    -> inner `model.onnx` (sha256 `220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079`)
  - embedding: `speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
    (sha256 `1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b`) — the misspelled tag is correct.
- The sherpa embedding model is used ONLY for the anonymous clustering inside sherpa; segment
  NAMING uses the existing SpeechBrain ECAPA path, so `enroll/*.npy` + `scripts/calibrate.py` still apply.
