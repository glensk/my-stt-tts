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

- PyPI `sherpa-onnx` Apache-2.0, no torch, macOS arm64 wheels. **PINNED `==1.10.46`** — see
  the "Live-on-Mac dylib pin" section below for why a newer wheel breaks at runtime.
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

## Live-on-Mac dylib pin (resolved 2026-06-21)

The merged feature degraded to single-speaker on this Mac because `import sherpa_onnx`
crashed with a dlopen error under `uv sync --extra all`:

```text
Library not loaded: @rpath/libonnxruntime.1.24.4.dylib  (no such file)
```

**Root cause** — NOT an ABI clash with the co-installed `onnxruntime`, but a *mispackaged
wheel*. The macOS arm64 `sherpa-onnx` wheel changed at 1.12.26: 1.10.46 and earlier
**self-bundle** their onnxruntime (`libonnxruntime.1.17.1.dylib` shipped inside
`sherpa_onnx/lib/`, the dir the `.so` rpaths to — 18 MB wheel). From 1.12.26 the wheel
dropped the bundled dylib (2 MB) yet its `.so` still hard-links
`@rpath/libonnxruntime.1.24.4.dylib`. The standalone `onnxruntime` pip package installs
its dylib into `onnxruntime/capi/`, which is **not** on sherpa's rpath
(`@loader_path` + `@loader_path/sherpa_onnx/lib`). dyld searches rpath *directories on
disk*, not already-loaded images, so even `onnxruntime==1.24.4` co-installed +
`import onnxruntime` first does NOT satisfy the link → dlopen fails.

**Fix** — pin `diarize = ["sherpa-onnx==1.10.46"]` (last self-bundled macOS arm64 wheel).
Its bundled `libonnxruntime.1.17.1.dylib` has a distinct leaf name from the standalone
`onnxruntime` that openWakeWord (`wake`) loads, so **both coexist in one process**. The
1.10.46 `__init__.py` exports the full diarization API used by `diarize.py`
(`OfflineSpeakerDiarization{,Config}`, `OfflineSpeakerSegmentationPyannoteModelConfig`,
`SpeakerEmbeddingExtractorConfig`, `FastClusteringConfig`).

**Verified end-to-end (one process):** `import sherpa_onnx` (1.10.46) + the diarizer built,
models auto-downloaded + checksum-verified, and run on a synthetic 2-speaker clip →
**2 segments / 2 anonymous speakers**; AND `openWakeWord` (onnxruntime 1.26.0) scored a clip
without error — no dlopen crash. `onnxruntime` itself is left unpinned (any `>=1.10,<2`
works for openWakeWord now that sherpa carries its own).

Do NOT bump sherpa-onnx to 1.12.x+ until upstream ships a self-bundled or correctly-rpathed
macOS wheel — `diarize._sherpa_importable()` is the runtime guard that would catch a regression
(it degrades to single-speaker rather than crashing the turn).
