# Within-turn speaker diarization (G7+)

Split a single captured turn that holds **multiple voices + a TV** into per-speaker
segments, each labelled with an enrolled name (or dropped as `unknown`), instead of
collapsing into one unattributed transcript.

- **Segmentation** is [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) offline
  speaker diarization (Apache-2.0, pure ONNX, no torch, arm64 wheels, no Hugging Face
  token). It groups the turn's frames into anonymous local speakers
  `[(start_s, end_s, local_id)]`.
- **Naming reuses the existing SpeechBrain ECAPA path** — each segment is sliced,
  embedded with the same `EcapaEmbedder`, and matched against your `enroll/<name>.npy`
  centroids with the same threshold/margin as single-speaker ID. So enrollment and
  `scripts/calibrate.py` keep working unchanged.
- Background TV / overlapped speech naturally scores **below `SPEAKER_THRESHOLD`**, is
  labelled `unknown`, and is **dropped for command routing** — only segments attributed
  to a known, enrolled speaker can drive an LLM / music intent.

It is **off by default** and a **strict superset of speaker ID**: it only becomes
active when `SPEAKER_DIARIZE` *and* `SPEAKER_ID` are on, voices are enrolled, and
sherpa-onnx + the diarization models are present. If anything is missing it falls back
to today's single-speaker behaviour — it **never crashes a turn**.

## Enable it

```commands
# 1. Install the diarize extra (sherpa-onnx) alongside speaker (speechbrain/ECAPA):
uv sync --extra speaker --extra diarize      # or: uv sync --extra all

# 2. Enrol every household member (see below), then turn both flags on:
SPEAKER_ID=true SPEAKER_DIARIZE=true ./mstt --wake
```

> [!note] sherpa-onnx is pinned to `1.10.46` on purpose
> The `diarize` extra installs `sherpa-onnx==1.10.46` — the last macOS arm64 wheel that
> *self-bundles* its onnxruntime. Newer wheels (1.12.26+) ship a `.so` that hard-links a
> `libonnxruntime.*.dylib` they no longer include, so they fail to load on macOS even with
> the standalone `onnxruntime` (the `wake`/`turn` extras) installed. 1.10.46 carries its own
> dylib, so diarization and the wake word run together in one process. Do not bump it until
> upstream fixes the macOS wheel — if you do and it can't load, diarization safely falls back
> to single-speaker (it never crashes a turn).

On first use the two ONNX models auto-download + **checksum-verify** into the
gitignored `models/`:

| Model        | File                                                                    |
|:-------------|:------------------------------------------------------------------------|
| segmentation | `models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx` (pyannote 3.0) |
| embedding    | `models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`      |

The sherpa **embedding** model is used only for sherpa's internal clustering (grouping
frames into anonymous local speakers); segment **naming** is done by your ECAPA
centroids, not by this model.

## Enrol your family (per language)

Diarization names a segment by matching it to an enrolled centroid, so accuracy depends
on good enrollment. Record clips for each person **in every language they use** — ECAPA
is largely language-independent, but a child's German clips and French clips together
give a sturdier centroid:

```commands
uv run scripts/enroll.py albert --clips 6      # records 6 short clips, averages -> enroll/albert.npy
uv run scripts/enroll.py wife   --clips 6
uv run scripts/enroll.py jakob  --clips 6      # re-enrol children every few months — their voices drift
```

Each `<name>.npy` is one L2-normalized centroid (a `<name>/` directory of multiple
`*.npy` embeddings is averaged). `enroll/` is gitignored — **voiceprints are biometric,
keep them local**. Then tune the threshold against held-out clips so strangers / TV are
rejected:

```commands
uv run scripts/calibrate.py --enroll enroll --tests tests_audio
# -> recommends a SPEAKER_THRESHOLD that accepts zero impostors
```

## Tuning

| Env                         | Default | Effect                                                              |
|:----------------------------|:--------|:--------------------------------------------------------------------|
| `SPEAKER_DIARIZE`           | `false` | master switch (needs `SPEAKER_ID=true` too)                         |
| `DIARIZE_NUM_SPEAKERS`      | `-1`    | `-1` = auto-detect the count; or a fixed household size (e.g. `4`)  |
| `DIARIZE_CLUSTER_THRESHOLD` | `0.5`   | auto-count sensitivity — smaller splits into more speakers          |
| `DIARIZE_MIN_SEGMENT_S`     | `0.4`   | drop diarized segments shorter than this (noise / clipped frames)   |
| `SPEAKER_THRESHOLD`         | `0.45`  | min ECAPA cosine to accept a name (raise to reject TV more harshly) |

## What you get

When diarization splits a turn, the loop emits **one `speaker` bus event per segment**
and a **speaker-labelled transcript per known segment** (e.g. `[albert] turn on the
light`, `[wife] …`), rather than one speaker for the whole turn. parakeet returns no
per-word timestamps in this pipeline, so attribution is **segment-level** (which
diarized span was whom), not word-level. The command is routed to the enrolled person
who spoke the longest; a turn that is **only** unknown / TV voices is dropped.
