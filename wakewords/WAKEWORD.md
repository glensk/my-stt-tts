# Wake words

`my-stt-tts --wake` listens for a wake word and then records your request. The
repo can **ship several pre-trained wake-word models** in `wakewords/` so you just
**pick one** — no path editing, no training. You can still train your own (below).

## Pre-shipped wake words

Several trained models are shipped in this folder as `wakewords/<name>.onnx`
(e.g. `maziko.onnx`, `nexus.onnx`, `alexa.onnx`, `jarvis.onnx`, `computer.onnx`).
**Selecting a wake word is just choosing its name** — the model path is derived
automatically as `wakewords/<name>.onnx`. Discovery is generic: whatever `.onnx`
models are actually present are offered (run `--settings` to see the live list).

Three equivalent ways to select one:

- **Web UI** — open `--browser`; the **Wake phrase** field is a **dropdown** of the
  available wake words. Choosing one applies it live (pick *custom…* to type your own).
- **CLI flag** — `./mstt --wake --wake-word jarvis` (sets the phrase + derives the path).
- **Env var** — `WAKE_PHRASE=jarvis ./mstt --wake` (same derivation).

Check what is selected and present:

```commands
./mstt --settings        # shows: phrase, model path, whether the file exists,
                         #        and the [available] wake words discovered on disk
```

To use a custom-trained model anywhere on disk, set an explicit path — it overrides
the name-based derivation: `WAKE_MODEL_PATH=/path/model.onnx` (or `--wake-model-path`).

> "alexa" and "jarvis" are third-party trademarks; any such models are provided as
> community-trained wake words for personal use only, not affiliated with their owners.

The `.onnx`/`.tflite` files are gitignored (large, machine-generated) and committed
separately; if none are present yet, train one as below.

## Training a wake word (e.g. "maziko")

openWakeWord trains a model from **synthetic speech (no recordings needed)** in about
an hour on a free Colab GPU.

## Steps

1. Open openWakeWord's **automatic model training** notebook in Google Colab
   (Runtime → change runtime type → GPU):
   <https://github.com/dscripka/openWakeWord> →
   `notebooks/automatic_model_training.ipynb`
   (a Colab badge is in the repo README).
2. Set the target phrase to **`maziko`** (e.g. `target_word = "maziko"`). The
   notebook synthesises thousands of TTS pronunciations of "maziko" plus negative
   clips and trains a small classifier. Increase the sample count / training steps
   for higher accuracy.
3. **Run all cells** (~1 h). Download the resulting **`maziko.onnx`** (and
   optionally the `.tflite`).
4. Put it here: **`wakewords/maziko.onnx`** (the `.onnx`/`.tflite` files are
   gitignored — they're large and machine-generated).
5. Test it offline against a recording of yourself saying "maziko":
   `uv run scripts/test_wakeword.py wakewords/maziko.onnx my_maziko.wav`
6. Run it: `./mstt --wake` — say "maziko", then speak your request.

## Tuning

- **False activations** (fires too easily): raise the threshold —
  `WAKE_THRESHOLD=0.6 ./mstt --wake` (or edit `wake_threshold`).
- **Missed activations**: lower it (e.g. `0.4`), or retrain with more samples.
- Use a different file with `WAKE_MODEL_PATH=/path/to/model.onnx`.

## Notes

- On Apple Silicon, openWakeWord runs on the **ONNX** backend (no tflite wheel),
  which `--extra wake` installs (`openwakeword`, `onnxruntime`).
- "maziko" is a good choice: multi-syllabic and uncommon, so it rarely false-fires.
- Frames are 80 ms (1280 samples at 16 kHz); that's what `--wake` feeds the model.

## Trained model (maziko)

`maziko.onnx` was trained on the CSCS RunAI cluster (NVIDIA GH200, namespace
`runai-test-test3`) with openWakeWord on 2026-06-18 — no Colab needed. The `.onnx`
lives in this folder but is **gitignored** (generated binary).

| Metric | Value |
|:-------|:------|
| Accuracy            | 0.878 |
| Recall              | 0.760 |
| False positives/hr  | 0.0   |
| Size / format       | ~201 KB · ONNX IR v7 (PyTorch 2.6.0 export) |

Both openWakeWord targets (accuracy ≥ 0.7, recall ≥ 0.5) are exceeded. If live
testing shows missed activations, lower `WAKE_THRESHOLD` or retrain with more
steps/samples. The model is gitignored — retrain via the steps above, or copy
`/output/maziko.onnx` out of the training pod with `kubectl cp`.
