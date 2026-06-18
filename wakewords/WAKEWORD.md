# Training the "maziko" wake word

`my-stt-tts --wake` needs a custom wake-word model at `wakewords/maziko.onnx`.
openWakeWord trains one from **synthetic speech (no recordings needed)** in about
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
steps/samples. To re-fetch the trained model from the (still-running) pod:

```commands

kubectl --context sdsc-fqdn -n runai-test-test3 cp \
  maziko-wakeword-0-0:/output/maziko.onnx wakewords/maziko.onnx
```
