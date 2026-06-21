# Wake words

`my-stt-tts --wake` listens for a wake word and then records your request. The
repo can **ship several pre-trained wake-word models** in `wakewords/` so you just
**pick one** — no path editing, no training. You can still train your own (below).

## Pre-shipped wake words

Several models ship in this folder as `wakewords/<name>.onnx`. **Selecting a wake word
is just choosing its name** — the model path is derived automatically as
`wakewords/<name>.onnx`. Discovery is generic: whatever `.onnx` models are present are
offered (run `--settings` to see the live list, with reliability tiers).

There are two groups:

- **Official openWakeWord models** (the *recommended reliable* choice) — `hey_jarvis`,
  `hey_mycroft`, `alexa`. These are openWakeWord's own **extensively-trained** models
  (trained on far more data than the self-trained set below), fetched with
  `uv run scripts/fetch_official_wakewords.py`. They are the **green-tier** default
  recommendation: if you don't have a strong reason to use a custom word, pick one of
  these. (The official `alexa` **replaces** the older self-trained `alexa`.)
- **Self-trained models** — `maziko, nexus, jarvis, computer, athena, nova, luna, sage,
  orion` (trained on a GPU via openWakeWord, so they carry no third-party model
  licence). Quality varies — see the recall column and the tiers below.

### Reliability tiers

The GUI colours each wake word by a **reliability tier** (also in `--settings` /
`settings_dict.wake_word_info`):

- 🟢 **green** — an **official** extensively-trained model, **or** measured recall ≥ 0.70.
  The reliable choice.
- 🟠 **orange** — measured recall in **[0.50, 0.70)**. Usable, may miss some activations;
  lower `WAKE_THRESHOLD` if so.
- 🔴 **red** — measured recall **< 0.50**, or a self-trained model with **unrecorded**
  recall. Not recommended; expect missed activations. Prefer a green model or retrain.

Training metrics (accuracy / recall / false-positives-per-hour; openWakeWord targets are
accuracy ≥ 0.7 and recall ≥ 0.5):

| Wake word     | Source         | Recall | Tier    | FP/hr |
|:--------------|:---------------|:-------|:--------|:------|
| `hey_jarvis`  | official OWW   | n/a    | 🟢 green | 0.0   |
| `hey_mycroft` | official OWW   | n/a    | 🟢 green | 0.0   |
| `alexa`       | official OWW   | n/a    | 🟢 green | 0.0   |
| `maziko`      | self-trained   | 0.76   | 🟢 green | 0.0   |
| `nova`        | self-trained   | 0.71   | 🟢 green | 0.0   |
| `athena`      | self-trained   | 0.71   | 🟢 green | 0.0   |
| `orion`       | self-trained   | 0.70   | 🟢 green | 0.0   |
| `computer`    | self-trained   | 0.64   | 🟠 orange| 0.0   |
| `luna`        | self-trained   | 0.52   | 🟠 orange| 0.0   |
| `sage`        | self-trained   | 0.45   | 🔴 red   | 0.0   |
| `nexus`       | self-trained   | n/a    | 🔴 red   | 0.0   |
| `jarvis`      | self-trained   | n/a    | 🔴 red   | 0.0   |

`sage` (recall 0.45) and `luna` (0.52) are short words that are harder to discriminate,
so they may miss more activations — lower `WAKE_THRESHOLD` if so. The self-trained
`nexus`/`jarvis` metrics weren't captured at train time, so they land in the red tier —
prefer an official green model (e.g. **`hey_jarvis`** instead of `jarvis`). `alexa` and
`jarvis` are third-party trademarks, shipped as community models for personal use only.

### Shipping / refreshing the official models

The official `.onnx` models are placed into this folder by:

```commands
uv run scripts/fetch_official_wakewords.py          # copy official OWW models -> wakewords/
uv run scripts/fetch_official_wakewords.py --list   # show what the installed openwakeword offers
```

The script prefers `openwakeword.utils.download_models()` when present, otherwise copies
the weights bundled with the installed `openwakeword`. The shared melspectrogram /
embedding feature models openWakeWord needs are loaded from the installed package at
runtime — they do **not** live in `wakewords/`.

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

## Zero-train custom words (sherpa-onnx KeywordSpotter)

You can add a custom wake word **without training anything** — no GPU, no Colab, no
`.onnx` to generate. A second detector, the **sherpa-onnx KeywordSpotter (KWS)**, is
**open-vocabulary**: you type the phrase and it spots it. It runs as an **OR'd** detector
alongside openWakeWord for **custom / self-trained** words — the word fires if EITHER
detector fires — and the detection event's `detector` field names the winner
(`"oww"` | `"kws"`).

> [!important] Official words are NEVER touched
> The official models (`hey_jarvis`, `alexa`, `hey_mycroft`) fire 99-100% on real voices,
> so KWS is **never** applied to them — they stay openWakeWord-only and byte-identical.
> KWS is purely an extra path for custom words.

**How it works.** KWS reuses the **same** `sherpa-onnx==1.10.46` already pinned for the
`diarize` extra (zero new dependency). The GigaSpeech English zipformer-transducer KWS
model (encoder/decoder/joiner ONNX + `tokens.txt` + `bpe.model`) is auto-downloaded and
checksum-verified into the gitignored `models/` on first custom-word use. Each phrase is
uppercased and BPE-segmented against the model's own vocabulary, then registered with a
per-keyword **boost** and **threshold**; you can register **multiple spellings** (accent
variants) that all map to one logical word.

**Use it.**

1. Install the backend: `uv sync --extra diarize` (or `--extra all`).
2. Pick any phrase as the wake word — e.g. `WAKE_PHRASE=maziko ./mstt --wake`. For a word
   with no trained `.onnx`, KWS alone serves it; for one that also has an `.onnx`
   (like `maziko`), BOTH run and either can fire.
3. Tune recall with `KWS_BOOST` (higher = fires more easily) and `KWS_THRESHOLD`
   (lower = accepts a weaker match). Add accent variants via `KWS_SPELLINGS`:
   `KWS_SPELLINGS='maziko=ma zi ko|ma tsi ko' ./mstt --wake`.
4. `--settings` shows, per word, which detector serves it (`oww` | `oww+kws`) and whether
   KWS is available (`kws_available` / `kws_enabled` in `settings_dict`).

> [!warning] HONEST limit — KWS does not "fix" a non-native maziko
> GigaSpeech is **English**. On Albert's real `maziko` recordings, KWS recovers **one**
> openWakeWord-dead clip (boost 4.0 / threshold 0.1) but not all of them — a non-native
> accent on a non-English word stays hard. KWS's real win is **zero-train custom words**,
> not rescuing a poorly-recalled trained word. For a reliable word, prefer an official
> model. See `PLAN_kws_detector.md` for the per-clip A/B table.

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
