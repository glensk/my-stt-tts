# PLAN — sherpa-onnx KeywordSpotter as OR'd wake detector (round-1 port)

## Session

Resume: `c --resume <session-id>` (worktree `.worktree-kws`, branch `kws-detector`)

## Goal

Add sherpa-onnx `KeywordSpotter` (KWS) as a **second, OR'd** wake detector for
**custom / self-trained words only**. Official words (hey_jarvis/alexa/hey_mycroft)
stay openWakeWord-only — byte-identical behaviour. Zero new dependency
(`sherpa-onnx==1.10.46` already pinned for `diarize`). Python only; a GUI agent
consumes the shared contract in parallel.

## Key facts established

- KWS English model: `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2`
  (archive sha256 `f170013b…6561a`). Uses the **int8** encoder/decoder/joiner ONNX
  - `tokens.txt` + `bpe.model`. Vocab is **UPPERCASE** GigaSpeech BPE (500 pieces).
- Keyword line format: `<bpe tokens> :<boost> #<threshold> @<label>`. Build tokens
  via `text2token(texts, tokens, tokens_type="bpe", bpe_model=...)` — BUT it eagerly
  imports `pypinyin` (not a dep). For BPE we call **sentencepiece directly** (bundled
  with sherpa) on UPPERCASED text — identical output, no pypinyin.
- `create_stream(keywords=...)` overrides keywords per stream → no on-disk keywords
  file churn. Result is **transient**: poll `get_result(s)` after each `decode_stream`
  and `reset_stream` on a hit.
- Coexistence proven: standalone onnxruntime (oWW) + sherpa-bundled onnxruntime
  1.17.1 load + run in one process, no dlopen clash.
- A/B on Albert's 6 real maziko clips (oWW threshold 0.4/phases 8 vs KWS):
  - oWW fires 1/6 (2517da05 @0.67); 5 dead (~0.001–0.002).
  - KWS @ boost 4.0/thr 0.1 fires 2/6 — recovers **d03f2ad3** (oWW 0.0015, dead),
    one of the two flagged clips. Does NOT recover e79574a0 or the other 3.
  - Honest verdict: KWS adds zero-train custom words + recovers 1 oWW-dead clip; it
    does NOT fully fix maziko (GigaSpeech is English; non-native accent still hard).

## Plan

- [x] Config: `kws_enabled` (env KWS_ENABLED, default true), model paths/urls/sha256,
      per-word `kws_boost` (1.5) + `kws_threshold` (0.25), `kws_spellings` map.
      from_env / validate.
- [x] `kws.py` — `SherpaKws` wrapping `KeywordSpotter`; lazy + checksum-verified
      model fetch mirroring diarize.py; keyword build (boost/threshold/multi-spelling
      /@label) via sentencepiece BPE; `detect/last_score/reset` matching `WakeWord`;
      ~0.66 s trailing-silence flush; graceful unavailable (no-op).
- [x] OR-combine routing: official → oWW only; custom + kws on/available → both, fire
      if either; report `detector`. `score_wake_clip_combined` returns KWS conf/detector
      for custom words.
- [x] settings_dict: `kws_available`, `kws_enabled`, `wake_word_info[w]["detector"]`.
- [x] events: `detector` on wake_test_result / detection events.
- [x] .env.example + WAKEWORD.md (zero-train custom words) + PLAN_wake_checker_loop.md
      (round-1 log).
- [x] Tests (mock sherpa where model absent; gate real-model tests on availability).
- [x] Lint clean; core 848 → 891 (no regression); full 852 → 897.
- [ ] Commit clean feat:/test: to kws-detector; report A/B honestly.
