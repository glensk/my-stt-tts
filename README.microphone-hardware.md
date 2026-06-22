# Microphone Hardware and Voice Detection Reliability

How much does the input **microphone hardware** actually affect reliable wake-word
detection and speech-to-text (STT) in this project? Is the MacBook Pro built-in
mic good enough? Would a different/better mic bring a large advantage?

> **Headline answer.** For the way this assistant is used today — a person sitting
> *at* the MacBook, close-talk (≈ 0.3–1 m) — the built-in mic is **good enough**,
> and a better mic buys you **little**. The bottleneck we measured is wake-word
> *recall* (the openWakeWord model not firing), not capture quality, and the macOS
> "Voice Isolation" / browser AGC processing actively **hurts** wake detection.
> Better hardware moves the needle a **lot** only when you change the *scenario* to
> across-the-room / far-field use — then a beamforming mic array (ReSpeaker-class)
> is a genuine step change. For close-talk, the wins are in **software** (gain,
> enrollment, wake model), not in the microphone.

---

## TL;DR recommendations

| Scenario | Keep / buy | Expected benefit | Rough cost |
| :------------------------------- | :--------------------------------------- | :----------------------- | :----------- |
| Sitting at the Mac (today's use) | **Keep built-in mic**; fix software first | Negligible from hardware | CHF 0 |
| Want a clean desk close-talk win | Cheap USB cardioid (AT2020USB / NT-USB) | Moderate (close SNR + rejection) | ~CHF 90–180 |
| Across-the-room / smart-speaker | **ReSpeaker XVF3800 4-mic array** | **Large** (far-field, beamforming) | ~CHF 50–60 |
| On-person, hands-free, best SNR | Lavalier / headset mic | Largest raw SNR, but tethered | ~CHF 30–250 |

The single highest-leverage action for *this* project is **not** a microphone — it is
turning off OS/browser voice processing and fixing wake-word recall in software
(gain, re-enrollment, model retrain). See [Interaction with our pipeline](#5-interaction-with-our-pipeline-where-hardware-actually-helps).

---

## 1. What mic properties actually matter

Voice detection reliability is not one number. Five hardware properties dominate,
and they matter *differently* for wake-word spotting vs. STT.

### 1.1 Signal-to-noise ratio (SNR) and self-noise

SNR is the ratio of captured voice to the microphone's own electronic hiss
(self-noise / noise floor). It is the property that most directly predicts
transcription accuracy. Practical thresholds from the recording world: aim for
**≥ 65 dB (A-weighted)**; below ~60 dB you hear hiss in silent pauses, and
near-field/headset mics target **> 70–80 dB**
([NearStream](https://www.nearstream.us/blog/external-microphone-laptop-audio-truth),
[Saramonic lavalier guide](https://www.saramonic.com/blogs/lavalier-microphone-ultimate-buying-guide)).
Self-noise of **16–20 dBA** is typical for entry USB condensers, while studio mics
reach **4–5 dBA**
([Sweetwater: mic self-noise](https://www.sweetwater.com/insync/david-stewart-guide-specs-microphone-self-noise/)).

A laptop's tiny embedded electret sits far from the mouth, near fans and the
keyboard, and is omnidirectional — independent testing found premium business
laptops show **> 22 dB SNR degradation** vs. an entry USB mic, and that entry USB
mics deliver **15–25 dB higher SNR** plus directional rejection
([NearStream](https://www.nearstream.us/blog/external-microphone-laptop-audio-truth)).

### 1.2 Sensitivity (and why absolute level is *not* SNR)

Sensitivity is how much electrical level a given sound pressure produces. It sets
the *absolute* recorded level (the "7–38 %" we saw on server capture), but it is
**not** quality: a USB mic with a built-in preamp can read "loud" while a quiet,
high-sensitivity capsule can have a far cleaner signal. What matters for detection
is SNR, not raw level — and low level is usually fixable in software with gain.

### 1.3 Frequency response

Speech intelligibility lives in **300–3400 Hz** (telephone band); modern wideband
STT uses the full band up to **8 kHz** (16 kHz sampling, as this project does:
48 kHz capture → 16 kHz resample). Any mic — including the MacBook's — covers this
band fine. Frequency response is rarely the limiting factor for either wake-word or
STT; SNR and directionality are.

### 1.4 Directionality (omni vs. cardioid vs. beamforming array)

- **Omnidirectional** (most laptop capsules, lavaliers): picks up the whole room
  equally → more noise and reverberation.
- **Cardioid** (desk USB mics): rejects sound from behind, improving the
  voice-to-room ratio for a seated talker.
- **Beamforming array** (smart speakers, ReSpeaker): multiple capsules combined in
  DSP to electronically "steer" a pickup lobe at the talker and null noise. For
  far-field, **target-source SNR rises roughly linearly with the number of
  microphones**
  ([SILICON SOURCE: mic-array basics](https://sistc.com/microphone-array-basics-beamforming-snr/)),
  which is why every commercial wake-word device ("Alexa", "Hey Cortana") uses an
  array front-end
  ([Far-field ASR survey, arXiv 2009.09395](https://arxiv.org/pdf/2009.09395)).

### 1.5 Near-field vs. far-field — the inverse-square law

This is the property that decides whether hardware matters *a lot* or *a little*.
In a free field, SPL drops **~6 dB every time distance doubles**
([Audio University](https://audiouniversityonline.com/inverse-square-law-of-sound/),
[Engineering Toolbox](https://www.engineeringtoolbox.com/inverse-square-law-d_890.html)).
Going from 0.3 m (at the laptop) to 3 m (across the room) is ~3.3 doublings ≈
**~20 dB lost voice level** — and in a real room, reflections add reverberation on
top. Far-field ASR research puts the operating range at **1–10 m** and reports word
error rate (WER) climbing from **~2.5 % close-mic to 15–20 % far-field/reverberant**
([Far-field ASR survey](https://arxiv.org/pdf/2009.09395),
[lavalier-vs-recorder comparison](https://www.umevo.ai/blogs/ume-all-posts/lavalier-mics-vs-ai-voice-recorders-which-is-better-for-creators)).
This ~20 dB gap is exactly the budget a beamforming array exists to recover.

### 1.6 Hardware/OS audio processing (AGC, NS, AEC) — double-edged

On-mic or in-OS processing — automatic gain control (AGC), noise suppression (NS),
acoustic echo cancellation (AEC) — *helps* STT and far-field, but can **hurt** a
wake-word model. A wake model is trained on a specific front-end; the wake model
"ingests the audio front-end (AFE)-processed audio and the AFE is hardware
dependent, so the WW performance is sensitive to any change in the AFE"
([Amazon: Front-end Gain Invariant Modeling for Wake Word Spotting, arXiv 2010.06676](https://arxiv.org/pdf/2010.06676)).
Change the processing and you push inference **out of distribution (OOD)** — which
is precisely what we observed with macOS Voice Isolation and browser AGC+NS+EC.

---

## 2. Is the MacBook built-in mic good?

Yes — for close talk. Modern MacBook Pro (2018+) ships a **three-mic array with
directional beamforming and a high SNR**, marketed as "studio-quality." The
geometry lets macOS prioritise the user's voice while suppressing keyboard noise
and room echo
([Challix mic teardown/location](https://challix.com/blogs/apple-questions/where-is-the-microphone-on-macbook-pro-and-macbook-air),
[NearStream MacBook-mic review](https://www.nearstream.us/blog/macbook-built-in-mic-vs-am25x-review)).

### Strengths

- Genuinely good **close-talk** capture; the array + beamforming + AEC are tuned
  for a seated user. We confirmed this empirically: the built-in mic captured
  usable audio and `hey_jarvis` fired at **~0.99** confidence.
- Built-in **AEC** is a real asset if the assistant also plays audio (barge-in).
- Zero cost, zero cabling, always present.

### Weaknesses

- **Far-field is weak.** A 2-capsule-class consumer array can't recover the ~20 dB
  loss across a room the way a dedicated 4-mic circular array can.
- **The OS processing is the trap.** macOS exposes Mic Modes — *Standard*,
  *Voice Isolation* ("filter out background sounds"), *Wide Spectrum* — and there is
  no clean global off-switch; it is plumbed per-app and engages during calls/voice
  contexts ([Apple: Use Mic Modes on your Mac](https://support.apple.com/guide/mac-help/use-mic-modes-on-your-mac-mchle82b42f0/mac);
  user complaints that it "isolates, gates, silences or garbles" intended audio:
  [Universal Audio support](https://help.uaudio.com/hc/en-us/articles/26704705384980-The-Audio-From-My-Interface-Is-Garbled-or-Even-Silent-on-macOS-Mic-Mode-Setting),
  [JustAskJimVO](https://justaskjimvo.studio/macos-mic-mode-mishaps/)).
  For an OOD wake model, this ML-based isolation **removes the very acoustic cues
  the model keys on** — matching our finding that Voice Isolation and browser
  AGC+NS+EC *reduce* wake detection.

**Verdict:** the *capsule* is fine; what hurts you is the *processing path*, and
that is a software setting, not a hardware limit.

---

## 3. Would a better mic help — and which kind?

Quantitative comparison. "Expected benefit here" is judged against *this* project's
pipeline (openWakeWord recall-limited; parakeet-mlx is level-robust).

| Mic type | Typical SNR | Directionality | Far-field (across room) | On-device DSP | Cost (CHF) | Best use case | Expected benefit *here* |
| :----------------------- | :---------- | :-------------- | :--------------------------- | :----------------- | :--------- | :---------------------------- | :------------------------------------- |
| MacBook built-in array | High (close) | 3-mic beamform | Weak (~20 dB loss @ 3 m) | macOS Mic Modes (AEC, isolation) | 0 (owned) | Seated close-talk | Baseline |
| USB cardioid desk mic | +15–25 dB vs laptop | Cardioid | Poor (still near-field) | None / minimal | 90–180 | Clean desk dictation | **Moderate** (close SNR + rear rejection) |
| Far-field 4-mic array (ReSpeaker XVF3800) | ~61–63 dB capsule, 60 dB AGC | Circular beamforming + DoA | **Good (≤ 5 m, 360°)** | AEC, AGC, NS, dereverb, VAD, beamforming | ~50–60 | Room / smart-speaker | **Large** (only real far-field fix) |
| Lavalier / headset | > 70–80 dB at mouth | Omni (lav) / cardioid (headset) | N/A (worn) | None | 30–250 | Hands-free on-person, noisy rooms | **Large raw SNR**, but tethered/worn |

Sources for the figures: ReSpeaker
([Seeed v2.0 wiki](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/),
[XVF3800 product page](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html),
[CNX Software XVF3800 launch](https://www.cnx-software.com/2025/07/29/respeaker-xmos-xvf3800-4-mic-array-board-features-esp32-s3-module-works-over-usb/));
USB-vs-laptop deltas
([NearStream](https://www.nearstream.us/blog/external-microphone-laptop-audio-truth));
lavalier/headset SNR
([Saramonic](https://www.saramonic.com/blogs/lavalier-microphone-ultimate-buying-guide),
[Shure: lav vs headset](https://www.shure.com/en-US/performance-production/louder/fundamentals-choosing-between-lavalier-and-headset-mics)).

### When each gives a LARGE advantage vs. marginal

- **USB cardioid desk mic** — *marginal-to-moderate.* It raises close-talk SNR and
  rejects rear noise, but the Mac is *already* near-field with beamforming, so the
  delta for a seated user is real but not transformative. Helps most if your room
  is noisy (fan, family) and you sit at a fixed spot. STT cleanliness improves more
  than wake recall does.
- **ReSpeaker far-field array** — *large, but only for the far-field scenario.* It
  exists to recover the ~20 dB / reverberation loss across a room with multi-mic
  beamforming + dereverberation. If you want to say the wake word from the kitchen,
  this is the only option that fundamentally changes the physics. At a desk it adds
  little over the built-in (and its own AGC/NS could re-introduce the OOD problem
  for the wake model — see §5).
- **Lavalier / headset** — *largest raw SNR* because the capsule is centimetres
  from the mouth, defeating the inverse-square law entirely; near-field capture
  "drastically reduces WER"
  ([umevo](https://www.umevo.ai/blogs/ume-all-posts/lavalier-mics-vs-ai-voice-recorders-which-is-better-for-creators)).
  But it must be *worn*, which defeats the point of an ambient assistant. Great for
  a noisy environment where you'll wear it anyway; wrong tool for hands-free room
  use.

---

## 4. The acoustic budget, made concrete

A simple level budget explains why scenario beats hardware:

- **0.3 m (typing distance):** reference level. Built-in array is in its comfort
  zone; SNR is high; wake fires (~0.99 observed).
- **1 m (leaning back):** ~ −10 dB vs. 0.3 m. Still workable; software gain covers it.
- **3 m (across a small room):** ~ −20 dB + reverberation. This is where a
  single-point mic (built-in or USB cardioid) struggles and a **beamforming array**
  earns its keep.

The ~6 dB-per-doubling rule
([Audio University](https://audiouniversityonline.com/inverse-square-law-of-sound/))
is the whole story: hardware that changes *where* the effective capture point is
(an array steering a lobe, or a lav at the mouth) beats hardware that just swaps one
fixed-position capsule for a slightly better fixed-position capsule.

---

## 5. Interaction with our pipeline: where hardware actually helps

This is the honest part. Our measured reality:

1. **Wake-word recall is the bottleneck, not capture level.** openWakeWord is
   level-sensitive but has no internal gain normalization; the failures we saw were
   the model *not recognising* good audio, not the mic failing to capture it
   (`hey_jarvis` reached ~0.99 on built-in audio). A better mic does **not** fix a
   recall miss — that is a *model/threshold/enrollment* problem.
2. **STT is already level-robust.** parakeet-mlx transcribes fine at the modest
   levels (~7–38 %) the built-in mic delivers. So the classic "better SNR → lower
   WER" argument, while true in general, has **little headroom to exploit** in our
   close-talk case.
3. **OS/browser processing hurts the wake model.** macOS Voice Isolation and
   browser AGC+NS+EC pushed the wake model out of distribution — consistent with the
   front-end-dependence literature
   ([arXiv 2010.06676](https://arxiv.org/pdf/2010.06676)). Counter-intuitively,
   *less* processing on the close-talk path is better for wake detection.

**So when does hardware move the needle?**

- **Close-talk Mac use (today):** Software dominates. Disable Voice Isolation
  (Standard mic mode), bypass browser AGC/NS/EC, add capture gain, re-enroll or
  retrain the wake model. A new mic is a distraction here — *little benefit*.
- **Room / smart-speaker use (future):** Physics dominates. No software trick
  recovers 20 dB + reverberation from a far-field single point. A **beamforming
  array is a large, real win** — but note it ships its *own* AGC/NS, so for the wake
  model you'd want to validate (or disable) that processing too, or retrain on the
  array's front-end. Hardware and the OOD problem are *coupled*: every processing
  front-end is a distribution the wake model must match.

---

## 6. Concrete recommendations for this project

**For Albert's current setup (MacBook, seated, household):**

1. **Keep the built-in mic.** It captures clean close-talk audio; the array + AEC
   are assets. Do **not** spend money to chase a wake-recall problem that hardware
   can't fix.
2. **Fix software first (free, highest leverage):** force macOS **Standard** mic
   mode (no Voice Isolation), bypass browser AGC/NS/EC on the capture path, apply
   capture gain, and address wake recall via threshold tuning + re-enrollment +
   (if needed) retraining the wake model on the actual mic's front-end.
3. **Optional, if you want a cleaner desk experience:** a **cheap USB cardioid**
   (Audio-Technica AT2020USB-class, ~CHF 90–180,
   [AT2020](https://www.audio-technica.com/en-us/at2020)) gives +15–25 dB SNR and
   rear-noise rejection for a fixed seated position. Moderate, not transformative —
   buy it for call/recording quality, not for wake reliability.
4. **Only if you pivot to across-the-room / smart-speaker use:** buy a
   **ReSpeaker XVF3800 4-mic array** (~CHF 50–60,
   [Seeed](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html)).
   This is the single hardware purchase that would deliver a **large** benefit —
   but only for far-field, and only after you validate its onboard DSP against the
   wake model (retrain on its front-end if recall regresses).

**Bottom line:** the MacBook built-in mic is good enough for the close-talk use the
assistant has today; better hardware brings only marginal gains here. The big
hardware win exists, but it is **scenario-gated** — it appears the moment you ask
the assistant to hear you from across the room, and the right tool for that is a
beamforming far-field array, not a fancier desk mic.

---

## Sources

- [Far-Field Automatic Speech Recognition (survey), arXiv 2009.09395](https://arxiv.org/pdf/2009.09395)
- [On Front-end Gain Invariant Modeling for Wake Word Spotting (Amazon), arXiv 2010.06676](https://arxiv.org/pdf/2010.06676)
- [What Is a Microphone Array? Beamforming and SNR — SILICON SOURCE](https://sistc.com/microphone-array-basics-beamforming-snr/)
- [Inverse Square Law of Sound — Audio University](https://audiouniversityonline.com/inverse-square-law-of-sound/)
- [Sound Propagation: the Inverse Square Law — Engineering Toolbox](https://www.engineeringtoolbox.com/inverse-square-law-d_890.html)
- [External Microphone for Laptop: The Honest Truth — NearStream](https://www.nearstream.us/blog/external-microphone-laptop-audio-truth)
- [Is Your MacBook's "Studio Mic" Actually Good Enough? — NearStream](https://www.nearstream.us/blog/macbook-built-in-mic-vs-am25x-review)
- [Where Is the Microphone on MacBook Pro & MacBook Air? — Challix](https://challix.com/blogs/apple-questions/where-is-the-microphone-on-macbook-pro-and-macbook-air)
- [Microphone Self-Noise — Sweetwater](https://www.sweetwater.com/insync/david-stewart-guide-specs-microphone-self-noise/)
- [Lavalier Microphone Ultimate Buying Guide — Saramonic](https://www.saramonic.com/blogs/lavalier-microphone-ultimate-buying-guide)
- [Choosing Between Lavalier and Headset Mics — Shure](https://www.shure.com/en-US/performance-production/louder/fundamentals-choosing-between-lavalier-and-headset-mics)
- [Lavalier Mics vs. AI Voice Recorders — umevo](https://www.umevo.ai/blogs/ume-all-posts/lavalier-mics-vs-ai-voice-recorders-which-is-better-for-creators)
- [Audio-Technica AT2020 cardioid condenser](https://www.audio-technica.com/en-us/at2020)
- [ReSpeaker Mic Array v2.0 (XVF3000) — Seeed Studio Wiki](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/)
- [ReSpeaker XVF3800 USB 4-Mic Array — Seeed Studio](https://www.seeedstudio.com/ReSpeaker-XVF3800-USB-Mic-Array-p-6488.html)
- [ReSpeaker XVF3800 launch — CNX Software](https://www.cnx-software.com/2025/07/29/respeaker-xmos-xvf3800-4-mic-array-board-features-esp32-s3-module-works-over-usb/)
- [Use Mic Modes on your Mac — Apple Support](https://support.apple.com/guide/mac-help/use-mic-modes-on-your-mac-mchle82b42f0/mac)
- [macOS Mic Mode garbles audio — Universal Audio Support](https://help.uaudio.com/hc/en-us/articles/26704705384980-The-Audio-From-My-Interface-Is-Garbled-or-Even-Silent-on-macOS-Mic-Mode-Setting)
- [MacOS "Mic Mode" Mishaps — JustAskJimVO](https://justaskjimvo.studio/macos-mic-mode-mishaps/)
