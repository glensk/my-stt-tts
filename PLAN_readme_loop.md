# PLAN — README Fanciness Loop (public-facing styling)

## Session

Resume: `c --resume e7dfe88f-9001-4138-8cfc-1f8789653cc6`

## Ownership

The public-facing GitHub **`README.md`** (and its assets) is owned by a **dedicated
"README designer" agent role**. All public-facing / styling changes route through it —
never ad-hoc edits. It runs **its own checker loop, independent** of the pipecat repo
loop in `PLAN_checker_loop.md`.

## Goal

An **award-winning, super-fancy, customer-facing** README — a sales-pitch landing page
for the *"speak to your agent / talk to your computer, it talks back"* product angle.
A fresh **fancy-checker** compares our README against the **fanciest READMEs on the
internet** and is forced to pick which is fancier; **as long as any competitor README is
fancier, the designer iterates.** Loop until ours wins (or ties at the top).

## Spec (must satisfy)

- [ ] Sales-pitch / customer-oriented tone; non-technical first. Lead with *what it can
  do* in short plain-language phrases — not how it works.
- [ ] **Feature list line-by-line at the very top** (emoji + short phrase each).
- [ ] The two showcase links **line by line, control room FIRST**:
  - `🖥️ See the control room →` (the live `--browser` GUI, demo mode)
  - `🔊 Hear the voices →` (live voice-sample gallery)
- [ ] Each capability: a simple one-liner, with a **`<details>` folded-by-default**
  "technical details" section underneath holding the deep specifics.
- [ ] Fancy + GitHub-safe: centered hero banner (committed self-contained SVG), badge row,
  `<picture>` light/dark, collapsibles, tables, emoji. **No `<style>`/JS** (GitHub strips them).
- [ ] **Accurate** — only real features; honest (macOS Apple-Silicon, prototype status).
- [ ] Keep it polished on every future change (the designer's standing remit).

## Round log

### Round 1 — designer redesign → merged (`1f44a29`)

Full customer-facing rewrite: centered `<picture>` hero (committed `docs/assets/hero-{dark,light}.svg`),
badge row, two showcase links line-by-line (control room first), 10-line plain-language feature
list, 8 capabilities each with a folded `<details>` technical block, quick-start + folded deep
install, privacy + mermaid pipeline. Benchmarked vs othneildrew/Best-README-Template and
twentyhq/twenty; matches their structure and exceeds on progressive disclosure + bespoke SVG.
Also caught + removed a real overclaim (Kokoro TTS — no runtime code). Lints clean via a repo
`.markdownlint.jsonc`.

### Fancy-checker round 1 → WINNER: lobehub/lobe-chat

Ours judged the most *cohesive/tasteful*, but lobe-chat wins on **motion + imagery** we lack:
embedded demo video, per-feature screenshots, animation, social-proof widgets. Round-2 plan
(browser-free techniques that actually render/animate on GitHub):

- [x] **Animate the hero SVGs** with SMIL (63 `<animate>` each: looping waveform bars, pulsing
  dots, blinking cursor, sweeping gradient) — motion, no JS; renders via `<img>` on GitHub.
- [x] **Animated "control room" SVG** (`docs/assets/control-room.svg` — oscilloscope sweep +
  6 cycling state chips + typing transcript) embedded near the top: motion + product-imagery in one.
- [x] **Typing-SVG tagline** (readme-typing-svg, palette-matched) cycling the pitch lines.
- [x] **Capsule-render section dividers** + **anchored centered TOC** + 5 "↑ back to top" links.
- SKIP: star-history (repo is new — a flat 0-star chart hurts) and award widgets (none — won't fabricate).
- DEFER to user: a *real* spoken-demo video (GitHub inline player) — needs a real mic recording.

Round-2 merged at `bb854f2` (all self-hosted SVGs validated + SMIL-animated; lint clean).

### Fancy-checker round 2 → still lobe-chat, but on motion + cohesion we now MATCH/BEAT it

Judge: our hand-authored SMIL is "arguably more sophisticated than anything lobe-chat ships as a
vector"; we match/beat on polish + cohesion. The ONLY remaining gap is **authentic full-motion
video + real product screenshots**, which the checker states **requires a human** (a genuine
recording of the app working). It agreed we should NOT fake awards or add a flat 0-star chart.

**Software ceiling reached.** Remaining items need the user:

- [ ] **(user)** record a ~20–40 s clip of `./mstt --wake --barge-in always --browser` and drag it
  into a GitHub issue/PR comment → drop the resulting `user-attachments` URL in the README → native
  inline `<video>` player (the single biggest wow element).
- [ ] **(user, optional)** per-feature screenshots/clips.

(Tried auto-capturing the GUI demo as a GIF via the shared browser — the browser-automation command
was declined, so authentic media is left to the user, exactly as the checker recommended.)

Round 3 (final automatable polish) — **DONE, merged `ac20732`**: added a contrib.rocks contributor
collage and a ready-to-fill demo-video slot (with paste-the-`user-attachments`-URL instructions).

**Loop status: RESTING at its honest software ceiling.** The README is award-tier and matches/beats
the canonical fanciest README (lobe-chat) on motion + cohesion. The only further gain is a real
demo video — a human task (see the two `(user)` items above). Reopen this loop (designer round 4)
only after the user records that clip, or if requirements change. The designer remains the standing
owner of README styling.
