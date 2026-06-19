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

- [ ] **Animate the hero SVGs** with SMIL (looping waveform bars, pulsing status dots, sweeping
  gradient) — motion, no JS; renders via `<img>` on GitHub.
- [ ] **Animated "control room" SVG mock** (on-brand oscilloscope sweep + cycling state chips +
  transcript) embedded near the top — our motion + product-imagery in one.
- [ ] **Typing-SVG tagline** (readme-typing-svg, palette-matched) cycling the pitch lines.
- [ ] **Capsule-render section dividers** + **anchored centered TOC** + "↑ back to top" links.
- SKIP: star-history (repo is new — a flat 0-star chart hurts) and award widgets (none — won't fabricate).
- DEFER to user: a *real* spoken-demo video (GitHub inline player) — needs a real mic recording.

Then re-run the fancy-checker (round 2). Loop until ours wins or ties at the top.
