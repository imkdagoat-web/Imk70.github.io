# CONTRAST — *Stay seen.*

A one-tap, pure black-and-white reflex game for phones.

## The idea

Black and white bands keep falling past you. You are a dot that is either black or white.
**Tap to flip your color. If you ever match the ground under you, you blend in and vanish.**

That's the whole rule. You can explain it in one sentence, and your thumb gets it within three seconds.

## Why it's addictive

- **One input, zero menus:** tap to start, tap to play, tap to retry. You're back in a run in under a second.
- **Every tap matters:** a tap at the wrong time also kills you, so you can't just mash. Each run is a rhythm you have to *read*.
- **Perfect timing pays:** tap within ±50 ms of an edge for **PERFECT**. Chaining perfects builds a combo worth up to +5 a band. Score chasers go for streaks; casual players just survive.
- **A short grace window (110 ms):** a slightly late tap still counts, so deaths feel like *your* fault and you want another go.
- **Ramping tempo:** the speed rises and bands shrink. After about 12 points, surprise **bursts** of thin stripes demand rapid double and triple taps.
- **Readable failure:** when you blend in, your dot literally disappears (with a red warning ring), so you always know why you died.
- **Best score** is saved on the device, with a **NEW BEST** badge.

## High-contrast art direction

- Only `#000` and `#fff`, plus **one** accent (`#ff2d55`) used for rewards and warnings.
- All HUD text uses the `difference` blend mode, so it inverts against whatever band is behind it and is always readable.
- Big, heavy type, with no gradients, textures or gray UI. It suits OLED screens, sunlight and low-vision players.
- Haptics on flip, perfect and death (where supported), plus tiny synth blips. No asset files.

## Play

Open `contrast/index.html` on your phone (add it to your home screen for full screen).
Controls: **tap** anywhere, or **Space / ↑ / Enter** on desktop.

## Where it could go next

- **Daily seed:** everyone plays the same band pattern each day, with a shareable result like `CONTRAST #142 — 87 ⚫⚪`.
- **Modes:** *Zen* (no death, just perfects), *Blind* (bands appear only right before they hit you), *Tri* (a third, accent band that needs a long-press).
- **Cosmetics only:** dot shapes and trails earned through perfect streaks, keeping the world strictly black and white.
- **Ghost race:** show your best run as a faint marker of when you flipped.
