# KAGE (影) - *Stay in the dark.*

A silhouette stealth platformer for phones and desktop. Pure-black silhouettes stand against glowing, atmospheric skies, and the dark foliage is your armor. It always plays sideways: on a phone held upright, the whole game rotates so play stays horizontal.

## Core loop

Sneak through a level, collect scrolls, take guards down or slip past them, and reach the red torii gate.
Earn **ryo** (両), then spend it in the **Dojo** on new techniques and in **Skins** on new looks. Replay levels for all three stars.

## Stealth rules

- **Lantern light:** guards carry lanterns and sentry lanterns swing from ceilings, casting cones of light. While you're inside one, their meter fills (faster up close). A full meter means **SPOTTED**, and you go back to the last shrine lantern.
- **Foliage:** inside dense black foliage you vanish (only your glowing eyes show) and can't be seen, unless a guard is right next to you or you're dashing.
- **Noise:** running and long falls make sound rings. Guards inside a ring turn and come to look. To move silently, tilt the stick only part-way or hold Shift.
- **Takedowns:** STRIKE a guard from behind, from inside foliage, or land on his head. Takedowns add a slow-motion slash.
- **Guards** patrol, pause, turn, get suspicious ("?"), search where they last saw you, then go back to their route. **Watchers** stand still and turn around. **Sentry lanterns** swing and can't be taken down.

## Movement

- Running speeds up and slows down smoothly, and your jump height depends on how long you hold JUMP.
- A short "coyote time" lets you jump just after leaving a ledge, and a jump pressed slightly early still counts.
- **Wall slide** and **wall climb:** hold toward a wall and tap JUMP to climb it, or jump without holding a direction to kick away.
- **Sneak:** a slower, silent walk.
- Unlockable **Double Jump** and **Shadow Dash**.

## Progression

- **10 levels:** Ink & Shadow, Wall Run, Behind You, Spikes & Shade, Watchtower, Crossfire, The Long Drop, Rooftops, The Vault, Shogun's Keep.
- **3 stars per level:** cleared, every scroll, never spotted. Stars from different runs add up.
- **Dojo upgrades:** Double Jump, Shadow Dash, Smoke Bomb (2 ranks), Silent Step (2 ranks), Swift Feet (2 ranks), Long Blade.
- **Skins:** Kuro, Ronin, Kitsune, Oni, Sakura, Phantom, and Shogun (needs 27 stars).
- Progress saves in the browser.

## Art direction

- Silhouette style: everything that matters (ground, characters, foliage) is pure black against luminous, painterly skies, for maximum contrast.
- **5 moods** across the 10 missions: Moonlit Blue (crescent moon and light streaks), Dusk Amber (full moon and embers), Bamboo Mist (fireflies), Blood Sunset (red sun), and Violet Night (falling petals).
- Every sky has a glowing moon or sun, drifting clouds, and three parallax silhouette layers (mountains, pines, pagodas, bamboo, torii, broad trees) with fog in between.
- Levels are generated with grass, flowers, mushrooms, stones, hanging roots and vines, leafy foliage, and thorns with a glowing red tip.
- Light and glow: amber lantern cones that turn red as guards notice you, glowing eyes, a torii gate with glowing paper lanterns as the exit, and shrine lanterns that light up as checkpoints.
- Game feel: red headband ribbons that flow with movement, afterimages, dash streaks, dust, leaves that scatter as you push through foliage, a slow-motion slash on takedowns, screen shake, a vignette, and a mission title card.
- Generated music: koto-like plucks in a Japanese scale, taiko hits, a low drone and wind.
- The skins are drawn in code as vector art, so they animate with every pose. If you make hand-drawn skins later (in Blender, for example), export them as sprite sheets and swap them in for `drawNinja`.

## Controls

| | Touch | Keyboard |
|---|---|---|
| Move | left-side floating stick (tilt part-way to sneak) | ← → / A D |
| Jump | JUMP (or tap anywhere on the right side) | Space / ↑ / W |
| Sneak | small tilt | Shift / ↓ |
| Strike | STRIKE | X / J |
| Dash | DASH | C / K |
| Smoke | SMOKE | V / L |
| Pause | II | Esc / P |

Play by opening `kage/index.html`. On a phone held upright, the game rotates itself to play sideways. On Android, it also tries to go fullscreen and lock to landscape.
