# KAGE (影) - *Stay in the dark.*

A high-contrast 2D stealth platformer for phones and desktop. You're a ninja in a black-and-white world, and the shadows are your armor.

## Core loop

Sneak through a level, collect scrolls, take guards down or slip past them, and reach the red torii gate.
Earn **ryo** (両), then spend it in the **Dojo** on new techniques and in **Skins** on new looks. Replay levels for all three stars.

## Stealth rules

- **Vision cones:** guards and cameras show red cones. While you're inside one, their meter fills (faster up close). A full meter means **SPOTTED**, and you go back to the last flag.
- **Shadows:** in black shadow you turn grey and can't be seen, unless a guard is right next to you or you're dashing.
- **Noise:** running and long falls make sound rings. Guards inside a ring turn and come to look. To move silently, tilt the stick only part-way or hold Shift.
- **Takedowns:** STRIKE a guard from behind, from the shadows, or land on his head.
- **Guards** patrol, pause, turn, get suspicious ("?"), search where they last saw you, then go back to their route. **Watchers** stand still and turn around. **Cameras** sweep and can't be taken down.

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

- Only white paper, black ink and one red. Characters are drawn in a blend mode that inverts them, so they're black in the light and white in the shadow, and always readable.
- The skins are drawn in code as vector art, so they animate with every pose and stay sharp at any size. If you want hand-made skins later, model them in Blender, export sprite sheets, and swap them in for the `drawNinja` silhouette.

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

Play by opening `kage/index.html`. It works best with the phone held sideways.
