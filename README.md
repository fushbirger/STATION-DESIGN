# Train Game

A relaxing game about designing train stations out of building blocks. Right now it is at the earliest stage: the block system. Long-term plans include towns, railways, money, banking and stocks, mountains and seaside, and multiplayer.

## Current features

### World
- A 100x100 grid floor to build on.
- Free-fly first-person camera for viewing your build from any angle.

### Building blocks
- **Place blocks** on the grid cell you are aiming at (center of the screen).
- **Remove blocks** you are pointing at.
- **Stack blocks** upward and arrange them side by side.
- **Base shapes**: Square (cube) and Circle (cylinder).
- **Colors**: 6 preset colors, switchable at any time before placing a block.
- **Target marker**: a small ball shows where the next block will be placed, in the currently selected color.
- Active shape and block count are shown in the window title.

## Controls

| Input | Action |
|---|---|
| W A S D | Move forward, left, back, right |
| Space / Ctrl | Move up / down |
| Shift | Move faster |
| Mouse | Look around |
| Scroll | Zoom (FOV) |
| Left click | Place block |
| Right click | Remove block |
| 1 / 2 | Select shape: Square / Circle |
| C | Cycle color |
| Esc | Quit |

## Build

Requires CMake 3.10+, a C++17 compiler, OpenGL, GLEW and GLFW.

```
mkdir -p build && cd build
cmake ..
make
./engine
```

## Planned
- Block types: floor, wall, roof, pillar
- Shape morphing (rectangles, hexagons, ovals)
- Transparency and free color picker
- Railways, platforms, towns marked "OPEN FOR STATION"
- Money, banking, stock system
- Mountains and seaside
- Save / load, multiplayer