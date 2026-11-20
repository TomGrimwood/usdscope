# usdscope

A USD stage drawn fast for editing: its meshes in their materials' colours, what moves in it
and skinned characters at any time, read with pxr into plain arrays; and a viewport over
Newton's OpenGL viewer to look at them, pick points, drag handles and play a timeline.

Made for editors that place things in big scenes (whole towns, forests) and need to see them
at once and move round them smoothly, not for final pictures: a plain colour per material,
a sun and shadows, no textures.

## Modules

| Module | Does |
|---|---|
| `read` | `scene(stage, boxes)`: still meshes, each once with every copy's placing and colour; `moving(stage)`: the outermost time-varying prims, their meshes joined by colour, `.at(seconds)` placings and which are within their keys; `skinned(path)`: UsdSkel characters and rigid props posed `.at(seconds)`, read into memory so the file can be rewritten. Z up, metres. |
| `poses` | `Posed(skinned, joints)`: a file's character posed from arrays a pose at a time (`.pose(rotations, translations, placed)`), a character steered live, shown as a `Skinned` is; `stride`: one pose as a mesh. |
| `gamepad` | `Pad().poll()`: a game controller each frame as plain data (`PadState`: sticks with their dead middle out, triggers, buttons held and just pressed), plugged in or pulled out at any time. |
| `view` | `Viewport`: Newton's `ViewerGL` with the scene, moving things and characters drawn, overlays (points, lines, labels), picking by ray, handles to drag, a timeline. Needs the `view` extra. |

## Install

```bash
uv add usdscope            # reading: numpy, usd-core
uv add "usdscope[view]"    # and the viewport: newton, pyglet, imgui-bundle, pyopengl
```

On Linux the viewport's OpenGL goes through GLX (XWayland under Wayland); it sets
`PYOPENGL_PLATFORM=glx` there unless already set. On Windows it uses PyOpenGL's own pick.

## Commands

```bash
uv sync
make        # lint, format, types, the tests
make fmt
uv run python scripts/check.py   # what `make` runs, without make (Windows)
```

Licence: Apache-2.0.
