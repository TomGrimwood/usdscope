"""Newton's fog as far as the viewport says, not a fixed 20 to 200 m: its shader fogs whatever
is past 200 m to the fog's colour, so a scene framed from further off (a forest, a city) looks
empty, as if it had not loaded. `follow_far()` changes the shader's source, before a viewer is
made, to fog by a value of its own (`fog_far`, metres: half of it clear, all fog at it), and
`set_far` sets it, each frame (`Viewport.fog`; by default the camera's far distance, which
`Viewport.frame_box` sets to take in what it frames).

Newton's shader is left as it is if its fog is not found as written (a Newton that fogs
otherwise): `follow_far` says so and changes nothing."""

from __future__ import annotations

import contextlib
from typing import Any

#: Newton's fog, as its shape shader has it, and a uniform it declares, to add ours after.
FIXED = "    float fog_start = 20.0;\n    float fog_end   = 200.0;\n"
DECLARED = "uniform mat4 inverse_projection;\n"
#: Fog so far off there is none.
OFF = 1e9
#: Ours: where the fog is whole, metres; it starts halfway there.
FOGGED = "uniform float fog_far;\n"
FOLLOWING = "    float fog_start = 0.5 * fog_far;\n    float fog_end   = fog_far;\n"


def follow_far() -> bool:
    """Newton's shape shader fogging by `fog_far` (from now on, for viewers made after):
    whether it does now."""
    from newton._src.viewer.gl import shaders

    source = shaders.shape_fragment_shader
    if FOLLOWING in source:
        return True
    if FIXED not in source or DECLARED not in source:
        return False
    source = source.replace(FIXED, FOLLOWING).replace(DECLARED, DECLARED + FOGGED, 1)
    shaders.shape_fragment_shader = source
    return True


def set_far(viewer: Any, metres: float) -> None:  # noqa: ANN401
    """The fog whole `metres` off (half of that clear) in `viewer`'s shape shaders, those it
    has made so far."""
    renderer = getattr(viewer, "renderer", None)
    for name in ("_shape_shader", "_shape_transparent_shader"):
        shader = getattr(renderer, name, None)
        program = getattr(shader, "shader_program", None)
        if program is not None:
            # A shader without ours (Newton's own fog) is left be.
            with contextlib.suppress(KeyError, AttributeError):
                program["fog_far"] = float(metres)
