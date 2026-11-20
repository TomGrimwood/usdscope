"""A USD stage drawn fast for editing.

`read`: the stage into plain arrays (still meshes with their materials' colours, moving
things, skinned characters at any time); pxr and numpy only. `pick`: rays against them.
`shapes`: placings, frames along an axis, markers, colour ramps (numpy). `poses`: a skinned
character's stride as one mesh.
`view`: a viewport over Newton's OpenGL viewer to look at them, pick, drag, play, with
`input` (the app's mouse and keys first), `widgets` (a timeline, a pulsing pointer) and
`menus` (menus from paths); the `view` extra (newton, pyglet, imgui-bundle, pyopengl).
"""

from . import read

__all__ = ["read"]
