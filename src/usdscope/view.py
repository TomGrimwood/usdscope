"""A viewport for editing in a USD scene: Newton's OpenGL viewer (pyglet, Dear ImGui) with the
scene, its moving things and skinned characters drawn at a time, overlays, picking by ray,
handles dragged with the mouse, gizmos and keys, and a timeline.

    view = Viewport()
    view.show_scene(read.scene(stage))
    view.show_moving(read.moving(stage))
    view.on_click = lambda x, y, mods: print(view.ray(x, y))
    view.panel(lambda imgui: imgui.text("hello"))
    view.run()

Screen positions are Dear ImGui's: pixels of the window's framebuffer from its top left, as
mouse events, labels and panels use them. Placings are USD's matrices (row vectors); the world
is z up, metres. Mouse: a left press on something `on_press` grabs is the app's (dragged with
`on_drag`, let go with `on_release`; asked before a gizmo under the mouse, `gizmo_hovered`,
and no gizmo shown while it drags); otherwise a left drag looks round (Newton's), a press
and release in place is `on_click`; middle drag orbits; the wheel (toward the mouse) and W A S D,
Q E move at `view_speed` (Shift faster, Ctrl slower; Shift and the wheel set it).
Keys go to `on_key` first; Escape never closes the window, H never hides the panels.
"""

from __future__ import annotations

import os
import sys

# On Linux, OpenGL through GLX (XWayland under Wayland): PyOpenGL's own pick there has no
# context. Elsewhere PyOpenGL's own pick (WGL on Windows) is right.
if sys.platform == "linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import threading
import time
import traceback
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
import warp as wp
from newton import Axis
from newton.viewer import ViewerGL
from numpy.typing import NDArray

from . import fog, input, read, widgets
from .input import Mods, Presses
from .overlays import Label, Overlays, Shown, _Gizmo, _Moving, _Skinned, _usd_name
from .pick import hit
from .poses import Posed
from .read import RGB
from .shapes import instances, matrix
from .widgets import u32

if TYPE_CHECKING:
    from imgui_bundle.imgui import ImDrawList

    pass

__all__ = ["Label", "Mods", "Viewport", "hit", "instances", "matrix"]

LAMP: RGB = (1.0, 0.95, 0.7)
#: A screen position (x, y) and the modifier keys held.
Pointer = Callable[[float, float, "Mods"], None]


@dataclass
class Clock:
    """The viewport's time: seconds now, how long its timeline is, whether it plays and how
    fast; what the camera can keep up with (a function giving a point), whether it does (the
    timeline's Follow), and where it last kept up to."""

    time: float = 0.0
    duration: float = 0.0
    playing: bool = False
    speed: float = 1.0
    follow: Callable[[], NDArray | None] | None = None
    following: bool = False
    followed: NDArray[np.float64] | None = None


@dataclass
class Hooks:
    """What the app is told (`Viewport.on`): a left press it may grab (True: grabbed, the
    camera left alone), dragging and letting go of what it grabbed, a click and a right click
    in place, a key (True: taken), each error reported."""

    press: Callable[[float, float, Mods], bool] | None = None
    drag: Pointer | None = None
    release: Pointer | None = None
    click: Pointer | None = None
    context: Pointer | None = None
    key: Callable[[int, Mods], bool] | None = None
    report: Callable[[str], None] | None = None


class Viewport(Overlays):
    """Newton's ViewerGL (or ViewerRTX) as an editor's viewport; see the module's docstring.
    `sky`, `shadows`, `vsync` and `fog_shown`: how it draws to start with (Newton's sky and
    shadows, waiting for the screen's refresh, fog far off), each switched in its view options.

    `renderer="rtx"`: drawn by NVIDIA's RTX (ovrtx, the `rtx` extra), slower, with `background`
    (a USD file: the scene as it is, its own materials and lights) loaded before the first
    frame; everything else drawn the same. RTX holds its background still (no animation), and
    a layer cleared is hidden rather than freed. Its background's own animation (traffic)
    plays at the viewport's time, re-sampled each frame the time moves (slow on big scenes)."""

    def __init__(
        self,
        width: int = 1700,
        height: int = 950,
        *,
        headless: bool = False,
        log: str | Path | None = None,
        renderer: str = "gl",
        background: str | Path | None = None,
        sky: bool = True,
        shadows: bool = True,
        vsync: bool = True,
        fog_shown: bool = True,
    ) -> None:
        self.log = Path(log) if log else None
        self.headless = headless
        #: Set by `stop`: the frames end after this one.
        self.stopping = False
        self.rtx = renderer == "rtx"
        self.viewer: Any
        if self.rtx:
            import newton
            from newton.viewer import ViewerRTX

            self.viewer = ViewerRTX(width=width, height=height, headless=headless, up_axis="Z")
            if background is not None:
                self.viewer.add_background_usd(str(background))
            self.viewer.set_model(newton.ModelBuilder().finalize())
            self._animate_background()
        elif renderer == "gl":
            fogged = fog.follow_far()  # before the viewer makes its shaders
            self.viewer = ViewerGL(
                width=width, height=height, vsync=vsync and not headless, headless=headless
            )
            renderer_gl = getattr(self.viewer, "renderer", None)
            if renderer_gl is not None:  # Newton's sky and shadows, as asked
                renderer_gl.draw_sky, renderer_gl.draw_shadows = sky, shadows
        else:
            raise ValueError(f'A renderer is "gl" or "rtx", not {renderer!r}.')
        self.clock = Clock()
        self.on = Hooks()
        self.presses = Presses()
        #: How fast W A S D move the camera, metres a second.
        self.view_speed = input.SPEED
        #: Where the fog is whole, metres (half that clear): None, the camera's far distance;
        #: `fog.OFF`, none.
        self.fog: float | None = None if fog_shown else fog.OFF
        #: Whether Newton's frame rate shows at the top right.
        self.frame_rate = False
        self.shown = Shown()
        #: The only layers shown (`only`); None: all.
        self.solo: set[str] | None = None
        #: What `only` hid (store, name), to show again.
        self._hid: set[tuple[str, str]] = set()
        self._panels: list[Callable[[ModuleType], None]] = []
        self._reported = ""
        self._hooked = False
        threading.excepthook = self._thread_failed
        if headless and not self.rtx:
            self._headless_panels()
        self.viewer.register_ui_callback(
            self._ui, position="free"
        )  # RTX keeps it until its window opens
        self._hook()
        if renderer == "gl" and not fogged:
            self.say("Newton's fog is not as usdscope knows it: far scenes may look empty.")

    def _keep_panels(self) -> None:
        """The app's panels drawn still: Newton's viewer drops its "free" UI callbacks whenever
        a layer's model is cleared (`remove_layer`, as `clear` does), taking them with it."""
        gui = self.viewer.gui
        free = getattr(gui, "_ui_callbacks", {}).get("free") if gui is not None else None
        if free is not None and self._ui not in free:
            free.append(self._ui)

    def _headless_panels(self) -> None:
        """Panels off screen too, drawn into every frame (a script's pictures show them):
        Newton's viewer leaves them out headless, though its frames are drawn with OpenGL all
        the same. Without them when they cannot be made here."""
        if self.viewer.gui is not None:
            return
        try:
            from newton._src.viewer.viewer_gui import ViewerGui  # not exported by newton.viewer

            self.viewer.gui = ViewerGui(self.viewer, self.viewer.renderer.window)
            rendering = self.viewer._ui_populate_rendering_panel  # as Newton does on screen
            self.viewer.gui.register_ui_callback(rendering, position="rendering")
        except Exception as e:  # no imgui backend for an off-screen window here
            self.report(f"No panels off screen: {e}")

    # Busy, and errors.

    def busy(self, text: str | None) -> None:
        """Newton's loading splash in the middle of the view, `text` under it (what is being
        done, and how far: update it each frame to show it going); None takes it away. A
        window without a screen shows nothing."""
        gui = getattr(self.viewer, "gui", None)
        if gui is None or not getattr(gui, "is_available", False):
            return
        if text is None:
            gui.hide_loading_splash()
        else:
            gui.show_loading_splash(text)

    def say(self, text: str) -> None:
        """A line (or lines) for the terminal and the log file, when there is one, each with
        the time: what the app says, so a whole session can be read and shared later."""
        self._write(text, sys.stdout)

    def report(self, text: str) -> None:
        """An error said in the terminal and the log file, as `say`; the same again straight
        after (a panel failing each frame) said once."""
        if text == self._reported:
            return
        self._reported = text
        self._write(text, sys.stderr)
        if self.on.report:
            self.on.report(text)

    def _write(self, text: str, to: TextIO) -> None:
        stamp = time.strftime("%H:%M:%S")
        lines = [f"{stamp} {line}" for line in (text.splitlines() or [""])]
        print("\n".join(lines), file=to, flush=True)
        if self.log:
            with self.log.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    def _thread_failed(self, args: threading.ExceptHookArgs) -> None:
        self.report(
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        )

    # Drawing. Everything is drawn in a layer (Newton's), each cleared at once by `clear`.

    def clear(self, layer: str) -> None:
        """Everything drawn in `layer` gone: its GPU memory freed (GL), or hidden (RTX)."""
        if self.rtx:
            self._use(layer)
            for kind, name, mesh in self.shown.meshes.pop(layer, []):
                if kind == "mesh":
                    self.viewer.log_mesh(
                        name, wp.zeros(3, dtype=wp.vec3), wp.zeros(3, dtype=wp.int32), hidden=True
                    )
                elif kind == "instances":
                    self.viewer.log_instances(name, mesh, None, None, None, None, hidden=True)
                elif kind == "points":
                    self.viewer.log_points(name, None)
                elif kind == "lines":
                    self.viewer.log_lines(name, None, None, None)
        elif layer in self.viewer.layers:
            self.viewer.remove_layer(layer)
        self.shown.meshes.pop(layer, None)
        self.shown.sent = {k: v for k, v in self.shown.sent.items() if k[0] != layer}
        self.shown.markers.discard(layer)
        self.shown.ribbons.discard(layer)
        self.shown.layer = ""
        self.shown.moving = {k: v for k, v in self.shown.moving.items() if v.layer != layer}
        self.shown.skinned = {k: v for k, v in self.shown.skinned.items() if v.layer != layer}

    def only(self, layers: set[str] | None) -> None:
        """Only `layers` shown (each drawn layer, and labels of the same names); None: every
        one again. Nothing is cleared: hidden layers come back as they were."""
        self.solo = None if layers is None else set(layers)
        for layer in list(self.viewer.layers):
            self.viewer.set_layer_visible(layer, layers is None or layer in layers)
        self._hide_unshown()

    def _hide_unshown(self) -> None:
        """What is drawn in layers not shown (`solo`) hidden, and what this hid shown again
        once its layer is: Newton's viewer hides a hidden layer's model only, not the meshes,
        copies, lines and points logged into it. Each frame, as a drawing may be logged again
        shown; what was hidden already (a mesh only copied, say) is left so."""
        if self.solo is None and not self._hid:
            return
        prefixes = tuple(
            layer.name_prefix + "/"
            for key, layer in self.viewer.layers.items()
            if layer.name_prefix and self.solo is not None and key not in self.solo
        )
        for store in ("objects", "lines", "arrows"):
            for name, drawn in getattr(self.viewer, store, {}).items():
                hide = name.startswith(prefixes) if prefixes else False
                if hide and not drawn.hidden:
                    drawn.hidden = True
                    self._hid.add((store, name))
                elif not hide and (store, name) in self._hid:
                    drawn.hidden = False
                    self._hid.discard((store, name))
        if self.solo is None:  # all shown again; any left were destroyed
            self._hid.clear()

    def _use(self, layer: str) -> None:
        if self.shown.layer != layer:
            self.viewer.activate(layer)
            self.shown.layer = layer

    def show_scene(self, scene: read.Scene, name: str = "scene", layer: str = "scene") -> None:
        """Still meshes, each once, drawn at every copy's placing in its colour (those placed
        once joined a colour: `read.drawn`)."""
        self._use(layer)
        for k, mesh in enumerate(read.drawn(scene)):
            self._log_mesh(f"/{name}/mesh{k}", mesh.vertices, mesh.triangles, hidden=True)
            self._log_instances(f"/{name}/at{k}", f"/{name}/mesh{k}", mesh.placings, mesh.colours)

    def show_moving(
        self, moving: read.Moving, name: str = "moving", hold: bool = False, layer: str = "scene"
    ) -> None:
        """Moving things, placed at the viewport's time each frame (their pieces logged once);
        hidden outside their own keys unless `hold`."""
        self._use(layer)
        for k, piece in enumerate(moving.pieces):
            self._log_mesh(f"/{name}/mesh{k}", piece.vertices, piece.triangles, hidden=True)
        self.shown.moving[name] = _Moving(moving, hold, layer)

    def show_skinned(
        self,
        skinned: read.Skinned | Posed | None,
        name: str = "character",
        colour: RGB = (0.95, 0.45, 0.15),
        at: float | None = None,
        layer: str = "characters",
    ) -> None:
        """A skinned file posed at the viewport's time each frame (or always at `at` seconds),
        or one posed live (`Posed`: drawn again each time it is posed); None hides it."""
        self._use(layer)
        shown = self.shown.skinned.get(name)
        meshes = shown.meshes if shown else 0
        if skinned is None:
            self._hide_meshes(name, 0, meshes)
            meshes = 0
        self.shown.skinned[name] = _Skinned(skinned, colour, at, layer, meshes)

    def held(self) -> Mods:
        """The modifier keys held now."""
        gui = self.viewer.gui
        if gui is None or not gui.is_available:
            return Mods()
        io = gui.ui.io
        return Mods(io.key_shift, io.key_ctrl, io.key_alt)

    # The camera, the screen and picking.

    def camera_at(
        self, at: Sequence[float] | NDArray, pitch: float = -35.0, yaw: float = 45.0
    ) -> None:
        """The camera at `at` (x, y, z), pitched down `pitch` degrees and turned `yaw`."""
        self.viewer.set_camera(wp.vec3(*[float(v) for v in at]), pitch=pitch, yaw=yaw)

    def aim(self, target: Sequence[float] | NDArray) -> None:
        """The camera turned to face `target` (x, y, z), where it is."""
        self.viewer.camera.look_at(wp.vec3(*[float(v) for v in target]))

    def centre_on(self, at: Sequence[float] | NDArray, distance: float = 15.0) -> None:
        """The camera moved to look at `at` (x, y, z) from at most `distance` metres off,
        facing the way it faced (and down at least 25 degrees); it orbits round `at`."""
        camera = self.viewer.camera
        target = np.asarray(at, float)[:3]
        off = min(float(np.linalg.norm(target - np.array(camera.pos, float))), distance)
        pitch = min(float(camera.pitch), -25.0)
        camera.pitch = pitch
        self.camera_at(target - np.array(camera.get_front(), float) * off, pitch, camera.yaw)
        camera.sync_pivot_to_view(distance=off)

    def frame_box(
        self,
        low: Sequence[float] | NDArray,
        high: Sequence[float] | NDArray,
        inset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        """The camera looking down on a box from its south west at its middle, as near as it
        can be with all of the box in view: inside the screen less `inset` pixels at its left,
        top, right and bottom (where the app's panels are). The camera then draws at least as
        far as the far side of the box, and orbits round the box's middle."""
        lo, hi = np.asarray(low, float), np.asarray(high, float)
        middle = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        back = np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0)  # from the middle: south west, 35 deg up
        pitch = -float(np.degrees(np.arcsin(1 / np.sqrt(3.0))))
        width, height, _, _ = self._sizes()
        left, top, right, bottom = inset
        if width - left - right < 0.2 * width or height - top - bottom < 0.2 * height:
            left = top = right = bottom = 0.0  # too little left between them: the whole screen
        distance = max(float(np.linalg.norm(hi - lo)) / 2, 5.0)
        for _ in range(40):
            self.camera_at(middle + back * distance, pitch=pitch, yaw=45.0)
            at, ahead = self.screen(corners)
            if (
                ahead.all()
                and (at[:, 0] >= left).all()
                and (at[:, 0] <= width - right).all()
                and (at[:, 1] >= top).all()
                and (at[:, 1] <= height - bottom).all()
            ):
                break
            distance *= 1.15
        self.viewer.camera.sync_pivot_to_view(distance=distance)  # it orbits round the middle
        reach = distance + float(np.linalg.norm(hi - lo))
        self.viewer.camera.far = max(float(self.viewer.camera.far), reach)

    def ray(self, x: float, y: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The ray through a screen position (any numbers, numpy's too): (origin, unit
        direction)."""
        w, h, cw, ch = self._sizes()
        across, up = float(x) * cw / w, (h - float(y)) * ch / h  # plain floats: pyglet's maths
        origin, direction = self.viewer.camera.get_world_ray(across, up)
        return np.array(list(origin)[:3], float), np.array(list(direction)[:3], float)

    def screen(self, xyz: NDArray) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Where 3D points (n, 3) are on the screen (n, 2), and whether each is in front."""
        camera = self.viewer.camera
        w, h, _, _ = self._sizes()
        front, right, up = (
            np.array(list(v), float)
            for v in (camera.get_front(), camera.get_right(), camera.get_up())
        )
        d = np.asarray(xyz, float).reshape(-1, 3) - np.array(list(camera.pos)[:3], float)
        z = d @ front
        alpha = np.tan(np.radians(camera.fov) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (d @ right) / (z * alpha * camera.width / camera.height)
            v = (d @ up) / (z * alpha)
        return np.stack([(u + 1) / 2 * w, (1 - v) / 2 * h], 1), z > 0.05

    def nearest(self, x: float, y: float, xyz: NDArray | None, within: float = 14.0) -> int | None:
        """The point of `xyz` drawn nearest a screen position, if within `within` pixels."""
        if xyz is None or not len(xyz):
            return None
        at, ahead = self.screen(xyz)
        gap = np.where(ahead, np.hypot(at[:, 0] - x, at[:, 1] - y), np.inf)
        k = int(np.argmin(gap))
        return k if gap[k] <= within else None

    @property
    def size(self) -> tuple[float, float]:
        """The screen's width and height, as screen positions are given."""
        w, h, _, _ = self._sizes()
        return w, h

    def showing(self, name: str = "character") -> read.Skinned | Posed | None:
        """The skinned file shown under `name`, if any."""
        shown = self.shown.skinned.get(name)
        return shown.skinned if shown else None

    def _sizes(self) -> tuple[float, float, float, float]:
        """The screen's width and height (ImGui's), then the camera's."""
        camera = self.viewer.camera
        gui = self.viewer.gui
        if gui is not None and gui.is_available:
            size = gui.ui.io.display_size
            return float(size.x), float(size.y), float(camera.width), float(camera.height)
        return float(camera.width), float(camera.height), float(camera.width), float(camera.height)

    # Gizmos.

    def gizmo(
        self,
        name: str,
        placing: NDArray | None,
        *,
        translate: bool = True,
        rotate: bool = False,
        local: bool = False,
        scale: bool = False,
    ) -> None:
        """A gizmo at `placing` (4, 4: its translation and rotation) to drag; None: gone.
        `local`: its arrows and rings along `placing`'s own axes (its blue arrow up the thing
        however it is turned), else the world's. `scale`: ImGuizmo's scale handles in place
        of its arrows. Read where it was dragged with `gizmo_at`, how far it was scaled (since
        it was placed) with `gizmo_scale`."""
        if placing is None:
            self.shown.gizmos.pop(name, None)
            return
        giz = getattr(getattr(getattr(self.viewer, "gui", None), "ui", None), "giz", None)
        if isinstance(giz, _LocalAxes):
            giz.scales.pop(name, None)
        xform, _ = instances(np.asarray(placing, float)[None])
        transform = wp.transform(wp.vec3(*xform[0, :3].tolist()), wp.quat(*xform[0, 3:].tolist()))
        axes = [Axis.X, Axis.Y, Axis.Z]
        self.shown.gizmos[name] = _Gizmo(
            transform, axes if translate or scale else [], axes if rotate else [], local, scale
        )

    def gizmo_scale(self, name: str) -> NDArray[np.float64]:
        """How far the scale gizmo `name` has been dragged since it was placed: x, y, z."""
        giz = getattr(getattr(getattr(self.viewer, "gui", None), "ui", None), "giz", None)
        found = giz.scales.get(name) if isinstance(giz, _LocalAxes) else None
        return np.ones(3) if found is None else found.copy()

    def gizmo_at(self, name: str) -> NDArray[np.float64] | None:
        """Where the gizmo `name` is now (4, 4), as dragged."""
        found = self.shown.gizmos.get(name)
        if found is None:
            return None
        v = np.array(found.transform, float)  # position, then quaternion x y z w
        return matrix(v[:3], v[3:])

    @property
    def gizmo_hovered(self) -> bool:
        """Whether the mouse is over a gizmo now (a press there would drag it)."""
        gui = getattr(self.viewer, "gui", None)
        return bool(getattr(gui, "_gizmo_hovered", False))

    @property
    def gizmo_busy(self) -> bool:
        """Whether a gizmo is being dragged now."""
        return bool(getattr(self.viewer, "gizmo_is_using", False))

    # Panels.

    def view_options(self, imgui: ModuleType) -> None:
        """Newton's own view options, drawn in the app's panel: its Rendering Options (sky,
        shadows, wireframe, colours, whatever it adds there later), VSync, and its renderer's
        and camera's settings it has no widgets for; where the camera is."""
        viewer, gui = self.viewer, self.viewer.gui
        if gui is None:
            return
        edit = viewer._edit_display_option
        for draw in gui._ui_callbacks.get("rendering", []):
            draw(imgui)
        edit(viewer, "vsync", imgui.checkbox, "VSync")
        renderer = getattr(viewer, "renderer", None)
        if renderer is not None and hasattr(renderer, "exposure"):
            edit(renderer, "exposure", imgui.slider_float, "Exposure", 0.0, 5.0)
            edit(renderer, "shadow_extents", imgui.slider_float, "Shadow reach (m)", 1.0, 500.0)
        edit(viewer.camera, "fov", imgui.slider_float, "Field of view", 15.0, 90.0)
        on, metres = self.fog != fog.OFF, self.fog or float(viewer.camera.far)
        changed, on = imgui.checkbox("Fog", on)
        if changed:
            self.fog = None if on else fog.OFF
        if on:
            moved, metres = imgui.slider_float(
                "Fog distance (m)", metres, 20.0, 5000.0, "%.0f", imgui.SliderFlags_.logarithmic
            )
            if moved:
                self.fog = metres
        _, self.frame_rate = imgui.checkbox("Frame rate", self.frame_rate)
        camera = viewer.camera
        x, y, z = (float(v) for v in camera.pos)
        imgui.text(
            f"Camera {x:.1f}, {y:.1f}, {z:.1f} m  pitch {camera.pitch:.0f}  yaw {camera.yaw:.0f}"
        )

    def panel(self, draw: Callable[[ModuleType], None]) -> None:
        """`draw(imgui)` each frame: the app's own windows (imgui.begin ... imgui.end)."""
        self._panels.append(draw)

    def timeline(
        self,
        imgui: ModuleType,
        marks: Sequence[tuple[float, float, str]] = (),
        height: float = 92.0,
    ) -> None:
        """A timeline along the window's bottom (`widgets.timeline`)."""
        widgets.timeline(self, imgui, marks, height)

    # Running.

    def frame(self, dt: float) -> None:
        """One frame: time moves on if playing, what moves is placed, the scene drawn."""
        self._keep_panels()
        if self.clock.playing and self.clock.duration > 0:
            self.clock.time = (self.clock.time + dt * self.clock.speed) % self.clock.duration
        try:
            self._place()
        except Exception:
            self.report(traceback.format_exc())
        self._hide_unshown()
        far = float(self.viewer.camera.far)
        fog.set_far(self.viewer, far if self.fog is None else self.fog)
        self.viewer.begin_frame(self.clock.time)
        # While the app drags what it took hold of, no gizmo shows: none can start a drag too.
        for name, gizmo in ({} if self.presses.grabbed else self.shown.gizmos).items():
            self.viewer.log_gizmo(
                name, gizmo.transform, translate=gizmo.translate, rotate=gizmo.rotate
            )
        gizmos = self.shown.gizmos.items()
        _LocalAxes.over(
            self.viewer, {n for n, g in gizmos if g.local}, {n for n, g in gizmos if g.scale}
        )
        self.viewer.end_frame()
        self._hook()

    def run(self, each: Callable[[float], None] | None = None) -> None:
        """Frames until the window closes; `each(dt)` before each."""
        last = time.perf_counter()
        while self.viewer.is_running() and not self.stopping:
            now = time.perf_counter()
            dt, last = now - last, now
            if each:
                try:
                    each(dt)
                except Exception:
                    self.report(traceback.format_exc())
            self.frame(dt)
            if self.headless:  # nothing waits for a screen: no faster than 60 frames a second
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - now)))
        self.viewer.close()

    def picture(self, path: str | Path, ui: bool = True) -> None:
        """The last frame saved as an image."""
        Path(path).write_bytes(
            self.image(ui=ui, max_size=None, kind="PNG" if str(path).endswith(".png") else "JPEG")
        )

    def image(self, ui: bool = True, max_size: int | None = 1024, kind: str = "JPEG") -> bytes:
        """The last frame as an image file's bytes (JPEG or PNG), at most `max_size` pixels
        across; with the panels (`ui`) when there are any."""
        import io

        from PIL import Image

        frame: Any = self.viewer.get_frame() if self.rtx else self.viewer.get_frame(render_ui=ui)
        picture = Image.fromarray(frame.numpy())
        if max_size:
            picture.thumbnail((max_size, max_size))
        out = io.BytesIO()
        picture.convert("RGB").save(out, kind, quality=88)
        return out.getvalue()

    def close(self) -> None:
        self.viewer.close()

    def stop(self) -> None:
        """`run` ends after this frame and closes the window, as if it had been closed."""
        self.stopping = True

    # Inside.

    def _animate_background(self) -> None:
        """RTX's background animated at the viewport's time: each frame the time has moved, its
        time-sampled values re-sampled there (ovstage's `update_from_usd_time`), in the same
        write as the viewer's own changes for that frame (its scene changes wrapped)."""
        viewer = self.viewer
        applied: dict[str, float | None] = {"at": None}
        apply = viewer._apply_ovstage_population_changes

        def changes() -> None:
            import ovstage

            stage = getattr(viewer, "_ovstage", None)
            if stage is not None and applied["at"] != self.clock.time:
                ovstage.population.update_from_usd_time(
                    stage, viewer._ovstage_ordinal, float(self.clock.time)
                )
                applied["at"] = self.clock.time
            apply()

        viewer._apply_ovstage_population_changes = changes

    def _hook(self) -> None:
        """The app's panels instead of Newton's, its mouse and keys first: once the viewer has
        its panels (GL at once, RTX once its window opens at the first frame)."""
        gui = self.viewer.gui
        if self._hooked or gui is None or not gui.is_available:
            return
        gui._render_left_panel = lambda: None
        stats = gui._render_stats_overlay  # Newton's frame rate, at the top right when asked
        gui._render_stats_overlay = lambda: self.frame_rate and stats()
        input.wrap(self, gui)
        self._hooked = True

    def _hide_meshes(self, name: str, start: int, end: int) -> None:
        for k in range(start, end):
            self.viewer.log_mesh(
                _usd_name(f"/{name}/{k}"),
                wp.zeros(3, dtype=wp.vec3),
                wp.zeros(3, dtype=wp.int32),
                hidden=True,
            )

    def _place(self) -> None:
        """Moving things and characters where they are at `time` (only when that changed), and
        the camera after what it follows."""
        for name, shown in self.shown.moving.items():
            code = round(self.clock.time * shown.moving.rate)
            if code == shown.placed_at:
                continue
            self._use(shown.layer)
            frame = shown.moving.at(self.clock.time, shown.hold)
            first = shown.placed_at is None
            colours = [piece.colour for piece in shown.moving.pieces] if first else None
            self._log_pieces(f"/{name}", frame.placings, frame.shown, colours)
            self.points(f"/{name}/lights", frame.lights[frame.lit], 0.12, LAMP, shown.layer)
            shown.placed_at = code
        for name, shown in self.shown.skinned.items():
            if shown.skinned is None:
                continue
            when = self.clock.time if shown.at is None else shown.at
            poses = getattr(shown.skinned, "poses", 0)  # posed live (`poses.Posed`): each new pose
            if (
                shown.posed_at is not None
                and abs(shown.posed_at - when) < 1 / 120
                and poses == shown.poses
            ):
                continue
            shown.posed_at, shown.poses = when, poses
            self._use(shown.layer)
            meshes = shown.skinned.at(when)
            if meshes:  # one mesh: one upload a frame, its normals made on the GPU
                points, triangles = read.join(meshes)
                self.viewer.log_mesh(
                    _usd_name(f"/{name}/0"),
                    wp.array(points, dtype=wp.vec3),
                    wp.array(triangles.ravel(), dtype=wp.int32),
                    color=shown.colour,
                    dynamic=True,
                    backface_culling=False,
                )
            self._hide_meshes(name, 1 if meshes else 0, shown.meshes)
            shown.meshes = 1 if meshes else 0
        target = (
            self.clock.follow() if self.clock.follow is not None and self.clock.following else None
        )
        if target is None:
            self.clock.followed = None
            return
        if self.clock.followed is not None:
            step = np.asarray(target, float) - self.clock.followed
            self.viewer.camera.translate(wp.vec3(*step.tolist()))
        self.clock.followed = np.asarray(target, float)

    def _ui(self, imgui: ModuleType) -> None:
        for draw in self._panels:
            # A panel that raises part-way is reported, and what it began and did not end
            # (windows, tab bars, ids, styles) ended for it: the next panels and the frame go on.
            state = imgui.internal.ErrorRecoveryState()
            imgui.internal.error_recovery_store_state(state)
            try:
                draw(imgui)
            except Exception:
                self.report(traceback.format_exc())
                _recovered(imgui, state)
        if self.shown.labels:
            canvas = imgui.get_background_draw_list()  # over the 3D view, under the panels
            placed: list[tuple[float, float, float, float]] = []  # tags drawn: none overlap
            for name, items in self.shown.labels.items():
                if self.solo is not None and name not in self.solo:
                    continue
                at, ahead = self.screen(np.array([label.at for label in items]))
                for label, (x, y), front in zip(items, at.tolist(), ahead.tolist(), strict=True):
                    if front:
                        placed.append(_tag(imgui, canvas, float(x), float(y), label, placed))
        if self.presses.grabbed:  # an app's drag: followed here, as Newton's own drags are
            io = imgui.get_io()
            mods = Mods(io.key_shift, io.key_ctrl, io.key_alt)
            if imgui.is_mouse_down(0):
                if (io.mouse_delta.x or io.mouse_delta.y) and self.on.drag:
                    self.on.drag(io.mouse_pos.x, io.mouse_pos.y, mods)
            else:
                self.presses.grabbed = False
                if self.on.release:
                    self.on.release(io.mouse_pos.x, io.mouse_pos.y, mods)


#: A label's tag: its box's colour, how far its text is inside it and off its point (pixels).
TAG_FILL, TAG_PAD, TAG_OFF = (0.08, 0.09, 0.11, 0.82), 4.0, 9.0


def _tag(
    imgui: ModuleType,
    canvas: ImDrawList,
    x: float,
    y: float,
    label: Label,
    placed: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """A label drawn as a tag beside its point: a dot on the point, a dark rounded box bordered
    in the label's colour, its text in its ink (white) in it: readable over any scene. Moved down past the
    tags `placed` (left, top, right, bottom) it would overlap, a line back to its point. Returns
    where it went."""
    size = imgui.calc_text_size(label.text)
    left, top = x + TAG_OFF, y - size.y / 2 - TAG_PAD
    right, bottom = left + size.x + 2 * TAG_PAD, top + size.y + 2 * TAG_PAD
    for _ in range(len(placed)):  # down past each it overlaps
        hit = next(
            (b for b in placed if left < b[2] and b[0] < right and top < b[3] and b[1] < bottom),
            None,
        )
        if hit is None:
            break
        top, bottom = hit[3] + 2, hit[3] + 2 + (bottom - top)
    colour = u32(imgui, label.colour)
    middle = (top + bottom) / 2
    canvas.add_circle_filled(imgui.ImVec2(x, y), 3.5, colour)
    canvas.add_line(imgui.ImVec2(x, y), imgui.ImVec2(left, middle), colour, 1.5)
    canvas.add_rect_filled(
        imgui.ImVec2(left, top), imgui.ImVec2(right, bottom), u32(imgui, TAG_FILL), 4.0
    )
    canvas.add_rect(imgui.ImVec2(left, top), imgui.ImVec2(right, bottom), colour, 4.0, 1.2)
    canvas.add_text(
        imgui.ImVec2(left + TAG_PAD, top + TAG_PAD),
        u32(imgui, label.ink or (0.96, 0.96, 0.96)),
        label.text,
    )
    return left, top, right, bottom


class _LocalAxes:
    """ImGuizmo as Newton's viewer reaches it (`gui.ui.giz`), drawing the gizmos named in
    `local` along their own axes, and those in `scaled` with its scale handles in place of
    its arrows: Newton's viewer asks for the world's arrows and rings for every gizmo, and
    names each (`push_id`) just before drawing it. A scale dragged is kept in `scales` (from
    where the drag began: ImGuizmo scales the matrix it started from) and taken out of the
    matrix Newton reads back, which keeps only where it is and how it is turned. All else
    passes through."""

    def __init__(self, giz: Any) -> None:  # noqa: ANN401
        self._giz, self.local, self.scaled, self._now = giz, set(), set(), None
        self.scales: dict[str, NDArray[np.float64]] = {}

    @classmethod
    def over(cls, viewer: Any, local: set[str], scaled: AbstractSet[str] = frozenset()) -> None:  # noqa: ANN401
        """Put in place over `viewer`'s ImGuizmo (once), drawing `local` along their own axes
        and `scaled` with scale handles."""
        ui = getattr(getattr(viewer, "gui", None), "ui", None)
        if ui is None or not hasattr(ui, "giz"):
            return
        if not isinstance(ui.giz, cls):
            ui.giz = cls(ui.giz)
        ui.giz.local, ui.giz.scaled = local, set(scaled)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._giz, name)

    def push_id(self, id_: Any) -> Any:  # noqa: ANN401
        if isinstance(id_, str):
            self._now = id_
        return self._giz.push_id(id_)

    def _op(self, op: Any) -> Any:  # noqa: ANN401
        """`op`, or its scale twin for a gizmo drawn with scale handles."""
        if self._now not in self.scaled:
            return op
        o = self._giz.OPERATION
        twins = {o.translate: o.scale, o.translate_x: o.scale_x}
        twins |= {o.translate_y: o.scale_y, o.translate_z: o.scale_z}
        return next((b for a, b in twins.items() if int(a) == int(op)), op)

    def is_over(self, op: Any = None) -> Any:  # noqa: ANN401
        return self._giz.is_over() if op is None else self._giz.is_over(self._op(op))

    def manipulate(self, view: Any, proj: Any, op: Any, mode: Any, *rest: Any) -> Any:  # noqa: ANN401
        if self._now in self.local:
            mode = self._giz.MODE.local
        done = self._giz.manipulate(view, proj, self._op(op), mode, *rest)
        if self._now in self.scaled and rest:
            values = rest[0].values  # column by column: the axes, then where it is
            sizes = np.linalg.norm(values[:12].reshape(3, 4)[:, :3], axis=1)
            if self._giz.is_using() and np.all(sizes > 1e-9):
                self.scales[str(self._now)] = sizes
            for k in range(3):  # Newton is given it unscaled
                values[4 * k : 4 * k + 3] /= max(float(sizes[k]), 1e-9)
        return done


def _recovered(imgui: ModuleType, state: Any) -> None:  # noqa: ANN401
    """What a panel began after `state` (an `ErrorRecoveryState`) ended for it, quietly: no
    assert and no log line each frame (the panel's error is reported once already)."""
    io = imgui.get_io()
    said = (io.config_error_recovery_enable_assert, io.config_error_recovery_enable_debug_log)
    io.config_error_recovery_enable_assert = io.config_error_recovery_enable_debug_log = False
    try:
        imgui.internal.error_recovery_try_to_recover_state(state)
    finally:
        io.config_error_recovery_enable_assert, io.config_error_recovery_enable_debug_log = said
