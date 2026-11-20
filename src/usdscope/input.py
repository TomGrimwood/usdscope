"""The app's mouse and keys before Newton's viewer's: a press the app grabs never turns the
camera, a press and release in place is a click (left) or a context click (right), keys go
to the app first; Escape never closes the window, H never hides the panels. W A S D and
the wheel (toward the mouse) move the camera at the view speed (`Viewport.view_speed`):
Shift faster, Ctrl slower, Shift and the wheel set it; Ctrl and the wheel, the lens."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newton._src.viewer.viewer_gui import ViewerGui

    from .view import Viewport

#: Pixels a press may move and still be a click.
CLICK = 4
#: The view speed to start with, and the slowest and fastest it goes, metres a second.
SPEED, SLOWEST, FASTEST = 10.0, 0.25, 2000.0
#: How much faster Shift moves the camera, and slower Ctrl.
HELD = 5.0
#: A notch of the wheel with Shift: the view speed this much faster (or slower).
NOTCH = 1.25
#: A notch of the wheel moves the camera as far as the view speed goes in this many seconds.
WHEEL = 0.25


@dataclass
class Presses:
    """What the mouse is doing: whether the app grabbed the left press; where the left press
    and the right press (the camera's, or a click to be) went down."""

    grabbed: bool = False
    left: tuple[float, float] | None = None
    right: tuple[float, float] | None = None


@dataclass(frozen=True)
class Mods:
    """The modifier keys held."""

    shift: bool = False
    ctrl: bool = False
    alt: bool = False


def wrap(view: Viewport, gui: ViewerGui) -> None:
    """The app's mouse and keys before Newton's: a press the app grabs never turns the
    camera; a press and release in place is a click."""
    from pyglet.window import key, mouse

    press, release, keys, scroll, fly = (
        gui.handle_mouse_press,
        gui.handle_mouse_release,
        gui.handle_key_press,
        gui.handle_mouse_scroll,
        gui.update_camera_from_keys,
    )
    io = gui.ui.io

    def mods() -> Mods:
        return Mods(io.key_shift, io.key_ctrl, io.key_alt)

    def on_press(x: float, y: float, button: int, to_framebuffer_coords: Callable) -> None:
        # The app is asked first, even over a gizmo: what it takes hold of is its (a point under
        # a gizmo, say), and the gizmo hidden while it drags (`Viewport.frame`).
        over_ui = io.mouse_down_owned[0] or view.gizmo_busy
        if button == mouse.RIGHT and not io.mouse_down_owned[1]:
            view.presses.right = (io.mouse_pos.x, io.mouse_pos.y)
        if button == mouse.LEFT and not over_ui:
            mx, my = io.mouse_pos.x, io.mouse_pos.y
            grabbed = False
            try:
                grabbed = bool(view.on.press and view.on.press(mx, my, mods()))
            except Exception:
                view.report(traceback.format_exc())
            if grabbed:
                view.presses.grabbed = True
                return
            view.presses.left = (mx, my)
        press(x, y, button, to_framebuffer_coords)

    def on_release(x: float, y: float, button: int) -> None:
        release(x, y, button)
        if button == mouse.RIGHT and view.presses.right is not None:
            mx, my = io.mouse_pos.x, io.mouse_pos.y
            moved = abs(mx - view.presses.right[0]) + abs(my - view.presses.right[1])
            view.presses.right = None
            if moved <= CLICK and view.on.context:
                try:
                    view.on.context(mx, my, mods())
                except Exception:
                    view.report(traceback.format_exc())
            return
        if button != mouse.LEFT or view.presses.left is None:
            return
        mx, my = io.mouse_pos.x, io.mouse_pos.y
        moved = abs(mx - view.presses.left[0]) + abs(my - view.presses.left[1])
        view.presses.left = None
        if moved <= CLICK and view.on.click:
            try:
                view.on.click(mx, my, mods())
            except Exception:
                view.report(traceback.format_exc())

    def on_key(symbol: int, close_fn: Callable | None = None) -> None:
        if gui.is_keyboard_capturing():
            return
        try:
            if view.on.key and view.on.key(symbol, mods()):
                return
        except Exception:
            view.report(traceback.format_exc())
        if symbol == key.SPACE:
            view.clock.playing = not view.clock.playing
        elif symbol not in (key.ESCAPE, key.H):
            keys(symbol, None)

    def on_scroll(scroll_y: float, is_ctrl_down: bool = False) -> None:
        if is_ctrl_down or gui.should_ignore_mouse_input():
            scroll(scroll_y, is_ctrl_down)  # Newton's: the field of view, or nothing
        elif io.key_shift:
            view.view_speed = min(max(view.view_speed * NOTCH**scroll_y, SLOWEST), FASTEST)
        else:  # toward the mouse, as far as the view speed goes in `WHEEL` seconds a notch
            _, direction = view.ray(io.mouse_pos.x, io.mouse_pos.y)
            step = direction * view.view_speed * WHEEL * scroll_y
            camera = view.viewer.camera
            camera.translate(camera._as_vec3([float(v) for v in step]))
            view.viewer._camera_dirty = True

    def on_fly(dt: float, is_key_down: Callable[[int], bool]) -> None:
        held = HELD if io.key_shift else 1 / HELD if io.key_ctrl else 1.0
        view.viewer.camera_speed = view.view_speed * held
        fly(dt, is_key_down)

    gui.handle_mouse_press, gui.handle_mouse_release, gui.handle_key_press = (
        on_press,
        on_release,
        on_key,
    )
    gui.handle_mouse_scroll, gui.update_camera_from_keys = on_scroll, on_fly

    def no_escape(symbol: int, modifiers: int) -> bool | None:
        """pyglet's own window closes on Escape: not this one."""
        return True if symbol == key.ESCAPE else None

    renderer = getattr(view.viewer, "renderer", None)
    window = renderer.window if renderer is not None else getattr(view.viewer, "_window", None)
    if window is not None:
        window.push_handlers(on_key_press=no_escape)
