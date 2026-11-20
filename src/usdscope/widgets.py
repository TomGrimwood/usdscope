"""Dear ImGui widgets over a viewport: a timeline along its bottom, something pointed at with
a pulsing outline, colours as ImGui takes them."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from . import input

if TYPE_CHECKING:
    from .overlays import Colour
    from .view import Viewport

#: The speeds the timeline offers.
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def timeline(
    view: Viewport,
    imgui: ModuleType,
    marks: Sequence[tuple[float, float, str]] = (),
    height: float = 92.0,
) -> None:
    """A timeline along the window's bottom: play, speed, the time and a slider; `marks`
    (start, end, label) drawn under the slider (stops, gestures)."""
    w, h, _, _ = view._sizes()
    imgui.set_next_window_pos(imgui.ImVec2(8, h - height - 8), imgui.Cond_.always)
    imgui.set_next_window_size(imgui.ImVec2(w - 16, height), imgui.Cond_.always)
    flags = (
        imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
    )
    imgui.begin("##timeline", flags=flags)
    if imgui.button("Pause" if view.clock.playing else "Play", imgui.ImVec2(64, 0)):
        view.clock.playing = not view.clock.playing
    imgui.same_line()
    imgui.set_next_item_width(80)
    names = [f"{s:g}x" for s in SPEEDS]
    now = SPEEDS.index(view.clock.speed) if view.clock.speed in SPEEDS else SPEEDS.index(1.0)
    changed, picked = imgui.combo("##speed", now, names)
    if changed:
        view.clock.speed = SPEEDS[picked]
    if view.clock.follow is not None:
        imgui.same_line()
        _, view.clock.following = imgui.checkbox("Follow", view.clock.following)
    imgui.same_line()
    imgui.text(f"{view.clock.time:7.1f} / {view.clock.duration:.1f} s")
    view_speed(view, imgui, w)
    imgui.set_next_item_width(w - 40)
    changed, value = imgui.slider_float(
        "##time", view.clock.time, 0.0, max(view.clock.duration, 0.01), ""
    )
    if changed:
        view.clock.time = value
    if marks and view.clock.duration > 0:
        draw = imgui.get_window_draw_list()
        low = imgui.get_cursor_screen_pos()
        span = w - 40
        for a, b, label in marks:
            x0 = low.x + span * max(a, 0.0) / view.clock.duration
            x1 = max(low.x + span * min(b, view.clock.duration) / view.clock.duration, x0 + 2)
            draw.add_rect_filled(
                imgui.ImVec2(x0, low.y),
                imgui.ImVec2(x1, low.y + 14),
                u32(imgui, (1.0, 0.6, 0.2, 0.8)),
            )
            if x1 - x0 > 30:
                draw.add_text(imgui.ImVec2(x0 + 2, low.y), u32(imgui, (0.1, 0.1, 0.1, 1.0)), label)
    imgui.end()


def view_speed(view: Viewport, imgui: ModuleType, w: float) -> None:
    """The view speed at the line's right, to drag or type."""
    imgui.same_line(max(w - 200, 440))  # past play, speed, Follow and the time
    imgui.set_next_item_width(170)
    changed, speed = imgui.drag_float(
        "##view_speed",
        view.view_speed,
        0.02 * view.view_speed,
        input.SLOWEST,
        input.FASTEST,
        "View speed %.3g m/s",
    )
    if changed:
        view.view_speed = min(max(speed, input.SLOWEST), input.FASTEST)


def pulse(
    imgui: ModuleType,
    view: Viewport,
    label: str,
    at: Sequence[float] | None = None,
    rect: Sequence[float] | None = None,
    since: float = 0.0,
) -> None:
    """`label` pointed at over everything: a screen rectangle (x, y, w, h) outlined, or a 3D
    point `at` ringed, pulsing since `since` (`time.perf_counter`)."""
    glow = 0.55 + 0.45 * math.sin((time.perf_counter() - since) * 6.0)
    colour = u32(imgui, (1.0, 0.85, 0.1, glow))
    canvas = imgui.get_foreground_draw_list()
    if rect is not None:
        x, y, w, h = rect
        canvas.add_rect(
            imgui.ImVec2(x - 4, y - 4), imgui.ImVec2(x + w + 4, y + h + 4), colour, 6.0, 0, 3.0
        )
        canvas.add_text(imgui.ImVec2(x, y + h + 8), colour, label)
    elif at is not None:
        spot, ahead = view.screen(np.array([at], float))
        if ahead[0]:
            x, y = float(spot[0, 0]), float(spot[0, 1])
            canvas.add_circle(imgui.ImVec2(x, y), 26.0 + 6.0 * glow, colour, 0, 3.0)
            canvas.add_text(imgui.ImVec2(x + 34, y - 8), colour, label)


def u32(imgui: ModuleType, colour: Colour) -> int:
    rgba = (*colour, 1.0) if len(colour) == 3 else colour
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


def same_line_if(imgui: ModuleType, label: str, extra: float = 0.0) -> None:
    """Before an item labelled `label` (`extra` pixels more than its text and padding: a radio
    button's circle, say): on the line of the last item if it fits there, else on the next."""
    try:
        style = imgui.get_style()
        wanted = imgui.calc_text_size(label.split("##")[0]).x + 2 * style.frame_padding.x + extra
        last = imgui.get_item_rect_max().x + style.item_spacing.x
        right = imgui.get_window_pos().x + imgui.get_window_width() - style.window_padding.x
    except (AttributeError, TypeError):  # no layout to ask (a stand-in for Dear ImGui)
        imgui.same_line()
        return
    if last + wanted <= right:
        imgui.same_line()


def row(
    imgui: ModuleType, labels: list[str], button: Callable[[str], object] | None = None
) -> list[bool]:
    """Buttons (`button`: Dear ImGui's `button`, unless another) labelled `labels`, as many to a
    line as fit, the rest wrapped onto the next: whether each was pressed."""
    press = button or imgui.button
    pressed = []
    for k, label in enumerate(labels):
        if k:
            same_line_if(imgui, label)
        pressed.append(bool(press(label)))
    return pressed


def confirm(imgui: ModuleType, popup: str, question: str, yes: str) -> bool:
    """A small menu `popup` (opened with `imgui.open_popup(popup)`, drawn each frame here)
    asking `question`: True the frame `yes` is clicked; Cancel or a click away closes it."""
    chosen = False
    if imgui.begin_popup(popup):
        imgui.separator_text(question)
        chosen = bool(imgui.menu_item_simple(yes))
        imgui.menu_item_simple("Cancel")
        imgui.end_popup()
    return chosen
