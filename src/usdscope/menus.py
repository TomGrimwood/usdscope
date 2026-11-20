"""Menus from paths, for Dear ImGui: each entry says where it is ("Stop/tilt": the tilt item of
the Stop submenu; a last part "-" is a separator), what choosing it does, whether it is
ticked, and its key. Drawn inside an open popup or menu (`draw`), or chosen by path as a
script would (`find`).

    entries = [Entry("Stop/tilt", tilt), Entry("Stop/-", nothing), Entry("Remove", remove, key="Delete")]
    if imgui.begin_popup("context"):
        chosen = draw(imgui, "Point 2", entries)
        imgui.end_popup()
    if chosen:
        chosen.run()
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType


def nothing() -> None:
    """What a separator does."""


@dataclass(frozen=True)
class Entry:
    """A menu item: where it is, what choosing it does, whether it is ticked, its key."""

    path: str
    run: Callable[[], object] = nothing
    ticked: bool = False
    key: str | None = None

    @property
    def separator(self) -> bool:
        return self.path.rpartition("/")[2] == "-"


def separator(within: str = "") -> Entry:
    """A line between items, in the submenu `within` ("" the menu itself)."""
    return Entry(f"{within}/-" if within else "-")


def draw(
    imgui: ModuleType,
    title: str,
    entries: Sequence[Entry],
    colour: Callable[[str], tuple[float, float, float, float] | None] | None = None,
) -> Entry | None:
    """`entries` drawn under `title` (submenus by their paths' first parts, in the order they
    come); the one clicked, if any (not run: the caller runs it). `colour`: a label's colour
    (RGBA, 0 to 1), or None for the usual, for the title, submenus and items."""

    def coloured(label: str) -> bool:
        said = colour(label) if colour is not None else None
        if said is not None:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*said))
        return said is not None

    if coloured(title):
        imgui.separator_text(title)
        imgui.pop_style_color()
    else:
        imgui.separator_text(title)
    chosen = None
    within: str | None = None  # the submenu being drawn, "" for none
    opened = True
    for entry in entries:
        head, _, leaf = entry.path.rpartition("/")
        if head != within:
            if within and opened:
                imgui.end_menu()
            within = head
            pushed = bool(head) and coloured(head.rsplit("/", 1)[-1])
            opened = imgui.begin_menu(head) if head else True
            if pushed:
                imgui.pop_style_color()
        if not opened:
            continue
        if leaf == "-":
            imgui.separator()
        else:
            pushed = coloured(leaf)
            clicked = imgui.menu_item_simple(leaf, entry.key, entry.ticked)
            if pushed:
                imgui.pop_style_color()
            if clicked:
                chosen = entry
    if within and opened:
        imgui.end_menu()
    if not entries:
        imgui.text_disabled("Nothing here.")
    return chosen


def find(entries: Sequence[Entry], path: str) -> Entry:
    """The entry at `path`, as a click on it would choose it; ValueError saying what there is
    when there is none so."""
    found = next((e for e in entries if e.path == path and not e.separator), None)
    if found is None:
        there = ", ".join(e.path for e in entries if not e.separator)
        raise ValueError(f"No menu item {path!r} here: there are {there}.")
    return found
