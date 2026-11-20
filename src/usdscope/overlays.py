"""What a viewport draws over its scene, by name in a layer: meshes in place, markers on the
ground, ribbons along a path, points, lines, labels, placings of what was drawn; and how it
hands them to Newton's viewer (instanced meshes, named as USD prims for RTX).

`Overlays` is the viewport's drawing half (`view.Viewport` is one): it needs `viewer`, `_use`
(a layer made current), `_drawn` and the marker and ribbon layers it keeps.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp
from newton import Axis
from numpy.typing import NDArray

from . import read
from .poses import Posed
from .read import RGB, XYZ, Placings
from .shapes import instances, marker_meshes

Colour = RGB | tuple[float, float, float, float]


@dataclass(frozen=True)
class Label:
    """Text drawn over the scene at a 3D point: its tag's dot and border in `colour`, its text
    in `ink` (white when None)."""

    at: XYZ
    text: str
    colour: Colour = (1.0, 1.0, 1.0)
    ink: Colour | None = None


@dataclass
class _Moving:
    moving: read.Moving
    hold: bool
    layer: str
    placed_at: int | None = None  # the stage's time code last placed at


@dataclass
class _Skinned:
    skinned: read.Skinned | Posed | None
    colour: RGB
    at: float | None  # always posed at this time, else at the viewport's
    layer: str
    meshes: int = 0  # how many meshes have been logged under its name
    posed_at: float | None = None
    poses: int = 0  # how many times it had been posed live when last drawn (`poses.Posed`)


@dataclass
class _Gizmo:
    transform: Any  # a wp.transform, dragged in place by Newton's gizmo
    translate: list[Axis] = field(default_factory=list)
    rotate: list[Axis] = field(default_factory=list)
    local: bool = False  # its arrows and rings along its own axes, turning as it turns
    scale: bool = False  # ImGuizmo's scale handles instead of its arrows (`gizmo_scale`)


@dataclass
class Shown:
    """What a viewport has drawn: its labels, moving things, skinned characters and gizmos by
    name; the layer being drawn in; what each layer drew, (kind, name, its mesh), to hide where
    a layer cannot be freed (RTX); the layers that have markers or ribbons."""

    labels: dict[str, list[Label]] = field(default_factory=dict)
    moving: dict[str, _Moving] = field(default_factory=dict)
    skinned: dict[str, _Skinned] = field(default_factory=dict)
    gizmos: dict[str, _Gizmo] = field(default_factory=dict)
    layer: str = ""
    meshes: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    markers: set[str] = field(default_factory=set)
    ribbons: set[str] = field(default_factory=set)
    # What was last sent under each (layer, name), as a fingerprint: the same again is not.
    sent: dict[tuple[str, str], int] = field(default_factory=dict)


class Overlays:
    """Drawing by name: each call replaces what was drawn under its name (None: takes it away)."""

    viewer: Any
    shown: Shown

    def _use(self, layer: str) -> None:
        raise NotImplementedError

    def _unchanged(self, name: str, *what: object) -> bool:
        """Whether `what` is what was last drawn under `name` in the layer in use (then nothing
        need be sent again); remembered as drawn now if not."""
        made = hash(tuple(_fingerprint(x) for x in what))
        key = (self.shown.layer, name)
        if self.shown.sent.get(key) == made:
            return True
        self.shown.sent[key] = made
        return False

    def mesh(
        self,
        name: str,
        vertices: NDArray,
        triangles: NDArray,
        colour: RGB,
        opacity: float = 1.0,
        hidden: bool = False,
        placings: Placings | None = None,
        layer: str = "overlay",
    ) -> None:
        """A mesh drawn in one colour: at its own points, or a copy at each of `placings`."""
        self._use(layer)
        if placings is None:
            self._log_mesh(name, vertices, triangles, color=colour, opacity=opacity, hidden=hidden)
            return
        self._log_mesh(f"{name}/mesh", vertices, triangles, hidden=True)
        colours = np.tile(np.array(colour, np.float32), (len(placings), 1))
        self._log_instances(name, f"{name}/mesh", placings, colours, hidden=hidden)

    def markers(
        self,
        name: str,
        at: NDArray | None,
        colours: NDArray | RGB = (0.3, 1.0, 0.4),
        radius: float = 0.35,
        layer: str = "overlay",
    ) -> None:
        """Places marked on the ground at `at` (n, 3): each a bright ring lying flat, a soft
        halo round it and a dot in its middle, in its colour (one, or one each (n, 3)); None
        or none: gone."""
        name = _usd_name(name)
        self._use(layer)
        if self._unchanged(name, at, colours, radius) and layer in self.shown.markers:
            return
        if layer not in self.shown.markers:  # the marker's own meshes, once a layer
            for part, (v, t) in marker_meshes().items():
                self._log_mesh(f"/_marker/{part}", v, t, hidden=True)
            self.shown.markers.add(layer)
        n = 0 if at is None else len(at)
        spots = np.asarray(at if n else np.zeros((0, 3)), float).reshape(-1, 3)
        each = (
            np.broadcast_to(np.asarray(colours, np.float32), (n, 3))
            if n
            else np.zeros((0, 3), np.float32)
        )
        placings = np.tile(np.eye(4), (n, 1, 1))
        placings[:, :3, :3] *= radius
        placings[:, 3, :3] = spots + np.array([0.0, 0.0, 0.02])  # just above the ground
        for part, opacity, lift in (("halo", 0.28, 0.0), ("ring", 1.0, 0.005), ("dot", 0.9, 0.01)):
            moved = placings.copy()
            moved[:, 3, 2] += lift
            self._log_instances(
                f"{name}/{part}",
                f"/_marker/{part}",
                moved,
                np.ascontiguousarray(each),
                hidden=not n,
                opacities=np.full(n, opacity, np.float32),
            )

    def ribbon(
        self,
        name: str,
        line: NDArray | None,
        colours: NDArray | RGB = (1.0, 0.85, 0.2),
        width: float = 0.25,
        layer: str = "overlay",
        opacity: float = 0.9,
        ends: NDArray | None = None,
    ) -> None:
        """A line through `line`'s points (n, 3) drawn as a band `width` wide lying on the
        ground, each piece in its colour (one, or one each): a path. With `ends` (n, 3), pieces
        apart instead, from each of `line`'s points to its end. None or none: gone."""
        name = _usd_name(name)
        self._use(layer)
        if (
            self._unchanged(name, line, colours, width, opacity, ends)
            and layer in self.shown.ribbons
        ):
            return
        if layer not in self.shown.ribbons:
            square = np.array(
                [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]], np.float32
            )
            self._log_mesh(
                "/_ribbon/piece", square, np.array([[0, 1, 2], [0, 2, 3]], np.int32), hidden=True
            )
            self.shown.ribbons.add(layer)
        points = np.zeros((0, 3)) if line is None else np.asarray(line, float).reshape(-1, 3)
        if ends is not None:
            a, b = points, np.asarray(ends, float).reshape(-1, 3)
        else:
            a, b = points[:-1], points[1:]
        n = len(a)
        along = b - a
        length = np.linalg.norm(along, axis=1)
        x = along / np.maximum(length, 1e-9)[:, None]
        y = np.cross([0.0, 0.0, 1.0], x)
        y /= np.maximum(np.linalg.norm(y, axis=1), 1e-9)[:, None]
        z = np.cross(x, y)
        placings = np.tile(np.eye(4), (n, 1, 1))
        # Each piece a square stretched along it (a little past its ends: no gaps at the turns).
        placings[:, 0, :3] = x * (length + width * 0.5)[:, None]
        placings[:, 1, :3] = y * width
        placings[:, 2, :3] = z
        placings[:, 3, :3] = (a + b) / 2
        each = (
            np.broadcast_to(np.asarray(colours, np.float32), (n, 3))
            if n
            else np.zeros((0, 3), np.float32)
        )
        self._log_instances(
            name,
            "/_ribbon/piece",
            placings,
            np.ascontiguousarray(each),
            hidden=not n,
            opacities=np.full(n, opacity, np.float32),
        )

    def place(self, name: str, placings: Placings, layer: str = "overlay") -> None:
        """A mesh drawn with `mesh(..., placings=...)` moved to new placings (the same count),
        its colours kept: nothing else uploaded."""
        self._use(layer)
        self._log_instances(name, f"{name}/mesh", placings, None)

    def points(
        self,
        name: str,
        xyz: NDArray | None,
        radius: float = 0.2,
        colours: NDArray | RGB = (1.0, 1.0, 1.0),
        layer: str = "overlay",
    ) -> None:
        """Balls at `xyz` (n, 3), in one colour or one each (n, 3); None or none: gone."""
        name = _usd_name(name)
        self._use(layer)
        if self._unchanged(name, xyz, radius, colours):
            return
        self._keep("points", name)
        if xyz is None or not len(xyz):
            self.viewer.log_points(name, None)
            return
        at = np.asarray(xyz, np.float32).reshape(-1, 3)
        each = np.ascontiguousarray(np.broadcast_to(np.asarray(colours, np.float32), at.shape))
        self.viewer.log_points(
            name, wp.array(at, dtype=wp.vec3), float(radius), wp.array(each, dtype=wp.vec3)
        )

    def lines(
        self,
        name: str,
        starts: NDArray | None,
        ends: NDArray | None = None,
        colours: NDArray | RGB = (1.0, 1.0, 1.0),
        layer: str = "overlay",
    ) -> None:
        """Lines from `starts` to `ends` (n, 3); without `ends`, `starts` is a polyline. None or
        none: gone."""
        name = _usd_name(name)
        self._use(layer)
        if self._unchanged(name, starts, ends, colours):
            return
        self._keep("lines", name)
        if starts is None or len(starts) < (1 if ends is not None else 2):
            self.viewer.log_lines(name, None, None, None)
            return
        a = np.asarray(starts, np.float32)
        b = np.asarray(ends, np.float32) if ends is not None else a[1:]
        a = a if ends is not None else a[:-1]
        each = np.ascontiguousarray(np.broadcast_to(np.asarray(colours, np.float32), a.shape))
        self.viewer.log_lines(
            name,
            wp.array(a, dtype=wp.vec3),
            wp.array(b, dtype=wp.vec3),
            wp.array(each, dtype=wp.vec3),
        )

    def labels(self, name: str, items: Sequence[Label] | None) -> None:
        """Text at 3D points, drawn over the scene; None or none: gone."""
        if items:
            self.shown.labels[name] = list(items)
        else:
            self.shown.labels.pop(name, None)

    def _log_mesh(
        self,
        name: str,
        vertices: NDArray,
        triangles: NDArray,
        *,
        color: RGB | None = None,
        opacity: float | None = None,
        hidden: bool = False,
        dynamic: bool = False,
    ) -> None:
        name = _usd_name(name)
        self._keep("mesh", name)
        v = np.ascontiguousarray(vertices, np.float32)
        t = np.ascontiguousarray(np.asarray(triangles, np.int32).reshape(-1, 3))
        self.viewer.log_mesh(
            name,
            wp.array(v, dtype=wp.vec3),
            wp.array(t.ravel(), dtype=wp.int32),
            normals=wp.array(read.normals(v, t), dtype=wp.vec3),
            hidden=hidden,
            backface_culling=False,
            color=color,
            opacity=opacity,
            dynamic=dynamic,
        )

    def _log_instances(
        self,
        name: str,
        mesh: str,
        placings: Placings,
        colours: NDArray | None,
        hidden: bool = False,
        shown: NDArray[np.bool_] | None = None,
        opacities: NDArray[np.float32] | None = None,
    ) -> None:
        """Copies of a logged mesh at `placings`; `colours` None keeps their colours; those not
        `shown` shrunk away (the batch kept)."""
        name, mesh = _usd_name(name), _usd_name(mesh)
        self._keep("instances", name, mesh)
        xforms, scales = instances(placings)
        if shown is not None:
            scales[~shown] = 0.0
        self.viewer.log_instances(
            name,
            mesh,
            wp.array(xforms, dtype=wp.transform),
            wp.array(scales, dtype=wp.vec3),
            wp.array(np.ascontiguousarray(colours, np.float32), dtype=wp.vec3)
            if colours is not None
            else None,
            None,
            hidden=hidden,
            opacities=wp.array(opacities, dtype=wp.float32) if opacities is not None else None,
        )

    def _log_pieces(
        self,
        name: str,
        placings: Sequence[Placings],
        shown: Sequence[NDArray[np.bool_]],
        colours: Sequence[RGB] | None,
    ) -> None:
        """Copies of meshes `{name}/mesh{k}` at `placings[k]` as `{name}/at{k}`, those not
        `shown` shrunk away; `colours` (one a mesh) None keeps theirs. One upload for them all,
        each mesh given its slice: per mesh, a frame's arrays cost more than its copies."""
        counts = [len(p) for p in placings]
        if not sum(counts):
            return
        xforms, scales = instances(np.concatenate([np.asarray(p, float) for p in placings]))
        scales[~np.concatenate(shown)] = 0.0
        all_xforms = wp.array(xforms, dtype=wp.transform)
        all_scales = wp.array(scales, dtype=wp.vec3)
        start = 0
        for k, n in enumerate(counts):
            at, mesh = _usd_name(f"{name}/at{k}"), _usd_name(f"{name}/mesh{k}")
            self._keep("instances", at, mesh)
            tint = None
            if colours is not None:
                tint = wp.array(np.tile(np.array(colours[k], np.float32), (n, 1)), dtype=wp.vec3)
            end = start + n
            self.viewer.log_instances(
                at, mesh, all_xforms[start:end], all_scales[start:end], tint, None
            )
            start = end

    def _keep(self, kind: str, name: str, mesh: str = "") -> None:
        """What the layer in use has drawn, once each."""
        drawn = self.shown.meshes.setdefault(self.shown.layer, [])
        if (kind, name, mesh) not in drawn:
            drawn.append((kind, name, mesh))


def _fingerprint(x: object) -> object:
    """`x` as something to hash: an array by its shape, type and bytes."""
    if isinstance(x, np.ndarray):
        return (x.shape, x.dtype.str, hash(np.ascontiguousarray(x).tobytes()))
    if isinstance(x, (list, tuple)):
        return tuple(_fingerprint(v) for v in x)
    return x


def _usd_name(name: str) -> str:
    """A name as a USD prim path (RTX makes a prim of each thing drawn): each part made of
    letters, digits and underscores, never starting with a digit."""
    parts = [re.sub(r"[^A-Za-z0-9_]", "_", part) for part in name.strip("/").split("/") if part]
    return "/" + "/".join(f"_{part}" if part[0].isdigit() else part for part in parts)
