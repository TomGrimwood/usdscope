"""Picking without a window: which triangle a ray meets first. NumPy only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def hit(
    triangles: NDArray, origin: Sequence[float] | NDArray, direction: Sequence[float] | NDArray
) -> tuple[int, float]:
    """The first of `triangles` (n, 3, 3) a ray meets: (its index, how far), or (-1, inf). To
    cast many rays at the same triangles, `Triangles` once and its `hit`."""
    return _first(np.asarray(triangles, np.float64), origin, direction)


class Triangles:
    """Triangles to cast rays at many times (a navmesh): sorted into chunks of `CHUNK` near each
    other, each chunk's box tested first, then the triangles of the chunks a ray goes through.

        found = Triangles(navmesh_triangles)
        k, far = found.hit(origin, direction)   # as `hit`, the index the triangles were given in
    """

    CHUNK = 128

    def __init__(self, triangles: NDArray) -> None:
        tri = np.asarray(triangles, np.float64).reshape(-1, 3, 3)
        self.order = _morton(tri.mean(axis=1)[:, :2]) if len(tri) else np.zeros(0, np.int64)
        self.sorted = tri[self.order]
        n = len(tri)
        starts = np.arange(0, n, self.CHUNK)
        self.starts = starts
        corners = self.sorted.reshape(-1, 3)
        self.low = np.minimum.reduceat(corners, starts * 3) if n else np.zeros((0, 3))
        self.high = np.maximum.reduceat(corners, starts * 3) if n else np.zeros((0, 3))

    def hit(
        self, origin: Sequence[float] | NDArray, direction: Sequence[float] | NDArray
    ) -> tuple[int, float]:
        """The first triangle the ray meets: (its index, how far), or (-1, inf)."""
        origin, direction = np.asarray(origin, float), np.asarray(direction, float)
        through = _boxes(self.low, self.high, origin, direction)
        if not through.any():
            return -1, float("inf")
        chunks = np.flatnonzero(through)
        index = (chunks[:, None] * self.CHUNK + np.arange(self.CHUNK)).ravel()
        index = index[index < len(self.sorted)]
        k, far = _first(self.sorted[index], origin, direction)
        return (int(self.order[index[k]]), far) if k >= 0 else (-1, far)


def _first(
    tri: NDArray, origin: Sequence[float] | NDArray, direction: Sequence[float] | NDArray
) -> tuple[int, float]:
    """Möller-Trumbore against every one of `tri` at once: the nearest met, or (-1, inf)."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = b - a, c - a
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if not good.any():
        return -1, float("inf")
    k = int(np.flatnonzero(good)[np.argmin(t[good])])
    return k, float(t[k])


def _boxes(low: NDArray, high: NDArray, origin: NDArray, direction: NDArray) -> NDArray:
    """Which boxes (low, high: (n, 3)) a ray from `origin` along `direction` goes through."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / direction
        t0, t1 = (low - origin) * inverse, (high - origin) * inverse
    near = np.nanmax(np.minimum(t0, t1), axis=1)
    far = np.nanmin(np.maximum(t0, t1), axis=1)
    # A ray along a face's plane: inside the box on that axis or not at all.
    flat = direction == 0
    if flat.any():
        inside = ((origin >= low) & (origin <= high))[:, flat].all(axis=1)
        far = np.where(inside, far, -np.inf)
    return (far >= np.maximum(near, 0.0)) & (far >= 0)


def _morton(xy: NDArray) -> NDArray:
    """An order of points `xy` (n, 2) along a Z curve: points near each other near each other."""
    low, size = xy.min(axis=0), np.ptp(xy, axis=0).max() or 1.0
    cell = np.clip(((xy - low) / size * 65535).astype(np.uint64), 0, 65535)

    def spread(v: NDArray) -> NDArray:
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x33333333)
        return (v | (v << np.uint64(1))) & np.uint64(0x55555555)

    return np.argsort(spread(cell[:, 0]) | (spread(cell[:, 1]) << np.uint64(1)), kind="stable")


@dataclass(frozen=True)
class Hit:
    """Where a ray met a mesh: the point, how far along the ray, which mesh (its index as the
    picker was given them) and which copy of it (its placing's index)."""

    at: tuple[float, float, float]
    distance: float
    mesh: int
    copy: int


class Picker:
    """Rays against meshes placed many times (a scene's), without every triangle in the world:
    every copy's box in the world tested at once, then the triangles of the copies a ray goes
    through, nearest first, in their mesh's own frame (a big mesh's in chunks, `Triangles`).

        picker = Picker([(mesh.vertices, mesh.triangles, mesh.placings) for mesh in scene.meshes])
        found = picker.hit(origin, direction)   # a Hit (where, how far, which mesh and copy) or None
        found = picker.hit(origin, direction, far=4.5)   # only as far as that: a few copies
    """

    #: A mesh with more triangles than this is cast at in chunks (`Triangles`), made when first hit.
    BIG = 2048

    def __init__(self, meshes: Sequence[tuple[NDArray, NDArray, NDArray]]) -> None:
        self._triangles: dict[int, NDArray] = {}
        self._chunked: dict[int, Triangles] = {}
        lows, highs, which, inverses = [], [], [], []
        for index, (vertices, triangles, placings) in enumerate(meshes):
            v = np.asarray(vertices, np.float64)
            t = np.asarray(triangles, np.int64).reshape(-1, 3)
            if not len(t):
                continue
            mats = np.asarray(placings, np.float64).reshape(-1, 4, 4)
            if not len(mats):
                continue
            self._triangles[index] = v[t]
            lo, hi = v.min(0), v.max(0)
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            placed = np.einsum("kj,nji->nki", corners, mats[:, :3, :3]) + mats[:, None, 3, :3]
            lows.append(placed.min(axis=1))
            highs.append(placed.max(axis=1))
            which.append(np.stack([np.full(len(mats), index), np.arange(len(mats))], axis=1))
            inverses.append(np.linalg.inv(mats))
        self._low = np.concatenate(lows) if lows else np.zeros((0, 3))
        self._high = np.concatenate(highs) if highs else np.zeros((0, 3))
        self._which = np.concatenate(which) if which else np.zeros((0, 2), np.int64)
        self._inverse = np.concatenate(inverses) if inverses else np.zeros((0, 4, 4))

    def hit(
        self,
        origin: Sequence[float] | NDArray,
        direction: Sequence[float] | NDArray,
        far: float = float("inf"),
    ) -> Hit | None:
        """Where a ray first meets a mesh, how far along it (in the ray's own units: no further
        than `far`), and which mesh and copy of it (their indices, as given)."""
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        if not len(self._low):
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / d
            t0, t1 = (self._low - o) * inverse, (self._high - o) * inverse
        near = np.nanmax(np.minimum(t0, t1), axis=1)
        out = np.nanmin(np.maximum(t0, t1), axis=1)
        flat = d == 0
        if flat.any():  # a ray along a face's plane: inside the box on that axis or not at all
            inside = ((o >= self._low) & (o <= self._high))[:, flat].all(axis=1)
            out = np.where(inside, out, -np.inf)
        near = np.maximum(near, 0.0)
        through = np.flatnonzero((out >= near) & (near <= far))
        best, which = far, (-1, -1)
        o4, d4 = np.r_[o, 1.0], np.r_[d, 0.0]
        for k in through[np.argsort(near[through])]:
            if near[k] >= best:
                break
            index, copy = (int(v) for v in self._which[k])
            local_o = (o4 @ self._inverse[k])[:3]  # in the copy's frame: the same distances
            local_d = (d4 @ self._inverse[k])[:3]
            found, along = self._cast(index, local_o, local_d)
            if found >= 0 and along < best:
                best, which = along, (index, copy)
        if which[0] < 0:
            return None
        x, y, z = (o + best * d).tolist()
        return Hit((x, y, z), float(best), which[0], which[1])

    def _cast(self, index: int, origin: NDArray, direction: NDArray) -> tuple[int, float]:
        triangles = self._triangles[index]
        if len(triangles) <= self.BIG:
            return _first(triangles, origin, direction)
        if index not in self._chunked:
            self._chunked[index] = Triangles(triangles)
        return self._chunked[index].hit(origin, direction)


def footprint(
    vertices: NDArray, placing: NDArray
) -> tuple[tuple[float, float], float, float, float]:
    """A placed mesh's outline on the ground as a turned box: (its middle (x, y), its length,
    its width, the bearing of its length in degrees clockwise from +Y), the smallest box round
    it along its own main ways (its points' principal axes in plan)."""
    m = np.asarray(placing, float)
    xy = (np.asarray(vertices, float) @ m[:3, :3] + m[3, :3])[:, :2]
    middle = xy.mean(0)
    _, axes = np.linalg.eigh(np.cov((xy - middle).T) + np.eye(2) * 1e-12)
    along = axes[:, 1]  # the way it spreads most
    across = np.array([along[1], -along[0]])
    a, b = (xy - middle) @ along, (xy - middle) @ across
    centre = middle + along * (a.max() + a.min()) / 2 + across * (b.max() + b.min()) / 2
    bearing = float(np.degrees(np.arctan2(along[0], along[1])) % 180.0)
    return (float(centre[0]), float(centre[1])), float(np.ptp(a)), float(np.ptp(b)), bearing
