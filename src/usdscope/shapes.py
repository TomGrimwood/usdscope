"""Shapes and placings, plain numpy: USD placings as Newton's instance transforms and back, a
frame along an axis, a marker's flat parts, colours along a ramp. Knows nothing of viewers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .read import Placings

#: Viridis, five stops: dark purple, blue, teal, green, yellow (a ramp read the same in grey).
VIRIDIS = np.array(
    [
        (0.27, 0.0, 0.33),
        (0.23, 0.32, 0.55),
        (0.13, 0.57, 0.55),
        (0.37, 0.79, 0.38),
        (0.99, 0.91, 0.14),
    ]
)


def instances(placings: Placings) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """USD placings as Newton's transforms (n, 7: position, quaternion x y z w) and scales
    (n, 3); a mirrored placing flips x."""
    mats = np.asarray(placings, np.float64).reshape(-1, 4, 4)
    linear = mats[:, :3, :3]
    scales = np.linalg.norm(linear, axis=2)
    scales[np.linalg.det(linear) < 0, 0] *= -1
    r = np.transpose(linear / np.where(scales == 0, 1, scales)[:, :, None], (0, 2, 1))
    w = np.sqrt(np.maximum(0, 1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2])) / 2
    x = np.copysign(
        np.sqrt(np.maximum(0, 1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2])) / 2,
        r[:, 2, 1] - r[:, 1, 2],
    )
    y = np.copysign(
        np.sqrt(np.maximum(0, 1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2])) / 2,
        r[:, 0, 2] - r[:, 2, 0],
    )
    z = np.copysign(
        np.sqrt(np.maximum(0, 1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2])) / 2,
        r[:, 1, 0] - r[:, 0, 1],
    )
    q = np.stack([x, y, z, w], 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return np.concatenate([mats[:, 3, :3], q], 1).astype(np.float32), scales.astype(np.float32)


def matrix(
    position: Sequence[float] | NDArray, quaternion: Sequence[float] | NDArray
) -> NDArray[np.float64]:
    """A placing (4, 4; row vectors) from a position and a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in quaternion)
    turn = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    m = np.eye(4)
    m[:3, :3] = turn.T  # row vectors
    m[3, :3] = np.asarray(position, float)[:3]
    return m


def along(at: Sequence[float], axis: Sequence[float]) -> NDArray[np.float64]:
    """A placing (4, 4; row vectors) at `at` whose +Z runs along `axis` (any length), its +X
    level where it can be."""
    up = np.asarray(axis, float) / np.linalg.norm(axis)
    side = np.cross([0.0, 1.0, 0.0] if abs(up[1]) < 0.9 else [1.0, 0.0, 0.0], up)
    side /= np.linalg.norm(side)
    ahead = np.cross(up, side)
    placing = np.eye(4)
    placing[0, :3], placing[1, :3], placing[2, :3] = side, ahead, up
    placing[3, :3] = np.asarray(at, float)[:3]
    return placing


def ramp(share: Sequence[float] | NDArray, stops: NDArray = VIRIDIS) -> NDArray[np.float64]:
    """Colours (n, 3) a share of the way along `stops` (0 the first, 1 the last), each."""
    at = np.clip(np.asarray(share, float), 0, 1) * (len(stops) - 1)
    low = np.minimum(at.astype(int), len(stops) - 2)
    f = (at - low)[:, None]
    return stops[low] * (1 - f) + stops[low + 1] * f


def marker_meshes() -> dict[str, tuple[NDArray[np.float32], NDArray[np.int32]]]:
    """A marker's parts, flat on the ground, radius 1: a ring, a halo round it nearly twice as
    wide (faint), a dot in its middle."""
    turn = np.linspace(0.0, 2 * np.pi, 49)[:-1]
    circle = np.stack([np.cos(turn), np.sin(turn), np.zeros_like(turn)], 1)
    n = len(turn)
    k = np.arange(n)

    def annulus(inner: float, outer: float) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        v = np.concatenate([circle * inner, circle * outer]).astype(np.float32)
        t = np.concatenate(
            [
                np.stack([k, (k + 1) % n, n + k], 1),
                np.stack([(k + 1) % n, n + (k + 1) % n, n + k], 1),
            ]
        )
        return v, t.astype(np.int32)

    def disc(r: float) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
        v = np.concatenate([[[0.0, 0.0, 0.0]], circle * r]).astype(np.float32)
        t = np.stack([np.zeros(n, int), 1 + k, 1 + (k + 1) % n], 1)
        return v, t.astype(np.int32)

    return {"halo": annulus(1.0, 1.9), "ring": annulus(0.72, 1.0), "dot": disc(0.18)}


def arrows(
    at: NDArray,
    into: NDArray,
    out: NDArray,
    reach: float = 0.5,
    width: float = 0.12,
    head: float = 0.28,
    radius: float = 0.3,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Flat arrows lying on the ground (z up), one at each of `at` (n, 3): each comes in along
    `into` (n, 3), turns on a circular arc of `radius` (its middle on `at`) and goes out along
    `out` (n, 3); `reach` metres each side, a band `width` wide, a head `head` long. The arc's
    radius is kept over the band's width, so a sharp turn never folds the band over itself."""
    radius = max(radius, width)
    lines = [
        turning(p, _flat(a), _flat(b), reach, head, radius)
        for p, a, b in zip(
            np.asarray(at, float), np.asarray(into, float), np.asarray(out, float), strict=True
        )
    ]
    return arrows_along(lines, width, head)


def arrows_along(
    lines: Sequence[NDArray], width: float = 0.12, head: float = 0.28
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Flat arrows lying on the ground (z up), each along one of `lines` (each (m, 3), m >= 2)
    and pointing on past its end with a head `head` long: a band `width` wide, as one mesh
    facing up. A line bending tighter than the band's half width folds it: give lines that
    do not (a route with its corners rounded)."""
    vertices: list[NDArray] = []
    triangles: list[NDArray] = []
    for line in lines:
        line = np.asarray(line, float)
        ahead = np.gradient(line, axis=0)
        ahead[:, 2] = 0.0
        ahead /= np.maximum(np.linalg.norm(ahead, axis=1, keepdims=True), 1e-9)
        side = np.stack([-ahead[:, 1], ahead[:, 0], np.zeros(len(line))], axis=1)
        n, base = len(line), sum(len(v) for v in vertices)
        band = np.empty((2 * n, 3))
        band[0::2], band[1::2] = line - side * width / 2, line + side * width / 2
        k = base + 2 * np.arange(n - 1)[:, None]
        quads = np.concatenate([k + np.array([0, 2, 1]), k + np.array([1, 2, 3])])  # up
        end, half = line[-1], side[-1] * min(head * 0.6, 2.2 * width)
        tip = np.array([end - half, end + half, end + ahead[-1] * head])
        vertices += [band, tip]
        triangles += [quads, np.array([[1, 0, 2]]) + base + 2 * n]
    if not vertices:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.concatenate(vertices).astype(np.float32), np.concatenate(triangles).astype(np.int32)


def double_arrows(
    lines: Sequence[NDArray], width: float = 0.08, head: float = 0.16
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Flat arrows lying on the ground (z up) with a head at both ends, one along each of
    `lines` (each (m, 3), its ends the heads' points): handles that stretch (a straight line)
    or turn (an arc). A band `width` wide, heads `head` long, one mesh facing up."""
    parts = []
    for line in lines:
        line = np.asarray(line, float)
        total = _length(line)
        if total <= 2 * head:
            continue
        middle = along_line(line, total - head, total - 2 * head, 0.0, steps=max(len(line), 8))
        band, tris = _band(middle, width)
        ahead, back = _way(middle[-1] - middle[-2]), _way(middle[0] - middle[1])
        heads = []
        for end, way in ((middle[-1], ahead), (middle[0], back)):
            side = np.array([-way[1], way[0], 0.0]) * width * 1.6
            heads.append(np.array([end - side, end + side, end + way * head]))
        n = len(band)
        parts.append(
            (
                np.vstack([band, *heads]),
                np.vstack([tris, np.array([[n + 1, n, n + 2], [n + 4, n + 3, n + 5]])]),
            )
        )
    return _joined(parts)


def _band(line: NDArray, width: float) -> tuple[NDArray, NDArray]:
    """A band `width` wide along `line` (n, 3), facing up (local indices)."""
    ahead = np.gradient(line, axis=0)
    ahead[:, 2] = 0.0
    ahead /= np.maximum(np.linalg.norm(ahead, axis=1, keepdims=True), 1e-9)
    side = np.stack([-ahead[:, 1], ahead[:, 0], np.zeros(len(line))], axis=1) * width / 2
    band = np.empty((2 * len(line), 3))
    band[0::2], band[1::2] = line - side, line + side
    k = 2 * np.arange(len(line) - 1)[:, None]
    return band, np.concatenate([k + np.array([0, 2, 1]), k + np.array([1, 2, 3])])


def _way(d: NDArray) -> NDArray:
    flat = np.array([d[0], d[1], 0.0])
    return flat / max(float(np.linalg.norm(flat)), 1e-9)


def streaks(
    lines: Sequence[NDArray],
    share: float,
    length: float = 0.4,
    width: float = 0.06,
    head: float = 0.12,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Thin streaks lying flat (z up), one on each of `lines` (each (m, 3)): its front `share`
    (0 to 1) of the way along it (from just before its start to its end), as
    `streaks_along` draws them. With `share` going from 0 to 1 they travel along the lines."""
    parts = [
        streaks_along(line, [share * (_length(line) + length) - head], length, width, head, 0.0)
        for line in lines
    ]
    return _joined(parts)


def streaks_along(
    line: NDArray,
    fronts: Sequence[float],
    length: float = 0.4,
    width: float = 0.06,
    head: float = 0.12,
    fade: float = 1.0,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Thin streaks lying flat (z up) along polyline `line` (m, 3), each with its front at one
    of `fronts` (metres along it): `length` long, narrowing to nothing at its tail, `width`
    wide at its front, with a small head `head` long pointing on; each shrunk to nothing
    within `fade` metres of the line's ends (0: none). Fronts off the line: none drawn."""
    line = np.asarray(line, float)
    total = _length(line)
    parts = []
    for front in fronts:
        back = max(front - length, 0.0)
        front = min(front, total)
        if front - back <= 0.02:
            continue
        scale = 1.0 if fade <= 0 else float(np.clip(min(front, total - back) / fade, 0.0, 1.0))
        if scale <= 0.02:
            continue
        piece = along_line(line, front, front - back, 0.0, steps=10)
        parts.append(_streak(piece, width * scale, head * scale))
    return _joined(parts)


def rounded(line: NDArray, radius: float, steps: int = 8) -> NDArray:
    """Polyline `line` (m, 3) with each corner cut by a curve about `radius` round (less where
    its sides are short), so a path along it swings round corners instead of kinking."""
    line = np.asarray(line, float)
    if len(line) < 3:
        return line
    out = [line[0]]
    for a, p, b in zip(line[:-2], line[1:-1], line[2:], strict=True):
        into, away = p - a, b - p
        la, lb = float(np.linalg.norm(into[:2])), float(np.linalg.norm(away[:2]))
        if la < 1e-6 or lb < 1e-6:
            continue
        turn = float(np.arccos(np.clip(into[:2] @ away[:2] / (la * lb), -1.0, 1.0)))
        if turn < np.radians(2):
            out.append(p)
            continue
        cut = min(radius * np.tan(turn / 2), la / 2, lb / 2)
        start, end = p - into / la * cut, p + away / lb * cut
        t = np.linspace(0.0, 1.0, steps)[:, None]
        out.extend((1 - t) ** 2 * start + 2 * (1 - t) * t * p + t**2 * end)
    out.append(line[-1])
    return np.array(out)


def _streak(piece: NDArray, width: float, head: float) -> tuple[NDArray, NDArray]:
    """One streak along `piece` (n, 3), narrowing to its tail, with its head (local indices)."""
    ahead = np.gradient(piece, axis=0)
    ahead[:, 2] = 0.0
    ahead /= np.maximum(np.linalg.norm(ahead, axis=1, keepdims=True), 1e-9)
    side = np.stack([-ahead[:, 1], ahead[:, 0], np.zeros(len(piece))], axis=1)
    taper = np.linspace(0.0, 1.0, len(piece))[:, None] ** 1.5 * width / 2
    n = len(piece)
    band = np.empty((2 * n, 3))
    band[0::2], band[1::2] = piece - side * taper, piece + side * taper
    k = 2 * np.arange(n - 1)[:, None]
    quads = np.concatenate([k + np.array([0, 2, 1]), k + np.array([1, 2, 3])])
    end, half = piece[-1], side[-1] * width * 1.4
    tip = np.array([end - half, end + half, end + ahead[-1] * head])
    return np.vstack([band, tip]), np.vstack([quads, np.array([[1, 0, 2]]) + 2 * n])


def _joined(
    parts: Sequence[tuple[NDArray, NDArray]],
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Meshes (vertices, triangles) as one."""
    if not parts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    offsets = np.cumsum([0, *(len(v) for v, _ in parts[:-1])])
    vertices = np.concatenate([v for v, _ in parts]).astype(np.float32)
    triangles = np.concatenate([t + o for (_, t), o in zip(parts, offsets, strict=True)])
    return vertices, triangles.astype(np.int32)


def _length(line: NDArray) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(line, float), axis=0), axis=1).sum())


def along_line(line: NDArray, at: float, behind: float, ahead: float, steps: int = 16) -> NDArray:
    """The piece of polyline `line` (m, 3) from `behind` metres before to `ahead` metres after
    the point `at` metres along it (kept within it), as `steps` points evenly along."""
    line = np.asarray(line, float)
    gone = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    s = np.clip(np.linspace(at - behind, at + ahead, steps), 0.0, gone[-1])
    return np.stack([np.interp(s, gone, line[:, i]) for i in range(3)], axis=1)


def turning(
    p: NDArray, a: NDArray, b: NDArray, reach: float, head: float, radius: float
) -> NDArray:
    """The middle line of an arrow at `p` (3,): straight in along `a`, round a circular arc of
    `radius` whose middle is on `p`, straight out along `b` (each flat, of length 1), `reach`
    metres each side, stopping `head` short of its point (`arrows_along` draws it)."""
    turn = float(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))  # signed, left positive
    half_arc = radius * abs(turn) / 2
    steps = max(2, int(abs(turn) / np.radians(10)) + 2)
    headings = turn * np.linspace(0.0, 1.0, steps)
    cos, sin = np.cos(headings), np.sin(headings)
    ways = np.stack([a[0] * cos - a[1] * sin, a[0] * sin + a[1] * cos, np.zeros(steps)], axis=1)
    sign = 1.0 if turn >= 0 else -1.0
    mid = ways[steps // 2] if steps % 2 else _flat(ways[steps // 2 - 1] + ways[steps // 2])
    left = lambda w: np.stack([-w[..., 1], w[..., 0], np.zeros(w.shape[:-1])], axis=-1)  # noqa: E731
    centre = p + sign * radius * left(mid)
    arc = centre - sign * radius * left(ways) if abs(turn) > 1e-3 else p[None]
    into = max(reach - half_arc, 0.08)
    away = max(reach - half_arc - head, 0.04)
    return np.concatenate([arc[:1] - a * into, arc, arc[-1:] + b * away])


def _flat(way: NDArray) -> NDArray:
    """A way along the ground, of length 1 (+Y when it has none)."""
    flat = np.array([way[0], way[1], 0.0])
    size = np.linalg.norm(flat)
    return flat / size if size > 1e-9 else np.array([0.0, 1.0, 0.0])
