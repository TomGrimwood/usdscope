"""A USD stage read into plain arrays to draw: its still meshes (each mesh once, every copy's
placing and colour), what moves in it (each thing's meshes joined by colour, placed at any
time), and skinned characters posed at any time. Z up, metres, whatever the stage's own axes
and units; placings are USD's matrices (row vectors: p' = p M).

    stage = Usd.Stage.Open("scene.usd")
    still = read.scene(stage)              # Scene: meshes, each with its copies' placings, colours
    moving = read.moving(stage)            # Moving: .at(seconds) -> Placement
    walker = read.skinned("walk.usdc")     # Skinned: .at(seconds) -> [(points, triangles)]

A colour is a material's own plain colour: a preview surface's diffuse colour or the average
colour of its diffuse texture; an MDL material's base colour or texture (OmniPBR's and
OmniUber's input names); else the mesh's displayColor; else grey. Colours are the screen's
(sRGB), as a viewer draws them.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdSkel, UsdUtils, Vt

from .kept import KEPT, kept

RGB = tuple[float, float, float]
XYZ = tuple[float, float, float]
#: A box in the world: (its low corner, its high corner).
Box = tuple[XYZ, XYZ]
Points = NDArray[np.float32]  # (v, 3)
Triangles = NDArray[np.int32]  # (t, 3), indices into points
Placings = NDArray[np.float64]  # (n, 4, 4), USD's matrices (row vectors)
Mask = NDArray[np.bool_]  # (n,)
MeshArrays = tuple[Points, Triangles]

GREY: RGB = (0.72, 0.70, 0.66)
#: An MDL material's inputs holding its base colour, and its base colour's texture, by name.
MDL_COLOURS = (
    "diffuse_color_constant",
    "diffuse_reflection_color",
    "base_color",
    "albedo_constant",
    "diffuse_tint",
)
MDL_TEXTURES = (
    "diffuse_texture",
    "base_color_image",
    "albedo_texture",
    "diffuse_reflection_color_image",
)


@dataclass(frozen=True)
class Look:
    """What a material shows: a plain colour, or a texture whose average colour is shown."""

    colour: RGB | None = None
    texture: str | None = None


@dataclass
class Mesh:
    """One mesh: its own points and triangles, and each copy's placing and colour."""

    vertices: Points  # the mesh's own frame
    triangles: Triangles
    placings: Placings  # into the world (z up, metres)
    colours: NDArray[np.float32]  # (n, 3) sRGB


@dataclass
class Scene:
    meshes: list[Mesh]
    materials: int = 0
    textures: int = 0

    @property
    def copies(self) -> int:
        return sum(len(m.placings) for m in self.meshes)

    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """The box every copy of every mesh fills, (low, high) corners in the world; None when
        there is nothing. Each copy's own box turned and moved, so a little larger than tight."""
        corners = []
        for mesh in self.meshes:
            if not len(mesh.vertices) or not len(mesh.placings):
                continue
            lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
            box = np.array(
                [
                    [x, y, z, 1.0]
                    for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1])
                    for z in (lo[2], hi[2])
                ]
            )
            placed = np.einsum("ki,nij->nkj", box, np.asarray(mesh.placings, np.float64))
            corners.append(placed[..., :3].reshape(-1, 3))
        if not corners:
            return None
        every = np.concatenate(corners)
        return every.min(axis=0), every.max(axis=0)


@dataclass
class Piece:
    """The meshes of one colour of a moving thing (or of things alike), joined in its frame."""

    vertices: Points  # the thing's own frame
    triangles: Triangles
    colour: RGB
    things: NDArray[np.int64]  # the things (indices into `Moving.paths`) made of it


@dataclass
class Placement:
    """Where a stage's moving things are at a time: each piece's things' placings and whether
    each is shown, and the lights they carry."""

    placings: list[Placings]
    shown: list[Mask]
    lights: NDArray[np.float64]  # (m, 3)
    lit: Mask


@dataclass
class Moving:
    """A stage's moving things: its outermost prims whose placing changes over time, each with
    what it carries. `at` places them at a time; a thing is shown only within its own keyed
    span unless `hold` (USD holds a value before its first key and after its last)."""

    stage: Usd.Stage  # kept: its prims are placed each time (a dropped stage's prims are empty)
    prims: list[Usd.Prim]
    pieces: list[Piece]
    spans: NDArray[np.float64]  # (n, 2) seconds: each thing's first and last key
    lights: NDArray[np.float64]  # (m, 4) homogeneous points in their thing's frame
    light_things: NDArray[np.int64]  # (m,)
    rate: float
    convert: NDArray[np.float64] = field(default_factory=lambda: np.eye(4))

    @property
    def paths(self) -> list[str]:
        return [str(p.GetPath()) for p in self.prims]

    @property
    def seconds(self) -> float:
        """When the last thing stops."""
        return float(self.spans[:, 1].max()) if len(self.spans) else 0.0

    def placings(self, seconds: float) -> Placings:
        """Each thing's placing at `seconds`."""
        cache = UsdGeom.XformCache(seconds * self.rate)
        world = np.array([np.array(cache.GetLocalToWorldTransform(p)) for p in self.prims])
        return world.reshape(-1, 4, 4) @ self.convert

    def at(self, seconds: float, hold: bool = False) -> Placement:
        world = self.placings(seconds)
        shown = np.ones(len(self.prims), bool)
        if not hold and len(self.spans):
            shown = (self.spans[:, 0] <= seconds) & (seconds <= self.spans[:, 1])
        lights = np.zeros((0, 3))
        if len(self.light_things):
            lights = np.einsum("ni,nij->nj", self.lights, world[self.light_things])[:, :3]
        return Placement(
            [world[p.things] for p in self.pieces],
            [shown[p.things] for p in self.pieces],
            lights,
            shown[self.light_things],
        )


@dataclass
class _Target:
    """A skinned mesh: how it is skinned, its rest points, its triangles."""

    query: UsdSkel.SkinningQuery
    rest: Vt.Vec3fArray
    triangles: Triangles


@dataclass
class _Skeleton:
    query: UsdSkel.SkeletonQuery
    prim: Usd.Prim
    targets: list[_Target]


@dataclass
class _Rigid:
    """A rigid prop: its meshes in its own frame."""

    prim: Usd.Prim
    meshes: list[MeshArrays]


@dataclass
class Skinned:
    """A file's skinned characters (UsdSkel) and its rigid props, posed at any time."""

    stage: Usd.Stage
    rate: float
    start: float  # seconds
    end: float
    skeletons: list[_Skeleton] = field(default_factory=list)
    rigid: list[_Rigid] = field(default_factory=list)
    convert: NDArray[np.float64] = field(default_factory=lambda: np.eye(4))

    def at(self, seconds: float) -> list[MeshArrays]:
        """Every skinned mesh in the world, then every rigid prop's meshes, at `seconds` (held
        at the file's first and last)."""
        code = min(max(seconds, self.start), self.end) * self.rate
        cache = UsdGeom.XformCache(code)
        out: list[MeshArrays] = []
        for skeleton in self.skeletons:
            xforms = skeleton.query.ComputeSkinningTransforms(code)
            world = np.array(cache.GetLocalToWorldTransform(skeleton.prim)) @ self.convert
            for target in skeleton.targets:
                points = Vt.Vec3fArray(target.rest)
                target.query.ComputeSkinnedPoints(xforms, points, code)
                out.append((_placed(np.asarray(points, np.float64), world), target.triangles))
        for prop in self.rigid:
            world = np.array(cache.GetLocalToWorldTransform(prop.prim)) @ self.convert
            out.extend((_placed(v, world), t) for v, t in prop.meshes)
        return out


def scene(
    stage: Usd.Stage,
    boxes: Iterable[Box] | None = None,
    *,
    margin: float = 1.0,
    moving: bool = False,
    textures: bool = True,
    without: Iterable[str] = (),
    log: Callable[[str], object] | None = None,
) -> Scene:
    """`_scene`, or what it read before from the stage's layers as they are (`kept`)."""
    boxes, without = list(boxes) if boxes is not None else None, list(without)
    settings = (repr(boxes), margin, moving, textures, tuple(without))
    return kept(
        "scene",
        stage,
        settings,
        lambda: _scene(
            stage, boxes, margin=margin, moving=moving, textures=textures, without=without, log=log
        ),
    )


def _scene(
    stage: Usd.Stage,
    boxes: Iterable[Box] | None = None,
    *,
    margin: float = 1.0,
    moving: bool = False,
    textures: bool = True,
    without: Iterable[str] = (),
    log: Callable[[str], object] | None = None,
) -> Scene:
    """The stage's still meshes: each mesh once with every copy's placing and colour; with
    `boxes` (z up, metres), only copies reaching within `margin` of one. Left out: what is
    invisible, guides and proxies, point instancers, (unless `moving`) what moves, which
    `moving()` reads, and the prims at `without`'s paths with all under them (a sky dome).
    `textures`: a textured material's colour is its texture's average (Pillow), else grey."""
    convert = _convert(stage)
    low, high = _bounds(boxes, margin)
    cache = UsdGeom.XformCache()
    meshes: dict[str, MeshArrays | None] = {}
    placings: dict[str, list[NDArray[np.float64]]] = {}
    looks_of: dict[str, list[Look]] = {}
    materials: dict[str, Look | None] = {}
    left_out = {Sdf.Path(p) for p in without}
    walk = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in walk:
        if (
            prim.GetPath() in left_out
            or not _shown(prim)
            or prim.IsA(UsdGeom.PointInstancer)
            or (not moving and _moves(prim))
        ):
            walk.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        key = _key(prim)
        if key not in meshes:
            meshes[key] = _mesh(prim)
        found = meshes[key]
        if found is None:
            continue
        placing = np.array(cache.GetLocalToWorldTransform(prim)) @ convert
        if not _reaches(found[0], placing, low, high):
            continue
        placings.setdefault(key, []).append(placing)
        looks_of.setdefault(key, []).append(_look(prim, materials))
    colour = _colours([look for looks in looks_of.values() for look in looks], textures)
    out = []
    for key, placed in placings.items():
        found = meshes[key]
        if found is not None:
            colours = np.array([colour(look) for look in looks_of[key]], np.float32)
            out.append(Mesh(found[0], found[1], np.array(placed), colours))
    looks = [look for look in materials.values() if look is not None]
    made = Scene(out, len(looks), len({look.texture for look in looks if look.texture}))
    if log:
        log(f"{len(out)} meshes, {made.copies} copies, {made.materials} materials")
    return made


#: The finest moving things are drawn, metres (`moving`).
DETAIL = 0.03


def moving(
    stage: Usd.Stage, *, colours: int = 1, textures: bool = True, detail: float | None = DETAIL
) -> Moving:
    """`_moving`, or what it read before from the stage's layers as they are (`kept`): kept
    without the stage, its things found again on it by their paths."""

    def bare() -> tuple:
        m = _moving(stage, colours=colours, textures=textures, detail=detail)
        return m.paths, m.pieces, m.spans, m.lights, m.light_things, m.rate, m.convert

    paths, *rest = kept("moving", stage, (colours, textures, detail), bare)
    return Moving(stage, [stage.GetPrimAtPath(p) for p in paths], *rest)


def _moving(
    stage: Usd.Stage, *, colours: int = 1, textures: bool = True, detail: float | None = DETAIL
) -> Moving:
    """The stage's moving things: its outermost prims whose placing may change over time,
    each one's meshes (and lights) read once in its own frame and joined into a piece a
    colour, its `colours` colours covering the most (the rest in the first: a car's paint);
    things made alike (a model's copies) share their pieces. Each piece is placed each frame:
    fewer colours, faster. `detail`: the finest each piece is drawn, metres (`simplified`; a
    city's cars come with 300,000 triangles each); None: as they are."""
    convert = _convert(stage)
    rate = stage.GetTimeCodesPerSecond()
    prims: list[Usd.Prim] = []
    walk = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in walk:
        if not _shown(prim):
            walk.PruneChildren()
        elif _moves(prim):
            prims.append(prim)
            walk.PruneChildren()
    spans = np.array([_span(p, rate) for p in prims], float).reshape(-1, 2)
    cache = UsdGeom.XformCache(0)
    meshes: dict[str, MeshArrays | None] = {}
    materials: dict[str, Look | None] = {}
    parts: list[list[tuple[str, NDArray[np.float64], Look]]] = []  # each thing's meshes, where
    lights: list[NDArray[np.float64]] = []
    light_things: list[int] = []
    for k, root in enumerate(prims):
        into = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(root)))
        mine = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not _shown(prim):
                continue
            if prim.IsA(UsdGeom.Mesh):
                key = _key(prim)
                if key not in meshes:
                    meshes[key] = _mesh(prim)
                if meshes[key] is not None:
                    relative = np.array(cache.GetLocalToWorldTransform(prim)) @ into
                    mine.append((key, relative, _look(prim, materials)))
            elif prim.GetTypeName().endswith("Light"):
                lights.append(np.array(cache.GetLocalToWorldTransform(prim))[3] @ into)
                light_things.append(k)
        parts.append(mine)
    colour = _colours([look for mine in parts for _, _, look in mine], textures)
    alike: dict[tuple, list[int]] = {}  # things made of the same meshes, placed the same
    for k, mine in enumerate(parts):
        if mine:
            sign = tuple((key, rel.round(3).tobytes(), look) for key, rel, look in mine)
            alike.setdefault(sign, []).append(k)
    pieces = []
    areas: dict[str, float] = {}
    for things in alike.values():
        by_colour: dict[RGB, list[MeshArrays]] = {}
        surface: dict[RGB, float] = {}
        for key, relative, look in parts[things[0]]:
            found = meshes[key]
            if found is not None:
                c = colour(look)
                by_colour.setdefault(c, []).append((_placed(found[0], relative), found[1]))
                if key not in areas:  # a mesh's surface once, however many times it is placed
                    areas[key] = _area([found])
                grown = abs(float(np.linalg.det(relative[:3, :3]))) ** (2 / 3)
                surface[c] = surface.get(c, 0.0) + areas[key] * grown
        ranked = sorted(by_colour.items(), key=lambda item: -surface[item[0]])
        kept = [(c, list(joined)) for c, joined in ranked[: max(colours, 1)]]
        for _, joined in ranked[max(colours, 1) :]:
            kept[0][1].extend(joined)
        for c, joined in kept:
            vertices, triangles = join(joined)
            if detail:
                vertices, triangles = simplified(vertices, triangles, detail)
            pieces.append(Piece(vertices, triangles, c, np.array(things, np.int64)))
    return Moving(
        stage,
        prims,
        pieces,
        spans,
        np.array(lights, float).reshape(-1, 4),
        np.array(light_things, np.int64),
        rate,
        convert,
    )


def skinned(path: str) -> Skinned:
    """The skinned characters and rigid props of the file at `path`, read into memory so the
    file can be replaced (a new bake) while this is shown. A rigid prop is any outermost prim
    whose placing changes over time, its meshes but those of skinned characters (a character
    is often carried by a moving prim above it: its body is skinned, not rigid)."""
    source = Sdf.Layer.FindOrOpen(str(path))
    if source is None:
        raise FileNotFoundError(f"No USD file at {path}.")
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.TransferContent(source)

    def anchored(asset: str) -> str:
        """An asset path as the file had it, now from wherever the file is."""
        return Sdf.ComputeAssetPathRelativeToLayer(source, asset) if asset else asset

    UsdUtils.ModifyAssetPaths(layer, anchored)
    rate = source.timeCodesPerSecond
    start, end = source.startTimeCode / rate, source.endTimeCode / rate
    stage = Usd.Stage.Open(layer)
    cache = UsdSkel.Cache()
    skeletons: list[_Skeleton] = []
    characters: list[Sdf.Path] = []
    for root in (p for p in stage.Traverse() if p.IsA(UsdSkel.Root)):
        cache.Populate(UsdSkel.Root(root), Usd.TraverseInstanceProxies())
        characters.append(root.GetPath())
        for prim in (p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton)):
            skeleton = UsdSkel.Skeleton(prim)
            binding = cache.ComputeSkelBinding(
                UsdSkel.Root(root), skeleton, Usd.TraverseInstanceProxies()
            )
            targets = []
            for target in binding.GetSkinningTargets():
                mesh = UsdGeom.Mesh(target.GetPrim())
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                targets.append(_Target(target, mesh.GetPointsAttr().Get(), fans(counts, indices)))
            skeletons.append(_Skeleton(cache.GetSkelQuery(skeleton), prim, targets))
    rigid: list[_Rigid] = []
    first = UsdGeom.XformCache(start * rate)
    walk = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in walk:
        if any(prim.GetPath().HasPrefix(c) for c in characters):
            walk.PruneChildren()
        elif _moves(prim):
            into = np.linalg.inv(np.array(first.GetLocalToWorldTransform(prim)))
            meshes: list[MeshArrays] = []
            parts = iter(Usd.PrimRange(prim))
            for part in parts:
                if any(part.GetPath().HasPrefix(c) for c in characters):
                    parts.PruneChildren()  # a character it carries: skinned, not rigid
                    continue
                if part.IsA(UsdGeom.Mesh) and _shown(part) and (found := _mesh(part)) is not None:
                    relative = np.array(first.GetLocalToWorldTransform(part)) @ into
                    meshes.append((_placed(found[0], relative), found[1]))
            if meshes:
                rigid.append(_Rigid(prim, meshes))
            walk.PruneChildren()
    return Skinned(stage, rate, start, end, skeletons, rigid, _convert(stage))


# Geometry.


def fans(counts: Sequence[int] | NDArray, indices: Sequence[int] | NDArray) -> Triangles:
    """USD's faces (any number of corners each) as triangles, each face a fan from its first
    corner."""
    counts = np.asarray(counts, np.int64)
    indices = np.asarray(indices, np.int64)
    if not len(counts):
        return np.zeros((0, 3), np.int32)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    many = np.maximum(counts - 2, 0)
    face = np.repeat(np.arange(len(counts)), many)
    k = np.arange(many.sum()) - np.repeat(np.cumsum(many) - many, many)
    first = starts[face]
    corners = [indices[first], indices[first + k + 1], indices[first + k + 2]]
    return np.stack(corners, 1).astype(np.int32)


def normals(vertices: NDArray, triangles: NDArray) -> Points:
    """Each vertex's normal: its faces', weighted by area."""
    v, t = np.asarray(vertices), np.asarray(triangles)
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    out = np.zeros((len(v), 3))
    flat = t.ravel()
    for axis in range(3):
        out[:, axis] = np.bincount(flat, weights=np.repeat(n[:, axis], 3), minlength=len(v))
    size = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return (out / size).astype(np.float32)


def _placed(points: NDArray, placing: NDArray[np.float64]) -> Points:
    """Points moved by a placing (row vectors)."""
    return (points @ placing[:3, :3] + placing[3, :3]).astype(np.float32)


def _area(meshes: list[MeshArrays], most: int = 4096) -> float:
    """How much surface meshes have, square metres: from at most `most` of each one's
    triangles, evenly spread (enough to rank colours by)."""
    total = 0.0
    for v, t in meshes:
        step = max(1, len(t) // most)
        some = t[::step]
        e1, e2 = v[some[:, 1]] - v[some[:, 0]], v[some[:, 2]] - v[some[:, 0]]
        total += (
            float(np.linalg.norm(np.cross(e1, e2), axis=1).sum()) / 2 * len(t) / max(len(some), 1)
        )
    return total


def simplified(vertices: Points, triangles: Triangles, cell: float) -> MeshArrays:
    """A mesh with its points snapped to a grid `cell` metres a side, those in one cell merged
    (the first of them kept) and the triangles that fold to a line or a point dropped: its
    shape kept to about `cell`, its detail finer than that gone."""
    if not len(vertices):
        return vertices, triangles
    grid = np.floor(np.asarray(vertices, float) / cell).astype(np.int64)
    grid -= grid.min(axis=0)
    span = grid.max(axis=0) + 1
    keys = grid[:, 0] + span[0] * (grid[:, 1] + span[1] * grid[:, 2])
    _, first, merged = np.unique(keys, return_index=True, return_inverse=True)
    t = merged.ravel()[triangles]
    kept = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    return vertices[first], t[kept].astype(np.int32)


def join(meshes: Sequence[MeshArrays]) -> MeshArrays:
    """Meshes as one: their points one after another, their triangles moved on to match."""
    offsets = np.cumsum([0] + [len(v) for v, _ in meshes[:-1]])
    points = np.concatenate([v for v, _ in meshes]).astype(np.float32)
    triangles = [t + o for (_, t), o in zip(meshes, offsets, strict=True)]
    return points, np.concatenate(triangles).astype(np.int32)


def drawn(scene: Scene) -> list[Mesh]:
    """A scene's meshes as drawn: those placed once joined into one mesh a colour (their points
    moved to where they are placed), the rest as they are. A draw costs the same whatever its
    size, and a city has a thousand meshes placed once."""
    many = [m for m in scene.meshes if len(m.placings) != 1]
    once: dict[tuple[float, ...], list[MeshArrays]] = {}
    for m in scene.meshes:
        if len(m.placings) == 1:
            key = tuple(np.round(np.asarray(m.colours[0], float), 4))
            once.setdefault(key, []).append((_placed(m.vertices, m.placings[0]), m.triangles))
    joined = []
    for colour, parts in once.items():
        points, triangles = join(parts)
        joined.append(Mesh(points, triangles, np.eye(4)[None], np.array([colour], np.float32)))
    return many + joined


def _mesh(prim: Usd.Prim) -> MeshArrays | None:
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or counts is None or indices is None or not len(points) or not len(counts):
        return None
    return np.asarray(points, np.float32), fans(np.asarray(counts), np.asarray(indices))


def _key(prim: Usd.Prim) -> str:
    """A prim's own path, or for an instance's, its prototype's: the same mesh read once."""
    return str((prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim).GetPath())


def _convert(stage: Usd.Stage) -> NDArray[np.float64]:
    """The stage's units and axes into metres, z up (a matrix to put after USD's own)."""
    m = np.diag([UsdGeom.GetStageMetersPerUnit(stage)] * 3 + [1.0])
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        m = m @ np.array([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    return m


def _bounds(
    boxes: Iterable[Box] | None, margin: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The low and high corners round `boxes`, `margin` out; everywhere without any."""
    found = list(boxes or ())
    if not found:
        return np.full(3, -np.inf), np.full(3, np.inf)
    low = np.min([b[0] for b in found], axis=0) - margin
    high = np.max([b[1] for b in found], axis=0) + margin
    return low, high


def _reaches(
    points: Points,
    placing: NDArray[np.float64],
    low: NDArray[np.float64],
    high: NDArray[np.float64],
) -> bool:
    """Whether a mesh placed by `placing` comes within the box from `low` to `high`."""
    if np.isinf(low).all():
        return True
    a, b = points.min(0), points.max(0)
    corners = np.array([[(a, b)[i][0], (a, b)[j][1], (a, b)[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])  # fmt: skip
    box = _placed(corners, placing)
    return not ((box.max(0) < low).any() or (box.min(0) > high).any())


def _shown(prim: Usd.Prim) -> bool:
    """Not invisible, and not a guide or proxy."""
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return True
    purpose = imageable.GetPurposeAttr().Get()
    if purpose in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
        return False
    return imageable.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible


def _moves(prim: Usd.Prim) -> bool:
    return prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).TransformMightBeTimeVarying()


def _span(prim: Usd.Prim, rate: float) -> tuple[float, float]:
    """A moving prim's first and last key, in seconds (its transform ops')."""
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    times = [t for op in ops for t in op.GetTimeSamples()]
    return (min(times) / rate, max(times) / rate) if times else (-np.inf, np.inf)


# Colours.


def _look(prim: Usd.Prim, materials: dict[str, Look | None]) -> Look:
    """What a mesh shows most: its material's look, else its displayColor, else grey."""
    material = _bound(prim)
    if material:
        found = material.GetPrim()
        key = _key(found)
        if key not in materials:
            materials[key] = _material_look(material)
        look = materials[key]
        if look is not None:
            return look
    shown = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    colours = shown.Get() if shown else None
    if colours is not None and len(colours):
        return Look(colour=srgb(colours[0]))
    return Look(colour=GREY)


def _bound(prim: Usd.Prim) -> UsdShade.Material | None:
    """The material a mesh shows most: its own binding, else its largest subset's."""
    api = UsdShade.MaterialBindingAPI(prim)
    material, _ = api.ComputeBoundMaterial()
    if material:
        return material
    best, most = None, 0
    for subset in api.GetMaterialBindSubsets():
        faces = len(subset.GetIndicesAttr().Get() or [])
        found, _ = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
        if found and faces > most:
            best, most = found, faces
    return best


def _material_look(material: UsdShade.Material) -> Look | None:
    """A material's look from its surface shader: a preview surface's, else an MDL one's."""
    for context in ("", "mdl"):
        surface = material.ComputeSurfaceSource(context)[0]
        if not surface:
            continue
        preview = surface.GetIdAttr().Get() == "UsdPreviewSurface"
        colours, textures = (("diffuseColor",), ()) if preview else (MDL_COLOURS, MDL_TEXTURES)
        for name in textures:
            given = surface.GetInput(name)
            value = given.Get() if given else None
            if value is not None and getattr(value, "resolvedPath", ""):
                return Look(texture=value.resolvedPath)
        for name in colours:
            given = surface.GetInput(name)
            if not given:
                continue
            if given.HasConnectedSource():  # a texture's file feeds it
                source = given.GetConnectedSources()[0][0].source
                file = UsdShade.Shader(source.GetPrim()).GetInput("file")
                value = file.Get() if file else None
                if value is not None and (value.resolvedPath or value.path):
                    return Look(texture=value.resolvedPath or value.path)
                continue
            value = given.Get()
            if value is not None:
                return Look(colour=srgb(value))
    return None


def _colours(looks: Sequence[Look], textures: bool) -> Callable[[Look], RGB]:
    """A function from a look to its colour, the textures among `looks` averaged at once (those
    averaged before, unchanged since, as `COLOURS` keeps them)."""
    files = sorted({look.texture for look in looks if look.texture}) if textures else []
    kept = _kept_colours()
    stamps = {f: _stamp(f) for f in files}
    missing = [f for f in files if stamps[f] not in kept]
    with ThreadPoolExecutor(16) as pool:
        for f, c in zip(missing, pool.map(texture_colour, missing), strict=True):
            kept[stamps[f]] = c
    if missing:
        _keep_colours(kept)
    averaged = {f: kept[stamps[f]] for f in files}

    def colour(look: Look) -> RGB:
        if look.colour is not None:
            return look.colour
        return averaged.get(look.texture or "") or GREY

    return colour


#: Where textures' average colours are kept between runs (in the temporary folder, by
#: `kept`'s): a texture opened once.
COLOURS = KEPT.parent / "colours.json"


def _stamp(path: str) -> str:
    """A texture as it is now: its path, size and when it last changed."""
    try:
        s = os.stat(path)
    except OSError:
        return path
    return f"{path}|{s.st_size}|{s.st_mtime_ns}"


def _kept_colours() -> dict[str, RGB | None]:
    try:
        return {
            k: tuple(v) if v else None
            for k, v in json.loads(COLOURS.read_text(encoding="utf-8")).items()
        }
    except (OSError, ValueError):
        return {}


def _keep_colours(kept: dict[str, RGB | None]) -> None:
    try:
        COLOURS.parent.mkdir(parents=True, exist_ok=True)
        part = COLOURS.with_suffix(".part")
        part.write_text(json.dumps(kept), encoding="utf-8")
        part.replace(COLOURS)  # whole, or not at all
    except OSError:
        pass  # kept next time, then


def texture_colour(path: str) -> RGB | None:
    """An image's average colour (sRGB, 0 to 1), from a small copy of it; None if unreadable."""
    try:
        from PIL import Image

        image = Image.open(path)
        image.draft("RGB", (64, 64))
        small = np.asarray(image.convert("RGB").resize((8, 8)), float)
    except Exception:
        return None
    r, g, b = small.reshape(-1, 3).mean(0) / 255
    return float(r), float(g), float(b)


def srgb(linear: Sequence[float]) -> RGB:
    """A linear colour as the screen's (sRGB)."""
    c = np.clip(np.asarray(linear, float)[:3], 0, 1)
    r, g, b = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return float(r), float(g), float(b)
