"""A skinned character's poses: one as a mesh to place, mid-stride, standing on the origin,
facing +Y (`stride`: to walk it along a route while its real animation is not there yet); or
posed from arrays a pose at a time (`Posed`: a character steered live), shown as a `Skinned`."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .read import Skinned


def stride(skinned: Skinned) -> tuple[NDArray[np.float32], NDArray[np.int32]] | None:
    """Vertices and triangles of `skinned` mid-stride (the first of nine times along its
    animation where it moves on more than 0.2 m in 0.3 s), turned so its stride is along +Y,
    its feet on the origin; None when it has no animation or no mesh."""
    start, end = skinned.start, skinned.end
    if end <= start:
        return None
    step = np.zeros(2)
    now: list = []
    for f in np.linspace(0.1, 0.9, 9):
        t = start + f * (end - start)
        now, later = skinned.at(t), skinned.at(t + 0.3)
        if not now:
            return None
        here = np.concatenate([m for m, _ in now])[:, :2].mean(0)
        step = np.concatenate([m for m, _ in later])[:, :2].mean(0) - here
        if np.linalg.norm(step) > 0.2:
            break
    points = np.concatenate([m for m, _ in now]).astype(np.float64)
    offsets = np.cumsum([0] + [len(m) for m, _ in now[:-1]])
    triangles = np.concatenate([t + o for (_, t), o in zip(now, offsets, strict=True)])
    centre = np.r_[points[:, :2].mean(0), points[:, 2].min()]
    a = math.atan2(step[0], step[1])  # the stride turned onto +Y
    c, s = math.cos(a), math.sin(a)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return ((points - centre) @ turn.T).astype(np.float32), triangles.astype(np.int32)


def _matrices(rotations: NDArray, translations: NDArray) -> NDArray[np.float64]:
    """Local transforms (J, 4, 4; USD's row vectors) from rotations (J, 4: w first) and
    translations (J, 3)."""
    q = np.asarray(rotations, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q.T
    m = np.zeros((len(q), 4, 4))
    m[:, 0, :3] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)
    m[:, 1, :3] = np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], 1)
    m[:, 2, :3] = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    m[:, 3, :3] = translations
    m[:, 3, 3] = 1.0
    return m


class Posed:
    """A skinned file's first character posed from arrays a pose at a time (`pose`), as a
    character steered live is: shown by a viewport as a `Skinned` is (`at`, `start`, `end`),
    drawn again whenever it is posed again (`poses` counts them). `joints` names the joints
    of the poses it will be given, in their order (UsdSkel's paths: those of the file's
    skeleton, in any order; the file's own if not given)."""

    def __init__(self, skinned: Skinned, joints: list[str] | None = None) -> None:
        from pxr import UsdSkel, Vt

        self._UsdSkel, self._Vt = UsdSkel, Vt
        if not skinned.skeletons:
            raise ValueError("The file has no skinned character to pose.")
        self.skinned = skinned
        self.skeleton = skinned.skeletons[0]
        query = self.skeleton.query
        self.topology = query.GetTopology()
        order = [str(j) for j in query.GetJointOrder()]
        given = order if joints is None else [str(j) for j in joints]
        missing = [j for j in order if j not in given]
        if missing:
            raise ValueError(f"The poses have no {missing[0]} (and {len(missing) - 1} more).")
        #: For each of the file's joints, its place in the poses given.
        self.index = np.array([given.index(j) for j in order])
        binds = np.array(query.GetJointWorldBindTransforms(), float)
        self.unbind = np.linalg.inv(binds)
        self.rate, self.start, self.end = skinned.rate, 0.0, 0.0
        self.meshes: list[tuple[NDArray[np.float32], NDArray[np.int32]]] = []
        self.poses = 0

    def pose(self, rotations: NDArray, translations: NDArray, placed: NDArray) -> None:
        """Posed with its joints' local `rotations` (J, 4, w first) and `translations` (J, 3),
        in the order it was given, its skeleton's space placed in the world by `placed` (4, 4;
        the scene's: metres, Z up, the file's own units in it)."""
        local = _matrices(
            np.asarray(rotations)[self.index], np.asarray(translations, float)[self.index]
        )
        Vt = self._Vt
        skel = np.array(
            self._UsdSkel.ConcatJointTransforms(self.topology, Vt.Matrix4dArray.FromNumpy(local)),
            float,
        )
        xforms = Vt.Matrix4dArray.FromNumpy(np.einsum("jab,jbc->jac", self.unbind, skel))
        world = np.asarray(placed, float)  # the scene's own: metres, Z up, as given
        meshes = []
        for target in self.skeleton.targets:
            points = Vt.Vec3fArray(target.rest)
            target.query.ComputeSkinnedPoints(xforms, points)
            p = np.asarray(points, np.float64)
            meshes.append(((p @ world[:3, :3] + world[3, :3]).astype(np.float32), target.triangles))
        self.meshes = meshes
        self.poses += 1

    def at(self, seconds: float) -> list[tuple[NDArray[np.float32], NDArray[np.int32]]]:
        """Its meshes as last posed, whatever the time."""
        return self.meshes
