"""`usdscope.read` on small stages made here."""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel, Vt

from usdscope import read

CUBE_POINTS = [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]
CUBE_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def _cube(stage, path: str, colour=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in CUBE_POINTS])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([i for f in CUBE_FACES for i in f])
    if colour is not None:
        mesh.CreateDisplayColorPrimvar().Set([Gf.Vec3f(*colour)])
    return mesh


def _preview(stage, path: str, colour=None, texture: str | None = None):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if texture:
        image = UsdShade.Shader.Define(stage, f"{path}/Image")
        image.CreateIdAttr("UsdUVTexture")
        image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        diffuse.ConnectToSource(image.ConnectableAPI(), "rgb")
    elif colour is not None:
        diffuse.Set(Gf.Vec3f(*colour))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _stage():
    """A stage in memory in metres, z up (USD's own default is centimetres)."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    return stage


def test_faces_are_fanned_into_triangles():
    t = read.fans(np.array([4, 3]), np.array([0, 1, 2, 3, 4, 5, 6]))
    assert t.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6]]


def test_an_instanced_mesh_is_read_once_in_metres_z_up_with_its_materials_colour():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    proto = UsdGeom.Xform.Define(stage, "/Proto")
    cube = _cube(stage, "/Proto/Cube")
    material = _preview(stage, "/Proto/Looks/Grey", colour=(0.5, 0.5, 0.5))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    proto.GetPrim().SetSpecifier(Sdf.SpecifierClass)
    for k, x in enumerate((0.0, 300.0)):
        copy = UsdGeom.Xform.Define(stage, f"/Copy{k}")
        copy.AddTranslateOp().Set(Gf.Vec3d(x, 200, 0))
        copy.GetPrim().GetInherits().AddInherit("/Proto")
        copy.GetPrim().SetInstanceable(True)
    found = read.scene(stage)
    assert len(found.meshes) == 1 and found.copies == 2
    mesh = found.meshes[0]
    # 300 cm along x is 3 m; 200 cm up y is 2 m up z.
    assert np.allclose(
        sorted(mesh.placings[:, 3, :3].tolist()), [[0, -0, 2], [3, -0, 2]], atol=1e-9
    )
    world = mesh.vertices @ mesh.placings[0][:3, :3] + mesh.placings[0][3, :3]
    assert np.allclose(np.ptp(world, 0), [0.01, 0.01, 0.01])
    assert np.allclose(mesh.colours, read.srgb([0.5] * 3), atol=1e-6)


def test_what_is_hidden_or_a_guide_is_left_out_and_so_is_what_moves_unless_asked():
    stage = _stage()
    _cube(stage, "/Shown")
    UsdGeom.Imageable(_cube(stage, "/Hidden")).MakeInvisible()
    UsdGeom.Imageable(_cube(stage, "/Guide")).CreatePurposeAttr(UsdGeom.Tokens.guide)
    car = UsdGeom.Xform.Define(stage, "/Car")
    op = car.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(10, 0, 0), 24)
    _cube(stage, "/Car/Body")
    assert read.scene(stage).copies == 1
    assert read.scene(stage, moving=True).copies == 2


def test_prims_said_are_left_out_with_all_under_them():
    stage = _stage()
    _cube(stage, "/Shown")
    UsdGeom.Xform.Define(stage, "/Sky")
    _cube(stage, "/Sky/Dome")
    assert read.scene(stage).copies == 2
    assert read.scene(stage, without=["/Sky"]).copies == 1


def test_boxes_keep_only_the_copies_that_reach_them():
    stage = _stage()
    for k, x in enumerate((0.0, 50.0)):
        xf = UsdGeom.Xform.Define(stage, f"/At{k}")
        xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
        _cube(stage, f"/At{k}/Cube")
    found = read.scene(stage, [((-2.0, -2, -2), (2.0, 2, 2))])
    assert found.copies == 1


def test_a_texture_gives_its_average_colour(tmp_path):
    from PIL import Image

    png = tmp_path / "red.png"
    Image.new("RGB", (16, 16), (200, 40, 40)).save(png)
    stage = _stage()
    cube = _cube(stage, "/Cube")
    material = _preview(stage, "/Looks/Red", texture=str(png))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    colour = read.scene(stage).meshes[0].colours[0]
    assert np.allclose(colour, np.array([200, 40, 40]) / 255, atol=1e-3)


def test_moving_things_are_placed_at_a_time_and_shown_only_within_their_keys():
    stage = _stage()
    stage.SetTimeCodesPerSecond(24)
    model = UsdGeom.Xform.Define(stage, "/Model")  # a car model, two cars made of it
    _cube(stage, "/Model/Body", colour=(1, 0, 0))
    _cube(stage, "/Model/Glass", colour=(0, 0, 1))
    UsdLux.SphereLight.Define(stage, "/Model/Lamp").AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    model.GetPrim().SetSpecifier(Sdf.SpecifierClass)
    for k in range(2):
        car = UsdGeom.Xform.Define(stage, f"/Car{k}")
        op = car.AddTranslateOp()
        op.Set(Gf.Vec3d(0, 5 * k, 0), 24)
        op.Set(Gf.Vec3d(10, 5 * k, 0), 48)
        car.GetPrim().GetInherits().AddInherit("/Model")
        car.GetPrim().SetInstanceable(True)
    found = read.moving(stage)
    assert found.paths == ["/Car0", "/Car1"]
    assert np.allclose(found.spans, [[1, 2], [1, 2]]) and found.seconds == 2
    # Alike cars share their pieces: one a colour, each made of both cars.
    assert len(found.pieces) == 1 and len(found.pieces[0].things) == 2  # one colour by default
    split = read.moving(stage, colours=2)
    assert len(split.pieces) == 2 and all(len(p.things) == 2 for p in split.pieces)
    assert {tuple(np.round(p.colour, 3)) for p in split.pieces} == {(1, 0, 0), (0, 0, 1)}
    frame = found.at(1.5)
    assert np.allclose(frame.placings[0][:, 3, :3], [[5, 0, 0], [5, 5, 0]]) and frame.shown[0].all()
    assert np.allclose(frame.lights, [[6, 0, 0], [6, 5, 0]]) and frame.lit.all()
    assert not found.at(3.0).shown[0].any()
    held = found.at(3.0, hold=True)
    assert held.shown[0].all() and np.allclose(held.placings[0][:, 3, 0], 10)


def _skinned_arm(path: str) -> None:
    """Two joints along x, a two-quad strip bound to them, the elbow turned 90 degrees at t=1 s."""
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Carried by a moving prim above it, as a bake keys a walker: its body is still skinned.
    carrier = UsdGeom.Xform.Define(stage, "/Carrier")
    op = carrier.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(0, 0, 0), 24)
    stage.GetRootLayer().defaultPrim = "Carrier"
    root = UsdSkel.Root.Define(stage, "/Carrier/Arm")
    skeleton = UsdSkel.Skeleton.Define(stage, "/Carrier/Arm/Skel")
    joints = ["shoulder", "shoulder/elbow"]
    skeleton.CreateJointsAttr(joints)
    rest = [Gf.Matrix4d(1), Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(1, 0, 0))]
    skeleton.CreateBindTransformsAttr([rest[0], rest[1]])
    skeleton.CreateRestTransformsAttr(rest)
    animation = UsdSkel.Animation.Define(stage, "/Carrier/Arm/Skel/Anim")
    animation.CreateJointsAttr(joints)
    animation.CreateTranslationsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0)])
    animation.CreateScalesAttr([Gf.Vec3h(1, 1, 1)] * 2)
    turn = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90).GetQuat())
    rotations = animation.CreateRotationsAttr()
    rotations.Set([Gf.Quatf(1), Gf.Quatf(1)], 0)
    rotations.Set([Gf.Quatf(1), turn], 24)
    UsdSkel.BindingAPI.Apply(skeleton.GetPrim()).CreateAnimationSourceRel().SetTargets(
        [animation.GetPath()]
    )
    mesh = UsdGeom.Mesh.Define(stage, "/Carrier/Arm/Strip")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4, 4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 4, 3, 1, 2, 5, 4])
    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateJointIndicesPrimvar(False, 1).Set(Vt.IntArray([0, 1, 1, 0, 1, 1]))
    binding.CreateJointWeightsPrimvar(False, 1).Set(Vt.FloatArray([1] * 6))
    binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1))
    UsdSkel.BindingAPI.Apply(root.GetPrim())
    stage.GetRootLayer().Save()


def test_a_skinned_character_is_posed_at_a_time_and_its_file_can_be_baked_again(tmp_path):
    path = str(tmp_path / "arm.usda")
    _skinned_arm(path)
    arm = read.skinned(path)
    assert (arm.start, arm.end) == (0, 1)
    ((points, triangles),) = arm.at(0.0)
    assert len(triangles) == 4 and np.allclose(points[2], (2, 0, 0), atol=1e-5)
    ((points, _),) = arm.at(1.0)  # the elbow turned: the hand points up y
    assert np.allclose(points[2], (1, 1, 0), atol=1e-5)
    # A new bake replaces the file while this one is shown.
    assert Sdf.Layer.Find(path) is None, "the file is still open"
    _skinned_arm(path)
    ((points, _),) = arm.at(1.0)
    assert np.allclose(points[2], (1, 1, 0), atol=1e-5)


def test_a_scene_says_the_box_its_copies_fill():
    cube = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    moved = np.eye(4)[None].repeat(2, 0)
    moved[1, 3, :3] = (10, 20, 5)
    mesh = read.Mesh(cube, np.array([[0, 1, 2]], np.int32), moved, np.ones((2, 3), np.float32))
    found = read.Scene([mesh]).bounds()
    assert found is not None
    low, high = found
    assert np.allclose(low, (0, 0, 0)) and np.allclose(high, (11, 21, 6))
    assert read.Scene([]).bounds() is None


def test_triangles_in_chunks_find_what_every_triangle_tested_finds():
    from usdscope.pick import Triangles, hit

    rng = np.random.default_rng(3)
    centres = rng.uniform(0, 100, (3000, 3)) * [1, 1, 0.05]
    tri = centres[:, None, :] + rng.uniform(-1, 1, (3000, 3, 3))
    found = Triangles(tri)
    for _ in range(200):
        origin = rng.uniform(0, 100, 3) + np.array([0.0, 0.0, 30.0])
        direction = rng.normal(size=3) - [0, 0, 2]
        assert found.hit(origin, direction) == hit(tri, origin, direction)
    k = int(np.argmin(np.linalg.norm(centres[:, :2] - 50, axis=1)))
    down = (*tri[k].mean(axis=0)[:2], 30.0)  # straight down onto a triangle
    assert (
        found.hit(down, (0, 0, -1)) == hit(tri, down, (0, 0, -1))
        and found.hit(down, (0, 0, -1))[0] >= 0
    )
    assert found.hit(down, (0, 0, 1)) == (-1, float("inf"))


def test_meshes_placed_once_are_drawn_joined_a_colour_the_rest_as_they_are():
    tri = np.array([[0, 1, 2]], np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    moved = np.eye(4)
    moved[3, :3] = (10, 0, 0)
    red, blue = np.array([[1, 0, 0]], np.float32), np.array([[0, 0, 1]], np.float32)
    scene = read.Scene(
        [
            read.Mesh(points, tri, np.eye(4)[None], red),
            read.Mesh(points, tri, moved[None], red),
            read.Mesh(points, tri, moved[None], blue),
            read.Mesh(points, tri, np.stack([np.eye(4), moved]), np.vstack([red, red])),
        ]
    )
    drawn = read.drawn(scene)
    assert len(drawn) == 3  # the twice-placed one, the reds joined, the blue
    assert len(drawn[0].placings) == 2
    reds = next(m for m in drawn[1:] if m.colours[0][0] == 1)
    assert len(reds.vertices) == 6 and reds.triangles.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert reds.vertices[3:, 0].tolist() == [10.0, 11.0, 10.0]  # where it was placed


def test_a_read_is_kept_until_a_layer_of_its_stage_changes(tmp_path):
    from usdscope import kept

    file = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(file))
    UsdGeom.Mesh.Define(stage, "/Ground")
    stage.Save()
    reads = []

    def read():
        reads.append(1)
        return len(reads)

    assert kept.kept("count", stage, (), read) == 1
    assert kept.kept("count", stage, (), read) == 1  # kept: not read again
    assert kept.kept("count", stage, ("other",), read) == 2  # read otherwise: afresh
    UsdGeom.Mesh.Define(stage, "/Wall")
    stage.Save()
    assert kept.kept("count", stage, (), read) == 3  # a layer changed: afresh


def test_reads_are_kept_in_a_folder_named_for_the_user_on_any_machine(monkeypatch):
    from usdscope import kept

    monkeypatch.setattr(kept.getpass, "getuser", lambda: "OFFICE\\Ann Lee")  # as Windows says
    assert kept._user() == "OFFICE_Ann_Lee"

    def nobody():
        raise OSError("No username set in the environment")

    monkeypatch.setattr(kept.getpass, "getuser", nobody)
    assert kept._user() == "user"


def test_a_character_posed_from_arrays_is_skinned_as_its_file_poses_it(tmp_path):
    from usdscope.poses import Posed

    path = str(tmp_path / "arm.usda")
    _skinned_arm(path)
    arm = read.skinned(path)
    # Given in the other order: the elbow first.
    posed = Posed(arm, ["shoulder/elbow", "shoulder"])
    turn = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # 90 degrees about z
    placed = np.eye(4)
    placed[3, :3] = (0, 0, 5)  # the skeleton five metres up
    posed.pose(np.array([turn, [1.0, 0, 0, 0]]), np.array([[1.0, 0, 0], [0, 0, 0]]), placed)
    ((points, triangles),) = posed.at(123.0)
    assert len(triangles) == 4 and posed.poses == 1
    assert np.allclose(points[2], (1, 1, 5), atol=1e-5)  # as the file has it at 1 s, raised
    assert np.allclose(points[0], (0, 0, 5), atol=1e-5)
