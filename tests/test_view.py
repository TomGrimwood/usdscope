"""`usdscope.view`: the maths without a window, and a viewport drawn headless (a GPU)."""

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from usdscope import pick, read

view = pytest.importorskip("usdscope.view")


def test_placings_become_newtons_transforms_and_back():
    turn = view.matrix(
        [1, 2, 3], [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)]
    )  # 45 degrees about z
    xforms, scales = view.instances(turn[None])
    assert np.allclose(scales, 1) and np.allclose(xforms[0, :3], [1, 2, 3])
    assert np.allclose(view.matrix(xforms[0, :3], xforms[0, 3:]), turn, atol=1e-6)
    # Row vectors: x turned 45 degrees about z points between x and y.
    assert np.allclose(np.array([1, 0, 0]) @ turn[:3, :3], [np.sqrt(0.5), np.sqrt(0.5), 0])


def test_a_ray_finds_the_nearest_triangle():
    ground = np.array(
        [[[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[-10, -10, 2], [10, -10, 2], [0, 10, 2]]],
        float,
    )
    k, far = pick.hit(ground, [0, 0, 5], [0, 0, -1])
    assert (k, far) == (1, 3.0)
    assert pick.hit(ground, [0, 0, 5], [0, 0, 1])[0] == -1


@pytest.fixture(scope="module")
def viewport():
    try:
        v = view.Viewport(640, 360, headless=True)
    except Exception as e:  # no GPU or no OpenGL here
        pytest.skip(f"no headless viewer: {e}")
    yield v
    v.close()


def test_a_scene_is_drawn_and_screen_positions_and_rays_agree(viewport, tmp_path):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    cube = UsdGeom.Cube.Define(stage, "/Box")  # not a mesh: left out
    mesh = UsdGeom.Mesh.Define(stage, "/Ground")
    mesh.CreatePointsAttr([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDisplayColorPrimvar().Set([Gf.Vec3f(0.2, 0.6, 0.2)])
    assert cube
    viewport.show_scene(read.scene(stage))
    viewport.camera_at([-8, -8, 6])
    viewport.frame(0.0)
    origin, direction = viewport.ray(200.0, 150.0)
    at, ahead = viewport.screen(origin + 10 * direction)
    assert ahead[0] and np.allclose(at[0], [200, 150], atol=0.5)
    viewport.picture(tmp_path / "ground.jpg", ui=False)
    assert (tmp_path / "ground.jpg").stat().st_size > 1000


def test_a_picker_finds_the_nearest_copy_a_ray_meets():
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    low, high = np.eye(4), np.eye(4)
    high[3, :3] = (0.0, 0.0, 3.0)  # the same square 3 m up
    big = np.diag([4.0, 4.0, 4.0, 1.0])  # and one four times the size, on the ground
    big[3, :3] = (10.0, 10.0, 0.0)
    picker = pick.Picker([(square, triangles, np.array([low, high, big]))])
    found = picker.hit((0.5, 0.5, 10.0), (0, 0, -1))
    assert found is not None and np.allclose(found.at, (0.5, 0.5, 3.0)) and found.distance == 7.0
    assert (found.mesh, found.copy) == (0, 1)  # the copy 3 m up
    found = picker.hit((13.0, 13.0, 10.0), (0, 0, -1))
    assert found is not None and np.allclose(found.at, (13, 13, 0)) and found.copy == 2
    assert picker.hit((5.0, 5.0, 10.0), (0, 0, -1)) is None


def test_markers_lie_on_the_ground_and_go(viewport):
    at = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    viewport.markers("/spots", at, np.array([[0.3, 1.0, 0.4], [1.0, 0.3, 0.3]]), 0.4, layer="marks")
    viewport.frame(0.0)
    viewport.markers("/spots", None, layer="marks")  # gone
    viewport.frame(0.0)
    viewport.clear("marks")
    viewport.markers("/spots", at[:1], layer="marks")  # drawn again in a cleared layer
    viewport.frame(0.0)


def test_a_ribbon_is_laid_along_a_path_in_its_colours(viewport):
    line = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    viewport.ribbon("/path", line, np.array([[0.2, 0.2, 0.8], [0.9, 0.9, 0.2]]), 0.3, layer="paths")
    viewport.frame(0.0)
    viewport.ribbon("/path", None, layer="paths")
    viewport.frame(0.0)


def test_a_placed_meshs_footprint_is_a_box_along_its_own_ways():
    long_box = np.array([[x, y, z] for x in (0, 4) for y in (0, 1) for z in (0, 1)], float)
    turn = view.matrix([10, 5, 0], [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)])  # 45 degrees
    (cx, cy), length, width, bearing = pick.footprint(long_box, turn)
    assert np.isclose(length, 4) and np.isclose(width, 1)
    assert np.isclose(bearing % 90, 45) and np.isclose(np.hypot(cx - 10, cy - 5), np.hypot(2, 0.5))


def test_menus_are_drawn_from_paths_and_chosen_by_them():
    from usdscope import menus

    ran = []
    entries = [
        menus.Entry("Stop/tilt", lambda: ran.append("tilt")),
        menus.separator("Stop"),
        menus.Entry("Remove", lambda: ran.append("remove"), key="Delete"),
    ]
    menus.find(entries, "Stop/tilt").run()
    assert ran == ["tilt"]
    with pytest.raises(ValueError, match="there are Stop/tilt, Remove"):
        menus.find(entries, "Stop/-")


def test_a_frame_runs_along_its_axis_and_a_ramp_runs_between_its_stops():
    from usdscope import shapes

    placing = shapes.along((1.0, 2.0, 3.0), (0.0, 0.0, 2.0))
    assert np.allclose(placing[2, :3], (0, 0, 1)) and np.allclose(placing[3, :3], (1, 2, 3))
    assert np.allclose(np.cross(placing[0, :3], placing[1, :3]), placing[2, :3])  # right-handed
    ends = shapes.ramp([0.0, 1.0])
    assert np.allclose(ends, shapes.VIRIDIS[[0, -1]])


def test_the_panels_stay_when_a_layer_is_cleared(viewport):
    drawn = []
    viewport.panel(lambda imgui: drawn.append(1))
    viewport.markers("/marks", np.zeros((1, 3)), (1.0, 1.0, 1.0), 0.5, layer="walk")
    viewport.frame(0.0)
    viewport.clear("walk")  # Newton drops its free UI callbacks with the layer's model
    before = len(drawn)
    viewport.frame(0.0)
    viewport.image(ui=True)
    assert len(drawn) > before


def test_a_box_is_framed_all_of_it_inside_what_the_panels_leave(viewport):
    width, height = viewport.size
    left, top, right, bottom = 0.25 * width, 0.05 * height, 0.25 * width, 0.12 * height
    corners = np.array([[x, y, z] for x in (70, 150) for y in (40, 120) for z in (6, 30)], float)
    viewport.frame_box(corners.min(axis=0), corners.max(axis=0), inset=(left, top, right, bottom))
    at, ahead = viewport.screen(corners)
    assert ahead.all() and (at[:, 0] >= left).all() and (at[:, 0] <= width - right).all()
    assert (at[:, 1] >= top).all() and (at[:, 1] <= height - bottom).all()
    assert (at[:, 0].max() - at[:, 0].min()) > 0.3 * width  # near: filling it, not far off


def test_a_box_framed_from_far_off_is_drawn_to_its_far_side(viewport):
    viewport.frame_box((-300.0, -300.0, 0.0), (300.0, 300.0, 20.0))
    position = np.array(viewport.viewer.camera.pos, float)
    farthest = np.linalg.norm(position - np.array([300.0, 300.0, 0.0]))
    assert viewport.viewer.camera.far >= farthest


def test_a_framed_box_is_orbited_round_its_middle(viewport):
    viewport.frame_box((-300.0, -300.0, 0.0), (300.0, 300.0, 20.0))
    assert np.allclose(np.array(viewport.viewer.camera.pivot, float), [0, 0, 0], atol=1e-3)


def test_the_keys_and_the_wheel_move_at_the_view_speed(viewport):
    from usdscope import input

    gui = viewport.viewer.gui
    if not viewport._hooked:
        pytest.skip("no panels headless here")
    viewport.view_speed = 7.0
    gui.update_camera_from_keys(0.1, lambda k: False)
    assert viewport.viewer.camera_speed == pytest.approx(7.0)
    before = np.array(viewport.viewer.camera.pos, float)
    for _ in range(100):  # the wheel never stops short
        gui.handle_mouse_scroll(1.0)
    moved = np.linalg.norm(np.array(viewport.viewer.camera.pos, float) - before)
    assert moved == pytest.approx(100 * 7.0 * input.WHEEL, rel=1e-3)
    viewport.view_speed = input.SPEED


def test_newtons_view_options_draw_in_an_apps_panel(viewport):
    if not viewport._hooked:
        pytest.skip("no panels headless here")
    viewport.panel(viewport.view_options)
    viewport.frame_rate = True
    viewport._reported = ""
    viewport.frame(0.0)
    assert viewport._reported == ""  # drawn without an error
    viewport.frame_rate = False
    viewport._panels.remove(viewport.view_options)


def test_centring_on_a_point_moves_the_camera_to_it_facing_as_it_did(viewport):
    viewport.camera_at([-500.0, 0.0, 300.0], pitch=-30.0, yaw=90.0)
    yaw = viewport.viewer.camera.yaw
    viewport.centre_on((40.0, 20.0, 0.0))
    camera = viewport.viewer.camera
    assert np.linalg.norm(np.array(camera.pos, float) - [40, 20, 0]) == pytest.approx(15.0)
    assert camera.yaw == pytest.approx(yaw)
    assert np.allclose(np.array(camera.pivot, float), [40, 20, 0], atol=1e-3)
    width, height = viewport.size
    at, ahead = viewport.screen(np.array([[40.0, 20.0, 0.0]]))
    assert ahead[0] and np.allclose(at[0], [width / 2, height / 2], atol=2.0)


def test_the_same_overlay_again_is_not_sent_again(viewport):
    sent = []
    log = viewport.viewer.log_instances
    viewport.viewer.log_instances = lambda *a, **k: (sent.append(a[0]), log(*a, **k))[1]
    at = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    viewport.markers("/m", at, layer="walk")
    first = len(sent)
    viewport.markers("/m", at.copy(), layer="walk")
    assert len(sent) == first  # nothing new: nothing sent
    viewport.markers("/m", at + 1.0, layer="walk")
    assert len(sent) == 2 * first
    viewport.clear("walk")
    viewport.markers("/m", at + 1.0, layer="walk")  # cleared: drawn again
    assert len(sent) == 3 * first


def test_a_local_gizmo_is_drawn_along_its_own_axes():
    from types import SimpleNamespace

    from usdscope.view import _LocalAxes

    drawn = []
    giz = SimpleNamespace(
        MODE=SimpleNamespace(world="world", local="local"),
        push_id=lambda i: None,
        manipulate=lambda view, proj, op, mode, *rest: drawn.append(mode),
    )
    viewer = SimpleNamespace(gui=SimpleNamespace(ui=SimpleNamespace(giz=giz)))
    _LocalAxes.over(viewer, {"pole"})
    for name in ("pole", "point"):
        viewer.gui.ui.giz.push_id(name)
        viewer.gui.ui.giz.manipulate(None, None, 0, giz.MODE.world, None)
    assert drawn == ["local", "world"]
    _LocalAxes.over(viewer, set())  # put in place once, its names kept up to date
    assert viewer.gui.ui.giz.local == set() and viewer.gui.ui.giz._giz is giz


def test_a_scaled_gizmo_draws_scale_handles_and_keeps_the_scale_out_of_newtons_matrix():
    from types import SimpleNamespace

    from usdscope.view import _LocalAxes

    ops = SimpleNamespace(translate=1, translate_x=2, translate_y=3, translate_z=4)
    ops.scale, ops.scale_x, ops.scale_y, ops.scale_z = 11, 12, 13, 14
    asked = []

    def manipulate(view, proj, op, mode, matrix, *rest):
        asked.append(op)
        if op == ops.scale_x:  # dragged: twice as long along x
            matrix.values[0:3] *= 2.0

    giz = SimpleNamespace(
        OPERATION=ops,
        MODE=SimpleNamespace(world="world", local="local"),
        push_id=lambda i: None,
        manipulate=manipulate,
        is_using=lambda: True,
    )
    viewer = SimpleNamespace(gui=SimpleNamespace(ui=SimpleNamespace(giz=giz)))
    _LocalAxes.over(viewer, set(), {"area"})
    wrapped = viewer.gui.ui.giz
    matrix = SimpleNamespace(values=np.eye(4).flatten(order="F"))
    wrapped.push_id("area")
    wrapped.manipulate(None, None, ops.translate_x, "world", matrix)
    wrapped.push_id("point")
    wrapped.manipulate(None, None, ops.translate_x, "world", matrix)
    assert asked == [ops.scale_x, ops.translate_x]  # scale handles for it alone
    assert wrapped.scales["area"] == pytest.approx([2.0, 1.0, 1.0])
    assert np.allclose(matrix.values, np.eye(4).flatten(order="F"))  # Newton: unscaled


def test_the_fog_is_where_the_viewport_says(viewport):
    from usdscope import fog

    viewport.frame(0.0)
    program = viewport.viewer.renderer._shape_shader.shader_program
    assert program["fog_far"] == pytest.approx(viewport.viewer.camera.far)  # by default
    viewport.fog = 300.0
    viewport.frame(0.0)
    assert program["fog_far"] == pytest.approx(300.0)
    viewport.fog = fog.OFF
    viewport.frame(0.0)
    assert program["fog_far"] == pytest.approx(fog.OFF)
    viewport.fog = None


def test_stop_ends_the_frames_after_the_one_it_was_called_in():
    try:
        v = view.Viewport(320, 180, headless=True)
    except Exception as e:  # no GPU or no OpenGL here
        pytest.skip(f"no headless viewer: {e}")
    frames = []

    def each(dt: float) -> None:
        frames.append(dt)
        v.stop()

    v.run(each)  # returns, the window closed
    assert len(frames) == 1 and v.stopping


def test_a_panel_that_raises_part_way_is_reported_and_the_frame_goes_on(viewport):
    said = []

    def broken(imgui) -> None:
        imgui.begin("broken")
        imgui.begin_tab_bar("tabs")
        imgui.push_id("deep")
        raise ValueError("gnss2 is off")

    reported, viewport.report = viewport.report, said.append
    viewport.panel(broken)
    try:
        viewport.frame(0.0)
        viewport.frame(0.0)  # again: imgui would assert "Missing End()" had it not been ended
    finally:
        viewport._panels.remove(broken)
        viewport.report = reported
    assert len(said) == 2 and "gnss2 is off" in said[0]


def test_only_hides_what_other_layers_drew_and_shows_it_again_as_it_was(viewport):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    ground = UsdGeom.Mesh.Define(stage, "/Ground")
    ground.CreatePointsAttr([(-5, -5, 0), (5, -5, 0), (5, 5, 0), (-5, 5, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    viewport.show_scene(read.scene(stage), name="solo", layer="soloscene")
    viewport.lines("/solo/line", np.array([[0.0, 0, 0], [1, 0, 0]]), layer="sololines")
    viewport.frame(0.0)
    objects, lines = viewport.viewer.objects, viewport.viewer.lines
    copies = next(
        o for n, o in objects.items() if n.startswith("/layers/soloscene/") and "/at" in n
    )
    only_copied = next(
        o for n, o in objects.items() if n.startswith("/layers/soloscene/") and "/mesh" in n
    )
    line = next(o for n, o in lines.items() if n.startswith("/layers/sololines/"))
    assert not copies.hidden and only_copied.hidden and not line.hidden

    viewport.only({"sololines"})
    viewport.frame(0.0)
    assert copies.hidden and not line.hidden  # the scene's copies hidden, not just its model

    viewport.only(None)
    viewport.frame(0.0)
    assert not copies.hidden and only_copied.hidden and not line.hidden
    viewport.clear("soloscene")
    viewport.clear("sololines")
