import numpy as np
import pytest

from usdscope import shapes


def test_an_arrow_bends_from_the_way_in_to_the_way_out_and_points_out():
    at = np.array([[0.0, 0.0, 0.0]])
    v, t = shapes.arrows(at, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    assert t.max() < len(v) and (v[:, 2] == 0).all()
    tip = v[np.argmax(v[:, 0])]  # its head's point, furthest along the way out
    assert tip[0] > 0.45 and abs(tip[1]) < 0.15  # out along +X (the arc a little aside)
    assert v[:, 1].min() < -0.4  # its tail back along the way in
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    assert (np.cross(b - a, c - a)[:, 2] > 0).all()  # every triangle faces up
    none = shapes.arrows(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)))
    assert none[0].shape == (0, 3) and none[1].shape == (0, 3)


def test_a_sharp_turn_does_not_fold_the_arrow_over_itself():
    at = np.zeros((1, 3))
    for degrees in (0, 45, 90, 150, 179, -120):
        r = np.radians(degrees)
        out = np.array([[-np.sin(r), np.cos(r), 0.0]])
        v, t = shapes.arrows(at, np.array([[0.0, 1.0, 0.0]]), out)
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        assert (np.cross(b - a, c - a)[:, 2] > 0).all(), degrees  # none flipped: no fold
        tip = v[-1]
        assert np.dot(tip[:2], out[0, :2]) > 0.2, degrees  # it points the way out


def test_an_arrow_along_a_line_follows_it_and_points_on_past_its_end():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    piece = shapes.along_line(line, 1.0, 0.5, 0.5, steps=5)
    assert np.allclose(piece[0], [0.5, 0.0, 0.0]) and np.linalg.norm(piece[-1] - piece[0]) < 1.0
    v, t = shapes.arrows_along([piece], width=0.1, head=0.2)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    assert (np.cross(b - a, c - a)[:, 2] > 0).all()
    assert v[-1][0] > piece[-1][0] and v[-1][1] > piece[-1][1]  # its tip on past the end


def test_a_streak_travels_along_its_line_narrow_at_its_tail():
    line = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    fronts = []
    for share in (0.3, 0.6, 0.9):
        v, t = shapes.streaks([line], share)
        assert len(t) and t.max() < len(v)
        fronts.append(v[:, 0].max())
        assert abs(v[0, 1]) < 1e-6  # its tail comes to a point
    assert fronts[0] < fronts[1] < fronts[2]  # it moves on, the way the line goes
    assert shapes.streaks([line], 0.0)[0].shape == (0, 3)  # not yet on it


def test_streaks_flow_along_a_line_its_corners_rounded():
    line = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0]])
    smooth = shapes.rounded(line, 1.0)
    assert len(smooth) > 3 and np.allclose(smooth[0], line[0]) and np.allclose(smooth[-1], line[-1])
    assert np.linalg.norm(smooth[:, :2] - [4.0, 0.0], axis=1).min() > 0.2  # the corner cut
    v, t = shapes.streaks_along(smooth, [1.0, 3.0, 5.0, 99.0], fade=0.5)
    assert len(t) and t.max() < len(v)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    assert (np.cross(b - a, c - a)[:, 2] > -1e-9).all()  # all facing up


def test_a_double_arrow_has_a_head_at_each_end_and_faces_up():
    straight = np.array([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    a = np.radians(np.linspace(-30, 30, 9))
    arc = np.stack([np.sin(a), np.cos(a), np.zeros(9)], axis=1)
    v, t = shapes.double_arrows([straight, arc])
    p, q, r = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    assert (np.cross(q - p, r - p)[:, 2] > 0).all()
    v, _ = shapes.double_arrows([straight])
    assert v[:, 0].max() == pytest.approx(0.4, abs=1e-6) and v[:, 0].min() < -0.39  # both ends
