"""A game controller's state as plain data."""

import math

from usdscope.gamepad import DEAD, PadState, dead_zoned


def test_a_stick_resting_near_its_middle_reads_as_nothing():
    assert dead_zoned(0.05, -0.08) == (0.0, 0.0)
    assert dead_zoned(DEAD * 0.99, 0.0) == (0.0, 0.0)


def test_a_stick_pushed_reads_its_way_and_at_most_one_long():
    x, y = dead_zoned(0.0, 1.0)
    assert (x, y) == (0.0, 1.0)
    x, y = dead_zoned(0.9, 0.9)  # a corner reads past 1: kept to 1, the way kept
    assert math.isclose(math.hypot(x, y), 1.0) and math.isclose(x, y)
    x, _ = dead_zoned(DEAD + (1 - DEAD) / 2, 0.0)  # half way past the dead middle: half
    assert math.isclose(x, 0.5)


def test_a_state_says_what_is_held_and_what_just_went_down():
    s = PadState(held=frozenset({"x", "a"}), down=frozenset({"a"}))
    assert s.holds("x") and not s.pressed("x") and s.pressed("a")
