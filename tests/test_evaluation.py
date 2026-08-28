import numpy as np

from so101_sim.constants import CUBE_HALF, ZONE_RADIUS
from so101_sim.evaluation import base_corners_xy, evaluate_placement

RESTING = CUBE_HALF
STILL = np.zeros(3)


def test_centred_cube_passes():
    r = evaluate_placement(np.array([0.2, 0.0, RESTING]), 0.0, STILL, np.array([0.2, 0.0]))
    assert r.success and r.reason == "ok"


def test_cube_with_a_corner_outside_the_perimeter_fails():
    # Centre inside the zone, but far enough out that a corner escapes.
    # The cube is yawed 45 deg so a corner points straight along the offset --
    # with an axis-aligned cube the furthest corner is off-axis and this same
    # centre offset still fits inside the perimeter.
    offset = ZONE_RADIUS - CUBE_HALF * np.sqrt(2) + 0.003
    r = evaluate_placement(np.array([0.2 + offset, 0.0, RESTING]), np.pi / 4, STILL,
                           np.array([0.2, 0.0]))
    assert not r.success
    assert r.center_in_zone, "the lenient metric should still register a pass"
    assert "perimeter" in r.reason


def test_airborne_cube_fails():
    r = evaluate_placement(np.array([0.2, 0.0, 0.10]), 0.0, STILL, np.array([0.2, 0.0]))
    assert not r.success and "resting" in r.reason


def test_moving_cube_fails():
    r = evaluate_placement(np.array([0.2, 0.0, RESTING]), 0.0, np.array([0.5, 0, 0]),
                           np.array([0.2, 0.0]))
    assert not r.success and "moving" in r.reason


def test_corner_reach_is_rotation_invariant():
    """A square's furthest corner is always half-diagonal from its own centre."""
    centre = np.array([0.2, 0.0])
    d_axis = np.linalg.norm(base_corners_xy(centre, 0.0) - centre, axis=1).max()
    d_diag = np.linalg.norm(base_corners_xy(centre, np.pi / 4) - centre, axis=1).max()
    assert np.isclose(d_axis, d_diag), "corner distance is rotation invariant for a square"
    assert np.isclose(d_axis, CUBE_HALF * np.sqrt(2))


def test_corners_are_four_distinct_points():
    c = base_corners_xy(np.array([0.1, 0.2]), 0.3)
    assert c.shape == (4, 2)
    assert len({tuple(np.round(p, 6)) for p in c}) == 4
