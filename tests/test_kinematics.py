import numpy as np
import pytest

from so101_sim.constants import GRASP_Z, HOME_QPOS, HOVER_Z, WORKSPACE_R_MAX, WORKSPACE_R_MIN
from so101_sim.kinematics import SO101Kinematics, rotation_error, top_down_pose
from so101_sim.randomization import sample_episode
from so101_sim.scene_builder import build_model


@pytest.fixture(scope="module")
def ik():
    return SO101Kinematics(build_model(sample_episode(0)))


def test_top_down_pose_is_a_rotation():
    for yaw in np.linspace(-np.pi, np.pi, 9):
        R = top_down_pose(np.zeros(3), yaw)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
        # z axis must point straight down (the approach direction)
        assert np.allclose(R[:, 2], [0, 0, -1])


def test_rotation_error_is_zero_for_identical_frames():
    R = top_down_pose(np.zeros(3), 0.4)
    assert np.linalg.norm(rotation_error(R, R)) < 1e-9


def test_fk_matches_ik_solution(ik):
    target = np.array([0.22, 0.05, GRASP_Z])
    res = ik.solve_top_down(target, 0.2, HOME_QPOS[:5])
    assert res.converged
    pos, rot = ik.fk(res.qpos)
    assert np.linalg.norm(pos - target) < 1e-3
    assert np.linalg.norm(rotation_error(rot, top_down_pose(target, 0.2))) < 2e-2


def test_ik_respects_joint_limits(ik):
    res = ik.solve_top_down(np.array([0.24, -0.1, GRASP_Z]), 0.0, HOME_QPOS[:5])
    assert np.all(res.qpos >= ik.arm_lo - 1e-9)
    assert np.all(res.qpos <= ik.arm_hi + 1e-9)


def test_ik_converges_across_the_declared_workspace(ik):
    """The declared workspace must actually be solvable -- this is the guard
    that stops someone widening WORKSPACE_R_MAX without re-measuring."""
    failures = 0
    total = 0
    for r in np.linspace(WORKSPACE_R_MIN, WORKSPACE_R_MAX, 6):
        for yaw in np.linspace(-np.deg2rad(60), np.deg2rad(60), 7):
            for z in (GRASP_Z, HOVER_Z):
                p = np.array([r * np.cos(yaw), r * np.sin(yaw), z])
                total += 1
                failures += not ik.solve_top_down(p, yaw, HOME_QPOS[:5]).converged
    assert failures == 0, f"{failures}/{total} workspace poses unreachable"


def test_ik_degrades_gracefully_outside_reach(ik):
    """Far out of reach the solver must return a clamped best effort, not blow up."""
    res = ik.solve_top_down(np.array([1.5, 0.0, 0.3]), 0.0, HOME_QPOS[:5])
    assert not res.converged
    assert np.all(np.isfinite(res.qpos))
