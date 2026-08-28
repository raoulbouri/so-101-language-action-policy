"""Guards on the observation/action alignment contract.

`action[t]` must be the command issued FROM `obs[t]`. If this ever slips by one
step, ACT trains on a lagged target and the failure is silent -- the loss still
goes down. Hence these tests.
"""

import numpy as np
import pytest

from so101_sim.constants import ALL_JOINTS, HOME_QPOS, PHYSICS_SUBSTEPS
from so101_sim.episode_runner import EpisodeRunner
from so101_sim.randomization import sample_episode


@pytest.fixture(scope="module")
def episode():
    runner = EpisodeRunner(render=False)
    ep = runner.run(sample_episode(3))
    runner.close()
    return ep


def test_all_streams_share_one_length(episode):
    n = len(episode)
    for arr in (episode.qpos, episode.qvel, episode.action,
                episode.tcp_pose, episode.phase_ids, episode.object_poses):
        assert arr.shape[0] == n


def test_first_observation_is_the_home_pose(episode):
    """obs[0] is recorded before any command, so it must still be home."""
    assert np.allclose(episode.qpos[0], HOME_QPOS, atol=2e-2)


def test_action_leads_the_observation(episode):
    """obs[t+1] must be closer to action[t] than obs[t] was.

    This is the directional check that catches an off-by-one: if the streams
    were swapped or shifted, the commanded target would trail the state instead
    of leading it.

    Arm joints only: the gripper is deliberately commanded past the cube's
    surface during the grasp, so it physically cannot converge on its target and
    would drag the statistic down for a legitimate reason. The exact-replay test
    below is the strict form of this contract.
    """
    q, a = episode.qpos[:, :5], episode.action[:, :5]
    before = np.linalg.norm(a[:-1] - q[:-1], axis=1)
    after = np.linalg.norm(a[:-1] - q[1:], axis=1)
    moving = before > 1e-4
    improved = (after[moving] <= before[moving] + 1e-6).mean()
    assert improved > 0.90, f"only {improved:.1%} of steps moved toward the command"


def test_observation_is_recorded_before_the_command(episode):
    """A replay of the recorded actions must reproduce the recorded states.

    This is the strongest form of the contract: stepping the simulator with
    action[t] from obs[t] has to land on obs[t+1].
    """
    import mujoco

    from so101_sim.scene_builder import build_model

    model = build_model(episode.spec)
    data = mujoco.MjData(model)
    adr = np.array([model.joint(n).qposadr[0] for n in ALL_JOINTS], dtype=int)

    data.qpos[adr] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS
    mujoco.mj_forward(model, data)
    for _ in range(int(0.4 * 50) * PHYSICS_SUBSTEPS):
        mujoco.mj_step(model, data)

    for t in range(min(len(episode), 120)):
        assert np.allclose(data.qpos[adr], episode.qpos[t], atol=1e-4), \
            f"replay diverged from the recorded observation at step {t}"
        data.ctrl[:] = episode.action[t]
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(model, data)


def test_phase_labels_are_contiguous_and_ordered(episode):
    ids = episode.phase_ids
    changes = ids[1:][ids[1:] != ids[:-1]]
    assert list(changes) == sorted(set(changes)), "phases must not interleave"
    assert ids[0] == 0


def test_actions_stay_inside_the_control_range(episode):

    from so101_sim.scene_builder import build_model

    model = build_model(episode.spec)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    assert np.all(episode.action >= lo - 1e-6)
    assert np.all(episode.action <= hi + 1e-6)


def test_actions_are_smooth(episode):
    """Quintic blending should leave no step discontinuity at phase joins."""
    step = np.abs(np.diff(episode.action, axis=0)).max()
    assert step < 0.12, f"largest single-step joint jump {step:.3f} rad"
