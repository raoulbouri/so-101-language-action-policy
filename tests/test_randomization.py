import numpy as np
import pytest

from so101_sim.constants import (
    MIN_ENTITY_SEPARATION,
    WORKSPACE_R_MAX,
    WORKSPACE_R_MIN,
    WORKSPACE_YAW,
)
from so101_sim.randomization import sample_episode


@pytest.mark.parametrize("seed", [0, 1, 17, 123, 4096])
def test_entities_land_inside_the_declared_workspace(seed):
    spec = sample_episode(seed)
    for e in list(spec.objects) + list(spec.zones):
        r = float(np.hypot(*e.pos))
        yaw = float(np.arctan2(e.pos[1], e.pos[0]))
        assert WORKSPACE_R_MIN - 1e-9 <= r <= WORKSPACE_R_MAX + 1e-9
        assert abs(yaw) <= WORKSPACE_YAW + 1e-9


@pytest.mark.parametrize("seed", [0, 3, 55, 777])
def test_entities_do_not_overlap(seed):
    spec = sample_episode(seed)
    pts = [e.pos for e in list(spec.objects) + list(spec.zones)]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) >= \
                MIN_ENTITY_SEPARATION - 1e-9


def test_counts_stay_within_the_directive_range():
    for seed in range(200):
        spec = sample_episode(seed)
        assert 2 <= len(spec.objects) <= 3
        assert 2 <= len(spec.zones) <= 3
        assert len({o.color for o in spec.objects}) == len(spec.objects)
        assert len({z.color for z in spec.zones}) == len(spec.zones)


def test_seeding_is_reproducible():
    assert sample_episode(42).to_dict() == sample_episode(42).to_dict()
    assert sample_episode(42).to_dict() != sample_episode(43).to_dict()


def test_instruction_names_the_actual_target():
    spec = sample_episode(11)
    assert spec.target_object.label in spec.instruction
    assert spec.target_zone.label in spec.instruction


def test_every_seed_in_a_large_range_lays_out():
    for seed in range(1000):
        sample_episode(seed)
