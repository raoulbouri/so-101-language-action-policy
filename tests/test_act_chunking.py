"""Tier 0.5.A gates: on-the-fly chunking and temporal ensembling (T2-T5).

These guard three deviations from the reference ACT/LeRobot implementation that
were found by reading the sources, all logged in docs/ISSUES.md:
  ISSUE-007 ensembler weighting was inverted vs the ACT paper
  ISSUE-008 chunk size 32 vs the ACT/LeRobot default of 100
  ISSUE-009 ensembling on by default, contrary to LeRobot
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from so101_act.config import Config
from so101_act.data import (
    SO101ACTDataset,
    compute_normalizer,
    make_splits,
    scan_episodes,
)
from so101_act.rollout import TemporalEnsembler

HDF5 = Path("data/train_1200.hdf5")
pytestmark = pytest.mark.skipif(not HDF5.exists(), reason="dataset not generated")


@pytest.fixture(scope="module")
def prepared():
    eps, _ = scan_episodes(HDF5)
    sp = make_splits(eps, seed=0)
    norm = compute_normalizer(HDF5, eps, sp["train"], max_episodes=20)
    return eps, sp, norm


# --- T2: on-the-fly chunks reproduce the stored ones exactly at k=32 --------
def test_onthefly_chunks_match_stored_action_chunk_at_k32(prepared):
    eps, sp, norm = prepared
    ei = sp["train"][0]
    ds = SO101ACTDataset(HDF5, eps, [ei], norm, chunk_size=32)
    with h5py.File(HDF5, "r") as f:
        g = f[eps[ei].name]
        T = g["action"].shape[0]
        for t in (0, 7, T // 2, T - 40, T - 1):
            idx = ds._index.index((ei, t))
            s = ds[idx]
            got = norm.denorm_action(s["action_chunk"].numpy())
            assert np.allclose(got, g["action_chunk"][t], atol=1e-4), f"t={t}"
            assert np.array_equal(
                s["action_chunk_mask"].numpy(), g["action_chunk_mask"][t]
            ), f"mask mismatch at t={t}"


# --- T3: k=100 slices are correct, including the padded tail ---------------
@pytest.mark.parametrize("k", [32, 64, 100])
def test_chunk_equals_action_slice_for_any_k(prepared, k):
    eps, sp, norm = prepared
    ei = sp["train"][0]
    ds = SO101ACTDataset(HDF5, eps, [ei], norm, chunk_size=k)
    with h5py.File(HDF5, "r") as f:
        actions = f[eps[ei].name]["action"][:]
    T = actions.shape[0]
    for t in (0, T // 3, T - k // 2, T - 1):
        idx = ds._index.index((ei, t))
        s = ds[idx]
        chunk = norm.denorm_action(s["action_chunk"].numpy())
        mask = s["action_chunk_mask"].numpy()
        n_real = min(k, T - t)
        assert chunk.shape == (k, 6)
        assert int(mask.sum()) == n_real, f"k={k} t={t}"
        assert np.allclose(chunk[:n_real], actions[t:t + n_real], atol=1e-4)
        # padded tail repeats the final real action, and is masked out
        if n_real < k:
            assert np.allclose(chunk[n_real:], actions[t + n_real - 1], atol=1e-4)
            assert not mask[n_real:].any()


def test_phase_chunk_length_follows_chunk_size(prepared):
    eps, sp, norm = prepared
    for k in (32, 100):
        ds = SO101ACTDataset(HDF5, eps, sp["train"][:1], norm, chunk_size=k)
        assert ds[0]["phase_chunk"].shape == (k,)


# --- T4: ensembler weights the OLDEST prediction highest -------------------
def test_ensembler_weights_oldest_highest():
    """ACT: w_i = exp(-m*i) with w_0 the weight of the OLDEST action."""
    k, m = 4, 0.5
    ens = TemporalEnsembler(chunk=k, action_dim=1, m=m)
    # Three overlapping chunks; each predicts a distinct constant for step 3.
    ens.add(1, np.array([[0.0], [0.0], [100.0], [0.0]]))   # oldest -> predicts 100 at t=3
    ens.add(2, np.array([[0.0], [10.0], [0.0], [0.0]]))    #        -> predicts 10  at t=3
    ens.add(3, np.array([[1.0], [0.0], [0.0], [0.0]]))     # newest -> predicts 1   at t=3
    out = float(ens.action_for(3)[0])
    w = np.exp(-m * np.arange(3)); w /= w.sum()
    expected = w[0] * 100.0 + w[1] * 10.0 + w[2] * 1.0
    assert np.isclose(out, expected), f"{out} != {expected}"
    # The oldest prediction must dominate.
    assert w[0] > w[-1]


def test_ensembler_single_chunk_is_passthrough():
    ens = TemporalEnsembler(chunk=4, action_dim=2, m=0.01)
    ch = np.arange(8, dtype=np.float64).reshape(4, 2)
    ens.add(0, ch)
    for h in range(4):
        assert np.allclose(ens.action_for(h), ch[h])


def test_ensembler_evicts_stale_chunks():
    ens = TemporalEnsembler(chunk=3, action_dim=1, m=0.01)
    for t in range(10):
        ens.add(t, np.zeros((3, 1)))
    assert len(ens.buf) <= 3


# --- config defaults must match the reference implementation ---------------
def test_config_defaults_match_act_lerobot():
    c = Config()
    assert c.chunk_size == 100, "ACT/LeRobot default is 100"
    assert c.use_ensembling is False, "LeRobot defaults temporal ensembling OFF"


def test_dataset_default_chunk_matches_config_default():
    """Guard against ISSUE-011: a dataset built at a different k than the model
    fails deep inside the CVAE positional embedding, far from the real cause."""
    from so101_act.data import DEFAULT_CHUNK_SIZE
    assert Config().chunk_size == DEFAULT_CHUNK_SIZE
