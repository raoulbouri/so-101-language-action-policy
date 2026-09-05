"""Delta action targets (ISSUE-017).

With absolute targets the recorded action is ~96% given by the observed joint
state, so "hold the current position" is a strong L1 minimiser and the trained
arm never moves. These tests pin the delta representation that removes the
shortcut, and the round-trip that rollout depends on.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from so101_act.data import (
    SO101ACTDataset,
    collate,
    compute_normalizer,
    make_splits,
    scan_episodes,
)

HDF5 = Path("data/train_1200.hdf5")
pytestmark = pytest.mark.skipif(not HDF5.exists(), reason="dataset not generated")
K = 16


@pytest.fixture(scope="module")
def prepared():
    eps, _ = scan_episodes(HDF5)
    sp = make_splits(eps, mode="iid", val_frac=0.1, test_frac=0.1, seed=0)
    norm = compute_normalizer(HDF5, eps, sp["train"], max_episodes=20, chunk_size=K)
    return eps, sp, norm


def test_absolute_normalizer_has_no_delta_stats(prepared):
    eps, sp, _ = prepared
    n = compute_normalizer(HDF5, eps, sp["train"], max_episodes=8)
    assert not n.has_delta()
    with pytest.raises(ValueError, match="no delta statistics"):
        n.norm_delta(np.zeros((K, 6), dtype=np.float32))


def test_delta_stats_shape_and_growth(prepared):
    _, _, norm = prepared
    assert norm.delta_std.shape == (K, 6)
    # The spread of the delta grows with the horizon -- that growth is exactly
    # why the statistics are per-horizon rather than pooled.
    assert norm.delta_std.mean(1)[K - 1] > 3 * norm.delta_std.mean(1)[0]


def test_delta_target_roundtrips_to_absolute_actions(prepared):
    """The inverse rollout uses must reconstruct the true joint targets."""
    eps, sp, norm = prepared
    ei = sp["train"][0]
    ds = SO101ACTDataset(HDF5, eps, [ei], norm, chunk_size=K, action_repr="delta")
    import h5py
    with h5py.File(HDF5, "r") as f:
        g = f[eps[ei].name]
        for t in (0, 50, 200):
            s = ds[ds._index.index((ei, t))]
            got = norm.denorm_chunk(
                s["action_chunk"].numpy(), s["qpos_raw"].numpy(), "delta")
            want = g["action"][t : t + K]
            assert np.allclose(got[: len(want)], want, atol=1e-4)


def test_holding_position_is_a_bad_minimiser_under_delta(prepared):
    """The whole point: the copycat shortcut must stop paying.

    Under absolute targets, predicting qpos[t] for every horizon scores better
    than a trained model at h=0. Under delta targets the same policy is the
    zero vector, and it must cost a large, roughly horizon-independent L1.
    """
    eps, sp, norm = prepared
    idx = sp["train"][:6]
    abs_norm = compute_normalizer(HDF5, eps, sp["train"], max_episodes=20)
    d_abs = SO101ACTDataset(HDF5, eps, idx, abs_norm, chunk_size=K)
    d_del = SO101ACTDataset(HDF5, eps, idx, norm, chunk_size=K, action_repr="delta")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(d_abs), 256, replace=False)
    b_abs = collate([d_abs[int(i)] for i in pick])
    b_del = collate([d_del[int(i)] for i in pick])

    def hold_cost(batch, normalizer, repr_):
        q = batch["qpos_raw"].numpy()
        hold_raw = np.repeat(q[:, None, :], K, axis=1)
        if repr_ == "delta":
            hold = (np.zeros_like(hold_raw) - normalizer.delta_mean[None, :K]) \
                / normalizer.delta_std[None, :K]
        else:
            hold = (hold_raw - normalizer.action_mean) / normalizer.action_std
        tgt = batch["action_chunk"].numpy()
        m = batch["action_chunk_mask"].numpy()[..., None]
        per_h = (np.abs(hold - tgt) * m).sum(axis=(0, 2)) / np.maximum(
            m.sum(axis=(0, 2)) * 6, 1e-9)
        return per_h

    c_abs = hold_cost(b_abs, abs_norm, "absolute")
    c_del = hold_cost(b_del, norm, "delta")

    # Absolute: the shortcut is nearly free at short horizons.
    assert c_abs[0] < 0.1
    # Delta: it costs a lot at EVERY horizon, including h=0.
    assert c_del.min() > 0.3
    assert c_del[0] > 5 * c_abs[0]


def test_dataset_rejects_delta_without_stats(prepared):
    eps, sp, _ = prepared
    n = compute_normalizer(HDF5, eps, sp["train"], max_episodes=8)
    with pytest.raises(ValueError):
        SO101ACTDataset(HDF5, eps, sp["train"][:1], n, chunk_size=K,
                        action_repr="delta")


class _HoldPositionModel(torch.nn.Module):
    """A policy that commands the current joint pose at every horizon.

    This is exactly the failure mode ISSUE-017 describes: it scores well on L1,
    because the pose is most of the action, but the arm never moves.
    """

    def __init__(self, norm, chunk, action_repr):
        super().__init__()
        self.norm, self.chunk, self.action_repr = norm, chunk, action_repr

    def forward(self, batch, sample_latent=None):
        q = batch["qpos_raw"].numpy()
        hold = np.repeat(q[:, None, :], self.chunk, axis=1)
        if self.action_repr == "delta":
            out = (np.zeros_like(hold) - self.norm.delta_mean[None, : self.chunk]) \
                / self.norm.delta_std[None, : self.chunk]
        else:
            out = (hold - self.norm.action_mean) / self.norm.action_std
        return {"actions": torch.as_tensor(out, dtype=torch.float32),
                "mu": None, "logvar": None}


def _gate(norm, eps, idx, repr_):
    from torch.utils.data import DataLoader

    from so101_act.evaluate import shortcut_baselines

    ds = SO101ACTDataset(HDF5, eps, idx, norm, chunk_size=K, action_repr=repr_)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    collate_fn=collate)
    return shortcut_baselines(_HoldPositionModel(norm, K, repr_), dl, "cpu",
                              norm=norm, action_repr=repr_, max_batches=8)


@pytest.mark.parametrize("repr_", ["absolute", "delta"])
def test_gate_rejects_a_policy_that_holds_position(prepared, repr_):
    """L1 alone passes this policy; the gate must not.

    Measured on the real e2_clip_100k checkpoint, beating the identity
    predictor on L1 by 76% overall was compatible with a rollout that moved the
    cube 0.0 mm. `lead_ratio` is what separates them.
    """
    eps, sp, norm = prepared
    r = _gate(norm, eps, sp["val"][:4], repr_)
    assert r["lead_ratio"] < 0.1
    assert r["under_driving"] is True
    assert r["passes"] is False


def test_gate_reports_a_sane_expert_lead(prepared):
    """Sanity: the expert's own command lead is the ~1 deg ISSUE-017 measured."""
    eps, sp, norm = prepared
    r = _gate(norm, eps, sp["val"][:4], "absolute")
    # Averaged over horizons 0..K-1 the gap exceeds the one-step 1.06 deg.
    assert 0.5 < np.degrees(r["expert_lead_rad"]) < 20.0


def test_delta_stats_must_cover_the_chunk(prepared):
    eps, sp, norm = prepared
    with pytest.raises(ValueError, match="horizons"):
        SO101ACTDataset(HDF5, eps, sp["train"][:1], norm, chunk_size=K + 8,
                        action_repr="delta")
