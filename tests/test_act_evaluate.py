"""Correctness checks on offline_metrics -- catches the two real bugs found in
review: per-joint MAE silently omitting radians, and per-phase MAE bucketing by
the anchor's phase instead of each predicted action's own phase.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from so101_act.config import Config
from so101_act.data import (
    SO101ACTDataset,
    collate,
    compute_normalizer,
    make_splits,
    scan_episodes,
)
from so101_act.evaluate import PHASE_NAMES, offline_metrics
from so101_act.model import build_model

HDF5 = Path("data/train_1200.hdf5")
pytestmark = pytest.mark.skipif(not HDF5.exists(), reason="dataset not generated")


@pytest.fixture(scope="module")
def setup():
    eps, _ = scan_episodes(HDF5)
    sp = make_splits(eps, seed=0)
    norm = compute_normalizer(HDF5, eps, sp["train"], max_episodes=20)
    ds = SO101ACTDataset(HDF5, eps, sp["val"][:2], norm, conditioning="clip")
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    cfg = Config(conditioning="clip", pretrained_backbone=False, hidden_dim=64,
                dim_feedforward=128, nheads=4, enc_layers=1, dec_layers=1,
                cvae_enc_layers=1)
    model = build_model(cfg)
    return eps, sp, norm, ds, loader, model, cfg


def test_phase_chunk_looks_ahead_not_just_at_the_anchor(setup):
    """A chunk starting near the end of a short phase must show later phases too."""
    eps, sp, _, ds, *_ = setup
    ei = sp["val"][0]
    # `grasp` is the shortest phase; find a t where the anchor is in grasp but
    # the chunk runs past it.
    import h5py
    with h5py.File(HDF5, "r") as f:
        raw_phase = f[eps[ei].name]["phase"][:]
    grasp_id = PHASE_NAMES.index("grasp")
    grasp_ts = np.where(raw_phase == grasp_id)[0]
    assert len(grasp_ts) > 0
    t = int(grasp_ts[-1])  # last timestep still labelled grasp

    idx = ds._index.index((sp["train"][0] if False else ei, t))  # find within this ds's index
    sample = ds[idx]
    phases_in_chunk = set(sample["phase_chunk"].tolist())
    assert sample["phase"].item() == grasp_id, "anchor should be the last grasp step"
    assert phases_in_chunk != {grasp_id}, (
        "chunk starting at the last grasp step must contain later phases too -- "
        "bucketing by the anchor phase alone would mislabel this data"
    )


def test_phase_chunk_matches_raw_hdf5_slice(setup):
    eps, sp, _, ds, *_ = setup
    ei = sp["val"][0]
    import h5py
    with h5py.File(HDF5, "r") as f:
        raw = f[eps[ei].name]["phase"][:]
    t = 50
    idx = ds._index.index((ei, t))
    sample = ds[idx]
    expected = raw[t : t + ds.chunk_size]
    assert np.array_equal(sample["phase_chunk"][: len(expected)].numpy(), expected)


def test_offline_metrics_reports_per_joint_radians(setup):
    _, _, norm, _, loader, model, cfg = setup
    out = offline_metrics(model, loader, torch.device("cpu"), cfg.kl_weight, norm,
                          max_batches=2)
    for j, entry in out["per_joint_mae"].items():
        assert "normalized" in entry and "radians" in entry, j
        assert entry["radians"] >= 0 and entry["normalized"] >= 0


def test_per_joint_radians_equals_normalized_times_std(setup):
    _, _, norm, _, loader, model, cfg = setup
    out = offline_metrics(model, loader, torch.device("cpu"), cfg.kl_weight, norm,
                          max_batches=2)
    from so101_act.data import IMAGENET_MEAN  # noqa: F401  (import sanity only)
    from so101_act.evaluate import JOINT_NAMES
    for j, name in enumerate(JOINT_NAMES):
        entry = out["per_joint_mae"][name]
        assert abs(entry["radians"] - entry["normalized"] * float(norm.action_std[j])) < 1e-6


def test_per_phase_mae_uses_target_phase_not_anchor_phase(setup):
    """A hand-built batch where the anchor phase differs from every chunk step's
    own phase must attribute error to the CHUNK's phases, not the anchor's."""
    _, _, norm, _, _, model, cfg = setup
    B, K = 2, model.chunk
    batch = {
        "scene_image": torch.randn(B, 3, 128, 128),
        "wrist_image": torch.randn(B, 3, 128, 128),
        "qpos": torch.randn(B, 6),
        "language_embedding": torch.nn.functional.normalize(torch.randn(B, 512), dim=1),
        "task_id": torch.zeros(B, dtype=torch.long),
        "action_chunk": torch.zeros(B, K, 6),
        "action_chunk_mask": torch.ones(B, K, dtype=torch.bool),
        # anchor phase is "hover_and_center" (0) for both samples...
        "phase": torch.zeros(B, dtype=torch.long),
        # ...but every predicted step in the chunk is actually "grasp" (2).
        "phase_chunk": torch.full((B, K), PHASE_NAMES.index("grasp"), dtype=torch.long),
    }
    loader = [batch]
    out = offline_metrics(model, loader, torch.device("cpu"), cfg.kl_weight, norm)
    assert out["per_phase_mae"]["grasp"] is not None
    assert out["per_phase_mae"]["hover_and_center"] is None, (
        "no chunk step is actually in hover_and_center -- bucketing by the "
        "anchor's phase would have wrongly attributed all error there"
    )
