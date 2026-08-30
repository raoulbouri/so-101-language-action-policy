"""Dataset, split and normalization checks (validation items 1-2)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from so101_act.data import (
    SO101ACTDataset,
    collate,
    compute_normalizer,
    make_splits,
    parse_instruction,
    scan_episodes,
)

HDF5 = Path("data/train_1200.hdf5")
pytestmark = pytest.mark.skipif(not HDF5.exists(), reason="dataset not generated")


@pytest.fixture(scope="module")
def scanned():
    eps, task_ids = scan_episodes(HDF5)
    return eps, task_ids


@pytest.fixture(scope="module")
def prepared(scanned):
    eps, _ = scanned
    sp = make_splits(eps, mode="iid", seed=0)
    norm = compute_normalizer(HDF5, eps, sp["train"], max_episodes=60)
    return eps, sp, norm


def test_instruction_parsing():
    assert parse_instruction("take the red cube and place it in the green circle") == ("red", "green")
    with pytest.raises(ValueError):
        parse_instruction("do something else entirely")


def test_task_ids_cover_the_full_matrix(scanned):
    _, task_ids = scanned
    assert len(task_ids) == 16, "4 cube colours x 4 zone colours"
    assert sorted(task_ids.values()) == list(range(16))


def test_splits_are_episode_level_and_disjoint(prepared):
    eps, sp, _ = prepared
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == len(eps)
    assert len(tr) > len(va) and len(tr) > len(te)


def test_compositional_split_holds_out_whole_combinations(scanned):
    eps, _ = scanned
    hold = (("orange", "pink"),)
    sp = make_splits(eps, mode="compositional", holdout_combos=hold, seed=0)
    train_combos = {(eps[i].cube, eps[i].zone) for i in sp["train"]}
    test_combos = {(eps[i].cube, eps[i].zone) for i in sp["test"]}
    assert ("orange", "pink") not in train_combos, "held-out combo leaked into train"
    assert test_combos == {("orange", "pink")}
    # Both concepts must still be learnable individually.
    assert any(c == "orange" for c, _ in train_combos)
    assert any(z == "pink" for _, z in train_combos)


def test_compositional_split_rejects_removing_a_concept_entirely(scanned):
    eps, _ = scanned
    all_pink = tuple((c, "pink") for c in ("red", "blue", "yellow", "orange"))
    with pytest.raises(ValueError, match="removes a concept"):
        make_splits(eps, mode="compositional", holdout_combos=all_pink, seed=0)


def test_normalizer_uses_training_episodes_only(prepared):
    _, _, norm = prepared
    # Statistics must be finite, and std floored above zero.
    for arr in (norm.qpos_mean, norm.qpos_std, norm.action_mean, norm.action_std):
        assert np.isfinite(arr).all()
        assert arr.shape == (6,)
    assert (norm.qpos_std > 0).all() and (norm.action_std > 0).all()


def test_normalizer_roundtrip(prepared):
    _, _, norm = prepared
    a = np.random.randn(5, 6).astype(np.float32)
    assert np.allclose(norm.denorm_action(norm.norm_action(a)), a, atol=1e-4)
    t = torch.randn(5, 6)
    assert torch.allclose(
        torch.as_tensor(norm.denorm_action(norm.norm_action(t.numpy()))), t, atol=1e-4
    )


def test_batch_shapes_match_the_contract(prepared):
    eps, sp, norm = prepared
    # k comes from the dataset, not a literal: the ACT default moved 32 -> 100
    # and a hardcoded shape here would fail for a reason unrelated to the
    # contract being tested. See ISSUE-011.
    ds = SO101ACTDataset(HDF5, eps, sp["train"][:4], norm, conditioning="clip")
    k = ds.chunk_size
    batch = collate([ds[i] for i in range(6)])
    assert batch["scene_image"].shape == (6, 3, 128, 128)
    assert batch["wrist_image"].shape == (6, 3, 128, 128)
    assert batch["qpos"].shape == (6, 6)
    assert batch["language_embedding"].shape == (6, 512)
    assert batch["action_chunk"].shape == (6, k, 6)
    assert batch["action_chunk_mask"].shape == (6, k)
    assert batch["phase"].shape == (6,)
    assert batch["action_chunk_mask"].dtype == torch.bool
    assert isinstance(batch["instruction"], list) and len(batch["instruction"]) == 6


def test_actions_are_not_shifted_relative_to_observations(prepared):
    """The chunk at t must start with the action recorded at t -- no roll."""
    import h5py

    eps, sp, norm = prepared
    ds = SO101ACTDataset(HDF5, eps, sp["train"][:2], norm)
    ei, t = ds._index[100]
    s = ds[100]
    with h5py.File(HDF5, "r") as f:
        g = f[eps[ei].name]
        raw_action_t = g["action"][t]
        raw_chunk_t0 = g["action_chunk"][t][0]
    assert np.allclose(raw_action_t, raw_chunk_t0), "stored chunk is already shifted"
    # and the dataset's normalised copy denormalises back to it
    assert np.allclose(
        norm.denorm_action(s["action_chunk"][0].numpy()), raw_action_t, atol=1e-4
    )


def test_mask_marks_exactly_the_valid_tail(prepared):
    eps, sp, norm = prepared
    ds = SO101ACTDataset(HDF5, eps, sp["train"][:1], norm)
    k = ds.chunk_size
    T = eps[sp["train"][0]].length
    for t in (0, T // 2, T - 1, T - 5):
        s = ds[t]
        expected = min(k, T - t)
        assert int(s["action_chunk_mask"].sum()) == expected, f"t={t} k={k}"


def test_language_shuffle_changes_the_embedding_not_the_observation(prepared):
    eps, sp, norm = prepared
    idx = sp["train"][:20]
    plain = SO101ACTDataset(HDF5, eps, idx, norm, conditioning="clip")
    shuf = SO101ACTDataset(HDF5, eps, idx, norm, conditioning="clip",
                           shuffle_language=True, language_seed=1)
    n_diff = 0
    for i in range(0, 400, 37):
        a, b = plain[i], shuf[i]
        assert torch.equal(a["scene_image"], b["scene_image"]), "observation changed"
        assert torch.equal(a["qpos"], b["qpos"])
        if not torch.equal(a["language_embedding"], b["language_embedding"]):
            n_diff += 1
    assert n_diff > 0, "shuffling never changed the instruction"
