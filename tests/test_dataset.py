import h5py
import numpy as np
import pytest

from so101_sim.episode_runner import EpisodeRunner
from so101_sim.language import HashingTextEncoder
from so101_sim.randomization import sample_episode
from so101_sim.recorder import DatasetWriter, build_action_chunks


def test_action_chunks_are_correctly_indexed():
    actions = np.arange(20 * 6, dtype=np.float32).reshape(20, 6)
    chunks, mask = build_action_chunks(actions, 5)
    assert chunks.shape == (20, 5, 6)
    # an interior step holds exactly the next k actions
    assert np.allclose(chunks[3], actions[3:8])
    assert mask[3].all()
    # the tail is padded by repeating the final action, and masked out
    assert np.allclose(chunks[-1, 0], actions[-1])
    assert mask[-1, 0] and not mask[-1, 1:].any()
    assert np.allclose(chunks[-1, 1:], actions[-1])


def test_chunk_mask_counts_the_remaining_real_actions():
    actions = np.zeros((10, 6), np.float32)
    _, mask = build_action_chunks(actions, 4)
    assert mask.sum(axis=1).tolist() == [4, 4, 4, 4, 4, 4, 4, 3, 2, 1]


def test_writer_refuses_to_clobber_an_existing_dataset(tmp_path):
    """Two runs on one path previously left the .hdf5 and its report disagreeing."""
    path = tmp_path / "ds.hdf5"
    DatasetWriter(path).finalize()
    with pytest.raises(FileExistsError):
        DatasetWriter(path)
    DatasetWriter(path, overwrite=True).finalize()  # explicit opt-in is fine


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    runner = EpisodeRunner(image_size=(64, 80), render=True)
    ep = runner.run(sample_episode(2))
    runner.close()
    enc = HashingTextEncoder()
    path = tmp_path_factory.mktemp("ds") / "test.hdf5"
    with DatasetWriter(path, chunk_size=8, encoder_name=enc.name,
                       embed_dim=enc.dim, image_size=(64, 80)) as w:
        w.add_episode(ep, enc.encode([ep.instruction])[0])
    return path, ep


def test_hdf5_contains_every_required_key(written):
    path, ep = written
    with h5py.File(path) as f:
        g = f["episode_000000"]
        for key in ("obs/scene_image", "obs/wrist_image", "obs/qpos", "obs/qvel",
                    "obs/tcp_pose", "action", "action_chunk", "action_chunk_mask",
                    "phase", "object_poses", "language_instruction",
                    "language_embedding"):
            assert key in g, f"missing {key}"
        assert g["obs/scene_image"].shape == (len(ep), 64, 80, 3)
        assert g["action"].shape == (len(ep), 6)
        assert g["action_chunk"].shape == (len(ep), 8, 6)
        assert g["language_instruction"][()].decode() == ep.instruction
        assert f["metadata"].attrs["n_episodes"] == 1
        assert f["metadata"].attrs["fps"] == 50.0


def test_metadata_records_which_text_encoder_was_used(written):
    path, _ = written
    with h5py.File(path) as f:
        assert f["metadata"].attrs["text_encoder"] == "hashing-fallback"


def test_images_are_not_blank(written):
    path, _ = written
    with h5py.File(path) as f:
        for key in ("obs/scene_image", "obs/wrist_image"):
            frames = f[f"episode_000000/{key}"][:]
            assert frames.std() > 5.0, f"{key} looks blank"


def test_distractors_are_recorded_for_language_grounding(written):
    path, _ = written
    with h5py.File(path) as f:
        labels = [s.decode() if isinstance(s, bytes) else s
                  for s in f["episode_000000"].attrs["object_labels"]]
        assert len(labels) >= 2
        assert f["episode_000000"].attrs["target_object_label"] in labels
