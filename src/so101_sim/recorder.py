"""HDF5 dataset writer.

Layout (one group per episode, mirroring the standard ACT / LeRobot HDF5
convention so an existing ACT dataloader needs only the extra language keys):

    /metadata                       attrs: fps, encoder, dims, chunk size, ...
    /episode_000000/
        obs/scene_image     (T, H, W, 3)  uint8
        obs/wrist_image     (T, H, W, 3)  uint8
        obs/qpos            (T, 6)        float32
        obs/qvel            (T, 6)        float32
        obs/tcp_pose        (T, 7)        float32
        action              (T, 6)        float32   target joint positions
        action_chunk        (T, k, 6)     float32   k-step lookahead for ACT
        action_chunk_mask   (T, k)        bool      False where padded
        phase               (T,)          int8
        object_poses        (T, n, 7)     float32
        language_instruction            scalar utf-8 string
        language_embedding  (D,)          float32
      attrs: seed, success, instruction, target names, zone geometry, ...

Images are stored gzip-compressed and chunked one frame at a time, which is the
access pattern a dataloader actually uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .constants import IMAGE_SIZE, N_DOF, RECORD_HZ
from .episode_runner import Episode
from .expert_policy import PHASES


def build_action_chunks(actions: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Materialise the k-step action chunks ACT is trained on.

    ``chunk[t]`` is ``actions[t : t+k]``. Near the end of the episode there are
    fewer than k real actions left, so the tail is padded by repeating the final
    action and the corresponding mask entries are set False -- repeating rather
    than zero-filling keeps the padded targets on the manifold, and the mask
    lets the loss ignore them entirely.
    """
    n_steps, n_dof = actions.shape
    chunks = np.empty((n_steps, chunk_size, n_dof), dtype=np.float32)
    mask = np.zeros((n_steps, chunk_size), dtype=bool)
    for t in range(n_steps):
        end = min(t + chunk_size, n_steps)
        valid = end - t
        chunks[t, :valid] = actions[t:end]
        chunks[t, valid:] = actions[end - 1]
        mask[t, :valid] = True
    return chunks, mask


class DatasetWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int = 32,
        encoder_name: str = "unknown",
        embed_dim: int = 512,
        image_size: tuple[int, int] = IMAGE_SIZE,
        compression: str | None = "gzip",
        overwrite: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # h5py's "w" mode truncates without complaint. Two collection runs
        # pointed at one path will then interleave and leave the .hdf5 and its
        # JSON report describing different things, with nothing to flag it.
        # Refuse by default rather than silently clobber.
        if self.path.exists() and not overwrite:
            raise FileExistsError(
                f"{self.path} already exists. Pass --overwrite to replace it "
                f"(or choose another --out); refusing to truncate silently."
            )
        self.chunk_size = chunk_size
        self.compression = compression
        self._file = h5py.File(self.path, "w")
        self._n = 0
        self._closed = False

        meta = self._file.create_group("metadata")
        meta.attrs["fps"] = float(RECORD_HZ)
        meta.attrs["action_chunk_size"] = int(chunk_size)
        meta.attrs["n_dof"] = int(N_DOF)
        meta.attrs["image_height"] = int(image_size[0])
        meta.attrs["image_width"] = int(image_size[1])
        meta.attrs["text_encoder"] = encoder_name
        meta.attrs["language_embedding_dim"] = int(embed_dim)
        meta.attrs["phases"] = np.array(PHASES, dtype=h5py.string_dtype())
        meta.attrs["action_space"] = "target joint positions (5 arm + 1 gripper), radians"
        meta.attrs["frame_alignment"] = (
            "action[t] is the command issued from obs[t]; obs[t+1] is the result"
        )

    # ------------------------------------------------------------------
    def add_episode(self, episode: Episode, embedding: np.ndarray) -> str:
        name = f"episode_{self._n:06d}"
        grp = self._file.create_group(name)
        obs = grp.create_group("obs")

        def img(dset_name: str, array: np.ndarray):
            obs.create_dataset(
                dset_name,
                data=array,
                dtype="uint8",
                compression=self.compression,
                chunks=(1,) + array.shape[1:],
            )

        img("scene_image", episode.scene_image)
        img("wrist_image", episode.wrist_image)
        obs.create_dataset("qpos", data=episode.qpos, dtype="float32")
        obs.create_dataset("qvel", data=episode.qvel, dtype="float32")
        obs.create_dataset("tcp_pose", data=episode.tcp_pose, dtype="float32")

        grp.create_dataset("action", data=episode.action, dtype="float32")
        chunks, mask = build_action_chunks(episode.action, self.chunk_size)
        grp.create_dataset("action_chunk", data=chunks, dtype="float32",
                           compression=self.compression)
        grp.create_dataset("action_chunk_mask", data=mask, dtype=bool,
                           compression=self.compression)
        grp.create_dataset("phase", data=episode.phase_ids, dtype="int8")
        grp.create_dataset("object_poses", data=episode.object_poses, dtype="float32")

        grp.create_dataset("language_instruction", data=episode.instruction,
                           dtype=h5py.string_dtype())
        grp.create_dataset("language_embedding", data=np.asarray(embedding, np.float32))

        spec = episode.spec
        grp.attrs["seed"] = int(spec.seed)
        grp.attrs["instruction"] = episode.instruction
        grp.attrs["success"] = bool(episode.success.success)
        grp.attrs["success_reason"] = episode.success.reason
        grp.attrs["episode_length"] = len(episode)
        grp.attrs["target_object"] = spec.target_object.name
        grp.attrs["target_object_label"] = spec.target_object.label
        grp.attrs["target_zone"] = spec.target_zone.name
        grp.attrs["target_zone_label"] = spec.target_zone.label
        grp.attrs["object_names"] = np.array([o.name for o in spec.objects],
                                             dtype=h5py.string_dtype())
        grp.attrs["object_labels"] = np.array([o.label for o in spec.objects],
                                              dtype=h5py.string_dtype())
        grp.attrs["zone_labels"] = np.array([z.label for z in spec.zones],
                                            dtype=h5py.string_dtype())
        grp.attrs["zone_positions"] = np.array([z.pos for z in spec.zones], dtype=np.float32)
        for key, value in episode.success.to_dict().items():
            grp.attrs[f"eval_{key}"] = value

        self._n += 1
        return name

    # ------------------------------------------------------------------
    def finalize(self, extra: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._file["metadata"].attrs["n_episodes"] = self._n
        for key, value in (extra or {}).items():
            self._file["metadata"].attrs[key] = value
        self._file.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finalize()
