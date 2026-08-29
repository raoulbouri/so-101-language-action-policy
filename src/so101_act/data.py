"""Lazy HDF5 dataset, episode-level splits, and train-only normalization.

Three invariants this module exists to protect, each of which silently ruins
training if broken:

1. **No action shift.** `action_chunk[t]` is stored as `action[t:t+k]`, and
   `action[t]` is the command issued *from* `obs[t]`. We index both at the same
   `t` and never roll either stream.
2. **Episode-level splits.** Timesteps within one episode are near-duplicates;
   splitting at timestep level leaks the test set into training and makes every
   metric meaningless.
3. **Train-only normalization.** Statistics are computed over training episodes
   alone, then applied everywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics: the ResNet18 backbone is pretrained, so inputs must be
# normalised the way it was trained.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_INSTR_RE = re.compile(r"take the (\w+) cube and place it in the (\w+) circle")


def parse_instruction(text: str) -> tuple[str, str]:
    """('take the red cube and place it in the green circle') -> ('red','green')."""
    m = _INSTR_RE.match(text.strip())
    if not m:
        raise ValueError(f"unparseable instruction: {text!r}")
    return m.group(1), m.group(2)


@dataclass
class Normalizer:
    """Per-dimension standardisation for qpos and actions."""

    qpos_mean: np.ndarray
    qpos_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def norm_qpos(self, q):
        return (q - self.qpos_mean) / self.qpos_std

    def norm_action(self, a):
        return (a - self.action_mean) / self.action_std

    def denorm_action(self, a):
        """Used at rollout time to turn network output back into joint targets."""
        if isinstance(a, torch.Tensor):
            mean = torch.as_tensor(self.action_mean, dtype=a.dtype, device=a.device)
            std = torch.as_tensor(self.action_std, dtype=a.dtype, device=a.device)
            return a * std + mean
        return a * self.action_std + self.action_mean

    def to_dict(self) -> dict[str, list[float]]:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        d = json.loads(Path(path).read_text())
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


# ---------------------------------------------------------------------------
# Episode index and splits
# ---------------------------------------------------------------------------
@dataclass
class EpisodeInfo:
    name: str
    seed: int
    length: int
    instruction: str
    cube: str
    zone: str
    task_id: int


def scan_episodes(hdf5_path: str | Path) -> tuple[list[EpisodeInfo], dict[tuple[str, str], int]]:
    """Read episode-level metadata once; no pixel data is touched."""
    with h5py.File(hdf5_path, "r") as f:
        names = sorted(k for k in f if k.startswith("episode_"))
        raw = []
        for n in names:
            g = f[n]
            instr = str(g.attrs["instruction"])
            cube, zone = parse_instruction(instr)
            raw.append((n, int(g.attrs["seed"]), int(g["action"].shape[0]), instr, cube, zone))

    combos = sorted({(c, z) for *_, c, z in raw})
    task_ids = {combo: i for i, combo in enumerate(combos)}
    eps = [
        EpisodeInfo(n, s, L, instr, c, z, task_ids[(c, z)])
        for (n, s, L, instr, c, z) in raw
    ]
    return eps, task_ids


def make_splits(
    episodes: list[EpisodeInfo],
    *,
    mode: str = "iid",
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    holdout_combos: tuple[tuple[str, str], ...] = (),
    seed: int = 0,
) -> dict[str, list[int]]:
    """Episode-level splits. Returns indices into `episodes`.

    `iid` shuffles episodes. `compositional` (E5) sends every episode whose
    (cube, zone) pair is held out to *test*, and splits the rest normally -- so
    the model sees both concepts individually during training but never that
    combination.
    """
    rng = np.random.default_rng(seed)
    n = len(episodes)

    if mode == "compositional":
        if not holdout_combos:
            raise ValueError("compositional split needs holdout_combos")
        hold = {tuple(c) for c in holdout_combos}
        seen_cubes = {e.cube for e in episodes if (e.cube, e.zone) not in hold}
        seen_zones = {e.zone for e in episodes if (e.cube, e.zone) not in hold}
        for cube, zone in hold:
            if cube not in seen_cubes or zone not in seen_zones:
                raise ValueError(
                    f"holdout ({cube},{zone}) removes a concept entirely; "
                    "both must still appear in training via other combinations"
                )
        test = [i for i, e in enumerate(episodes) if (e.cube, e.zone) in hold]
        rest = [i for i, e in enumerate(episodes) if (e.cube, e.zone) not in hold]
        rng.shuffle(rest)
        n_val = round(val_frac * len(rest))
        return {"train": sorted(rest[n_val:]), "val": sorted(rest[:n_val]),
                "test": sorted(test)}

    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = round(test_frac * n)
    n_val = round(val_frac * n)
    return {
        "test": sorted(idx[:n_test].tolist()),
        "val": sorted(idx[n_test:n_test + n_val].tolist()),
        "train": sorted(idx[n_test + n_val:].tolist()),
    }


def compute_normalizer(
    hdf5_path: str | Path,
    episodes: list[EpisodeInfo],
    train_idx: list[int],
    *,
    max_episodes: int = 400,
    seed: int = 0,
) -> Normalizer:
    """Standardisation statistics from TRAINING episodes only.

    Subsamples episodes for speed; qpos/action arrays are tiny so every timestep
    of the sampled episodes is used.
    """
    rng = np.random.default_rng(seed)
    pick = list(train_idx)
    if len(pick) > max_episodes:
        pick = rng.choice(pick, size=max_episodes, replace=False).tolist()

    qs, as_ = [], []
    with h5py.File(hdf5_path, "r") as f:
        for i in pick:
            g = f[episodes[i].name]
            qs.append(g["obs/qpos"][:])
            as_.append(g["action"][:])
    q = np.concatenate(qs).astype(np.float64)
    a = np.concatenate(as_).astype(np.float64)

    # Floor the std so a near-constant joint cannot blow up into huge values.
    return Normalizer(
        qpos_mean=q.mean(0).astype(np.float32),
        qpos_std=np.maximum(q.std(0), 1e-3).astype(np.float32),
        action_mean=a.mean(0).astype(np.float32),
        action_std=np.maximum(a.std(0), 1e-3).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SO101ACTDataset(Dataset):
    """One sample per (episode, timestep).

    The HDF5 handle is opened lazily *per worker process*: h5py handles cannot
    be shared across a fork, and holding one open in the parent then forking is
    the classic way to get silent corruption or a hang.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        episodes: list[EpisodeInfo],
        indices: list[int],
        normalizer: Normalizer,
        *,
        chunk_size: int = 32,
        cameras: tuple[str, ...] = ("scene_image", "wrist_image"),
        conditioning: str = "none",
        shuffle_language: bool = False,
        language_seed: int = 0,
    ):
        self.hdf5_path = str(hdf5_path)
        self.episodes = episodes
        self.indices = list(indices)
        self.norm = normalizer
        self.chunk_size = chunk_size
        self.cameras = tuple(cameras)
        self.conditioning = conditioning
        self.shuffle_language = shuffle_language
        self._f: h5py.File | None = None

        # Flat (episode_index, timestep) index.
        self._index: list[tuple[int, int]] = [
            (ei, t) for ei in self.indices for t in range(episodes[ei].length)
        ]

        # E3: a fixed permutation of episodes, so each observation is paired
        # with another episode's instruction. Deterministic, so the shuffled
        # run is reproducible.
        rng = np.random.default_rng(language_seed)
        perm = rng.permutation(len(self.indices))
        self._lang_swap = {self.indices[i]: self.indices[perm[i]]
                           for i in range(len(self.indices))}

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.hdf5_path, "r")
        return self._f

    def _image(self, g, cam: str, t: int) -> np.ndarray:
        img = g[f"obs/{cam}"][t].astype(np.float32) / 255.0     # (H,W,3)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(img.transpose(2, 0, 1))     # (3,H,W)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | str]:
        ei, t = self._index[i]
        info = self.episodes[ei]
        f = self._handle()
        g = f[info.name]

        sample: dict = {
            cam: torch.from_numpy(self._image(g, cam, t)) for cam in self.cameras
        }
        sample["qpos"] = torch.from_numpy(
            self.norm.norm_qpos(g["obs/qpos"][t].astype(np.float32))
        )
        sample["action_chunk"] = torch.from_numpy(
            self.norm.norm_action(g["action_chunk"][t].astype(np.float32))
        )
        sample["action_chunk_mask"] = torch.from_numpy(
            g["action_chunk_mask"][t].astype(np.bool_)
        )
        sample["phase"] = torch.tensor(int(g["phase"][t]), dtype=torch.long)

        # Language. Under shuffle (E3) the embedding comes from a *different*
        # episode while the observation stays put.
        lang_ei = self._lang_swap[ei] if self.shuffle_language else ei
        lang_info = self.episodes[lang_ei]
        sample["language_embedding"] = torch.from_numpy(
            f[lang_info.name]["language_embedding"][:].astype(np.float32)
        )
        sample["task_id"] = torch.tensor(lang_info.task_id, dtype=torch.long)
        sample["instruction"] = lang_info.instruction
        sample["episode_index"] = torch.tensor(ei, dtype=torch.long)
        sample["timestep"] = torch.tensor(t, dtype=torch.long)
        return sample


def collate(batch: list[dict]) -> dict:
    """Default collation, but keep `instruction` as a plain list of strings."""
    out = {}
    for k in batch[0]:
        if k == "instruction":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
