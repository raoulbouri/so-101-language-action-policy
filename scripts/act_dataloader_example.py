#!/usr/bin/env python3
"""Reference PyTorch Dataset showing how this HDF5 feeds a language-conditioned ACT.

This is deliberately a *reference*, not a training script -- no ACT model is
vendored in this repo. It exists to pin down the contract: what a sample looks
like, and where `language_embedding` is meant to enter the network.

Where the language token goes (per the directive):

    z ~ CVAE_encoder([cls, lang_token, qpos_token, action_tokens...])
    a_hat = Transformer_decoder(query=chunk_positions,
                                memory=[visual_tokens..., lang_token, qpos_token, z])

i.e. the same projected embedding is prepended as one extra token to BOTH the
CVAE encoder input and the decoder memory. It is frozen -- only the projection
into the model width is learned.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - reference code
    torch = None
    Dataset = object


class SO101LangACTDataset(Dataset):
    """One sample per timestep: (images, qpos, language) -> action chunk."""

    def __init__(self, path: str, camera_keys=("scene_image", "wrist_image")):
        self.path = path
        self.camera_keys = camera_keys
        self._file: h5py.File | None = None
        with h5py.File(path, "r") as f:
            self.chunk_size = int(f["metadata"].attrs["action_chunk_size"])
            self.text_encoder = f["metadata"].attrs["text_encoder"]
            self.index: list[tuple[str, int]] = [
                (name, t)
                for name in sorted(k for k in f if k.startswith("episode_"))
                for t in range(f[name]["action"].shape[0])
            ]
        if str(self.text_encoder) == "hashing-fallback":
            print("WARNING: this dataset was built with the hashing fallback "
                  "encoder. Its language embeddings carry no semantics.")

    def _handle(self) -> h5py.File:
        # Opened lazily so the dataset survives fork-based DataLoader workers.
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        name, t = self.index[i]
        g = self._handle()[name]

        images = np.stack([g[f"obs/{k}"][t] for k in self.camera_keys])  # (V,H,W,3)
        images = images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0  # (V,3,H,W)

        sample = {
            "images": images,                              # (V, 3, H, W)
            "qpos": g["obs/qpos"][t].astype(np.float32),    # (6,)
            "action_chunk": g["action_chunk"][t].astype(np.float32),   # (k, 6)
            "is_pad": ~g["action_chunk_mask"][t],           # (k,) True where padded
            "language_embedding": g["language_embedding"][:].astype(np.float32),
            "language_instruction": g.attrs["instruction"],
        }
        if torch is not None:
            sample = {
                k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in sample.items()
            }
        return sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    args = ap.parse_args()

    ds = SO101LangACTDataset(args.path)
    print(f"{len(ds)} timesteps, chunk size {ds.chunk_size}, "
          f"encoder {ds.text_encoder}")
    s = ds[0]
    for k, v in s.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"  {k:22s} {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
