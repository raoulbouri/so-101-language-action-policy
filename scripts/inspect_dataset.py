#!/usr/bin/env python3
"""Print the structure and a sanity summary of a generated dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"=== {args.path}  ({args.path.stat().st_size / 1e6:.1f} MB) ===\n")
        print("metadata:")
        for k, v in sorted(f["metadata"].attrs.items()):
            print(f"  {k:26s} {v}")

        names = sorted(k for k in f if k.startswith("episode_"))
        print(f"\n{len(names)} episodes\n")

        lengths, successes, instructions = [], [], set()
        for name in names:
            g = f[name]
            lengths.append(g.attrs["episode_length"])
            successes.append(bool(g.attrs["success"]))
            instructions.append if False else instructions.add(g.attrs["instruction"])

        print(f"episode length: min {min(lengths)} max {max(lengths)} "
              f"mean {np.mean(lengths):.0f}")
        print(f"success: {sum(successes)}/{len(successes)}")
        print(f"distinct instructions: {len(instructions)}")

        print(f"\n--- first {args.episodes} episodes ---")
        for name in names[:args.episodes]:
            g = f[name]
            print(f"\n{name}: seed={g.attrs['seed']}  \"{g.attrs['instruction']}\"")
            print(f"  distractor objects: {[s.decode() if isinstance(s, bytes) else s for s in g.attrs['object_labels']]}")
            print(f"  zones             : {[s.decode() if isinstance(s, bytes) else s for s in g.attrs['zone_labels']]}")
            for key in ("obs/scene_image", "obs/wrist_image", "obs/qpos", "obs/qvel",
                        "obs/tcp_pose", "action", "action_chunk", "action_chunk_mask",
                        "phase", "object_poses", "language_embedding"):
                d = g[key]
                print(f"  {key:22s} {d.shape!s:20s} {d.dtype}")
            emb = g["language_embedding"][:]
            print(f"  embedding norm: {np.linalg.norm(emb):.4f}")
            print(f"  centre error  : {g.attrs['eval_center_distance'] * 1000:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
