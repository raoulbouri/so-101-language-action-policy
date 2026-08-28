"""Merge several dataset shards into one file.

Collecting a few thousand episodes in a single process is fragile: one crash,
one killed job, and hours are gone. Collect in batches instead --

    for i in 0 200 400 600 800 1000; do
      make collect N=200 SEED=$i OUT=data/part_$i.hdf5
    done
    python -m so101_sim.cli.merge data/part_*.hdf5 --out data/train.hdf5

-- and merge at the end. Episode groups are renumbered contiguously; metadata is
taken from the first shard and its episode counters are recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def merge(shards: list[Path], out: Path, overwrite: bool = False) -> int:
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists; pass --overwrite to replace it")
    missing = [s for s in shards if not s.exists()]
    if missing:
        raise FileNotFoundError(f"missing shard(s): {missing}")

    written = 0
    attempted = 0
    succeeded = 0
    seen_seeds: set[int] = set()
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out, "w") as dst:
        for i, shard in enumerate(shards):
            with h5py.File(shard, "r") as src:
                if i == 0 and "metadata" in src:
                    src.copy("metadata", dst)
                meta = src["metadata"].attrs if "metadata" in src else {}
                attempted += int(meta.get("episodes_attempted", 0))
                succeeded += int(meta.get("episodes_successful", 0))
                names = sorted(k for k in src if k.startswith("episode_"))
                for name in names:
                    seed = int(src[name].attrs.get("seed", -1))
                    if seed in seen_seeds:
                        print(f"  ! skipping duplicate seed {seed} from {shard.name}")
                        continue
                    seen_seeds.add(seed)
                    src.copy(name, dst, name=f"episode_{written:06d}")
                    written += 1
                print(f"  {shard.name}: +{len(names)} episodes (total {written})")

        if "metadata" not in dst:
            dst.create_group("metadata")
        dst["metadata"].attrs["n_episodes"] = written
        if attempted:
            dst["metadata"].attrs["episodes_attempted"] = attempted
            dst["metadata"].attrs["episodes_successful"] = succeeded
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    n = merge(sorted(args.shards), args.out, args.overwrite)
    size = args.out.stat().st_size / 1e9
    print(f"\nmerged {len(args.shards)} shards -> {args.out} "
          f"({n} episodes, {size:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
