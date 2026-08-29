#!/usr/bin/env python3
"""Train ACT: baseline (E1), CLIP-conditioned (E2), or task-id (E6).

Must be a real module (not stdin/heredoc) because DataLoader workers are spawned
on macOS and need to re-import __main__.

    python scripts/act_train.py --conditioning none --steps 20000 --out runs/e1_baseline
    python scripts/act_train.py --conditioning clip --steps 20000 --out runs/e2_clip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from so101_act.config import Config
from so101_act.train import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hdf5", default="data/train_1200.hdf5")
    p.add_argument("--conditioning", default="none", choices=["none", "clip", "taskid"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-weight", type=float, default=10.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--split-mode", default="iid", choices=["iid", "compositional"])
    p.add_argument("--holdout", nargs="*", default=[],
                   help="E5 held-out combos, e.g. orange:pink")
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--time-only", type=int, default=0,
                   help="run N steps purely to measure throughput, then exit")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    combos = tuple(tuple(c.split(":")) for c in a.holdout)
    out = a.out or f"runs/act_{a.conditioning}_{a.split_mode}"

    cfg = Config(
        hdf5_path=a.hdf5, conditioning=a.conditioning, num_steps=a.steps,
        batch_size=a.batch_size, lr=a.lr, lr_backbone=a.lr, kl_weight=a.kl_weight,
        num_workers=a.num_workers, seed=a.seed, out_dir=out,
        split_mode=a.split_mode, holdout_combos=combos,
        val_every=a.val_every, log_every=a.log_every,
    )
    trainer = Trainer(cfg)

    if a.time_only:
        import time

        from so101_act.train import move
        dl = trainer.loader("train")
        it = iter(dl)
        for _ in range(3):
            trainer.train_step(move(next(it), trainer.device), 0)
        t0 = time.time()
        for i in range(a.time_only):
            trainer.train_step(move(next(it), trainer.device), i)
        dt = (time.time() - t0) / a.time_only
        print(f"\n{dt:.3f} s/step  batch={cfg.batch_size}  device={trainer.device}")
        for s in (2000, 20000, 100000):
            print(f"  {s:>6d} steps -> {s*dt/3600:5.2f} h")
        return 0

    summary = trainer.fit()
    print("\n" + json.dumps(summary, indent=2))
    Path(out, "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
