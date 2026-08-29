#!/usr/bin/env python3
"""Tiny-subset overfit: validation item 9, and the gate before full training.

If a model with ~84M parameters cannot drive the loss towards zero on a handful
of timesteps it has seen hundreds of times, something is structurally wrong --
a detached graph, a broken mask, a frozen module, a bad normalization. Catching
that here costs a minute; catching it after a multi-hour run costs the run.
"""

from __future__ import annotations

import argparse
import time

import torch

from so101_act.config import Config
from so101_act.data import (
    SO101ACTDataset,
    collate,
    compute_normalizer,
    make_splits,
    scan_episodes,
)
from so101_act.model import act_loss, build_model
from so101_act.train import move, pick_device, set_seed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hdf5", default="data/train_1200.hdf5")
    ap.add_argument("--conditioning", default="clip", choices=["none", "clip", "taskid"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-timesteps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="final/initial action-loss ratio required to pass")
    args = ap.parse_args()

    cfg = Config(hdf5_path=args.hdf5, conditioning=args.conditioning,
                 batch_size=args.batch_size, lr=args.lr)
    set_seed(cfg.seed)
    device = pick_device()

    episodes, _ = scan_episodes(cfg.hdf5_path)
    splits = make_splits(episodes, seed=cfg.seed)
    norm = compute_normalizer(cfg.hdf5_path, episodes, splits["train"], max_episodes=40)

    ds = SO101ACTDataset(cfg.hdf5_path, episodes, splits["train"][:1], norm,
                         conditioning=cfg.conditioning)
    # A fixed handful of timesteps from ONE episode.
    fixed = collate([ds[i] for i in range(0, args.n_timesteps * 20, 20)])
    fixed = move(fixed, device)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"device={device} conditioning={args.conditioning} "
          f"overfitting {fixed['qpos'].shape[0]} timesteps from 1 episode")
    model.train()
    first = None
    t0 = time.time()
    for step in range(args.steps):
        out = model(fixed)
        L = act_loss(out, fixed, cfg.kl_weight)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if first is None:
            first = L["action_loss"].item()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"  [{step:4d}] action {L['action_loss'].item():8.5f} "
                  f"kl {L['kl'].item():8.4f} total {L['loss'].item():9.4f}", flush=True)

    # Deterministic inference path (z = 0) is what actually matters at rollout.
    model.eval()
    with torch.no_grad():
        final = act_loss(model(fixed), fixed, cfg.kl_weight)["action_loss"].item()

    ratio = final / max(first, 1e-9)
    print(f"\ninitial action loss (train-mode) : {first:.5f}")
    print(f"final   action loss (eval, z=0)  : {final:.5f}")
    print(f"ratio                            : {ratio:.4f}  (need < {args.threshold})")
    print(f"elapsed                          : {time.time()-t0:.0f}s")
    ok = ratio < args.threshold
    print("\nGATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
