#!/usr/bin/env python3
"""Sweep the open-loop execution horizon on ALREADY-TRAINED checkpoints.

The policy predicts k=100 actions, but error grows steeply with horizon
(h=99 is ~5.8x h=0). Executing all 100 open-loop may therefore be worse than
replanning sooner. `n_action_steps` controls exactly that, and it is an
inference-time choice -- no retraining, no re-collection.

    python scripts/act_exec_sweep.py --run runs/e2_clip_k100 \
        --n-action-steps 1 10 25 50 100 --rollouts 20

Only closed-loop rollout depends on this setting, so offline metrics are not
recomputed. Every value is evaluated on the SAME seeds so the comparison is
paired rather than confounded by scene difficulty.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from so101_act.config import Config
from so101_act.data import Normalizer, make_splits, scan_episodes
from so101_act.model import build_model, upgrade_legacy_state_dict
from so101_act.rollout import rollout_episode, summarize
from so101_act.train import pick_device
from so101_sim.randomization import sample_episode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory with ckpt_best.pt")
    ap.add_argument("--n-action-steps", type=int, nargs="+",
                    default=[1, 10, 25, 50, 100],
                    help="execution horizons to sweep (1 = replan every step)")
    ap.add_argument("--with-ensembling", action="store_true",
                    help="additionally evaluate temporal ensembling")
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rd = Path(a.run)
    cfg = Config.load(rd / "config.json")
    norm = Normalizer.load(rd / "norm_stats.json")
    device = pick_device()
    ckpt_path = rd / "ckpt_best.pt"
    if not ckpt_path.exists():
        ckpt_path = rd / "ckpt_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(upgrade_legacy_state_dict(ckpt["model"]))
    model.eval()

    episodes, _ = scan_episodes(cfg.hdf5_path)
    splits = make_splits(episodes, mode=cfg.split_mode, val_frac=cfg.val_frac,
                         test_frac=cfg.test_frac,
                         holdout_combos=cfg.holdout_combos, seed=cfg.seed)
    # Same seeds for every setting -> paired comparison.
    sel = splits["test"][:a.rollouts]
    with h5py.File(cfg.hdf5_path, "r") as f:
        emb = {episodes[i].name: f[episodes[i].name]["language_embedding"][:]
               for i in sel}

    print(f"=== {rd.name}  ({cfg.conditioning}, k={cfg.chunk_size}, "
          f"{ckpt_path.name} @ step {ckpt.get('step')}) ===")
    print(f"{a.rollouts} episodes per setting, identical seeds\n")
    print(f"{'setting':22s} {'success':>9s} {'centre p50':>11s} {'centre min':>11s} {'cube moved':>11s}")

    settings = [(f"n_action_steps={n}",
                 {"use_ensembling": False, "n_action_steps": n})
                for n in a.n_action_steps]
    if a.with_ensembling:
        settings.append(("temporal ensembling",
                         {"use_ensembling": True, "n_action_steps": None}))

    results = {}
    for label, kw in settings:
        t0 = time.time()
        rows = []
        for i in sel:
            info = episodes[i]
            rows.append(rollout_episode(
                model, norm, device, info.seed,
                language_embedding=emb[info.name], task_id=info.task_id, **kw))
        s = summarize(rows)
        cd = np.array([r["center_distance"] for r in rows]) * 1000
        # Did the cube move at all? Compare against its start-to-zone distance.
        moved = []
        for r in rows:
            spec = sample_episode(r["seed"])
            d0 = np.linalg.norm(np.array(spec.target_object.pos)
                                - np.array(spec.target_zone.pos)) * 1000
            moved.append(abs(d0 - r["center_distance"] * 1000) > 5.0)
        results[label] = {"summary": s, "rows": rows,
                          "frac_cube_moved": float(np.mean(moved)),
                          "centre_p50_mm": float(np.median(cd)),
                          "centre_min_mm": float(cd.min()),
                          "seconds": time.time() - t0}
        print(f"{label:22s} {s['success_rate']:8.1%} {np.median(cd):10.1f}mm "
              f"{cd.min():10.1f}mm {np.mean(moved):10.1%}")

    out = Path(a.out or (rd / "exec_sweep.json"))
    out.write_text(json.dumps(
        {"run": str(rd), "conditioning": cfg.conditioning,
         "chunk_size": cfg.chunk_size, "step": ckpt.get("step"),
         "n_rollouts": a.rollouts, "results": results}, indent=2, default=float))
    print(f"\nwrote {out}")
    print("\nNote: 'cube moved' is the leading indicator. Success can stay 0 % while")
    print("the policy starts actually contacting the cube -- that is real progress")
    print("and would not show up in the success column alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
