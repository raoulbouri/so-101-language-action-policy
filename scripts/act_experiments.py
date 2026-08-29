#!/usr/bin/env python3
"""Run E1-E6 against trained checkpoints.

Offline metrics and the language-sensitivity tests (E3/E4) are cheap and always
run. Closed-loop rollouts are expensive, so they are opt-in via --rollouts.

    python scripts/act_experiments.py --runs runs/e1_baseline runs/e2_clip \
        --out runs/experiments.json --rollouts 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from so101_act.config import Config
from so101_act.data import (
    Normalizer,
    SO101ACTDataset,
    collate,
    make_splits,
    scan_episodes,
)
from so101_act.evaluate import counterfactual_divergence, offline_metrics
from so101_act.model import build_model
from so101_act.train import pick_device


def load_run(run_dir: Path, device):
    cfg = Config.load(run_dir / "config.json")
    norm = Normalizer.load(run_dir / "norm_stats.json")
    ckpt_path = run_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "ckpt_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, norm, model, ckpt.get("step", -1), ckpt_path.name


def make_loader(cfg, episodes, idx, norm, *, shuffle_language=False, bs=16, workers=0):
    ds = SO101ACTDataset(cfg.hdf5_path, episodes, idx, norm,
                         chunk_size=cfg.chunk_size, cameras=cfg.cameras,
                         conditioning=cfg.conditioning,
                         shuffle_language=shuffle_language, language_seed=cfg.seed)
    return ds, DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                          collate_fn=collate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/experiments.json")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--cf-samples", type=int, default=120)
    ap.add_argument("--rollouts", type=int, default=0,
                    help="closed-loop episodes per run (0 = offline only)")
    ap.add_argument("--wrong-language-rollouts", type=int, default=0)
    a = ap.parse_args(argv)

    device = pick_device()
    report: dict = {"device": str(device), "runs": {}}

    for rd in a.runs:
        rd = Path(rd)
        cfg, norm, model, step, ckpt_name = load_run(rd, device)
        episodes, _ = scan_episodes(cfg.hdf5_path)
        splits = make_splits(episodes, mode=cfg.split_mode, val_frac=cfg.val_frac,
                             test_frac=cfg.test_frac,
                             holdout_combos=cfg.holdout_combos, seed=cfg.seed)
        name = rd.name
        entry: dict = {"conditioning": cfg.conditioning, "split_mode": cfg.split_mode,
                       "checkpoint": ckpt_name, "step": step,
                       "n_test_episodes": len(splits["test"])}
        print(f"\n=== {name}  ({cfg.conditioning}, {ckpt_name} @ step {step}) ===")

        # --- offline (E1 / E2 / E5 depending on the run) -------------------
        _, test_loader = make_loader(cfg, episodes, splits["test"], norm)
        entry["offline_test"] = offline_metrics(
            model, test_loader, device, cfg.kl_weight, norm, a.max_batches)
        print(f"  masked L1 (test)          {entry['offline_test']['masked_l1']:.4f}")

        if cfg.use_language:
            # --- E3: shuffled language ------------------------------------
            _, shuf_loader = make_loader(cfg, episodes, splits["test"], norm,
                                         shuffle_language=True)
            entry["E3_shuffled_language"] = offline_metrics(
                model, shuf_loader, device, cfg.kl_weight, norm, a.max_batches)
            d = entry["E3_shuffled_language"]["masked_l1"]
            base = entry["offline_test"]["masked_l1"]
            entry["E3_degradation"] = d - base
            entry["E3_relative_degradation"] = (d - base) / max(base, 1e-9)
            print(f"  masked L1 (shuffled lang) {d:.4f}   "
                  f"degradation {d-base:+.4f} ({100*(d-base)/max(base,1e-9):+.1f}%)")

            # --- E4: counterfactual divergence ----------------------------
            ds, _ = make_loader(cfg, episodes, splits["test"], norm)
            entry["E4_counterfactual"] = counterfactual_divergence(
                model, ds, device, n_samples=a.cf_samples, seed=cfg.seed)
            e4 = entry["E4_counterfactual"]
            print(f"  E4 D(language swap)       {e4['D_language_swap']:.4f}  "
                  f"(same-language control {e4['D_same_language_control']:.6f})")

        # --- closed loop ---------------------------------------------------
        if a.rollouts:
            import h5py

            from so101_act.rollout import rollout_episode, summarize
            with h5py.File(cfg.hdf5_path, "r") as f:
                emb = {episodes[i].name: f[episodes[i].name]["language_embedding"][:]
                       for i in splits["test"][:a.rollouts]}
            rows = []
            for i in splits["test"][:a.rollouts]:
                info = episodes[i]
                rows.append(rollout_episode(
                    model, norm, device, info.seed,
                    language_embedding=emb[info.name], task_id=info.task_id))
            entry["closed_loop"] = summarize(rows)
            entry["closed_loop_rows"] = rows
            print(f"  closed-loop success       "
                  f"{entry['closed_loop']['success_rate']:.1%} over {len(rows)}")

        report["runs"][name] = entry

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
