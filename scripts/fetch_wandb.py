#!/usr/bin/env python3
"""Pull training curves from Weights & Biases and summarise them.

Reads the API key from WANDB_API_KEY (never a command-line flag -- a key on the
command line lands in shell history and in `ps` output on a shared machine).

    export WANDB_API_KEY=...
    python scripts/fetch_wandb.py --entity <you> --project so101-act

Writes the full history to JSON so the analysis is reproducible without
re-querying, and prints a health verdict per run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def summarise(name: str, hist: list[dict], cfg: dict, summary: dict) -> dict:
    """Reduce a run's history to the numbers that decide go/no-go."""
    def series(key):
        return [(r["_step"], r[key]) for r in hist if r.get(key) is not None]

    train = series("train/action_loss")
    val = series("val/action_loss")
    kl = series("train/kl")
    gn = series("train/grad_norm")

    out = {
        "conditioning": cfg.get("conditioning"),
        "steps_logged": max((s for s, _ in train), default=None),
        "n_params": cfg.get("n_params"),
        "final_train_action_loss": train[-1][1] if train else None,
        "best_val_action_loss": min((v for _, v in val), default=None),
        "final_val_action_loss": val[-1][1] if val else None,
        "final_kl": kl[-1][1] if kl else None,
        "final_grad_norm": gn[-1][1] if gn else None,
        "val_curve": [{"step": s, "val_action_loss": v} for s, v in val],
        "summary": {k: v for k, v in summary.items() if not k.startswith("_")},
    }

    # Health checks -- the same ones I would eyeball on the dashboard.
    flags = []
    if val and len(val) >= 2:
        if val[-1][1] > val[0][1]:
            flags.append("val loss ENDED HIGHER than it started")
        half = len(val) // 2
        if half and min(v for _, v in val[half:]) > min(v for _, v in val[:half]):
            flags.append("val loss stopped improving in the second half")
    if kl and kl[-1][1] < 1e-4:
        flags.append("KL collapsed to ~0 (posterior collapse: z carries nothing)")
    if gn and gn[-1][1] > 500:
        flags.append("grad norm still very large at the end")
    if train and val and val[-1][1] > 2 * train[-1][1]:
        flags.append("val >> train (overfitting)")
    out["flags"] = flags
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    ap.add_argument("--project", default="so101-act")
    ap.add_argument("--group", default="", help="filter to one group, e.g. e1-vs-e2")
    ap.add_argument("--out", default="runs/wandb_history.json")
    args = ap.parse_args()

    if not os.environ.get("WANDB_API_KEY") and not Path("~/.netrc").expanduser().exists():
        print("No WANDB_API_KEY set and no ~/.netrc. Export the key first.")
        return 2

    import wandb
    api = wandb.Api(timeout=30)
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    print(f"querying {path}" + (f" (group={args.group})" if args.group else ""))

    filters = {"group": args.group} if args.group else None
    runs = list(api.runs(path, filters=filters))
    if not runs:
        print("no runs found -- check --entity/--project/--group")
        return 1

    payload = {}
    for run in runs:
        hist = list(run.scan_history())
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        payload[run.name] = {
            "id": run.id, "state": run.state, "url": run.url,
            "created": str(run.created_at),
            "config": cfg, "history": hist,
            "analysis": summarise(run.name, hist, cfg, dict(run.summary)),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nwrote {args.out}\n")
    for name, r in payload.items():
        a = r["analysis"]
        print(f"=== {name}  [{r['state']}]  conditioning={a['conditioning']} ===")
        print(f"  steps logged        : {a['steps_logged']}")
        print(f"  final train action  : {a['final_train_action_loss']}")
        print(f"  best  val   action  : {a['best_val_action_loss']}")
        print(f"  final KL            : {a['final_kl']}")
        print(f"  final grad norm     : {a['final_grad_norm']}")
        for f in a["flags"]:
            print(f"  !! {f}")
        if not a["flags"]:
            print("  no health flags")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
