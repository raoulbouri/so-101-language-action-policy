"""Automated success evaluation over N distinct randomized seeds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..episode_runner import EpisodeRunner
from ..randomization import sample_episode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-seeds", type=int, default=100)
    p.add_argument("--start-seed", type=int, default=10_000,
                   help="held out from the default collection range")
    p.add_argument("--render", action="store_true",
                   help="also render cameras (slower; off by default)")
    p.add_argument("--report", type=Path, default=Path("data/eval_report.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = EpisodeRunner(render=args.render)

    rows, reasons = [], Counter()
    for seed in tqdm(range(args.start_seed, args.start_seed + args.num_seeds), desc="eval"):
        try:
            ep = runner.run(sample_episode(seed))
        except Exception as exc:  # noqa: BLE001
            reasons[f"exception: {type(exc).__name__}"] += 1
            rows.append({"seed": seed, "success": False, "reason": str(exc)})
            continue
        reasons[ep.success.reason] += 1
        rows.append({"seed": seed, "instruction": ep.instruction, **ep.success.to_dict()})
    runner.close()

    n = len(rows)
    n_pass = sum(r["success"] for r in rows)
    lenient = sum(r.get("center_in_zone", False) for r in rows)
    centre = np.array([r["center_distance"] for r in rows if "center_distance" in r])

    print(f"\nPASS {n_pass}/{n}  ({n_pass / max(n, 1):.1%})   "
          f"[lenient centre-in-zone: {lenient}/{n}]")
    if centre.size:
        print(f"centre error: median {np.median(centre) * 1000:.1f} mm | "
              f"p90 {np.percentile(centre, 90) * 1000:.1f} mm")
    print("outcome breakdown:")
    for reason, count in reasons.most_common():
        print(f"  {count:4d}  {reason}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({
        "num_seeds": n, "passed": n_pass, "pass_rate": n_pass / max(n, 1),
        "center_in_zone": lenient, "reasons": dict(reasons), "episodes": rows,
    }, indent=2))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
