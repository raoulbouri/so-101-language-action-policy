"""Render an episode to video for eyeballing the expert."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ..episode_runner import EpisodeRunner
from ..randomization import sample_episode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/episode.mp4"))
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--width", type=int, default=480)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = EpisodeRunner(image_size=(args.height, args.width), render=True)
    ep = runner.run(sample_episode(args.seed))
    runner.close()

    frames = [np.hstack([s, w]) for s, w in zip(ep.scene_image, ep.wrist_image)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=25)
    print(f'seed {args.seed}: "{ep.instruction}"')
    print(f"success={ep.success.success} ({ep.success.reason})  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
