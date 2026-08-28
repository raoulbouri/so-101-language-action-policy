"""Generate the language-conditioned demonstration dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from ..constants import IMAGE_SIZE
from ..episode_runner import EpisodeRunner
from ..language import CachedEncoder, build_text_encoder
from ..randomization import sample_episode
from ..recorder import DatasetWriter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/so101_lang_act.hdf5"))
    p.add_argument("--chunk-size", type=int, default=32,
                   help="ACT action-chunk length k")
    p.add_argument("--image-height", type=int, default=IMAGE_SIZE[0])
    p.add_argument("--image-width", type=int, default=IMAGE_SIZE[1])
    p.add_argument("--keep-failures", action="store_true",
                   help="record unsuccessful episodes too (default: drop them)")
    p.add_argument("--no-clip", action="store_true",
                   help="skip CLIP and use the deterministic hashing encoder")
    p.add_argument("--report", type=Path, default=None,
                   help="write a JSON collection report here")
    p.add_argument("--overwrite", action="store_true",
                   help="replace --out if it already exists")
    p.add_argument("--num-workers", type=int, default=1,
                   help="parallel worker processes. NOTE: on macOS MuJoCo's "
                        "offscreen renderer contends across processes -- measured "
                        "only ~1.3x at 4 workers. Expect better on Linux/EGL.")
    return p


def _main_parallel(args, image_size: tuple[int, int]) -> int:
    from ..parallel import collect_parallel

    seeds = list(range(args.start_seed, args.start_seed + args.num_episodes))
    t0 = time.time()
    result = collect_parallel(
        seeds,
        args.out,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        image_size=image_size,
        keep_failures=args.keep_failures,
        prefer_clip=not args.no_clip,
        overwrite=args.overwrite,
    )
    stats = result["stats"]
    elapsed = time.time() - t0
    rate = stats["succeeded"] / max(stats["attempted"], 1)
    print(f"\nwrote {result['merged']} episodes to {args.out} "
          f"({args.num_workers} workers)")
    print(f"expert success rate: {stats['succeeded']}/{stats['attempted']} ({rate:.1%})")
    print(f"errors: {stats['errors']}   elapsed: {elapsed:.1f}s "
          f"({elapsed / max(stats['attempted'], 1):.2f}s/episode)")
    print(f"text encoder: {result['encoder']}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"stats": stats, "elapsed_s": elapsed, "encoder": result["encoder"],
             "num_workers": args.num_workers, "episodes": result["episodes"]}, indent=2))
        print(f"report: {args.report}")
    return 0 if stats["errors"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_size = (args.image_height, args.image_width)

    if args.num_workers > 1:
        return _main_parallel(args, image_size)

    encoder = CachedEncoder(build_text_encoder(prefer_clip=not args.no_clip))
    runner = EpisodeRunner(image_size=image_size, render=True)

    stats = {"attempted": 0, "succeeded": 0, "written": 0, "errors": 0}
    per_episode = []
    t0 = time.time()

    with DatasetWriter(
        args.out,
        chunk_size=args.chunk_size,
        encoder_name=encoder.name,
        embed_dim=encoder.dim,
        image_size=image_size,
        overwrite=args.overwrite,
    ) as writer:
        seeds = range(args.start_seed, args.start_seed + args.num_episodes)
        for seed in tqdm(seeds, desc="episodes"):
            stats["attempted"] += 1
            try:
                episode = runner.run(sample_episode(seed))
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                per_episode.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                continue

            stats["succeeded"] += int(episode.success.success)
            per_episode.append({
                "seed": seed,
                "instruction": episode.instruction,
                "length": len(episode),
                **episode.success.to_dict(),
            })
            if episode.success.success or args.keep_failures:
                writer.add_episode(episode, encoder.encode_one(episode.instruction))
                stats["written"] += 1

        writer.finalize({
            "episodes_attempted": stats["attempted"],
            "episodes_successful": stats["succeeded"],
            "keep_failures": bool(args.keep_failures),
        })
    runner.close()

    elapsed = time.time() - t0
    rate = stats["succeeded"] / max(stats["attempted"], 1)
    print(f"\nwrote {stats['written']} episodes to {args.out}")
    print(f"expert success rate: {stats['succeeded']}/{stats['attempted']} ({rate:.1%})")
    print(f"errors: {stats['errors']}   elapsed: {elapsed:.1f}s "
          f"({elapsed / max(stats['attempted'], 1):.2f}s/episode)")
    print(f"text encoder: {encoder.name} (dim {encoder.dim})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(
            {"stats": stats, "elapsed_s": elapsed, "encoder": encoder.name,
             "episodes": per_episode}, indent=2))
        print(f"report: {args.report}")

    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
