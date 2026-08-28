"""Generate the language-conditioned demonstration dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
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
    p.add_argument("--skip-existing", action="store_true",
                   help="exit successfully if --out already exists; lets a batch "
                        "driver be re-run without redoing finished shards")
    p.add_argument("--plain-log", action="store_true",
                   help="one timestamped line per episode instead of a progress "
                        "bar. Auto-enabled when stdout is not a terminal, so "
                        "piping to tee produces a readable log.")
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


def _now() -> str:
    """Local wall-clock stamp for the log (tz-aware, so it survives DST)."""
    return datetime.now().astimezone().strftime("%H:%M:%S")


class _Progress:
    """Progress reporting that survives being piped to a file.

    tqdm redraws with carriage returns, which turns a tee'd log into one
    enormous line. When stdout is not a terminal we emit a timestamped line per
    episode instead, carrying the running rate and an ETA.
    """

    def __init__(self, seeds, plain: bool):
        self.seeds = list(seeds)
        self.total = len(self.seeds)
        self.plain = plain
        self.t0 = time.time()
        self.done = 0
        self._bar = None
        if not plain:
            self._bar = tqdm(total=self.total, desc="episodes")
        else:
            print(f"[{_now()}] collecting {self.total} episodes "
                  f"(seeds {self.seeds[0]}..{self.seeds[-1]})", flush=True)

    def update(self, seed: int, note: str) -> None:
        self.done += 1
        if self._bar is not None:
            self._bar.update(1)
            return
        elapsed = time.time() - self.t0
        rate = elapsed / self.done
        eta = timedelta(seconds=int(rate * (self.total - self.done)))
        print(f"[{_now()}] {self.done:5d}/{self.total} "
              f"seed {seed:<6d} {note:<34s} {rate:5.1f}s/ep  ETA {eta}", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_size = (args.image_height, args.image_width)

    if args.skip_existing and args.out.exists():
        print(f"{args.out} already exists -- skipping (--skip-existing)")
        return 0
    plain = args.plain_log or not sys.stdout.isatty()

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
        progress = _Progress(seeds, plain)
        for seed in seeds:
            stats["attempted"] += 1
            try:
                episode = runner.run(sample_episode(seed))
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                per_episode.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                progress.update(seed, f"ERROR {type(exc).__name__}")
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
            progress.update(
                seed,
                ("ok  " if episode.success.success else "FAIL") +
                f" centre {episode.success.center_distance * 1000:5.1f}mm",
            )
        progress.close()

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
