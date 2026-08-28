"""Sharded multiprocess collection.

Rendering, not physics, is the bottleneck: an episode costs ~0.19 s of stepping
and ~10 s of rendering two camera views for every one of its ~430 frames. That
work is embarrassingly parallel across seeds, so each worker simulates a
contiguous block of seeds and writes its own shard file. The parent then merges
the shards, which is pure I/O.

Shards rather than IPC on purpose: an uncompressed episode is ~200 MB of image
data, so shipping episodes back through a pipe would cost more than it saves.

Measured scaling caveat (macOS, M-series, 4 workers, 240x320, two cameras):
rendered episodes go from 10.15 s/ep serial to 7.79 s/ep -- a **1.3x** speedup,
not the ~4x the core count suggests. MuJoCo's offscreen renderer contends across
processes on macOS. Physics-only work (evaluation, `render=False`) parallelises
far better. On Linux with EGL, headless rendering is process-local and this path
should scale much closer to linearly. Default is therefore a single worker.

Text embeddings are computed **once in the parent** and passed down as a plain
{instruction: vector} dict. An instruction is a pure function of the seed, so it
needs no simulation to derive -- and loading a CLIP tower inside every worker
cost more than the parallelism gained back on small runs.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .episode_runner import EpisodeRunner
from .language import CachedEncoder, build_text_encoder
from .randomization import sample_episode
from .recorder import DatasetWriter


def _shard_path(out: Path, index: int) -> Path:
    return out.with_name(f"{out.stem}.part{index:03d}{out.suffix}")


def _collect_shard(job: dict[str, Any]) -> dict[str, Any]:
    seeds: list[int] = job["seeds"]
    out = Path(job["out"])
    image_size = tuple(job["image_size"])

    # Pre-computed in the parent: workers never import torch.
    embeddings: dict[str, np.ndarray] = job["embeddings"]
    runner = EpisodeRunner(image_size=image_size, render=True)

    stats = {"attempted": 0, "succeeded": 0, "written": 0, "errors": 0}
    episodes: list[dict[str, Any]] = []

    with DatasetWriter(out, chunk_size=job["chunk_size"], encoder_name=job["encoder_name"],
                       embed_dim=job["embed_dim"], image_size=image_size,
                       overwrite=True) as writer:
        for seed in seeds:
            stats["attempted"] += 1
            try:
                ep = runner.run(sample_episode(seed))
            except Exception as exc:  # noqa: BLE001
                stats["errors"] += 1
                episodes.append({"seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                continue
            stats["succeeded"] += int(ep.success.success)
            episodes.append({"seed": seed, "instruction": ep.instruction,
                             "length": len(ep), **ep.success.to_dict()})
            if ep.success.success or job["keep_failures"]:
                writer.add_episode(ep, embeddings[ep.instruction])
                stats["written"] += 1
    runner.close()
    return {"path": str(out), "stats": stats, "episodes": episodes}


def merge_shards(shards: list[Path], out: Path, extra: dict[str, Any] | None = None) -> int:
    """Concatenate shard files into one dataset, renumbering episode groups."""
    written = 0
    with h5py.File(out, "w") as dst:
        meta_copied = False
        for shard in shards:
            if not shard.exists():
                continue
            with h5py.File(shard, "r") as src:
                if not meta_copied and "metadata" in src:
                    src.copy("metadata", dst)
                    meta_copied = True
                for name in sorted(k for k in src if k.startswith("episode_")):
                    src.copy(name, dst, name=f"episode_{written:06d}")
                    written += 1
        if "metadata" not in dst:
            dst.create_group("metadata")
        dst["metadata"].attrs["n_episodes"] = written
        for key, value in (extra or {}).items():
            dst["metadata"].attrs[key] = value
    return written


def collect_parallel(
    seeds: list[int],
    out: Path,
    *,
    num_workers: int,
    chunk_size: int,
    image_size: tuple[int, int],
    keep_failures: bool,
    prefer_clip: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists. Pass --overwrite to replace it."
        )
    num_workers = max(1, min(num_workers, len(seeds)))

    # Derive every instruction from its seed and embed them all in one pass,
    # in this process, before any worker starts.
    encoder = CachedEncoder(build_text_encoder(prefer_clip=prefer_clip))
    embeddings: dict[str, np.ndarray] = {}
    for seed in seeds:
        try:
            text = sample_episode(seed).instruction
        except RuntimeError:
            continue
        if text not in embeddings:
            embeddings[text] = encoder.encode_one(text)

    blocks: list[list[int]] = [seeds[i::num_workers] for i in range(num_workers)]
    jobs = [
        {
            "seeds": block,
            "out": str(_shard_path(out, i)),
            "chunk_size": chunk_size,
            "image_size": image_size,
            "keep_failures": keep_failures,
            "embeddings": embeddings,
            "encoder_name": encoder.name,
            "embed_dim": encoder.dim,
        }
        for i, block in enumerate(blocks)
        if block
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(_collect_shard, jobs)

    totals = {"attempted": 0, "succeeded": 0, "written": 0, "errors": 0}
    episodes: list[dict[str, Any]] = []
    for res in results:
        for key in totals:
            totals[key] += res["stats"][key]
        episodes.extend(res["episodes"])

    shard_paths = [Path(r["path"]) for r in results]
    merged = merge_shards(shard_paths, out, extra={
        "episodes_attempted": totals["attempted"],
        "episodes_successful": totals["succeeded"],
        "keep_failures": bool(keep_failures),
    })
    for path in shard_paths:
        path.unlink(missing_ok=True)

    episodes.sort(key=lambda e: e["seed"])
    return {"stats": totals, "merged": merged, "episodes": episodes,
            "encoder": encoder.name}
