#!/usr/bin/env python3
"""Exit 0 if a shard is a COMPLETE collection run, 1 otherwise.

A shard killed mid-run is not obviously broken: h5py still closes the file, so
it opens fine and reports a plausible `n_episodes`. Resuming on file-existence
alone therefore skips it and silently yields a dataset with a hole in it.

The reliable marker is `episodes_attempted`, which is written only by the
successful-completion path (`DatasetWriter.finalize(extra=...)`). An interrupted
shard has `n_episodes` but not `episodes_attempted`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py


def shard_complete(path: Path, expected: int | None = None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as f:
            if "metadata" not in f:
                return False, "no metadata group"
            attrs = f["metadata"].attrs
            if "episodes_attempted" not in attrs:
                n = int(attrs.get("n_episodes", -1))
                return False, f"interrupted mid-run (has {n} episodes, never finalized)"
            attempted = int(attrs["episodes_attempted"])
            if expected is not None and attempted != expected:
                return False, f"attempted {attempted}, expected {expected}"
            return True, f"complete ({int(attrs.get('n_episodes', 0))} episodes kept)"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable: {type(exc).__name__}"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: shard_complete.py <shard.hdf5> [expected_episodes]")
        return 2
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ok, reason = shard_complete(Path(sys.argv[1]), expected)
    print(reason)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
