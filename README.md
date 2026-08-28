# so-101-sim

Language-conditioned imitation-learning data generation for the **LeRobot SO-101**
arm in MuJoCo.

Simulates an open-ended tabletop manipulation matrix —
*"Take [COLOR] [SHAPE] and place in [COLOR] [ZONE]"* — and emits an HDF5
dataset shaped for a language-conditioned **ACT** (Action Chunking with
Transformers) policy.

```
scene randomization -> language instruction -> DLS-IK expert -> 7-phase
quintic trajectory -> 30 Hz recording (scene cam + wrist cam + qpos + action)
-> HDF5 + frozen text embedding -> binary success validator
```

## Results

| Metric | Value |
| --- | --- |
| Expert success, held-out seeds 10000–10099 | **100 / 100** |
| Expert success, seeds 0–249 | **250 / 250** |
| Placement centre error | median 2.7 mm, p90 4.6 mm |
| Tests | 44 passing |
| Generated dataset | 40 episodes, 17 760 timesteps, 40/40 success, 15 distinct instructions |

Success is the directive's strict criterion: **every** base corner of the cube
inside the zone perimeter, cube resting on the table and at rest.

## Quickstart

```bash
make setup                 # uv venv on Python 3.13 + install with dev,text extras
make test                  # 43 unit and property tests
make eval                  # 100-seed expert success evaluation
make preview SEED=5        # render one episode to data/episode.mp4
make collect N=100 WORKERS=4
make inspect
```

Or directly:

```bash
uv run python -m so101_sim.cli.collect --num-episodes 100 --num-workers 4 \
    --out data/so101_lang_act.hdf5 --report data/collection_report.json
uv run python -m so101_sim.cli.evaluate --num-seeds 100
uv run python -m so101_sim.cli.visualize --seed 5
```

## What the robot actually does

The SO-101 is a **5-DoF** arm, so a general 6-DoF pose is unreachable. The
expert only ever commands *top-down* grasps, which is exactly the family the
arm can hit precisely (`shoulder_pan` picks the plane, three joints work in it,
`wrist_roll` spins the jaws). The usable straight-down workspace is a measured
annulus of r ∈ [0.16, 0.26] m — not the 0.478 m raw kinematic reach. See
[docs/DECISIONS.md](docs/DECISIONS.md) D-003 and D-004.

## Verifying the data

```bash
make verify                 # automated health checks on the stored dataset
make replay EP=0            # render a STORED episode with overlays
make viewer SEED=5          # interactive MuJoCo viewer (geometry inspection)
make inspect                # schema and summary
```

`replay_dataset.py` and `verify_dataset.py` read **only what is on disk** — they
never re-simulate. A policy trains on the bytes in the file, and those are what
can be silently wrong.

## Documentation

| File | Purpose |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | Working agreement for coding agents on this repo |
| [MEMORY.md](MEMORY.md) | Durable project facts that survive across sessions |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Running build log — what is done, in flight, not started |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map and data flow |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Design decisions and their rationale |
| [docs/ISSUES.md](docs/ISSUES.md) | Known bugs, misconceptions, and open risks |
| [docs/DATASET_SPEC.md](docs/DATASET_SPEC.md) | Exact on-disk HDF5 schema |
| [docs/README.md](docs/README.md) | Index of the above |

## Scope note

This repo is the **data-generation pipeline**. The directive's final section
describes modifying an ACT *training* codebase to consume `language_embedding`
as a conditioning token; no ACT model is vendored or trained here. The dataset is
shaped to make that change direct, and `scripts/act_dataloader_example.py` plus
`docs/DATASET_SPEC.md` pin down exactly how the embedding is meant to enter the
CVAE encoder and transformer decoder. See `docs/PROGRESS.md` for the full
in-scope / out-of-scope list.

## Assets

Robot model is the official `robotstudio_so101` MJCF from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
vendored under `assets/so101/` (Apache-2.0, license retained).
