# CLAUDE.md — working agreement for agents on this repo

Read this before touching anything. It exists so a coding agent does not
re-introduce bugs that have already been found and fixed here.

## What this project is

An automated MuJoCo pipeline that generates a **language-conditioned imitation
learning dataset** for the LeRobot **SO-101** arm. Task family:
`"take the [COLOR] [SHAPE] and place it in the [COLOR] [ZONE]"`. Output is an
HDF5 dataset shaped for a language-conditioned **ACT** policy.

## Hard rules

1. **Never hand-write robot MJCF.** The arm model is the vendored official
   `robotstudio_so101` package from `google-deepmind/mujoco_menagerie` under
   `assets/so101/`. Do not edit those files. The scene is built by *parsing and
   extending* that XML in `scene_builder.py`.
2. **No Denavit-Hartenberg tables, ever.** All kinematics go through MuJoCo's
   analytical API: `mj_jacSite`, `mj_kinematics`, `site_xpos`, `site_xmat`.
3. **Never change a physical constant without measuring first.** Every number in
   `constants.py` that matters carries a comment with the measurement that
   produced it. If you change one, re-run the measurement and update the
   comment. Guessing these numbers is how all three logged bugs happened.
4. **Do not widen the workspace without re-running the reachability test.**
   `tests/test_kinematics.py::test_ik_converges_across_the_declared_workspace`
   is the guard. The SO-101 is a **5-DoF** arm; a straight-down approach is only
   reachable in a narrow radius band that shrinks with height.
5. **Preserve frame alignment.** `action[t]` is the command issued *from*
   `obs[t]`. Observation is recorded before the command is applied. Control and
   recording run at the same rate on purpose — do not decouple them.
6. **Update the docs in the same change.** See below.

## The docs you must keep current

| File | Update when |
| --- | --- |
| `docs/PROGRESS.md` | Any component changes state (not started / in progress / done) |
| `docs/DECISIONS.md` | You make a design choice with a real alternative |
| `docs/ISSUES.md` | You find OR fix a bug, or spot a risk |
| `docs/DATASET_SPEC.md` | The on-disk schema changes in any way |
| `MEMORY.md` | A durable fact is established that a future session would otherwise re-derive |

Append, don't rewrite history. A fixed issue gets marked RESOLVED with the fix,
it does not get deleted — the record of *why* a number is what it is is the
point.

## Architecture in one paragraph

`randomization.py` turns a seed into an `EpisodeSpec` (which cubes, which zones,
where, and which pair the instruction names). `scene_builder.py` compiles that
spec plus the menagerie MJCF into an `MjModel`. `kinematics.py` provides FK and
a damped-least-squares IK on the `tcp_site`. `expert_policy.py` plans the
7-phase quintic trajectory. `episode_runner.py` executes it, records aligned
frames, and scores it via `evaluation.py`. `language.py` embeds the instruction
with a frozen CLIP text tower. `recorder.py` writes HDF5.

## Verifying a change

```bash
uv run pytest -q                                          # unit + property tests
uv run python -m so101_sim.cli.evaluate --num-seeds 100   # must stay at 100/100
uv run python -m so101_sim.cli.visualize --seed 5         # eyeball it
```

**Look at the video.** Three of the bugs in `docs/ISSUES.md` were invisible in
the pass/fail number and obvious in the render.
