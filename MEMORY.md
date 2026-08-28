# MEMORY.md — durable facts about this project

Things a future session would otherwise waste time re-deriving. Facts only;
narrative belongs in `docs/`.

## Environment

- Python **3.13** (Homebrew ARM64) via `uv`, venv at `.venv/`, deps in `pyproject.toml`.
- MuJoCo **3.12.0**, arm64. Renders offscreen on macOS with no extra setup.
- `torch` + `transformers` are in the optional `text` extra, used only for the
  frozen CLIP text encoder.

## Robot facts (measured, not assumed)

- Model: menagerie `robotstudio_so101`, vendored at `assets/so101/`. Apache-2.0.
- The arm is **5-DoF** (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll`) plus a `gripper` joint. `nq = nu = 6`.
- A general 6-DoF pose is therefore **not reachable**. The reachable family:
  `shoulder_pan` picks the vertical plane, three joints work in that plane, and
  `wrist_roll` spins the jaws about the tool axis. For a *straight-down*
  approach the wrist axis is vertical, so the full 6-DoF target is consistent —
  which is why the expert only ever commands top-down grasps.
- Raw kinematic reach is 0.478 m radially, but the usable **straight-down**
  workspace is only **r ∈ [0.16, 0.26] m, |yaw| ≤ 60°, z ≤ ~0.095 m**. Above
  z ≈ 0.10 m a straight-down approach is unreachable at *any* radius.
- Gripper joint → jaw separation: `-0.17 → 10.8 mm`, `0.0 → 21.7 mm`,
  `0.5 → 55.1 mm`, `1.74 → 120.5 mm`.
- Actuator force limit is ±2.94 N·m and it **does** saturate. A saturated
  actuator silently tracks its command badly — always check
  `data.actuator_force` before blaming the IK.
- The gripper body frame: local **+x** is the jaw opening direction, local
  **−z** points out along the fingers. The fixed jaw's contact pads sit at
  local x ≈ −0.010; its tip geoms reach ~22 mm beyond the TCP.

## Calibrated constants (do not guess these)

- `TCP_LOCAL_POS = (0.005, 0, -0.082)`, `quat = (0,1,0,0)`. The **x** offset is
  the critical one: below 0.004 the fixed jaw spawns inside the cube. Measured
  success over 8 seeds: `x=0.002 → 3/8`, `x≥0.004 → 8/8`.
- `GRASP_Z = 0.0235` is set by **jaw-tip clearance**, not cube mid-height.
  Aiming at the cube centre (0.0165) buries the tip 5.7 mm in the table.
- `HOME_QPOS = [0, -1.2, 0.4, 1.2, 0, GRIPPER_OPEN]` → TCP at (0.198, 0, 0.233).
  A low home pose parks the gripper *on the table* among the cubes.

## Current status

- Expert success: **100/100** on held-out seeds 10000–10099 (strict footprint
  criterion), median centre error 3.0 mm, p90 4.6 mm.
- Throughput: **0.2–0.4 s/episode** physics-only; **≈10 s/episode** with two
  240×320 cameras on an idle machine. Rendering is ~98% of collection cost.
- Multiprocess collection gives only **1.3×** at 4 workers on macOS — MuJoCo's
  offscreen renderer contends across processes. Expect better on Linux/EGL.
- **Check `uptime` before trusting any timing measurement here.** Orphaned pool
  workers from a killed parent once pushed load average to 28 and made every
  benchmark read ~15× slow, nearly producing a fabricated root cause.
- The generated dataset is `data/so101_lang_act.hdf5` (gitignored, 851 MB):
  40 episodes / seeds 0–39, 444 steps each, 17 760 timesteps, 40/40 success,
  15 distinct instructions, real CLIP ViT-B/32 embeddings.
- `DatasetWriter` refuses to open an existing path without `overwrite=True`
  (`--overwrite`), after two concurrent runs left the HDF5 and its JSON report
  disagreeing (ISSUE-006).
- `transformers` 5.x: `CLIPModel.get_text_features` returns a model-output
  object, not a tensor. Use `CLIPTextModelWithProjection(...).text_embeds`.
- MuJoCo `data.site_xpos` and friends return **live views**, not copies. Always
  `.copy()` before comparing state across steps.
