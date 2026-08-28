# Architecture

## Data flow

```
seed (int)
  │
  ├─ randomization.sample_episode ──► EpisodeSpec
  │      2–3 coloured cubes, 2–3 coloured zones, positions in the reachable
  │      annulus, plus which (object, zone) pair the instruction names
  │
  ├─ scene_builder.build_model ─────► mujoco.MjModel
  │      parses assets/so101/so101.xml, injects tcp_site into the gripper body,
  │      adds table + lights + scene camera + the spec's entities
  │
  ├─ expert_policy.ExpertPlanner ───► ExpertTrajectory  (T, 6) joint commands
  │      7 phases, quintic time scaling, DLS IK per Cartesian waypoint
  │
  ├─ episode_runner.EpisodeRunner ──► Episode
  │      steps physics 4x per control step, records obs BEFORE each command,
  │      renders scene_cam + wrist_cam, scores with evaluation.evaluate_placement
  │
  ├─ language.CachedEncoder ────────► (512,) frozen CLIP embedding
  │
  └─ recorder.DatasetWriter ────────► HDF5
```

## Modules

| Module | Responsibility |
| --- | --- |
| `constants.py` | Measured physical constants, workspace bounds, semantic vocabulary |
| `randomization.py` | Seed → `EpisodeSpec`; rejection sampling with layout restarts |
| `scene_builder.py` | `EpisodeSpec` + menagerie MJCF → compiled `MjModel` |
| `kinematics.py` | FK, DLS/Levenberg-Marquardt IK, `top_down_pose`, rotation error |
| `trajectory.py` | Quintic time scaling and joint/Cartesian segment interpolation |
| `expert_policy.py` | The 7-phase scripted expert |
| `episode_runner.py` | Physics rollout, observation recording, frame alignment |
| `evaluation.py` | Binary success validator (strict footprint containment) |
| `language.py` | Instruction text → frozen text embedding, with fallback |
| `recorder.py` | HDF5 writer, action chunking |
| `cli/collect.py` | Dataset generation entry point |
| `cli/evaluate.py` | N-seed success evaluation |
| `cli/visualize.py` | Episode → side-by-side mp4 |

## The seven phases

| # | Phase | Motion | Why it is separate |
| --- | --- | --- | --- |
| 1 | `hover_and_center` | **Joint-space** blend home → above cube, then a Cartesian centring hold | Joint space keeps the arm folded clear of the table (ISSUE-003); the hold squares the fixed jaw to the chosen cube face |
| 2 | `approach` | Straight vertical descent to `GRASP_Z` | Vertical keeps the fixed jaw aligned so it does not disturb the cube |
| 3 | `grasp` | Close jaws in place | Held stationary so the position servo can build force |
| 4 | `lift` | Straight vertical to `HOVER_Z` | Clears the tabletop before any horizontal motion |
| 5 | `transit` | Straight horizontal line to above the zone | Jaw yaw is carried over *relative to the radial axis*, so `wrist_roll` barely moves while carrying |
| 6 | `place` | Lower to `PLACE_Z` | Supported descent rather than a drop |
| 7 | `release_and_clear` | Open → retreat along −x_tcp → climb → joint-space blend home | Backing out over the flat fixed jaw avoids hooking the cube (D-008) |

## Coordinate conventions

- Table top is **z = 0**; the arm base is bolted to it, so all workspace numbers
  are base coordinates. The floor plane is 0.75 m below and non-colliding.
- **TCP frame**: `z` is the approach direction (out of the fingers), `x` is the
  jaw opening direction. The site carries `quat = (0,1,0,0)` — a 180° rotation
  about the gripper body's x axis — to produce exactly this.
- `jaw_yaw` is the world-frame azimuth of the TCP x axis, i.e. the direction the
  jaws open along.
