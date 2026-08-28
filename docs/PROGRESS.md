# Progress log

Status of every component in the directive, plus the running build log.
Update this file whenever a component changes state.

Last updated: 2026-08-27.

## Component status

| # | Directive requirement | Status | Where |
| --- | --- | --- | --- |
| 1 | Official SO-101 MJCF from `mujoco_menagerie` | ✅ done | `assets/so101/` (`robotstudio_so101`) |
| 2 | Tabletop scene with the arm mounted | ✅ done | `scene_builder.py` |
| 3 | 2–3 coloured object cubes | ✅ done | `randomization.py`, `scene_builder.py` |
| 4 | 2–3 flat, uncollidable target zones | ✅ done | cylinders with `contype=0 conaffinity=0` |
| 5 | Per-episode (x, y) randomization from a seed | ✅ done | `randomization.sample_episode` |
| 6 | TCP site at the fixed-jaw tool centre | ✅ done | injected into the `gripper` body |
| 7 | FK via `site_xpos` / `site_xmat` | ✅ done | `kinematics.SO101Kinematics.fk` |
| 8 | DLS / Levenberg-Marquardt IK on `mj_jacSite` | ✅ done | `kinematics.solve` |
| 9 | 7-phase quintic-interpolated expert | ✅ done | `expert_policy.py` |
| 10 | ≥30 Hz recording | ✅ done | 50 Hz, 1:1 with control |
| 11 | Scene + wrist images, `qpos` | ✅ done | `episode_runner.py` |
| 12 | Action chunks of target joint positions | ✅ done | `recorder.build_action_chunks` |
| 13 | `language_instruction` string | ✅ done | `randomization.EpisodeSpec.instruction` |
| 14 | `language_embedding` from a frozen encoder | ✅ done | frozen CLIP ViT-B/32, 512-d |
| 15 | HDF5 dataset | ✅ done | `recorder.DatasetWriter` |
| 16 | 100-seed evaluation harness + binary validator | ✅ done | `cli/evaluate.py`, `evaluation.py` |
| 17 | Throughput mitigation for the compute roadblock | ⚠️ partial | `parallel.py` — only 1.3× on macOS, renderer-bound (ISSUE-005) |
| 18 | Language-conditioned ACT **model** changes | ⛔ not started | out of scope — see below |

## Results

| Metric | Value |
| --- | --- |
| Expert success, seeds 10000–10099 (held out) | **100 / 100** |
| Expert success, seeds 0–249 | **250 / 250** |
| Placement centre error | median 2.7–3.0 mm, p90 4.6 mm |
| Unit + property tests | 44 passed |
| Throughput, physics only | ≈ 0.19 s/episode |
| Throughput, with 2-camera rendering | 10.04 s/episode measured over the 40-episode run on an idle machine; 27–33 s/episode observed under contention |
| Episode length | 444 steps = 8.9 s at 50 Hz |
| Dataset size | ≈ 22 MB/episode at 240×320, gzip |
| Dataset generated | `data/so101_lang_act.hdf5` — 40 episodes (seeds 0–39), 17 760 timesteps, 40/40 success, 15 distinct instructions, real CLIP embeddings, 851 MB |

## Build log

1. **Repo bootstrap.** Empty directory → `uv` project on Homebrew Python 3.13,
   MuJoCo 3.12.0 arm64.
2. **Asset retrieval.** Sparse-cloned `mujoco_menagerie`. Initially pulled
   `trs_so_arm100`, then found `robotstudio_so101` — the actual SO-101 — and
   switched to it (D-001).
3. **Model probing.** Measured joint ranges, actuator limits, jaw kinematics and
   reach *before* writing any code. Established that the arm is 5-DoF and that a
   general 6-DoF pose is unreachable (D-003).
4. **Reachability study.** Swept the top-down IK over radius, height and jaw yaw
   to find the honest workspace: r ∈ [0.16, 0.26], hover ≤ 0.070 (D-004).
5. **Scene builder, randomization, kinematics, trajectory, expert.** Layout
   restarts added after 7/300 seeds failed to place 6 entities.
6. **First end-to-end rollout: 2/5 success.** Debugging chain →
   **ISSUE-001** (jaw spawned inside the cube). Fixed by calibrating the TCP x
   offset empirically. → 40/40.
7. **100-seed evaluation: 93/100.** →
   **ISSUE-002** (grasp height buried the jaw tip in the table, saturating the
   actuators). Fixed by setting `GRASP_Z` from jaw-tip clearance. Precision
   improved, success unchanged at 93/100.
8. **Rendered and watched the video.** →
   **ISSUE-004** (scene camera framed the base, not the workspace; shadow acne).
   Invisible to the metrics, would have degraded every training image.
9. **Traced the residual failures to the first timestep.** →
   **ISSUE-003** (home pose parked the gripper *on the table*, and phase 1 dragged
   it across horizontally). Fixed the home pose and made phase 1 a joint-space
   blend. → **100/100**, confirmed on 250 further seeds.
10. **Test suite, docs, HDF5 writer, CLIP encoder.** `transformers` 5.x changed
    `get_text_features` to return an output object rather than a tensor; switched
    to `CLIPTextModelWithProjection`.
11. **Parallel collection** added after measuring 10 s/episode with rendering.
    Two rounds of debugging (ISSUE-005): per-worker CLIP loading, then a
    contaminated benchmark caused by orphaned pool workers pushing the load
    average to 28. Clean result: **1.30×** at 4 workers, renderer-bound.

## Not done / out of scope

- **Item 18 — modifying the ACT model itself.** The directive's final section
  describes changes to a *training* codebase (passing `language_embedding` into
  the CVAE encoder and transformer decoder as a conditioning token). This repo is
  the data-generation pipeline; no ACT training code was cloned or written here.
  The dataset is shaped to make that change straightforward, and
  `docs/DATASET_SPEC.md` records exactly how the embedding is meant to be
  consumed. Flagging rather than silently assuming it was in scope.
- **GPU-batched simulation.** The directive suggests GPU tensor arrays for a
  ~20× speedup. The implemented mitigation is CPU multiprocessing, which is the
  right first step here because the bottleneck is *rendering*, not stepping —
  MJX accelerates physics but does not render, so it would not address the actual
  cost on this workload. Noted rather than assumed away.
- **Policy training and evaluation.** Success numbers describe the scripted
  expert, not a learned policy (RISK-001).
- **More shapes than `cube` / `circle`** (RISK-003).
