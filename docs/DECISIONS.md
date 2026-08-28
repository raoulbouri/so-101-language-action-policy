# Design decisions

Each entry is a choice with a real alternative, the evidence behind it, and what
would make us revisit it.

---

### D-001 — Use `robotstudio_so101`, not `trs_so_arm100`

Menagerie ships both. `trs_so_arm100` is the older SO-ARM100; `robotstudio_so101`
is the SO-101 the directive actually names, derived from The Robot Studio's own
`so101_new_calib.xml`, and it already carries tuned gripper collision primitives
(`condim=6`, `priority=1`, elliptic friction cone) plus a wrist camera mount.
Vendored under `assets/so101/` with its Apache-2.0 licence.

*Revisit if:* menagerie publishes a newer calibration.

---

### D-002 — Build the scene by parsing the MJCF, not by `<include>`

The episode scene is produced by loading `so101.xml` with `ElementTree`, mutating
the tree, and compiling from a string.

The alternative — a static `scene.xml` with `<include file="so101.xml"/>` — was
rejected for one blocking reason: **an include cannot inject a site into a body
inside the included file**, and the TCP site must live in the `gripper` body. It
also makes per-episode object counts a file-editing problem instead of a data
problem.

*Cost:* the renderer must be rebuilt whenever the model is, since scene topology
changes between episodes.

---

### D-003 — Constrain the expert to top-down grasps

The SO-101 has five actuated arm joints, so a general 6-DoF pose is unreachable
and a naive 6-DoF IK target would make DLS return a least-squares compromise that
quietly corrupts *position* accuracy to chase an impossible orientation.

The reachable family is: `shoulder_pan` selects a vertical plane; three joints
work in that plane; `wrist_roll` spins the jaws about the tool axis. When the
approach points **straight down** the wrist axis is vertical, so `wrist_roll`
freely sets the jaw azimuth and the full 6-DoF target *is* consistent. Every
target the expert commands is built by `top_down_pose`, so the solver is always
given an exactly reachable pose and returns sub-millimetre solutions.

*Revisit if:* tilted approaches are needed to extend reach — they are consistent
too, but only if the jaw azimuth is left free rather than commanded.

---

### D-004 — Workspace annulus r ∈ [0.16, 0.26] m

Raw radial reach is 0.478 m, but that is irrelevant: the binding constraint is
that a straight-down approach is reachable only in a band that shrinks with
height. Measured convergence of the top-down IK over {grasp, mid, hover} heights
and ±45° of jaw yaw:

| annulus | hover z | convergence |
| --- | --- | --- |
| 0.16–0.30 | 0.070 | 89.9 % |
| **0.16–0.26** | **0.070** | **100.0 %** |

and a height sweep at fixed radius:

| z | 0.015 | 0.045 | 0.075 | 0.095 | 0.105 |
| --- | --- | --- | --- | --- | --- |
| reachable | 97 % | 94 % | 72 % | 40 % | **0 %** |

Hence `HOVER_Z = 0.070` — near the top of the envelope while retaining margin.
`tests/test_kinematics.py::test_ik_converges_across_the_declared_workspace`
locks this in so the bounds cannot be widened without the test failing.

---

### D-005 — Damped least squares rather than a plain pseudo-inverse

`(JᵀWJ + λ²I) dq = JᵀW e` with λ = 0.05, a diagonal weight down-weighting
rotation to 0.6, and a per-iteration step clip. The damping bounds `dq` when the
Jacobian loses rank near a singularity or a joint limit, trading a little
tracking accuracy for not exploding. Warm-starting each Cartesian waypoint from
the previous solution keeps the solver on one continuous IK branch instead of
flipping elbow configuration mid-motion.

---

### D-006 — Quintic (minimum-jerk) time scaling

`s(τ) = 10τ³ − 15τ⁴ + 6τ⁵` pins position, velocity **and** acceleration at both
ends of every segment. Cubic blends leave an acceleration step at each of the
seven phase boundaries, which shows up in the recorded action stream as a
discontinuity the policy then has to imitate. Segments deliberately exclude their
start frame so concatenation produces no duplicated timesteps.

---

### D-007 — Record at the control rate, 1:1

Control and recording both run at 50 Hz (above the 30 Hz floor the directive
sets), with exactly 4 physics substeps per control step at the model's 200 Hz
timestep. Recording at a *different* rate would force resampling one of the two
streams, which is precisely how an off-by-one action lag gets introduced. The
contract — `action[t]` is the command issued from `obs[t]` — is asserted in
`tests/test_alignment.py` and written into the HDF5 metadata.

---

### D-008 — Retreat over the fixed jaw

After releasing, the gripper translates ~45 mm along **−x_tcp** before climbing.
−x_tcp is the fixed-jaw side: that jaw is a flat plate, so backing out over it
slides cleanly off the cube. Retreating the other way, or lifting straight up
with the jaws still near the cube, risks the hooked moving jaw catching an edge
and dragging the cube out of the zone it was just placed in.

---

### D-009 — Frozen CLIP ViT-B/32 text tower, with a loud fallback

512-d L2-normalised embeddings, encoder frozen and cached per unique instruction
string. If `torch`/`transformers` are unavailable the pipeline degrades to a
deterministic hashing encoder so the physics side stays runnable — but the
encoder name is written into the dataset metadata, because training on hashed
vectors while believing they are CLIP would be an invisible disaster.

---

### D-010 — Store materialised action chunks

`action_chunk` is `(T, k, 6)` with an explicit `action_chunk_mask`. Chunks could
be sliced on the fly in the dataloader; materialising them costs disk but makes
the ACT contract explicit in the file and removes a place for the loader to get
the indexing subtly wrong. Tail padding repeats the final action rather than
zero-filling, keeping padded targets on the manifold, and the mask lets the loss
ignore them.

---

### D-011 — Strict footprint success criterion

PASS requires **every** base corner of the rotated cube inside the zone radius,
plus the cube resting on the table and at rest. The looser "centre inside the
zone" number is computed and reported alongside rather than substituted, so a
regression that degrades precision is visible as a gap between the two.
