# Issue log

Bugs found, their **root cause**, and the fix. Resolved issues stay here — the
record of why a constant has the value it has is the whole point of this file.

Every one of these was a case of a plausible-looking number being *assumed*
rather than measured, and every one produced silent quality loss rather than a
crash.

---

## ISSUE-001 — Fixed jaw spawned inside the cube (RESOLVED)

**Symptom.** Expert success ~40%. Failing episodes showed the cube nudged a few
centimetres and never lifted.

**Investigation.** `data.actuator_force` showed `shoulder_lift` pinned at its
2.94 N·m limit with a 0.1 rad standing position error, while the *commanded*
joint targets had sub-millimetre FK error. So the IK was right and something was
physically blocking the arm. Dumping `data.contact` at the stall showed
`fixed_jaw_box3 <-> object_1_geom` plus two gripper collision meshes pressing on
the cube at 6–7 N.

**Root cause.** `TCP_LOCAL_POS.x` was 0.001. The fixed jaw's contact pads sit at
gripper-local x ≈ −0.010, so with the TCP (the intended grasp centre) only 1 mm
out in +x, the jaw pad ended up ~3.5 mm *inside* the cube's footprint. Descending
jammed the jaw into the cube instead of straddling it.

**Fix.** Calibrated the offset empirically instead of deriving it: swept
`TCP_LOCAL_POS.x` end-to-end over 8 seeds. `x = 0.002 → 3/8`; `x ≥ 0.004 → 8/8`.
Set to `0.005` (mid-band). The z offset and grasp height barely mattered by
comparison — x was the entire story.

**Guard.** The measurement and its numbers are recorded in the `constants.py`
comment.

---

## ISSUE-002 — Grasp height buried the jaw tip in the tabletop (RESOLVED)

**Symptom.** After ISSUE-001, success 93/100. Remaining hard failures all had the
cube knocked 30–45 mm aside, and every seed showed actuators saturated at
2.94/2.94 N·m throughout the grasp.

**Investigation.** Measured the world-z of the lowest fixed-jaw geom while
holding the commanded grasp pose:

| `GRASP_Z` | lowest jaw tip | settled TCP sag | actuator load |
| --- | --- | --- | --- |
| 0.0165 | **−5.7 mm** (below the table) | +6.2 mm | 2.94 / 2.94 (saturated) |
| 0.0220 | −0.2 mm | +0.6 mm | 2.4–2.7 |
| 0.0260 | +3.7 mm | −0.1 mm | 0.3–0.5 |

**Root cause.** `GRASP_Z` was defined as `CUBE_HALF + 0.004`, i.e. aimed at the
cube's mid-height. But the fixed jaw's tip geoms extend ~22 mm *below* the TCP,
which is more than the 25 mm cube is tall. Aiming the TCP at the cube centre
necessarily drives the tip through the tabletop. The arm then fought the table,
saturated, and scraped cubes aside on the way down.

**Fix.** `GRASP_Z = 0.0235`, set by jaw-tip clearance (~1.5 mm above the table)
rather than by cube geometry. The jaw pads still straddle the cube's upper half.
`PLACE_Z = 0.0245` likewise.

**Effect.** Placement precision improved markedly (p90 centre error
11.2 mm → 6.3 mm) and actuator saturation disappeared, which also makes the
recorded actions physically meaningful rather than clipped.

**Note.** A subtle trap during this investigation: `data.site_xpos` returns a
**live view**, not a copy. An early probe compared a pre-step and post-step value
that were the same underlying array and reported a nonsensical result. Always
`.copy()` MuJoCo state you intend to compare across steps.

---

## ISSUE-003 — Home pose parked the gripper on the tabletop (RESOLVED)

**Symptom.** Persistent 7/100 failures. The target cube was displaced 31–45 mm
**before the descent phase began**.

**Investigation.** Logged the first timestep at which the cube moved: step 3–5 of
the *first* phase, with the TCP at ≈ (0.185, 0, 0.024). Forward kinematics of
`HOME_QPOS` returned TCP `(0.183, 0, 0.023)`.

**Root cause.** Two compounding mistakes. (a) The home configuration put the TCP
at z = 0.023 — the gripper resting *on the table*, inside the object annulus.
(b) The first phase interpolated a straight **Cartesian** line from home to the
hover point, so the gripper ploughed horizontally across the tabletop through
whatever cubes lay between, before it ever rose.

**Fix.** (a) `HOME_QPOS = [0, -1.2, 0.4, 1.2, 0, GRIPPER_OPEN]`, parking the TCP
at (0.198, 0, 0.233), well clear. (b) Phase 1 now blends home → hover in
**joint space**, which keeps the arm folded up during the traverse, followed by a
short Cartesian centring hold at the hover pose.

**Effect.** 93/100 → **100/100**, median centre error 3.0 mm.

---

## ISSUE-004 — Scene camera framing and shadow acne (RESOLVED)

**Symptom.** Not a failure at all — success metrics were unaffected. Visible only
by watching the render: the scene camera targeted the robot *base* from 0.6 m
with a 58° FOV, so the arm and the 25 mm cubes occupied a small central patch and
most pixels were empty table. MuJoCo's default shadow map also painted blotchy
shadow acne across the tabletop.

**Why it mattered.** Both defects would have been baked into every training
image, costing the policy visual resolution on exactly the objects it has to
ground language to, plus a spurious moving texture to overfit.

**Fix.** The camera now targets a massless body at the centre of the workspace
annulus from 0.47 m at 52° FOV, and both lights set `castshadow="false"`.

**Lesson.** Pass/fail metrics cannot see dataset quality. Render and *look*.

---

## ISSUE-005 — Parallel collection scales poorly on macOS (RESOLVED as documented)

**Symptom.** `--num-workers 8` reported **13.6 s/episode** against 10.1 s/episode
single-process — parallelism made collection *slower*.

**Two separate causes, found in order.**

1. **Every worker loaded its own CLIP tower.** With only 2 episodes per worker
   the model-load cost dominated. Fixed properly: an instruction is a pure
   function of its seed, so the parent now derives and embeds every instruction
   up front and passes a `{instruction: vector}` dict down. Workers never import
   torch.
2. **Benchmark contamination.** Subsequent runs suggested parallel rendering was
   *catastrophically* slow (~2 min/episode). That was wrong. Killed parent
   processes had left orphaned pool workers spinning; `uptime` showed a load
   average of **28** on an 8-core machine. Every measurement taken during that
   window — including the "physics-only is 3.0 s/ep" reading, against a true
   0.2–0.4 s/ep — was measuring contention, not the code.

**Clean measurement**, taken on a quiet machine after reaping the orphans:

| Mode | Throughput | Speedup |
| --- | --- | --- |
| rendered, serial | 10.15 s/ep | 1.0× |
| rendered, 4 workers | 7.79 s/ep | **1.30×** |

**Conclusion.** Parallel collection is correct and modestly faster, but MuJoCo's
offscreen renderer contends across processes on macOS, so it does not approach
the ~4× the core count suggests. Default is one worker; the caveat is documented
at the flag and in `parallel.py`. On Linux with EGL, headless rendering is
process-local and should scale far better.

**Lesson.** Always check machine load before trusting a performance number, and
reap orphaned pool workers when a parent is killed — a stale benchmark will
happily invent a root cause that does not exist.

---

## ISSUE-006 — Two collection runs raced on one output path (RESOLVED)

**Symptom.** After a collection run, `data/so101_lang_act.hdf5` contained **10**
episodes while `data/collection_report.json` sitting next to it described **40**.
Two artifacts of the same run, disagreeing.

**Root cause.** A long background collection was believed dead and a second run
was started against the same `--out` path. Both were alive. `h5py.File(path,
"w")` truncates without complaint, and nothing in the pipeline noticed that two
writers were pointed at one file.

**Why the data was not corrupted.** The second run `rm -f`'d the path before
opening it, so the first writer kept a handle to the now-*unlinked* inode. Its 40
episodes were written to a deleted file and freed on close. The surviving HDF5
was the second run's, complete and internally consistent — verified by reopening
it and checking group count, seeds and per-episode array shapes. That was luck,
not design: with a different ordering the two writers would have interleaved into
one file.

**Fix.** `DatasetWriter` now refuses to open an existing path unless
`overwrite=True` (`--overwrite` on the CLI), so a stale or concurrent run fails
loudly instead of silently clobbering. Guarded by
`tests/test_dataset.py::test_writer_refuses_to_clobber_an_existing_dataset`.

**Lesson.** "The process isn't in `pgrep`" is not proof it is gone, and a
truncating open is a silent destructive default. Also: when two artifacts that
describe the same run disagree, verify the *data*, do not reason about which
writer probably won.

---

## Open risks

- **RISK-001 — Success is measured on the scripted expert, not on a policy.**
  100/100 says the demonstrations are clean; it says nothing about whether an
  ACT policy trained on them generalises. Training is out of scope here.
- **RISK-002 — Near-miss margin is thin by construction.** Zone radius is 42 mm
  and the cube's base half-diagonal is 17.7 mm, so the strict criterion needs the
  centre within 24.3 mm. Current p90 is 4.6 mm, comfortable, but any regression
  in placement accuracy will show up as failures quickly. That is intentional —
  it is a sensitive canary.
- **RISK-003 — Single cube size and shape.** `OBJECT_SHAPES` and `ZONE_SHAPES`
  each hold one entry. The language template and the scene builder are written to
  take more, but only `cube`/`circle` are wired to geometry today. Adding a shape
  means adding its grasp geometry, not just a vocabulary word.
- **RISK-005 — Rendering, not physics, is the collection bottleneck.** Physics
  is ~0.2 s/episode; rendering two 240×320 cameras for ~430 frames is ~10 s. The
  directive suggests GPU tensor arrays for a ~20× gain, but MJX accelerates
  *stepping* and does not render, so it would not touch the dominant cost here.
  The real levers are a Linux/EGL host (where multiprocessing actually scales),
  smaller images, or batched GPU rendering.
- **RISK-004 — CLIP fallback is silent-ish.** If `transformers` is missing,
  `build_text_encoder` prints a warning and returns a deterministic *hashing*
  encoder whose vectors carry no semantics. The encoder name is written into the
  HDF5 metadata; a consumer must check it rather than assume CLIP.
