#!/usr/bin/env python3
"""Automated health checks on a generated dataset.

This reads only what is *stored on disk* -- it never re-simulates. That
distinction matters: re-running the simulator verifies the simulator, whereas a
policy trains on the bytes in the file, and those are what can be silently
wrong.

Every check prints PASS/FAIL with the measured number, so a regression shows up
as a changed value rather than a vague "looks fine".
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

CTRL_LO = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
CTRL_HI = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{name}: {detail}")

    def warn(self, name: str, detail: str) -> None:
        print(f"  [WARN] {name}  --  {detail}")
        self.warnings.append(f"{name}: {detail}")


def verify(path: Path, sample: int | None) -> int:
    c = Checker()
    with h5py.File(path, "r") as f:
        meta = f["metadata"].attrs
        eps = sorted(k for k in f if k.startswith("episode_"))
        if sample:
            eps = eps[:sample]

        print(f"\n=== {path.name}  ({path.stat().st_size / 1e6:.0f} MB, "
              f"{len(eps)} episodes checked) ===\n")

        # ---------------------------------------------------------- schema
        print("Schema and metadata")
        required = ["obs/scene_image", "obs/wrist_image", "obs/qpos", "obs/qvel",
                    "obs/tcp_pose", "action", "action_chunk", "action_chunk_mask",
                    "phase", "object_poses", "language_instruction",
                    "language_embedding"]
        missing = {k for e in eps for k in required if k not in f[e]}
        c.check("every episode has every required key", not missing, str(missing or ""))
        c.check("metadata n_episodes matches group count",
                int(meta["n_episodes"]) == len(sorted(k for k in f if k.startswith("episode_"))),
                f"attr={int(meta['n_episodes'])}")
        enc = str(meta["text_encoder"])
        c.check("a real text encoder was used (not the hashing fallback)",
                enc != "hashing-fallback", f"encoder={enc}")
        fps = float(meta["fps"])
        c.check("recording rate >= 30 Hz", fps >= 30.0, f"{fps:g} Hz")

        # ---------------------------------------------- per-episode gathers
        print("\nShapes and ranges")
        n_dof = int(meta["n_dof"])
        k = int(meta["action_chunk_size"])
        bad_shape, out_of_range, bad_chunk, bad_mask = [], [], [], []
        step_maxima: list[float] = []
        max_norm_err = 0.0
        blank_imgs, static_imgs, dup_cams = [], [], []
        succ, lengths, instructions = [], [], []
        emb_by_text: dict[str, np.ndarray] = {}
        emb_norm_bad, lang_mismatch, target_not_moved, distractor_moved = [], [], [], []
        phase_bad = []

        for e in eps:
            g = f[e]
            T = g["action"].shape[0]
            lengths.append(T)
            succ.append(bool(g.attrs["success"]))

            act = g["action"][:]
            qpos = g["obs/qpos"][:]
            if not (qpos.shape == (T, n_dof) and act.shape == (T, n_dof)
                    and g["obs/scene_image"].shape[0] == T
                    and g["obs/wrist_image"].shape[0] == T
                    and g["phase"].shape[0] == T):
                bad_shape.append(e)

            if (act < CTRL_LO - 1e-5).any() or (act > CTRL_HI + 1e-5).any():
                out_of_range.append(e)

            step_maxima.append(float(np.abs(np.diff(act, axis=0)).max()))

            # action chunks must be exact slices of the action stream
            ch = g["action_chunk"][:]
            mask = g["action_chunk_mask"][:]
            if not np.allclose(ch[:, 0], act):
                bad_chunk.append(e)
            expected = np.minimum(k, T - np.arange(T))
            if not (mask.sum(axis=1) == expected).all():
                bad_mask.append(e)

            # frame alignment, statistically: the command must lead the state
            before = np.linalg.norm(act[:-1, :5] - qpos[:-1, :5], axis=1)
            after = np.linalg.norm(act[:-1, :5] - qpos[1:, :5], axis=1)
            mv = before > 1e-4
            frac = float((after[mv] <= before[mv] + 1e-6).mean()) if mv.any() else 1.0
            max_norm_err = max(max_norm_err, 1.0 - frac)

            # images
            for cam, store in (("scene", "obs/scene_image"), ("wrist", "obs/wrist_image")):
                idx = np.linspace(0, T - 1, min(T, 12)).astype(int)
                frames = g[store][idx]
                if frames.std() < 5.0:
                    blank_imgs.append(f"{e}/{cam}")
                if float(np.abs(frames[1:].astype(np.int16)
                                - frames[:-1].astype(np.int16)).mean()) < 0.2:
                    static_imgs.append(f"{e}/{cam}")
            i0 = np.linspace(0, T - 1, min(T, 6)).astype(int)
            if np.array_equal(g["obs/scene_image"][i0], g["obs/wrist_image"][i0]):
                dup_cams.append(e)

            # phases must appear in order, no interleaving
            ids = g["phase"][:]
            changes = ids[1:][ids[1:] != ids[:-1]]
            if list(changes) != sorted(set(changes)) or ids[0] != 0:
                phase_bad.append(e)

            # language grounding
            text = str(g.attrs["instruction"])
            instructions.append(text)
            emb = g["language_embedding"][:]
            if abs(float(np.linalg.norm(emb)) - 1.0) > 1e-3:
                emb_norm_bad.append(e)
            if text in emb_by_text:
                if not np.allclose(emb_by_text[text], emb, atol=1e-6):
                    lang_mismatch.append(f"{e} (embedding differs for identical text)")
            else:
                emb_by_text[text] = emb
            tgt_label = str(g.attrs["target_object_label"])
            zone_label = str(g.attrs["target_zone_label"])
            if tgt_label not in text or zone_label not in text:
                lang_mismatch.append(f"{e} (instruction does not name its target)")

            # the demonstration must actually perform the named task
            names = [s.decode() if isinstance(s, bytes) else str(s)
                     for s in g.attrs["object_names"]]
            ti = names.index(str(g.attrs["target_object"]))
            poses = g["object_poses"][:]
            disp = np.linalg.norm(poses[-1, :, :2] - poses[0, :, :2], axis=1)
            if disp[ti] < 0.02:
                target_not_moved.append(e)
            others = [j for j in range(len(names)) if j != ti]
            if others and disp[others].max() > 0.02:
                distractor_moved.append(f"{e} ({disp[others].max() * 1000:.0f} mm)")

        c.check("array shapes are internally consistent", not bad_shape, str(bad_shape[:3]))
        c.check("all actions inside the actuator ctrlrange", not out_of_range,
                str(out_of_range[:3]))
        # Two different questions, so two thresholds. A genuine discontinuity --
        # a bad concatenation, a duplicated frame, an IK branch flip -- shows up
        # as a large isolated jump. What a *typical* episode does is bounded by
        # the quintic's peak velocity. Measured over 1200 episodes: p50 0.072,
        # p99 0.113, max 0.147 rad/step, where every episode above 0.12 is a
        # `transit` whose straight-line path passes near the base, making
        # shoulder_pan sweep quickly. That is real kinematics, not corruption,
        # so the tail is allowed while an outlier is still caught.
        sm = np.array(step_maxima)
        p99 = float(np.percentile(sm, 99))
        worst = float(sm.max())
        c.check("action stream is smooth (typical episode)", p99 < 0.12,
                f"p99 per-step joint jump {p99:.4f} rad")
        c.check("no action discontinuity (outlier check)", worst < 0.25,
                f"worst per-step joint jump {worst:.4f} rad "
                f"({int((sm > 0.12).sum())}/{len(sm)} episodes above 0.12)")

        print("\nACT contract")
        c.check("action_chunk[t,0] == action[t]", not bad_chunk, str(bad_chunk[:3]))
        c.check("action_chunk_mask counts the remaining real actions", not bad_mask,
                str(bad_mask[:3]))
        c.check("chunk size recorded in metadata", k > 1, f"k={k}")

        print("\nFrame alignment (the silent killer)")
        c.check("commanded action leads the observed state",
                max_norm_err < 0.12,
                f"worst episode: {100 * (1 - max_norm_err):.1f}% of moving steps "
                f"converge toward the command")

        print("\nImages")
        c.check("no blank camera streams", not blank_imgs, str(blank_imgs[:3]))
        c.check("cameras actually change over time", not static_imgs, str(static_imgs[:3]))
        c.check("scene and wrist views are distinct", not dup_cams, str(dup_cams[:3]))

        print("\nLanguage conditioning")
        c.check("embeddings are L2-normalised", not emb_norm_bad, str(emb_norm_bad[:3]))
        c.check("instruction names its own target object and zone", not lang_mismatch,
                str(lang_mismatch[:3]))
        uniq = len(set(instructions))
        c.check("more than one distinct instruction", uniq > 1,
                f"{uniq} distinct instructions over {len(eps)} episodes")
        if uniq < 8:
            c.warn("instruction diversity is low",
                   f"only {uniq} distinct commands -- a policy can memorise rather "
                   f"than ground language")

        print("\nDemonstration quality")
        c.check("phase labels are ordered and contiguous", not phase_bad, str(phase_bad[:3]))
        c.check("the target object actually moved", not target_not_moved,
                str(target_not_moved[:3]))
        c.check("all recorded episodes are successful", all(succ),
                f"{sum(succ)}/{len(succ)}")
        if distractor_moved:
            c.warn("distractor objects were disturbed", str(distractor_moved[:3]))

        print("\nDataset scale")
        total_steps = int(np.sum(lengths))
        print(f"  episodes {len(eps)} | timesteps {total_steps} | "
              f"length min {min(lengths)} max {max(lengths)}")
        print(f"  instructions: {dict(Counter(instructions).most_common(3))} ...")
        if len(eps) < 300:
            c.warn("episode count is below the published baseline",
                   f"{len(eps)} episodes; the reference blog needed 300 for a 53% "
                   f"policy and 1000 for 99%")

    print("\n" + "=" * 70)
    if c.failures:
        print(f"RESULT: {len(c.failures)} FAILED CHECK(S)")
        for x in c.failures:
            print(f"  - {x}")
    else:
        print("RESULT: all checks passed")
    if c.warnings:
        print(f"\n{len(c.warnings)} warning(s) -- not corruption, but read them:")
        for x in c.warnings:
            print(f"  - {x}")
    print("=" * 70)
    return 1 if c.failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--sample", type=int, default=None,
                    help="only check the first N episodes (faster)")
    args = ap.parse_args()
    return verify(args.path, args.sample)


if __name__ == "__main__":
    sys.exit(main())
