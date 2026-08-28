#!/usr/bin/env python3
"""Render a video FROM a stored episode -- not by re-simulating it.

The distinction is the whole point. `so101_sim.cli.visualize` re-runs the
simulator and shows you what the *simulator* does. This reads the pixels and
numbers that are actually on disk, which is what a policy will train on. If the
two ever disagree, the file is what matters.

Overlays, per frame:
  * the language instruction (the conditioning signal)
  * the current phase name
  * commanded action vs observed qpos, per joint, as paired bars
  * a gripper open/closed indicator

Watching the bars is the fastest way to spot a frame-alignment problem: the
commanded bar should lead the observed one, never trail it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

JOINTS = ["pan", "lift", "elbow", "wristF", "wristR", "grip"]
CTRL_LO = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
CTRL_HI = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


def compose(scene, wrist, instruction, phase, action, qpos, t, total, scale=2):
    h = scene.shape[0]
    strip = np.hstack([scene, wrist])
    img = Image.fromarray(strip).resize((strip.shape[1] * scale, h * scale),
                                        Image.NEAREST)
    W, H = img.size
    header, footer = 34, 96
    canvas = Image.new("RGB", (W, H + header + footer), (16, 17, 20))
    canvas.paste(img, (0, header))
    d = ImageDraw.Draw(canvas)

    d.text((8, 6), f'"{instruction}"', fill=(235, 235, 240))
    d.text((W - 150, 6), f"frame {t + 1}/{total}", fill=(150, 150, 160))
    d.text((8, header + 4), "scene_cam", fill=(255, 255, 120))
    d.text((W // 2 + 8, header + 4), "wrist_cam", fill=(255, 255, 120))

    y0 = H + header + 6
    d.text((8, y0), f"phase: {phase}", fill=(120, 220, 255))

    # Per-joint paired bars: commanded (top) vs observed (bottom), normalised
    # into the actuator range so all six are comparable at a glance.
    bar_x, bar_y, bar_w = 8, y0 + 18, W - 120
    a = (action - CTRL_LO) / (CTRL_HI - CTRL_LO)
    q = (qpos - CTRL_LO) / (CTRL_HI - CTRL_LO)
    for j in range(6):
        yy = bar_y + j * 11
        d.text((bar_x, yy - 2), JOINTS[j], fill=(170, 170, 180))
        x0 = bar_x + 46
        d.rectangle([x0, yy, x0 + bar_w, yy + 8], outline=(60, 62, 70))
        d.rectangle([x0, yy, x0 + int(bar_w * float(np.clip(a[j], 0, 1))), yy + 3],
                    fill=(255, 170, 60))       # commanded
        d.rectangle([x0, yy + 5, x0 + int(bar_w * float(np.clip(q[j], 0, 1))), yy + 8],
                    fill=(90, 200, 255))       # observed
    d.text((W - 66, bar_y), "cmd", fill=(255, 170, 60))
    d.text((W - 66, bar_y + 12), "obs", fill=(90, 200, 255))
    grip = "CLOSED" if action[5] < 0.3 else "OPEN"
    d.text((W - 66, bar_y + 30), grip,
           fill=(255, 110, 110) if grip == "CLOSED" else (140, 230, 140))
    return np.asarray(canvas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--episode", default="0",
                    help="episode index, or 'all' to concatenate every episode")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        names = sorted(k for k in f if k.startswith("episode_"))
        phases = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in f["metadata"].attrs["phases"]]
        chosen = names if args.episode == "all" else [names[int(args.episode)]]

        out = args.out or Path("data") / (
            f"replay_{'all' if args.episode == 'all' else chosen[0]}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)

        frames = []
        for name in chosen:
            g = f[name]
            instr = str(g.attrs["instruction"])
            scene, wrist = g["obs/scene_image"][:], g["obs/wrist_image"][:]
            action, qpos, ph = g["action"][:], g["obs/qpos"][:], g["phase"][:]
            T = action.shape[0]
            print(f"{name}: \"{instr}\"  T={T}  success={bool(g.attrs['success'])}")
            for t in range(T):
                frames.append(compose(scene[t], wrist[t], instr, phases[ph[t]],
                                      action[t], qpos[t], t, T, args.scale))

        imageio.mimsave(out, frames, fps=args.fps, macro_block_size=1)
    print(f"\nwrote {len(frames)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
