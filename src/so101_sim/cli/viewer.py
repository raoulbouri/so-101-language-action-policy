"""Interactive MuJoCo viewer: watch the expert run live, and scrub the scene.

Use this when you want to inspect *geometry* -- whether the jaws actually
straddle the cube, whether a tip is clipping the table. Two of the bugs in
docs/ISSUES.md were contact problems that were obvious here and invisible in
any scalar metric.

For checking the recorded dataset instead, use scripts/replay_dataset.py --
that reads the stored pixels rather than re-running physics.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..constants import ALL_JOINTS, CONTROL_HZ, HOME_QPOS, PHYSICS_SUBSTEPS
from ..expert_policy import PHASES, ExpertPlanner
from ..kinematics import SO101Kinematics
from ..randomization import sample_episode
from ..scene_builder import build_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0, help="playback rate")
    p.add_argument("--loop", action="store_true", help="restart when finished")
    return p


def main(argv: list[str] | None = None) -> int:
    import mujoco.viewer

    args = build_parser().parse_args(argv)
    spec = sample_episode(args.seed)
    print(f'seed {args.seed}: "{spec.instruction}"')
    print(f"  objects: {[o.label for o in spec.objects]}")
    print(f"  zones  : {[z.label for z in spec.zones]}")

    model = build_model(spec)
    data = mujoco.MjData(model)
    adr = np.array([model.joint(n).qposadr[0] for n in ALL_JOINTS], dtype=int)

    traj = ExpertPlanner(SO101Kinematics(model), control_hz=CONTROL_HZ).plan(
        np.array(spec.target_object.pos), spec.target_object.yaw,
        np.array(spec.target_zone.pos), q_home=HOME_QPOS)

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[adr] = HOME_QPOS
        data.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(model, data)

    reset()
    dt = 1.0 / (CONTROL_HZ * max(args.speed, 1e-3))
    print("\ndrag to orbit; press Tab for the control panel; Esc to quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0
        last_phase = -1
        while viewer.is_running():
            step_start = time.time()
            if t < len(traj):
                if traj.phase_ids[t] != last_phase:
                    last_phase = int(traj.phase_ids[t])
                    print(f"  [{t:4d}] {PHASES[last_phase]}")
                data.ctrl[:] = traj.qpos[t]
                t += 1
            elif args.loop:
                reset()
                t, last_phase = 0, -1
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - step_start)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
