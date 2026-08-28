"""Scripted expert that produces the demonstration trajectories.

The motion is laid out as seven named phases (see `PHASES`) and each phase is a
minimum-jerk segment, so the concatenated action stream is C2-continuous across
phase boundaries. Cartesian phases are solved waypoint-by-waypoint with the DLS
IK, warm-started from the previous solution, which both speeds the solver up and
keeps it on a single continuous IK branch instead of flipping elbow
configurations mid-motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    CONTROL_HZ,
    GRASP_Z,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_QPOS,
    HOVER_Z,
    PLACE_Z,
)
from .kinematics import SO101Kinematics
from .trajectory import quintic_segment, steps_for

PHASES = (
    "hover_and_center",
    "approach",
    "grasp",
    "lift",
    "transit",
    "place",
    "release_and_clear",
)

# Seconds per phase. Grasp/release are deliberately unhurried: the position
# actuator needs time to actually build force against the cube.
PHASE_DURATIONS = {
    "hover_and_center": 1.6,
    "approach": 1.0,
    "grasp": 0.7,
    "lift": 0.9,
    "transit": 1.6,
    "place": 1.0,
    "release_and_clear": 2.1,
}

RETREAT_DISTANCE = 0.045   # along -x_tcp, i.e. out over the flat fixed jaw


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def choose_jaw_yaw(object_xy, object_yaw: float) -> float:
    """Pick which of the cube's four equivalent grasp axes to use.

    A cube is 90-deg symmetric, so `object_yaw + k*pi/2` are all valid jaw
    alignments. We take the one closest to the arm's radial direction, which
    keeps `wrist_roll` near the middle of its range instead of winding up
    against a limit.
    """
    radial = np.arctan2(object_xy[1], object_xy[0])
    candidates = [object_yaw + k * np.pi / 2 for k in range(4)]
    return min(candidates, key=lambda a: abs(_wrap(a - radial)))


@dataclass
class ExpertTrajectory:
    """Joint-space command stream plus a per-step phase label."""

    qpos: np.ndarray          # (T, 6) arm joints + gripper
    phase_ids: np.ndarray     # (T,) index into PHASES
    ik_max_pos_error: float
    ik_failures: int

    def __len__(self) -> int:
        return int(self.qpos.shape[0])


class ExpertPlanner:
    def __init__(self, kinematics: SO101Kinematics, control_hz: float = CONTROL_HZ):
        self.ik = kinematics
        self.control_hz = control_hz

    # ------------------------------------------------------------------
    def plan(
        self,
        object_xy: np.ndarray,
        object_yaw: float,
        zone_xy: np.ndarray,
        q_home: np.ndarray | None = None,
    ) -> ExpertTrajectory:
        q_home = HOME_QPOS.copy() if q_home is None else np.asarray(q_home, float).copy()

        object_xy = np.asarray(object_xy, float)
        zone_xy = np.asarray(zone_xy, float)

        jaw_pick = choose_jaw_yaw(object_xy, object_yaw)
        # Carry the same jaw orientation *relative to the arm's radial axis*
        # over to the placement, so wrist_roll barely moves during transit.
        radial_pick = np.arctan2(object_xy[1], object_xy[0])
        radial_place = np.arctan2(zone_xy[1], zone_xy[0])
        jaw_place = jaw_pick + (radial_place - radial_pick)

        p_hover_pick = np.array([object_xy[0], object_xy[1], HOVER_Z])
        p_grasp = np.array([object_xy[0], object_xy[1], GRASP_Z])
        p_hover_place = np.array([zone_xy[0], zone_xy[1], HOVER_Z])
        p_place = np.array([zone_xy[0], zone_xy[1], PLACE_Z])

        # Retreat is a pure translation along -x_tcp (the flat fixed-jaw side).
        # Backing out this way slides the straight jaw off the cube instead of
        # dragging the hooked moving jaw across it.
        retreat_dir = np.array([np.cos(jaw_place), np.sin(jaw_place), 0.0])
        p_retreat = p_place - RETREAT_DISTANCE * retreat_dir

        chunks: list[np.ndarray] = []
        labels: list[int] = []
        max_pos_err = 0.0
        failures = 0
        q_arm = q_home[:5].copy()

        def cartesian_phase(name, p_from, yaw_from, p_to, yaw_to, grip_from, grip_to,
                            duration=None):
            nonlocal q_arm, max_pos_err, failures
            n = steps_for(PHASE_DURATIONS[name] if duration is None else duration,
                          self.control_hz)
            path = quintic_segment(
                np.concatenate([p_from, [yaw_from, grip_from]]),
                np.concatenate([p_to, [yaw_to, grip_to]]),
                n,
            )
            out = np.empty((n, 6))
            for i, row in enumerate(path):
                res = self.ik.solve_top_down(row[:3], float(row[3]), q_arm)
                if not res.converged:
                    failures += 1
                max_pos_err = max(max_pos_err, res.pos_error)
                q_arm = res.qpos
                out[i, :5] = q_arm
                out[i, 5] = row[4]
            chunks.append(out)
            labels.extend([PHASES.index(name)] * n)

        def gripper_phase(name: str, grip_from: float, grip_to: float):
            n = steps_for(PHASE_DURATIONS[name], self.control_hz)
            held = np.repeat(q_arm[None, :], n, axis=0)
            grip = quintic_segment(np.array([grip_from]), np.array([grip_to]), n)
            chunks.append(np.hstack([held, grip]))
            labels.extend([PHASES.index(name)] * n)

        # --- 1. hover & centre -----------------------------------------
        # (a) Home -> above the cube as a JOINT-space blend, not a straight
        #     Cartesian line. A Cartesian line from the parked pose sweeps the
        #     gripper down through the tabletop plane and scatters the cubes it
        #     passes; interpolating in joint space keeps the arm folded up and
        #     out of the way for the whole traverse.
        hover_ik = self.ik.solve_top_down(p_hover_pick, jaw_pick, q_home[:5])
        if not hover_ik.converged:
            failures += 1
        max_pos_err = max(max_pos_err, hover_ik.pos_error)
        n_reach = steps_for(PHASE_DURATIONS["hover_and_center"] * 0.7, self.control_hz)
        chunks.append(quintic_segment(
            np.concatenate([q_home[:5], [q_home[5]]]),
            np.concatenate([hover_ik.qpos, [GRIPPER_OPEN]]),
            n_reach,
        ))
        labels.extend([PHASES.index("hover_and_center")] * n_reach)
        q_arm = hover_ik.qpos.copy()
        # (b) brief centring hold: re-solve at the exact hover pose so the fixed
        #     jaw squares up to the cube face before descending.
        cartesian_phase("hover_and_center", p_hover_pick, jaw_pick, p_hover_pick, jaw_pick,
                        GRIPPER_OPEN, GRIPPER_OPEN,
                        duration=PHASE_DURATIONS["hover_and_center"] * 0.3)
        # --- 2. approach ------------------------------------------------
        cartesian_phase("approach", p_hover_pick, jaw_pick, p_grasp, jaw_pick,
                        GRIPPER_OPEN, GRIPPER_OPEN)
        # --- 3. grasp ---------------------------------------------------
        gripper_phase("grasp", GRIPPER_OPEN, GRIPPER_CLOSED)
        # --- 4. lift ----------------------------------------------------
        cartesian_phase("lift", p_grasp, jaw_pick, p_hover_pick, jaw_pick,
                        GRIPPER_CLOSED, GRIPPER_CLOSED)
        # --- 5. transit -------------------------------------------------
        cartesian_phase("transit", p_hover_pick, jaw_pick, p_hover_place, jaw_place,
                        GRIPPER_CLOSED, GRIPPER_CLOSED)
        # --- 6. supported placement -------------------------------------
        cartesian_phase("place", p_hover_place, jaw_place, p_place, jaw_place,
                        GRIPPER_CLOSED, GRIPPER_CLOSED)
        # --- 7. release & clear -----------------------------------------
        n_rel = steps_for(PHASE_DURATIONS["release_and_clear"], self.control_hz)
        third = max(n_rel // 4, 1)
        # (a) open the jaws in place
        grip = quintic_segment(np.array([GRIPPER_CLOSED]), np.array([GRIPPER_OPEN]), third)
        chunks.append(np.hstack([np.repeat(q_arm[None, :], third, axis=0), grip]))
        labels.extend([PHASES.index("release_and_clear")] * third)
        sub = PHASE_DURATIONS["release_and_clear"] / 4.0
        # (b) slide the fixed jaw straight back out from under the cube
        cartesian_phase("release_and_clear", p_place, jaw_place, p_retreat, jaw_place,
                        GRIPPER_OPEN, GRIPPER_OPEN, duration=sub)
        # (c) climb back clear of the tabletop
        p_up = p_retreat + np.array([0.0, 0.0, HOVER_Z - PLACE_Z])
        cartesian_phase("release_and_clear", p_retreat, jaw_place, p_up, jaw_place,
                        GRIPPER_OPEN, GRIPPER_OPEN, duration=sub)
        # (d) blend back to the home configuration in joint space
        n_home = steps_for(sub, self.control_hz)
        home_cmd = np.concatenate([q_arm, [GRIPPER_OPEN]])
        chunks.append(quintic_segment(home_cmd, q_home, n_home))
        labels.extend([PHASES.index("release_and_clear")] * n_home)
        q_arm = q_home[:5].copy()

        qpos = np.vstack(chunks)
        return ExpertTrajectory(
            qpos=qpos,
            phase_ids=np.array(labels, dtype=np.int8),
            ik_max_pos_error=max_pos_err,
            ik_failures=failures,
        )
