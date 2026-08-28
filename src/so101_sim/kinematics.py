"""Forward kinematics and a damped least-squares (Levenberg-Marquardt) IK
solver built on MuJoCo's analytical Jacobians.

No Denavit-Hartenberg tables anywhere: `mujoco.mj_jacSite` gives the exact
spatial Jacobian of the TCP site, and `mujoco.mj_forward` gives the exact
Cartesian pose, both straight out of the model tree.

A note on rank
--------------
The SO-101 arm has five actuated joints, so a general 6-DoF pose is *not*
reachable. The reachable family is: pick an azimuth with `shoulder_pan`, and
the remaining four joints work in that vertical plane. For a top-down grasp
(approach pointing straight down) the wrist axis becomes vertical, so
`wrist_roll` freely sets the jaw azimuth and the full 6-DoF target *is*
consistent. `top_down_pose` builds exactly those targets; feeding the solver
anything else will make DLS return a least-squares compromise instead of an
exact solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .constants import ARM_JOINTS, TCP_SITE


def top_down_pose(position: np.ndarray, jaw_yaw: float) -> np.ndarray:
    """Rotation matrix for a top-down grasp with the jaws opening along `jaw_yaw`.

    Returns the 3x3 desired TCP orientation whose z axis (approach) points
    straight down and whose x axis (jaw opening direction) lies at `jaw_yaw`
    in the world xy plane.
    """
    del position  # kept in the signature for readability at call sites
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.array([np.cos(jaw_yaw), np.sin(jaw_yaw), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Rotation vector taking `current` onto `desired`, expressed in world frame."""
    r_err = desired @ current.T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, r_err.flatten())
    axis_angle = np.empty(3)
    mujoco.mju_quat2Vel(axis_angle, quat, 1.0)
    return axis_angle


@dataclass
class IKResult:
    qpos: np.ndarray
    pos_error: float
    rot_error: float
    iterations: int
    converged: bool


class SO101Kinematics:
    """FK/IK helper bound to a compiled model.

    Operates on the arm joints only; the gripper joint is passed through
    untouched so grasping never fights the Cartesian solver.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._data = mujoco.MjData(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.site_id < 0:
            raise ValueError(f"model has no site named {TCP_SITE!r}")

        self.arm_qpos_adr = np.array(
            [model.joint(name).qposadr[0] for name in ARM_JOINTS], dtype=int
        )
        self.arm_dof_adr = np.array(
            [model.joint(name).dofadr[0] for name in ARM_JOINTS], dtype=int
        )
        self.arm_lo = np.array([model.joint(n).range[0] for n in ARM_JOINTS])
        self.arm_hi = np.array([model.joint(n).range[1] for n in ARM_JOINTS])

    # ---------------------------------------------------------------- FK
    def fk(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (TCP position, TCP 3x3 rotation) for the given arm joints."""
        d = self._data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.arm_qpos_adr] = arm_q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(3, 3).copy()

    # ---------------------------------------------------------------- IK
    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        *,
        max_iters: int = 200,
        pos_tol: float = 1e-3,
        rot_tol: float = 2e-2,
        damping: float = 5e-2,
        rot_weight: float = 0.6,
        step_clip: float = 0.25,
    ) -> IKResult:
        """Damped least-squares IK on the TCP site.

        Solves ``(J^T W J + lambda^2 I) dq = J^T W e`` each iteration, which is
        the Levenberg-Marquardt form of the pseudo-inverse. The damping term
        keeps `dq` bounded when the Jacobian loses rank near a singularity or
        at a joint limit, at the cost of a little tracking accuracy.
        """
        model, d = self.model, self._data
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.arm_lo, self.arm_hi)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        weights = np.concatenate([np.ones(3), np.full(3, rot_weight)])
        W = np.diag(weights)

        pos_err_norm = rot_err_norm = np.inf
        it = 0
        for it in range(1, max_iters + 1):
            mujoco.mj_resetData(model, d)
            d.qpos[self.arm_qpos_adr] = q
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)

            cur_pos = d.site_xpos[self.site_id]
            cur_rot = d.site_xmat[self.site_id].reshape(3, 3)

            e_pos = target_pos - cur_pos
            e_rot = rotation_error(cur_rot, target_rot)
            pos_err_norm = float(np.linalg.norm(e_pos))
            rot_err_norm = float(np.linalg.norm(e_rot))
            if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
                break

            mujoco.mj_jacSite(model, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp, jacr])[:, self.arm_dof_adr]
            e = np.concatenate([e_pos, e_rot])

            JtW = J.T @ W
            dq = np.linalg.solve(JtW @ J + (damping**2) * np.eye(J.shape[1]), JtW @ e)

            norm = np.linalg.norm(dq)
            if norm > step_clip:
                dq *= step_clip / norm
            q = np.clip(q + dq, self.arm_lo, self.arm_hi)

        return IKResult(
            qpos=q,
            pos_error=pos_err_norm,
            rot_error=rot_err_norm,
            iterations=it,
            converged=bool(pos_err_norm < pos_tol and rot_err_norm < rot_tol),
        )

    def solve_top_down(
        self, position: np.ndarray, jaw_yaw: float, q_init: np.ndarray, **kw
    ) -> IKResult:
        """Convenience wrapper for the reachable top-down grasp family."""
        return self.solve(np.asarray(position, float), top_down_pose(position, jaw_yaw), q_init, **kw)
