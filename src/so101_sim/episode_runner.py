"""Roll one episode of the scripted expert and record aligned frames.

Frame alignment contract (this is the part that quietly ruins ACT training if
you get it wrong):

    at record index t
        obs[t]     = the state observed BEFORE anything is commanded at step t
        action[t]  = the target joint position commanded at step t
        the simulator then advances, producing obs[t+1]

So `action[t]` is always the action *taken from* `obs[t]`. There is no
resampling between the control and record streams -- they run at the same rate
-- so no off-by-one can creep in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .constants import (
    ALL_JOINTS,
    CONTROL_HZ,
    HOME_QPOS,
    IMAGE_SIZE,
    PHYSICS_SUBSTEPS,
    SCENE_CAM,
    TCP_SITE,
    WRIST_CAM,
)
from .evaluation import SuccessReport, evaluate_placement
from .expert_policy import PHASES, ExpertPlanner
from .kinematics import SO101Kinematics
from .randomization import EpisodeSpec
from .scene_builder import build_model


@dataclass
class Episode:
    spec: EpisodeSpec
    instruction: str
    qpos: np.ndarray            # (T, 6) observed joint positions
    qvel: np.ndarray            # (T, 6)
    action: np.ndarray          # (T, 6) commanded target joint positions
    tcp_pose: np.ndarray        # (T, 7) xyz + wxyz quaternion
    scene_image: np.ndarray     # (T, H, W, 3) uint8
    wrist_image: np.ndarray     # (T, H, W, 3) uint8
    phase_ids: np.ndarray       # (T,)
    object_poses: np.ndarray    # (T, n_objects, 7)
    success: SuccessReport
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.qpos.shape[0])


class EpisodeRunner:
    """Owns a renderer across episodes -- creating one per episode is slow."""

    def __init__(self, image_size: tuple[int, int] = IMAGE_SIZE, render: bool = True):
        self.image_size = image_size
        self.render = render
        self._renderer: mujoco.Renderer | None = None
        self._renderer_model_id: int | None = None

    def _get_renderer(self, model: mujoco.MjModel) -> mujoco.Renderer:
        # Scene topology changes between episodes (different object counts), so
        # the renderer has to be rebuilt whenever the model does.
        if self._renderer is None or self._renderer_model_id != id(model):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(model, height=self.image_size[0],
                                             width=self.image_size[1])
            self._renderer_model_id = id(model)
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    def run(self, spec: EpisodeSpec, settle_seconds: float = 0.6) -> Episode:
        model = build_model(spec)
        data = mujoco.MjData(model)

        joint_qpos_adr = np.array([model.joint(n).qposadr[0] for n in ALL_JOINTS], dtype=int)
        joint_dof_adr = np.array([model.joint(n).dofadr[0] for n in ALL_JOINTS], dtype=int)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        obj_body_ids = [model.body(o.name).id for o in spec.objects]

        # Start at home, fully settled, so frame 0 is a valid static state.
        data.qpos[joint_qpos_adr] = HOME_QPOS
        data.ctrl[:] = HOME_QPOS
        mujoco.mj_forward(model, data)
        for _ in range(int(0.4 * CONTROL_HZ) * PHYSICS_SUBSTEPS):
            mujoco.mj_step(model, data)

        planner = ExpertPlanner(SO101Kinematics(model), control_hz=CONTROL_HZ)
        traj = planner.plan(
            np.array(spec.target_object.pos),
            spec.target_object.yaw,
            np.array(spec.target_zone.pos),
            q_home=HOME_QPOS,
        )

        renderer = self._get_renderer(model) if self.render else None
        h, w = self.image_size
        n = len(traj)

        qpos_buf = np.empty((n, 6), np.float32)
        qvel_buf = np.empty((n, 6), np.float32)
        act_buf = np.empty((n, 6), np.float32)
        tcp_buf = np.empty((n, 7), np.float32)
        obj_buf = np.empty((n, len(obj_body_ids), 7), np.float32)
        scene_buf = np.zeros((n, h, w, 3), np.uint8)
        wrist_buf = np.zeros((n, h, w, 3), np.uint8)

        quat = np.empty(4)
        for t in range(n):
            # --- observe BEFORE commanding ---------------------------------
            qpos_buf[t] = data.qpos[joint_qpos_adr]
            qvel_buf[t] = data.qvel[joint_dof_adr]
            mujoco.mju_mat2Quat(quat, data.site_xmat[site_id])
            tcp_buf[t, :3] = data.site_xpos[site_id]
            tcp_buf[t, 3:] = quat
            for k, bid in enumerate(obj_body_ids):
                obj_buf[t, k, :3] = data.xpos[bid]
                obj_buf[t, k, 3:] = data.xquat[bid]
            if renderer is not None:
                renderer.update_scene(data, camera=SCENE_CAM)
                scene_buf[t] = renderer.render()
                renderer.update_scene(data, camera=WRIST_CAM)
                wrist_buf[t] = renderer.render()

            # --- command, then advance -------------------------------------
            act_buf[t] = traj.qpos[t]
            data.ctrl[:] = traj.qpos[t]
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, data)

        # Let the scene come to rest before scoring; not part of the dataset.
        for _ in range(int(settle_seconds * CONTROL_HZ) * PHYSICS_SUBSTEPS):
            mujoco.mj_step(model, data)

        target_bid = model.body(spec.target_object.name).id
        cube_pos = data.xpos[target_bid].copy()
        cube_quat = data.xquat[target_bid].copy()
        cube_yaw = float(np.arctan2(
            2.0 * (cube_quat[0] * cube_quat[3] + cube_quat[1] * cube_quat[2]),
            1.0 - 2.0 * (cube_quat[2] ** 2 + cube_quat[3] ** 2),
        ))
        free_adr = model.joint(f"{spec.target_object.name}_free").dofadr[0]
        cube_vel = data.qvel[free_adr:free_adr + 3].copy()

        report = evaluate_placement(
            cube_pos, cube_yaw, cube_vel, np.array(spec.target_zone.pos)
        )

        return Episode(
            spec=spec,
            instruction=spec.instruction,
            qpos=qpos_buf,
            qvel=qvel_buf,
            action=act_buf,
            tcp_pose=tcp_buf,
            scene_image=scene_buf,
            wrist_image=wrist_buf,
            phase_ids=traj.phase_ids,
            object_poses=obj_buf,
            success=report,
            diagnostics={
                "ik_failures": int(traj.ik_failures),
                "ik_max_pos_error": float(traj.ik_max_pos_error),
                "n_steps": n,
                "phases": list(PHASES),
                "final_cube_pos": cube_pos.tolist(),
                "final_cube_yaw": cube_yaw,
            },
        )
