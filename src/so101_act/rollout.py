"""Closed-loop evaluation: run the policy in MuJoCo and score real success.

This is the only evidence that actually counts. Offline L1 measures agreement
with an expert on states the expert visited; success measures whether the policy
can reach the goal from its own induced state distribution, and -- when the
instruction is swapped -- whether it goes to the *named* target rather than a
memorised one.

Temporal ensembling is ACT's inference-time smoother: at each timestep the
policy predicts a full k-step chunk, so any given timestep is covered by up to k
overlapping predictions, which are averaged with exponential weights. The same
aggregation is used for every experiment.
"""

from __future__ import annotations

from collections import defaultdict, deque

import mujoco
import numpy as np
import torch

from so101_sim.constants import (
    ALL_JOINTS,
    HOME_QPOS,
    PHYSICS_SUBSTEPS,
    SCENE_CAM,
    WRIST_CAM,
)
from so101_sim.evaluation import evaluate_placement
from so101_sim.randomization import sample_episode
from so101_sim.scene_builder import build_model as build_scene

from .data import IMAGENET_MEAN, IMAGENET_STD


class TemporalEnsembler:
    """ACT's exponentially-weighted average over overlapping action chunks.

    The ACT paper defines the weights as ``w_i = exp(-m * i)`` where **``w_0`` is
    the weight of the OLDEST action** in the buffer, and smaller ``m`` means new
    observations are incorporated faster. An earlier version of this class had
    the ordering inverted -- it weighted the *newest* prediction highest -- which
    measurably degraded rollout (divergence at t=40 was 0.410 with ensembling
    against 0.259 when replanning every step). See ISSUE-007.
    """

    def __init__(self, chunk: int, action_dim: int, m: float = 0.01):
        self.chunk = chunk
        self.m = m
        self.buf: deque[tuple[int, np.ndarray]] = deque()
        self.action_dim = action_dim

    def add(self, t: int, chunk: np.ndarray) -> None:
        self.buf.append((t, chunk))
        while self.buf and self.buf[0][0] <= t - self.chunk:
            self.buf.popleft()

    def action_for(self, t: int) -> np.ndarray:
        # Buffer is append-ordered, so iterating it yields oldest -> newest.
        preds = []
        for t0, ch in self.buf:
            h = t - t0
            if 0 <= h < self.chunk:
                preds.append(ch[h])
        if not preds:
            return np.zeros(self.action_dim, dtype=np.float32)
        # i = 0 is the oldest surviving prediction, per the paper.
        w = np.exp(-self.m * np.arange(len(preds), dtype=np.float64))
        w /= w.sum()
        return (np.asarray(preds, dtype=np.float64) * w[:, None]).sum(0).astype(np.float32)


def _prep_image(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@torch.no_grad()
def rollout_episode(
    model, norm, device, seed: int, *,
    language_embedding: np.ndarray | None = None,
    task_id: int | None = None,
    max_steps: int = 500,
    image_size: int = 128,
    settle_seconds: float = 0.6,
    control_hz: float = 50.0,
    ensemble_m: float = 0.01,
    use_ensembling: bool = False,
    n_action_steps: int | None = None,
    renderer=None,
) -> dict:
    """Run one closed-loop episode. `language_embedding` overrides the episode's
    own instruction, which is how the wrong-language evaluation is done."""
    spec = sample_episode(seed)
    scene = build_scene(spec)
    data = mujoco.MjData(scene)

    adr = np.array([scene.joint(n).qposadr[0] for n in ALL_JOINTS], dtype=int)
    data.qpos[adr] = HOME_QPOS
    data.ctrl[:] = HOME_QPOS
    mujoco.mj_forward(scene, data)
    for _ in range(int(0.4 * control_hz) * PHYSICS_SUBSTEPS):
        mujoco.mj_step(scene, data)

    own = renderer is None
    if own:
        renderer = mujoco.Renderer(scene, height=image_size, width=image_size)

    if language_embedding is None:
        raise ValueError("language_embedding must be supplied by the caller")

    lang = torch.as_tensor(language_embedding, dtype=torch.float32,
                           device=device).unsqueeze(0)
    tid = (torch.tensor([task_id], device=device)
           if task_id is not None else torch.zeros(1, dtype=torch.long, device=device))

    # LeRobot's ACTConfig sets temporal_ensemble_coeff = None, i.e. ensembling is
    # OFF by default and the chunk is executed open-loop for n_action_steps
    # before replanning. Our own rollout measurements agreed: ensembling was
    # worse than every alternative. See ISSUE-009.
    ens = TemporalEnsembler(model.chunk, 6, m=ensemble_m) if use_ensembling else None
    n_exec = n_action_steps or model.chunk
    pending: np.ndarray | None = None
    model.eval()
    try:
        for t in range(max_steps):
            renderer.update_scene(data, camera=SCENE_CAM)
            scene_img = _prep_image(renderer.render())
            renderer.update_scene(data, camera=WRIST_CAM)
            wrist_img = _prep_image(renderer.render())
            qpos = data.qpos[adr].astype(np.float32)

            batch = {
                "scene_image": torch.from_numpy(scene_img).unsqueeze(0).to(device),
                "wrist_image": torch.from_numpy(wrist_img).unsqueeze(0).to(device),
                "qpos": torch.from_numpy(norm.norm_qpos(qpos)).unsqueeze(0).to(device),
                "language_embedding": lang,
                "task_id": tid,
            }
            if ens is not None:
                chunk = norm.denorm_action(model(batch)["actions"][0].cpu().numpy())
                ens.add(t, chunk)
                action = ens.action_for(t)
            else:
                # Replan every n_exec steps, execute the chunk open-loop between.
                if pending is None or t % n_exec == 0:
                    pending = norm.denorm_action(
                        model(batch)["actions"][0].cpu().numpy())
                action = pending[t % n_exec]
            data.ctrl[:] = action
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(scene, data)
    finally:
        if own:
            renderer.close()

    for _ in range(int(settle_seconds * control_hz) * PHYSICS_SUBSTEPS):
        mujoco.mj_step(scene, data)

    bid = scene.body(spec.target_object.name).id
    pos = data.xpos[bid].copy()
    quat = data.xquat[bid].copy()
    yaw = float(np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                           1 - 2 * (quat[2] ** 2 + quat[3] ** 2)))
    free = scene.joint(f"{spec.target_object.name}_free").dofadr[0]
    vel = data.qvel[free:free + 3].copy()
    report = evaluate_placement(pos, yaw, vel, np.array(spec.target_zone.pos))

    return {
        "seed": seed,
        "instruction": spec.instruction,
        "cube": spec.target_object.color,
        "zone": spec.target_zone.color,
        "success": bool(report.success),
        "reason": report.reason,
        "center_distance": float(report.center_distance),
    }


def summarize(results: list[dict]) -> dict:
    """Overall success plus the breakdowns required for the report."""
    n = len(results)
    def by(key):
        return {k: (sum(r["success"] for r in v) / len(v), len(v))
                for k, v in _group(results, key).items()}
    return {
        "n": n,
        "success_rate": sum(r["success"] for r in results) / max(n, 1),
        "per_instruction": by("instruction"),
        "per_cube_colour": by("cube"),
        "per_zone_colour": by("zone"),
        "failure_reasons": dict(_count(r["reason"] for r in results if not r["success"])),
    }


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def _count(it):
    out = defaultdict(int)
    for x in it:
        out[x] += 1
    return out
