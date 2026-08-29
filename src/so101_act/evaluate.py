"""Offline evaluation and the language-sensitivity experiments.

Offline L1 alone cannot demonstrate language grounding -- a model that ignores
the instruction can still score well on IID data by exploiting visual cues. The
experiments here are designed to separate those explanations:

  E3 (shuffle)        pair each observation with another episode's instruction.
                      A grounded model gets *worse*; a language-blind one does
                      not move.
  E4 (counterfactual) hold the observation exactly fixed and swap only the
                      instruction, measuring how far the predicted chunk moves.
                      A language-blind model gives D = 0 by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SO101ACTDataset, collate
from .model import act_loss

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
PHASE_NAMES = ["hover_and_center", "approach", "grasp", "lift",
               "transit", "place", "release_and_clear"]


def _move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def offline_metrics(model, loader: DataLoader, device, kl_weight: float, norm,
                    max_batches: int | None = None) -> dict:
    """Masked L1 overall, per joint, per phase, and per chunk-horizon step.

    Units: `masked_l1` and `error_vs_horizon` average over all 6 joints, whose
    normalized scales differ, so they are reported in *normalized* action units
    only -- a single "radians" number mixing joints with different std would not
    mean anything. `per_joint_mae` is reported in BOTH units: normalized (for
    cross-joint comparability) and radians (`* norm.action_std`, for physical
    interpretation), since that is the whole point of a per-joint breakdown.

    Per-phase bucketing uses `phase_chunk[h]` -- the phase of the action being
    PREDICTED at horizon `h` -- not the phase of the anchor observation at `t`.
    A chunk started late in a short phase (`grasp` is ~35 of 444 steps) can run
    into the next phase; bucketing by the anchor's phase would blend that
    error into the wrong phase.
    """
    model.eval()
    n_j, n_p, n_h = len(JOINT_NAMES), len(PHASE_NAMES), model.chunk
    std = np.asarray(norm.action_std, dtype=np.float64)

    sum_all = cnt_all = 0.0
    sum_j = np.zeros(n_j); cnt_j = 0.0
    sum_h = np.zeros(n_h); cnt_h = np.zeros(n_h)
    sum_p = np.zeros(n_p); cnt_p = np.zeros(n_p)
    kl_sum = 0.0; n_batch = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = _move(batch, device)
        out = model(batch)
        err = (out["actions"] - batch["action_chunk"]).abs()      # (B,k,6)
        m = batch["action_chunk_mask"].unsqueeze(-1).to(err.dtype)  # (B,k,1)
        masked = err * m                                          # (B,k,6)

        sum_all += masked.sum().item(); cnt_all += (m.sum() * n_j).item()
        sum_j += masked.sum(dim=(0, 1)).cpu().numpy()
        cnt_j += m.sum().item()   # same valid (batch,horizon) count for every joint
        sum_h += masked.sum(dim=(0, 2)).cpu().numpy()
        cnt_h += (m.squeeze(-1).sum(dim=0) * n_j).cpu().numpy()

        # Per-target-phase: bucket each (sample, horizon) error by the phase of
        # the action actually being predicted at that horizon step.
        phase_chunk = batch["phase_chunk"].cpu().numpy()          # (B,k)
        per_bh = masked.sum(dim=2).cpu().numpy()                  # (B,k) summed over joints
        cnt_bh = (m.squeeze(-1) * n_j).cpu().numpy()               # (B,k)
        for p in range(n_p):
            sel = phase_chunk == p
            if sel.any():
                sum_p[p] += per_bh[sel].sum()
                cnt_p[p] += cnt_bh[sel].sum()

        kl_sum += act_loss(out, batch, kl_weight)["kl"].item()
        n_batch += 1

    def safe(s, c):
        return s / np.maximum(c, 1e-9)

    per_joint_norm = safe(sum_j, cnt_j)
    return {
        "masked_l1": sum_all / max(cnt_all, 1e-9),
        "per_joint_mae": {
            JOINT_NAMES[j]: {
                "normalized": float(per_joint_norm[j]),
                "radians": float(per_joint_norm[j] * std[j]),
            } for j in range(n_j)
        },
        "per_phase_mae": {PHASE_NAMES[p]: (float(sum_p[p] / cnt_p[p]) if cnt_p[p] > 0 else None)
                          for p in range(n_p)},
        "error_vs_horizon": [float(x) for x in safe(sum_h, cnt_h)],
        "kl": kl_sum / max(n_batch, 1),
        "n_batches": n_batch,
    }


@torch.no_grad()
def counterfactual_divergence(model, dataset: SO101ACTDataset, device, *,
                              n_samples: int = 200, seed: int = 0) -> dict:
    """E4: same observation, different instruction.

        D = (1/k) * sum_h || pi(O, L1)_h - pi(O, L2)_h ||_2

    Reported alongside a same-language control: re-running with the *identical*
    instruction must give D = 0, which proves the metric is measuring the
    language swap and not sampling noise.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(dataset)
    picks = rng.choice(n, size=min(n_samples, n), replace=False)

    # Pool of distinct instruction embeddings present in the dataset.
    emb_pool, seen = [], set()
    for ei in dataset.indices:
        info = dataset.episodes[ei]
        if info.instruction not in seen:
            seen.add(info.instruction)
            s = dataset[dataset._index.index((ei, 0))]
            emb_pool.append((info.instruction, s["language_embedding"]))
    if len(emb_pool) < 2:
        raise ValueError("need at least two distinct instructions for E4")

    d_swap, d_same = [], []
    for i in picks:
        s = dataset[int(i)]
        base = collate([s])
        base = _move(base, device)
        a1 = model(base)["actions"]

        # control: identical language
        a_same = model(base)["actions"]
        d_same.append(torch.norm(a1 - a_same, dim=-1).mean().item())

        # swap: a different instruction, same observation
        cur = s["instruction"]
        alt = [e for t, e in emb_pool if t != cur]
        emb = alt[int(rng.integers(len(alt)))]
        b2 = dict(base)
        b2["language_embedding"] = emb.unsqueeze(0).to(device)
        a2 = model(b2)["actions"]
        d_swap.append(torch.norm(a1 - a2, dim=-1).mean().item())

    return {
        "D_language_swap": float(np.mean(d_swap)),
        "D_same_language_control": float(np.mean(d_same)),
        "D_swap_std": float(np.std(d_swap)),
        "n_samples": len(picks),
        "n_distinct_instructions": len(emb_pool),
    }


def save_report(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=float))
