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
                    max_batches: int | None = None, *,
                    action_repr: str = "absolute") -> dict:
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
    # Scale that turns a normalised error back into radians. Under delta targets
    # the scale is per-horizon, so it is applied inside the loop rather than to
    # the pooled per-joint mean -- averaging normalised errors across horizons
    # first and multiplying by a single std afterwards would be wrong.
    if action_repr == "delta":
        scale = np.asarray(norm.delta_std, dtype=np.float64)[:n_h]   # (k, 6)
    else:
        scale = np.broadcast_to(
            np.asarray(norm.action_std, dtype=np.float64), (n_h, n_j)).copy()
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)

    sum_all = cnt_all = 0.0
    sum_j = np.zeros(n_j); sum_j_rad = np.zeros(n_j); cnt_j = 0.0
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
        sum_j_rad += (masked * scale_t).sum(dim=(0, 1)).cpu().numpy()
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
    per_joint_rad = safe(sum_j_rad, cnt_j)
    return {
        "masked_l1": sum_all / max(cnt_all, 1e-9),
        "per_joint_mae": {
            JOINT_NAMES[j]: {
                "normalized": float(per_joint_norm[j]),
                "radians": float(per_joint_rad[j]),
            } for j in range(n_j)
        },
        "per_phase_mae": {PHASE_NAMES[p]: (float(sum_p[p] / cnt_p[p]) if cnt_p[p] > 0 else None)
                          for p in range(n_p)},
        "error_vs_horizon": [float(x) for x in safe(sum_h, cnt_h)],
        "kl": kl_sum / max(n_batch, 1),
        "n_batches": n_batch,
    }


@torch.no_grad()
def shortcut_baselines(model, loader: DataLoader, device, norm, *,
                       action_repr: str = "absolute",
                       max_batches: int | None = None) -> dict:
    """Does the model beat the two predictors that solve NOTHING?

    ISSUE-017: with absolute action targets the recorded action is ~96% given by
    the observed joint state, so "command the current position at every horizon"
    scores better than the trained model at h=0 and correlates ~1.0 with the
    expert chunk across scenes -- while producing an arm that never moves. Every
    offline metric in this file was blind to that. These two baselines are the
    floor a policy must clear before any closed-loop number is worth spending
    GPU time on:

      IDENTITY     command qpos[t] for every horizon. Uses no image, no
                   language. Under delta targets this is the zero vector.
      SCENE_BLIND  predict the mean target over the batch. Uses nothing at all.

    L1 against these is necessary but NOT sufficient, and this was measured:
    the e2_clip_100k checkpoint beats IDENTITY by 76% overall and by 41% at its
    worst horizon, yet its rollout moves the cube 0.0 mm. L1 is dominated by the
    bulk of the action, which the observation already supplies.

    `lead_ratio` is the metric that actually separates them. The expert's
    command sits AHEAD of the measured state -- that gap is the entire motion
    command, mean 1.016 deg against an action std of 24.67 deg. A policy that
    reproduces the pose but shrinks the gap does not move the arm. Measured on
    the failing checkpoint: commanded lead 0.027 deg against the expert's
    1.016 deg, a ratio of 0.03. Treat a lead_ratio far below 1 as a hard stop,
    whatever the L1 says.
    """
    model.eval()
    n_h = model.chunk
    tot = {"model": 0.0, "identity": 0.0, "scene_blind": 0.0}
    per_h = {k: np.zeros(n_h) for k in tot}
    cnt = 0.0; cnt_h = np.zeros(n_h)
    lead_pred = lead_true = lead_cnt = 0.0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = _move(batch, device)
        with torch.no_grad():
            pred = model(batch, sample_latent=False)["actions"]
        tgt = batch["action_chunk"]
        m = batch["action_chunk_mask"].unsqueeze(-1).to(tgt.dtype)

        # "Hold the current position", expressed in the training representation.
        q_raw = batch["qpos_raw"].cpu().numpy()                    # (B,6)
        hold_raw = np.repeat(q_raw[:, None, :], n_h, axis=1)       # (B,k,6)
        if action_repr == "delta":
            hold = np.zeros_like(hold_raw)
            hold = (hold - norm.delta_mean[None, :n_h]) / norm.delta_std[None, :n_h]
        else:
            hold = (hold_raw - norm.action_mean) / norm.action_std
        ident = torch.as_tensor(hold, dtype=tgt.dtype, device=device)
        blind = tgt.mean(dim=0, keepdim=True).expand_as(tgt)

        for name, p in (("model", pred), ("identity", ident), ("scene_blind", blind)):
            e = (p - tgt).abs() * m
            tot[name] += e.sum().item()
            per_h[name] += e.sum(dim=(0, 2)).cpu().numpy()
        cnt += (m.sum() * tgt.shape[-1]).item()
        cnt_h += (m.squeeze(-1).sum(dim=0) * tgt.shape[-1]).cpu().numpy()

        # How far ahead of the measured state does each one command? This is
        # the motion signal; L1 is dominated by the pose, which is free.
        pn = pred.cpu().numpy(); tn = tgt.cpu().numpy()
        if action_repr == "delta":
            lp = pn * norm.delta_std[None, :n_h] + norm.delta_mean[None, :n_h]
            lt = tn * norm.delta_std[None, :n_h] + norm.delta_mean[None, :n_h]
        else:
            a_s = np.asarray(norm.action_std); a_m = np.asarray(norm.action_mean)
            lp = (pn * a_s + a_m) - q_raw[:, None, :]
            lt = (tn * a_s + a_m) - q_raw[:, None, :]
        mm = m.cpu().numpy()
        lead_pred += float((np.abs(lp) * mm).sum())
        lead_true += float((np.abs(lt) * mm).sum())
        lead_cnt += float(mm.sum() * tgt.shape[-1])

    l1 = {k: v / max(cnt, 1e-9) for k, v in tot.items()}
    gain = {k: float(100.0 * (1.0 - l1["model"] / max(l1[k], 1e-9)))
            for k in ("identity", "scene_blind")}
    hz = {k: (per_h[k] / np.maximum(cnt_h, 1e-9)) for k in per_h}
    worst = float(min(
        100.0 * (1.0 - hz["model"][h] / max(hz["identity"][h], 1e-9))
        for h in range(n_h)))
    lp = lead_pred / max(lead_cnt, 1e-9)
    lt = lead_true / max(lead_cnt, 1e-9)
    ratio = lp / max(lt, 1e-9)
    return {
        "commanded_lead_rad": float(lp),
        "expert_lead_rad": float(lt),
        "lead_ratio": float(ratio),
        "under_driving": bool(ratio < 0.5),
        "l1": {k: float(v) for k, v in l1.items()},
        "gain_over_identity_pct": gain["identity"],
        "gain_over_scene_blind_pct": gain["scene_blind"],
        "worst_horizon_gain_over_identity_pct": worst,
        "beats_identity_at_every_horizon": bool(worst > 0.0),
        # The gate. L1 alone passes models whose arm never moves.
        "passes": bool(worst > 0.0 and ratio >= 0.5),
        "error_vs_horizon": {k: [float(x) for x in v] for k, v in hz.items()},
    }


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


# Windows chosen around what the observation can and cannot explain. At t=0-4
# the arm is at home with 2-3 cubes visible and ONLY the instruction says which
# to fetch, so a grounded policy must degrade sharply when language is shuffled.
# Later windows are progressively more determined by vision alone, because the
# arm is already committed to a target.
DEFAULT_WINDOWS = ((0, 5), (0, 20), (20, 60), (60, 150), (150, 10_000))


@torch.no_grad()
def windowed_language_sensitivity(
    model, hdf5_path, episodes, indices, norm, cfg, device, *,
    windows=DEFAULT_WINDOWS, max_samples: int = 640, batch_size: int = 32,
) -> dict:
    """E3, resolved by timestep window rather than averaged over the episode.

    Whole-episode E3 dilutes the signal badly: only ~4 % of timesteps are ones
    where language is the sole disambiguator, so a policy can ignore language
    entirely and still move the episode-average by a fraction of a percent. This
    reports the degradation where it should be largest.

    Uses stored observations, so it is valid even when closed-loop success is 0.
    """
    from .data import SO101ACTDataset, collate

    model.eval()

    def build(shuffle: bool):
        return SO101ACTDataset(
            hdf5_path, episodes, indices, norm, chunk_size=cfg.chunk_size,
            cameras=cfg.cameras, conditioning=cfg.conditioning,
            shuffle_language=shuffle, language_seed=cfg.seed,
            action_repr=getattr(cfg, "action_repr", "absolute"),
        )

    plain, shuf = build(False), build(True)

    def masked_l1(ds, lo, hi):
        idx = [i for i, (_, t) in enumerate(ds._index) if lo <= t < hi]
        if not idx:
            return None
        idx = idx[:max_samples]
        tot = cnt = 0.0
        for s in range(0, len(idx), batch_size):
            b = collate([ds[i] for i in idx[s:s + batch_size]])
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b)
            err = (out["actions"] - b["action_chunk"]).abs()
            m = b["action_chunk_mask"].unsqueeze(-1).to(err.dtype)
            tot += (err * m).sum().item()
            cnt += (m.sum() * err.shape[-1]).item()
        return tot / max(cnt, 1e-9)

    rows = {}
    for lo, hi in windows:
        n = masked_l1(plain, lo, hi)
        sh = masked_l1(shuf, lo, hi)
        if n is None:
            continue
        # Inclusive label for a half-open window: [lo, hi) -> "t=lo..hi-1".
        rows[f"t={lo}..{min(hi, 443) - 1}"] = {
            "normal": n, "shuffled": sh,
            "degradation": sh - n,
            "degradation_pct": 100.0 * (sh - n) / max(n, 1e-9),
        }
    return rows


def save_report(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=float))
