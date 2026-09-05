"""Configuration for ACT training.

One dataclass drives both experiments. The only difference between the baseline
and the language-conditioned model is `use_language` / `conditioning`, so every
other hyperparameter is guaranteed identical between them -- which is what makes
E1 vs E2 a controlled comparison rather than two unrelated runs.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Conditioning modes:
#   none    -- E1 baseline, pi(A | I, q, z)
#   clip    -- E2, pi(A | I, q, L, z) with frozen CLIP embedding
#   taskid  -- E6, same architecture but L is a learned embedding of the task id
#   film    -- Tier 1, CLIP modulates the visual backbone via FiLM (RT-1 style),
#              with NO language token
#   film_token -- both FiLM and the language token
Conditioning = Literal["none", "clip", "taskid", "film", "film_token"]

# Split modes:
#   iid           -- random episode-level split
#   compositional -- E5, whole (cube, zone) combinations held out for test
SplitMode = Literal["iid", "compositional"]

# Action representation:
#   absolute -- predict joint targets directly. This is what ACT/ALOHA does, but
#               it assumes the action differs meaningfully from the observed
#               state. Here the expert is a smooth quintic tracked by a stiff
#               position servo, so mean |action - qpos| is 1.06 deg against an
#               action std of 24.67 deg: the command-ahead term is 4.3% of the
#               action range (0.6% on shoulder_pan). The L1 optimum is then to
#               echo the current position, and the arm never moves. See
#               ISSUE-017.
#   delta    -- predict action[t+h] - qpos[t], normalised per horizon, so the
#               signal fills the target range at every h and "hold position"
#               stops being a good minimiser.
ActionRepr = Literal["absolute", "delta"]


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    hdf5_path: str = "data/train_1200.hdf5"
    # ACT and LeRobot both default to chunk_size = n_action_steps = 100. At
    # 50 Hz that is 2 s of lookahead (matching ALOHA) versus 0.64 s at k=32.
    # Long chunks are ACT's mechanism for suppressing compounding error, so a
    # smaller k gives away the method's main benefit. Chunks are sliced from the
    # dataset's `action` array, so k is not limited by the stored action_chunk.
    chunk_size: int = 100                # ACT's action-chunk length k
    # Actions executed open-loop before replanning. LeRobot sets this equal to
    # chunk_size; None means "use chunk_size".
    n_action_steps: int | None = None
    # LeRobot's temporal_ensemble_coeff defaults to None (ensembling OFF).
    # Measured here: ensembling was worse than every alternative. See ISSUE-009.
    use_ensembling: bool = False
    ensemble_m: float = 0.01
    image_size: int = 128
    cameras: tuple[str, ...] = ("scene_image", "wrist_image")

    split_mode: SplitMode = "iid"
    val_frac: float = 0.10
    test_frac: float = 0.10
    # E5: (cube colour, zone colour) pairs held out of train/val entirely.
    holdout_combos: tuple[tuple[str, str], ...] = ()

    # ---- model ------------------------------------------------------------
    conditioning: Conditioning = "none"
    # Defaults to "absolute" so configs written before ISSUE-017 keep loading
    # and evaluating exactly as they were trained. New runs should pass
    # --action-repr delta.
    action_repr: ActionRepr = "absolute"
    hidden_dim: int = 512                # ACT default d_model
    dim_feedforward: int = 3200          # ACT default
    nheads: int = 8
    enc_layers: int = 4                  # ACT default transformer encoder depth
    dec_layers: int = 7                  # ACT default transformer decoder depth
    latent_dim: int = 32                 # ACT default CVAE latent size
    cvae_enc_layers: int = 4
    dropout: float = 0.1
    backbone: str = "resnet18"
    pretrained_backbone: bool = True
    share_backbone: bool = True          # one backbone for both cameras
    lang_dim: int = 512                  # CLIP ViT-B/32 text embedding width
    n_task_ids: int = 16                 # 4 cube colours x 4 zone colours

    # ---- loss -------------------------------------------------------------
    kl_weight: float = 10.0              # ACT default beta

    # ---- optimisation -----------------------------------------------------
    lr: float = 1e-5                     # ACT default
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_steps: int = 100_000
    warmup_steps: int = 500
    grad_clip: float = 10.0
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"

    # ---- experiment tracking ----------------------------------------------
    # wandb is entirely optional: leave wandb_project empty to disable. The API
    # key is NEVER written to config.json (see to_dict) so a run directory can
    # be copied or committed without leaking a credential.
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_group: str = ""
    wandb_api_key: str = ""

    # ---- bookkeeping ------------------------------------------------------
    out_dir: str = "runs/act"
    log_every: int = 50
    val_every: int = 2000
    ckpt_every: int = 5000
    max_val_batches: int = 50
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def use_language(self) -> bool:
        return self.conditioning != "none"

    def to_dict(self, redact_secrets: bool = True) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if redact_secrets and d.get("wandb_api_key"):
            d["wandb_api_key"] = "<redacted>"
        return d

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=list))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        d = json.loads(Path(path).read_text())
        if d.get("wandb_api_key") == "<redacted>":
            d["wandb_api_key"] = ""
        d["cameras"] = tuple(d.get("cameras", ("scene_image", "wrist_image")))
        d["holdout_combos"] = tuple(tuple(c) for c in d.get("holdout_combos", ()))
        return cls(**d)
