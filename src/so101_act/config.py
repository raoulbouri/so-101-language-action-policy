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
Conditioning = Literal["none", "clip", "taskid"]

# Split modes:
#   iid           -- random episode-level split
#   compositional -- E5, whole (cube, zone) combinations held out for test
SplitMode = Literal["iid", "compositional"]


@dataclass
class Config:
    # ---- data -------------------------------------------------------------
    hdf5_path: str = "data/train_1200.hdf5"
    chunk_size: int = 32                 # ACT's action-chunk length k
    image_size: int = 128
    cameras: tuple[str, ...] = ("scene_image", "wrist_image")

    split_mode: SplitMode = "iid"
    val_frac: float = 0.10
    test_frac: float = 0.10
    # E5: (cube colour, zone colour) pairs held out of train/val entirely.
    holdout_combos: tuple[tuple[str, str], ...] = ()

    # ---- model ------------------------------------------------------------
    conditioning: Conditioning = "none"
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
