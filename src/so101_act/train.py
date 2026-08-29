"""Training loop for ACT, shared by the baseline and every language variant.

Nothing here branches on `use_language` except the model construction, so E1 and
E2 necessarily run with the same optimiser, schedule, budget, normalization and
splits. That is what makes the comparison controlled rather than anecdotal.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    SO101ACTDataset,
    collate,
    compute_normalizer,
    make_splits,
    scan_episodes,
)
from .model import act_loss, build_model


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay -- one schedule for every experiment."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    p = (step - cfg.warmup_steps) / max(cfg.num_steps - cfg.warmup_steps, 1)
    return cfg.lr * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))


def move(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.episodes, self.task_ids = scan_episodes(cfg.hdf5_path)
        self.splits = make_splits(
            self.episodes, mode=cfg.split_mode, val_frac=cfg.val_frac,
            test_frac=cfg.test_frac, holdout_combos=cfg.holdout_combos, seed=cfg.seed,
        )
        # Normalization from TRAIN episodes only.
        self.norm = compute_normalizer(
            cfg.hdf5_path, self.episodes, self.splits["train"], seed=cfg.seed
        )

        self.model = build_model(cfg).to(self.device)

        # ACT convention: a lower LR on the pretrained visual backbone.
        backbone_ids = {id(p) for p in self.model.backbones.parameters()}
        groups = [
            {"params": [p for p in self.model.parameters()
                        if id(p) not in backbone_ids and p.requires_grad],
             "lr": cfg.lr},
            {"params": [p for p in self.model.backbones.parameters() if p.requires_grad],
             "lr": cfg.lr_backbone},
        ]
        self.opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self._save_run_metadata()

    # ------------------------------------------------------------------
    def _save_run_metadata(self) -> None:
        """Everything needed to reproduce or audit this run."""
        self.cfg.save(self.out / "config.json")
        self.norm.save(self.out / "norm_stats.json")
        (self.out / "splits.json").write_text(json.dumps({
            k: [self.episodes[i].name for i in v] for k, v in self.splits.items()
        }, indent=2))
        (self.out / "run_meta.json").write_text(json.dumps({
            "git_hash": git_hash(),
            "seed": self.cfg.seed,
            "device": str(self.device),
            "task_ids": {f"{c}->{z}": i for (c, z), i in self.task_ids.items()},
            "n_train_episodes": len(self.splits["train"]),
            "n_val_episodes": len(self.splits["val"]),
            "n_test_episodes": len(self.splits["test"]),
            "n_params": sum(p.numel() for p in self.model.parameters()),
            "config": asdict(self.cfg),
        }, indent=2, default=list))

    def loader(self, split: str, *, shuffle=True, indices=None, batch_size=None,
               shuffle_language=False) -> DataLoader:
        idx = self.splits[split] if indices is None else indices
        ds = SO101ACTDataset(
            self.cfg.hdf5_path, self.episodes, idx, self.norm,
            chunk_size=self.cfg.chunk_size, cameras=self.cfg.cameras,
            conditioning=self.cfg.conditioning, shuffle_language=shuffle_language,
            language_seed=self.cfg.seed,
        )
        return DataLoader(
            ds, batch_size=batch_size or self.cfg.batch_size, shuffle=shuffle,
            num_workers=self.cfg.num_workers, collate_fn=collate,
            persistent_workers=self.cfg.num_workers > 0, drop_last=shuffle,
        )

    # ------------------------------------------------------------------
    def train_step(self, batch: dict, step: int) -> dict[str, float]:
        lr = lr_at(step, self.cfg)
        self.opt.param_groups[0]["lr"] = lr
        self.opt.param_groups[1]["lr"] = lr * (self.cfg.lr_backbone / self.cfg.lr)

        self.model.train()
        out = self.model(batch)
        losses = act_loss(out, batch, self.cfg.kl_weight)

        self.opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip
        )
        self.opt.step()
        return {
            "loss": losses["loss"].item(),
            "action_loss": losses["action_loss"].item(),
            "kl": losses["kl"].item(),
            "lr": lr,
            "grad_norm": float(gnorm),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        tot = {"action_loss": 0.0, "kl": 0.0}
        n = 0
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = move(batch, self.device)
            out = self.model(batch)
            L = act_loss(out, batch, self.cfg.kl_weight)
            tot["action_loss"] += L["action_loss"].item()
            tot["kl"] += L["kl"].item()
            n += 1
        return {k: v / max(n, 1) for k, v in tot.items()}

    def save_checkpoint(self, step: int, name: str = "latest") -> Path:
        p = self.out / f"ckpt_{name}.pt"
        torch.save({
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": asdict(self.cfg),
            "norm": self.norm.to_dict(),
            "git_hash": git_hash(),
        }, p)
        return p

    # ------------------------------------------------------------------
    def fit(self) -> dict:
        cfg = self.cfg
        train_loader = self.loader("train")
        val_loader = self.loader("val", shuffle=False)
        log_path = self.out / "train_log.jsonl"
        log_f = log_path.open("a")

        print(f"device={self.device} params={sum(p.numel() for p in self.model.parameters())/1e6:.1f}M "
              f"conditioning={cfg.conditioning}")
        print(f"episodes train/val/test = {len(self.splits['train'])}/"
              f"{len(self.splits['val'])}/{len(self.splits['test'])}")

        step = 0
        t0 = time.time()
        best_val = float("inf")
        it = iter(train_loader)
        while step < cfg.num_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            stats = self.train_step(move(batch, self.device), step)

            if step % cfg.log_every == 0:
                rec = {"step": step, "time": round(time.time() - t0, 1), **stats}
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                print(f"[{step:6d}] loss {stats['loss']:8.4f} | act {stats['action_loss']:7.4f} "
                      f"| kl {stats['kl']:7.4f} | lr {stats['lr']:.2e} "
                      f"| gnorm {stats['grad_norm']:6.2f} | {time.time()-t0:6.0f}s", flush=True)

            if step > 0 and step % cfg.val_every == 0:
                v = self.evaluate(val_loader, cfg.max_val_batches)
                rec = {"step": step, "val_action_loss": v["action_loss"], "val_kl": v["kl"]}
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                print(f"  val: action {v['action_loss']:.4f} kl {v['kl']:.4f}", flush=True)
                if v["action_loss"] < best_val:
                    best_val = v["action_loss"]
                    self.save_checkpoint(step, "best")

            if step > 0 and step % cfg.ckpt_every == 0:
                self.save_checkpoint(step, "latest")
            step += 1

        self.save_checkpoint(step, "final")
        log_f.close()
        return {"steps": step, "best_val_action_loss": best_val,
                "minutes": (time.time() - t0) / 60}
