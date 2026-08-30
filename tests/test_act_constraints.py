"""Mechanical checks on the task specification's hard constraints (section 13).

These encode the "Do NOT" list so a future edit cannot quietly violate it.
Several are deliberately structural (inspecting source / parameter names) rather
than behavioural, because the failure they guard against is someone *adding* a
forbidden input, which no behavioural test would notice.
"""

import inspect
from pathlib import Path

import torch

from so101_act import model as model_mod
from so101_act.config import Config
from so101_act.data import SO101ACTDataset
from so101_act.model import build_model

FORBIDDEN = ["object_poses", "target_object", "target_zone",
             "zone_labels", "object_labels"]


def _cfg(cond="clip"):
    return Config(conditioning=cond, pretrained_backbone=False, hidden_dim=64,
                  dim_feedforward=128, nheads=4, enc_layers=1, dec_layers=1,
                  cvae_enc_layers=1)


def test_policy_never_reads_privileged_state():
    """object_poses / target labels must not appear anywhere in the model."""
    src = inspect.getsource(model_mod)
    for key in FORBIDDEN:
        assert key not in src, f"{key} leaked into the policy"


def test_policy_never_reads_phase():
    """phase is a diagnostic label; feeding it would manufacture success."""
    assert "phase" not in inspect.getsource(model_mod)


def test_dataset_never_loads_privileged_state():
    src = inspect.getsource(SO101ACTDataset)
    for key in FORBIDDEN:
        assert key not in src, f"dataset exposes {key} to the policy"


def test_inference_uses_zero_latent_not_a_sample():
    m = build_model(_cfg()).eval()
    batch = {
        "scene_image": torch.randn(2, 3, 128, 128),
        "wrist_image": torch.randn(2, 3, 128, 128),
        "qpos": torch.randn(2, 6),
        "language_embedding": torch.nn.functional.normalize(torch.randn(2, 512), dim=1),
        "task_id": torch.zeros(2, dtype=torch.long),
        "action_chunk": torch.randn(2, 32, 6),
        "action_chunk_mask": torch.ones(2, 32, dtype=torch.bool),
    }
    with torch.no_grad():
        out = m(batch)
    # z = 0 means no posterior is computed and repeated calls agree exactly.
    assert out["mu"] is None and out["logvar"] is None
    with torch.no_grad():
        assert torch.equal(out["actions"], m(batch)["actions"])


def test_clip_is_frozen_only_the_projection_learns():
    m = build_model(_cfg("clip"))
    lang_params = [n for n, _ in m.named_parameters() if "lang" in n]
    # Only the small projection MLP, never a text encoder.
    assert all(n.startswith("lang_proj.") for n in lang_params), lang_params
    n_lang = sum(p.numel() for p in m.lang_proj.parameters())
    assert n_lang < 1_000_000, f"lang_proj too large ({n_lang}) -- is CLIP inside?"


def test_action_decoder_is_parallel_not_autoregressive():
    """ACT predicts the whole chunk in one shot from k learned queries."""
    m = build_model(_cfg())
    # One learned query per chunk step, produced in a single forward pass.
    assert m.query_emb.num_embeddings == m.chunk == m.cfg.chunk_size
    src = inspect.getsource(model_mod.ACT.forward)
    for tok in ("for h in", "while", "causal", "tgt_mask"):
        assert tok not in src, f"decoder looks autoregressive ({tok})"


def test_cvae_posterior_excludes_language():
    """q(z|A,qpos): language must not enter the posterior (see ACT_ARCHITECTURE)."""
    sig = inspect.signature(model_mod.CVAEEncoder.forward)
    assert list(sig.parameters)[1:] == ["actions", "qpos", "mask"], (
        "CVAE posterior signature changed -- if language was added, update "
        "docs/ACT_ARCHITECTURE.md with the rationale"
    )


def test_baseline_and_language_configs_differ_only_in_conditioning():
    """E1 vs E2 must be controlled: every other hyperparameter identical."""
    a, b = Config(conditioning="none"), Config(conditioning="clip")
    da, db = a.to_dict(), b.to_dict()
    differing = {k for k in da if da[k] != db[k]}
    assert differing == {"conditioning"}, f"uncontrolled differences: {differing}"


def test_wandb_api_key_is_never_persisted(tmp_path):
    """A run directory must be safe to copy or commit."""
    c = Config(wandb_project="p", wandb_api_key="SUPERSECRET")
    p = tmp_path / "config.json"
    c.save(p)
    assert "SUPERSECRET" not in p.read_text()
    assert Config.load(p).wandb_api_key == ""


def test_runs_directory_is_gitignored():
    """Checkpoints are hundreds of MB and fully reproducible; never commit them."""
    ignore = Path(".gitignore").read_text()
    assert "runs/" in ignore
