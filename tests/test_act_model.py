"""Model, loss and gradient checks (validation items 3-8, 10)."""

import numpy as np
import pytest
import torch

from so101_act.config import Config
from so101_act.model import act_loss, build_model

B, K, A = 3, 32, 6


def make_cfg(conditioning="none", **kw):
    # Small and untrained: these tests are about shapes and gradient flow, and a
    # pretrained download would make them slow and network-dependent.
    return Config(conditioning=conditioning, pretrained_backbone=False,
                  hidden_dim=64, dim_feedforward=128, nheads=4,
                  enc_layers=1, dec_layers=1, cvae_enc_layers=1, **kw)


def make_batch(seed=0, mask_tail=None):
    g = torch.Generator().manual_seed(seed)
    m = torch.ones(B, K, dtype=torch.bool)
    if mask_tail is not None:
        m[:, mask_tail:] = False
    return {
        "scene_image": torch.randn(B, 3, 128, 128, generator=g),
        "wrist_image": torch.randn(B, 3, 128, 128, generator=g),
        "qpos": torch.randn(B, A, generator=g),
        "language_embedding": torch.nn.functional.normalize(
            torch.randn(B, 512, generator=g), dim=1),
        "task_id": torch.randint(0, 16, (B,), generator=g),
        "action_chunk": torch.randn(B, K, A, generator=g),
        "action_chunk_mask": m,
    }


# --- items 3, 4, 5 --------------------------------------------------------
@pytest.mark.parametrize("cond", ["none", "clip", "taskid"])
def test_forward_produces_the_required_shape(cond):
    m = build_model(make_cfg(cond)).eval()
    with torch.no_grad():
        out = m(make_batch())
    assert out["actions"].shape == (B, K, A)
    assert torch.isfinite(out["actions"]).all()


def test_language_adds_exactly_one_token():
    base = build_model(make_cfg("none"))
    lang = build_model(make_cfg("clip"))
    z = torch.zeros(B, base.cfg.latent_dim)
    n_base = base.build_context(make_batch(), z).shape[1]
    n_lang = lang.build_context(make_batch(), z).shape[1]
    assert n_lang == n_base + 1, "language must add one token and nothing else"
    # 1 latent + 1 qpos + 16 scene + 16 wrist for 128x128 through ResNet18
    assert n_base == 34, f"unexpected baseline token count {n_base}"


def test_inference_uses_zero_latent_and_is_deterministic():
    m = build_model(make_cfg("clip")).eval()
    b = make_batch()
    with torch.no_grad():
        a1 = m(b)["actions"]
        a2 = m(b)["actions"]
    assert torch.allclose(a1, a2), "eval-mode forward must be deterministic (z=0)"


def test_training_mode_samples_latent_and_returns_posterior():
    m = build_model(make_cfg("clip")).train()
    out = m(make_batch())
    assert out["mu"] is not None and out["logvar"] is not None
    assert out["mu"].shape == (B, m.cfg.latent_dim)


# --- item 6: padded entries contribute zero loss ---------------------------
def test_padded_entries_contribute_zero_loss():
    cfg = make_cfg("clip")
    m = build_model(cfg).eval()
    batch = make_batch(mask_tail=10)          # only first 10 of 32 are valid
    with torch.no_grad():
        pred = m(batch)
    base = act_loss(pred, batch, cfg.kl_weight)["action_loss"]

    # Corrupt only the masked-out targets; the loss must not move at all.
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["action_chunk"][:, 10:] += 1000.0
    with torch.no_grad():
        alt = act_loss(pred, b2, cfg.kl_weight)["action_loss"]
    assert torch.allclose(base, alt), "padded entries leaked into the loss"


def test_loss_denominator_counts_only_valid_entries():
    cfg = make_cfg()
    pred = {"actions": torch.zeros(B, K, A), "mu": None, "logvar": None}
    batch = make_batch(mask_tail=8)
    batch["action_chunk"] = torch.ones(B, K, A)
    out = act_loss(pred, batch, cfg.kl_weight)
    # every valid element has |0-1| = 1, so the mean over valid elements is 1
    assert torch.allclose(out["action_loss"], torch.tensor(1.0), atol=1e-6)


# --- item 7: finite gradients ---------------------------------------------
@pytest.mark.parametrize("cond", ["none", "clip", "taskid"])
def test_gradients_are_finite_everywhere(cond):
    cfg = make_cfg(cond)
    m = build_model(cfg).train()
    out = act_loss(m(make_batch()), make_batch(), cfg.kl_weight)
    out["loss"].backward()
    bad = [n for n, p in m.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads in {bad[:5]}"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for _, p in m.named_parameters())


# --- item 8: the language projection actually trains -----------------------
def test_language_projection_receives_gradient():
    cfg = make_cfg("clip")
    m = build_model(cfg).train()
    out = act_loss(m(make_batch()), make_batch(), cfg.kl_weight)
    out["loss"].backward()
    grads = [p.grad.abs().sum().item() for p in m.lang_proj.parameters()
             if p.grad is not None]
    assert grads, "lang_proj received no gradient at all"
    assert sum(grads) > 0, "lang_proj gradient is exactly zero"


def test_clip_embedding_itself_is_not_a_parameter():
    """CLIP must stay frozen: the embedding is data, not something we learn."""
    m = build_model(make_cfg("clip"))
    names = [n for n, _ in m.named_parameters()]
    assert not any("clip" in n.lower() for n in names)
    assert sum(p.numel() for p in m.lang_proj.parameters()) < 1e6


# --- item 10: language changes predictions ---------------------------------
def test_swapping_language_changes_predictions_with_observation_fixed():
    torch.manual_seed(0)
    cfg = make_cfg("clip")
    m = build_model(cfg).eval()
    batch = make_batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    # same observation, different instruction
    b2["language_embedding"] = torch.nn.functional.normalize(
        torch.randn(B, 512, generator=torch.Generator().manual_seed(99)), dim=1)
    with torch.no_grad():
        a1, a2 = m(batch)["actions"], m(b2)["actions"]
    assert not torch.allclose(a1, a2), "language token has no effect on the output"


def test_baseline_ignores_language_by_construction():
    """The E1 baseline must be genuinely language-blind."""
    torch.manual_seed(0)
    m = build_model(make_cfg("none")).eval()
    batch = make_batch()
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    b2["language_embedding"] = torch.randn(B, 512)
    with torch.no_grad():
        assert torch.allclose(m(batch)["actions"], m(b2)["actions"])
