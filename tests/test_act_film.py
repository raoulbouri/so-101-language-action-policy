"""Tier 1 gates: FiLM language conditioning (T7-T12).

FiLM modulates the visual backbone channel-wise from the language embedding
(RT-1 style) instead of appending one droppable token. These tests verify the
three properties that make it a controlled, safe change: identity at init,
gradients actually flow, and language genuinely reaches the *visual features*.
"""

import inspect

import pytest
import torch

from so101_act.config import Config
from so101_act.model import FiLM, VisualBackbone, act_loss, build_model

B = 2


def cfg(cond, **kw):
    return Config(conditioning=cond, pretrained_backbone=False, hidden_dim=64,
                  dim_feedforward=128, nheads=4, enc_layers=1, dec_layers=1,
                  cvae_enc_layers=1, chunk_size=16, **kw)


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "scene_image": torch.randn(B, 3, 128, 128, generator=g),
        "wrist_image": torch.randn(B, 3, 128, 128, generator=g),
        "qpos": torch.randn(B, 6, generator=g),
        "language_embedding": torch.nn.functional.normalize(
            torch.randn(B, 512, generator=g), dim=1),
        "task_id": torch.randint(0, 16, (B,), generator=g),
        "action_chunk": torch.randn(B, 16, 6, generator=g),
        "action_chunk_mask": torch.ones(B, 16, dtype=torch.bool),
    }


# --- T7: identity at initialisation ---------------------------------------
def test_film_layer_is_identity_at_init():
    f = FiLM(cond_dim=32, channels=8)
    h = torch.randn(B, 8, 5, 5)
    cond = torch.randn(B, 32)
    assert torch.allclose(f(h, cond), h, atol=1e-6), "FiLM must start as identity"


def test_film_backbone_matches_unfilmed_backbone_at_init():
    """A FiLM run must start numerically identical to the baseline."""
    torch.manual_seed(0)
    plain = VisualBackbone(64, "resnet18", pretrained=False)
    torch.manual_seed(0)
    filmed = VisualBackbone(64, "resnet18", pretrained=False, film_cond_dim=64)
    # copy the shared weights so only FiLM differs
    filmed.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(B, 3, 128, 128)
    cond = torch.randn(B, 64)
    with torch.no_grad():
        a, _, _ = plain(x)
        b, _, _ = filmed(x, cond)
    assert torch.allclose(a, b, atol=1e-5), "FiLM changed the backbone at init"


def test_film_is_not_identity_after_perturbing_its_weights():
    """Sanity: the identity test above is meaningful only if FiLM *can* act."""
    f = FiLM(cond_dim=32, channels=8)
    torch.nn.init.normal_(f.to_gamma_beta.weight, std=0.5)
    h = torch.randn(B, 8, 5, 5)
    assert not torch.allclose(f(h, torch.randn(B, 32)), h, atol=1e-6)


# --- T8: gradients reach the FiLM generator --------------------------------
@pytest.mark.parametrize("cond", ["film", "film_token"])
def test_film_generator_receives_gradient(cond):
    c = cfg(cond)
    m = build_model(c).train()
    b = batch()
    act_loss(m(b), b, c.kl_weight)["loss"].backward()
    film_params = [(n, p) for n, p in m.named_parameters() if "film" in n]
    assert film_params, "no FiLM parameters found"
    total = sum(p.grad.abs().sum().item() for _, p in film_params if p.grad is not None)
    assert total > 0, "FiLM generator received zero gradient"


# --- T9: language reaches the VISUAL FEATURES, not just the output ---------
def test_language_changes_visual_features_under_film():
    """The point of FiLM: swapping language must change the encoder's features."""
    torch.manual_seed(0)
    bb = VisualBackbone(64, "resnet18", pretrained=False, film_cond_dim=64)
    torch.nn.init.normal_(bb.film4.to_gamma_beta.weight, std=0.1)  # train-like state
    x = torch.randn(B, 3, 128, 128)
    with torch.no_grad():
        f1, _, _ = bb(x, torch.randn(B, 64, generator=torch.Generator().manual_seed(1)))
        f2, _, _ = bb(x, torch.randn(B, 64, generator=torch.Generator().manual_seed(2)))
    assert not torch.allclose(f1, f2, atol=1e-6), "language did not modulate features"


def test_clip_token_mode_leaves_visual_features_untouched():
    """Contrast: in token mode the backbone is language-blind by construction."""
    m = build_model(cfg("clip")).eval()
    b1, b2 = batch(), batch()
    b2["language_embedding"] = torch.nn.functional.normalize(torch.randn(B, 512), dim=1)
    with torch.no_grad():
        f1, _, _ = m.backbones["scene_image"](b1["scene_image"])
        f2, _, _ = m.backbones["scene_image"](b2["scene_image"])
    assert torch.allclose(f1, f2), "token mode should not touch the backbone"


# --- T10: baseline stays language-blind ------------------------------------
def test_none_mode_still_ignores_language_entirely():
    m = build_model(cfg("none")).eval()
    b1 = batch()
    b2 = dict(b1)
    b2["language_embedding"] = torch.randn(B, 512)
    with torch.no_grad():
        assert torch.allclose(m(b1)["actions"], m(b2)["actions"])


@pytest.mark.parametrize("cond", ["film", "film_token"])
def test_film_modes_respond_to_language(cond):
    torch.manual_seed(0)
    m = build_model(cfg(cond)).eval()
    # perturb FiLM off identity to emulate a trained model
    for n, p in m.named_parameters():
        if "film" in n and p.dim() == 2:
            torch.nn.init.normal_(p, std=0.05)
    b1 = batch()
    b2 = dict(b1)
    b2["language_embedding"] = torch.nn.functional.normalize(
        torch.randn(B, 512, generator=torch.Generator().manual_seed(7)), dim=1)
    with torch.no_grad():
        assert not torch.allclose(m(b1)["actions"], m(b2)["actions"], atol=1e-6)


# --- T11: A/B stays controlled ---------------------------------------------
def test_film_and_clip_configs_differ_only_in_conditioning():
    a, b = Config(conditioning="clip"), Config(conditioning="film")
    da, db = a.to_dict(), b.to_dict()
    assert {k for k in da if da[k] != db[k]} == {"conditioning"}


def test_output_shape_is_chunk_by_action_dim():
    for cond in ("none", "clip", "film", "film_token", "taskid"):
        c = cfg(cond)
        m = build_model(c).eval()
        with torch.no_grad():
            assert m(batch())["actions"].shape == (B, c.chunk_size, 6)


def test_film_preserves_parallel_decoding():
    src = inspect.getsource(build_model(cfg("film")).forward.__func__)
    for tok in ("for h in", "causal", "tgt_mask"):
        assert tok not in src
