"""ACT (Action Chunking with Transformers), with optional language conditioning.

Faithful to Zhao et al. 2023 in structure: a ResNet18 visual backbone, a CVAE
whose encoder infers a style latent `z` from the action chunk, a Transformer
encoder over the multimodal context, and a Transformer decoder with `k` learned
action queries producing the whole chunk in parallel.

Language is added as **one extra token** and nothing else. The encoder context
goes from

    [latent, qpos, scene tokens..., wrist tokens...]
to
    [latent, qpos, language, scene tokens..., wrist tokens...]

which keeps every other tensor shape, hyperparameter and code path identical
between the baseline and the conditioned model.

CVAE posterior
--------------
`q(z | A, qpos)` is left unchanged -- language is deliberately *not* fed to the
posterior. If it were, `z` could carry task identity and the decoder could
satisfy the reconstruction loss through `z` while ignoring the language token,
which is precisely the confound the E3/E4 tests are meant to detect. Keeping
language out of the posterior means the only path from instruction to action is
the token we are trying to measure.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

# Modality ids for the learned modality embedding.
MOD_LATENT, MOD_STATE, MOD_LANG, MOD_SCENE, MOD_WRIST = 0, 1, 2, 3, 4
N_MODALITIES = 5


def sine_positional_2d(h: int, w: int, dim: int, device=None) -> Tensor:
    """Standard DETR-style 2D sine positional encoding -> (h*w, dim)."""
    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4 for 2D sine encoding")
    y = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
    x = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)
    quarter = dim // 4
    omega = torch.exp(
        torch.arange(quarter, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / max(quarter - 1, 1))
    )
    def enc(coord):
        a = coord.flatten().unsqueeze(1) * omega.unsqueeze(0)
        return torch.cat([a.sin(), a.cos()], dim=1)
    return torch.cat([enc(y), enc(x)], dim=1)          # (h*w, dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: ``h <- gamma(L) * h + beta(L)``.

    Channel-wise affine modulation of a conv feature map, conditioned on the
    language embedding. RT-1 conditions its visual backbone this way and
    outperforms BC-Z -- which appends language as a token, our `clip` mode --
    by 25% on seen tasks: early fusion lets the encoder extract task-relevant
    features rather than leaving language as one token among 35 that
    self-attention can learn to drop.

    Identity initialisation: gamma = 1 + W_g c and beta = W_b c with W_g, W_b
    zero-initialised, so at step 0 the layer is exactly the identity. This
    preserves the ImageNet pretraining intact and makes a FiLM run start
    numerically identical to the un-modulated baseline.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.channels = channels

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        d_gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma = 1.0 + d_gamma                       # identity at init
        return h * gamma[:, :, None, None] + beta[:, :, None, None]


class VisualBackbone(nn.Module):
    """ResNet18 truncated after layer4, projected to `hidden_dim`.

    For a 128x128 input ResNet18 downsamples by 32, giving a 4x4 feature map =
    **16 tokens per camera**.
    """

    def __init__(self, hidden_dim: int, name: str = "resnet18", pretrained: bool = True,
                 film_cond_dim: int | None = None):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        # Keep the stages addressable so FiLM can be applied between them.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.out_channels = 512 if name in ("resnet18", "resnet34") else 2048
        self.proj = nn.Conv2d(self.out_channels, hidden_dim, kernel_size=1)

        self.use_film = film_cond_dim is not None
        if self.use_film:
            ch2 = self.layer2[-1].conv2.out_channels
            ch3 = self.layer3[-1].conv2.out_channels
            ch4 = self.layer4[-1].conv2.out_channels
            self.film2 = FiLM(film_cond_dim, ch2)
            self.film3 = FiLM(film_cond_dim, ch3)
            self.film4 = FiLM(film_cond_dim, ch4)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, int, int]:
        h = self.layer1(self.stem(x))
        h = self.layer2(h)
        if self.use_film:
            h = self.film2(h, cond)
        h = self.layer3(h)
        if self.use_film:
            h = self.film3(h, cond)
        h = self.layer4(h)
        if self.use_film:
            h = self.film4(h, cond)
        f = self.proj(h)                             # (B, D, h, w)
        hh, ww = f.shape[2], f.shape[3]
        return f.flatten(2).permute(0, 2, 1), hh, ww  # (B, h*w, D)


class CVAEEncoder(nn.Module):
    """q(z | A, qpos): a small Transformer encoder over [CLS, qpos, action chunk]."""

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int, latent_dim: int,
                 nheads: int, layers: int, dim_ff: int, dropout: float, chunk: int):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, chunk + 2, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, nheads, dim_ff, dropout, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.to_latent = nn.Linear(hidden_dim, latent_dim * 2)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, actions: Tensor, qpos: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        b = actions.shape[0]
        tok = torch.cat([
            self.cls.expand(b, -1, -1),
            self.state_proj(qpos).unsqueeze(1),
            self.action_proj(actions),
        ], dim=1) + self.pos
        # Padded chunk entries must not inform the posterior either.
        key_pad = torch.cat([
            torch.zeros(b, 2, dtype=torch.bool, device=actions.device),
            ~mask,
        ], dim=1)
        h = self.encoder(tok, src_key_padding_mask=key_pad)[:, 0]
        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        return mu, logvar


class ACT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.chunk = cfg.chunk_size
        self.action_dim = 6
        self.state_dim = 6
        self.cameras = tuple(cfg.cameras)

        # --- conditioning mode ---------------------------------------------
        self.use_film = cfg.conditioning in ("film", "film_token")
        # `film` puts language ONLY in the backbone; `film_token` also keeps the
        # token, so the two paths can be attributed separately.
        self.use_lang_token = cfg.conditioning in ("clip", "taskid", "film_token")

        # --- visual -------------------------------------------------------
        film_dim = d if self.use_film else None
        if cfg.share_backbone:
            shared = VisualBackbone(d, cfg.backbone, cfg.pretrained_backbone, film_dim)
            self.backbones = nn.ModuleDict({c: shared for c in self.cameras})
        else:
            self.backbones = nn.ModuleDict(
                {c: VisualBackbone(d, cfg.backbone, cfg.pretrained_backbone, film_dim)
                 for c in self.cameras}
            )

        # --- conditioning tokens -------------------------------------------
        self.state_proj = nn.Linear(self.state_dim, d)
        self.latent_proj = nn.Linear(cfg.latent_dim, d)
        self.modality_emb = nn.Embedding(N_MODALITIES, d)

        self.use_language = cfg.use_language
        if cfg.conditioning in ("clip", "film", "film_token"):
            # Frozen CLIP vector -> one token. LayerNorm first because CLIP
            # embeddings are L2-normalised and therefore small in magnitude
            # relative to the other token streams.
            self.lang_proj = nn.Sequential(
                nn.LayerNorm(cfg.lang_dim),
                nn.Linear(cfg.lang_dim, d),
                nn.LayerNorm(d),
            )
        elif cfg.conditioning == "taskid":
            self.task_emb = nn.Embedding(cfg.n_task_ids, d)
            self.lang_proj = nn.Sequential(nn.LayerNorm(d))

        # --- CVAE ----------------------------------------------------------
        self.cvae = CVAEEncoder(
            self.action_dim, self.state_dim, d, cfg.latent_dim,
            cfg.nheads, cfg.cvae_enc_layers, cfg.dim_feedforward,
            cfg.dropout, cfg.chunk_size,
        )

        # --- transformer ----------------------------------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d, cfg.nheads, cfg.dim_feedforward, cfg.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.enc_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d, cfg.nheads, cfg.dim_feedforward, cfg.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_layers)

        self.query_emb = nn.Embedding(cfg.chunk_size, d)
        self.action_head = nn.Linear(d, self.action_dim)
        self._pos_cache: dict[tuple, Tensor] = {}

    # ---------------------------------------------------------------- utils
    def _visual_pos(self, h: int, w: int, device) -> Tensor:
        key = (h, w, self.cfg.hidden_dim, str(device))
        if key not in self._pos_cache:
            self._pos_cache[key] = sine_positional_2d(
                h, w, self.cfg.hidden_dim, device
            ).unsqueeze(0)
        return self._pos_cache[key]

    def _mod(self, mid: int, b: int, n: int, device) -> Tensor:
        return self.modality_emb(
            torch.full((1, 1), mid, dtype=torch.long, device=device)
        ).expand(b, n, -1)

    def build_context(self, batch: dict, z: Tensor) -> Tensor:
        """Assemble the multimodal token sequence fed to the Transformer encoder."""
        device = z.device
        b = z.shape[0]
        tokens = [
            self.latent_proj(z).unsqueeze(1) + self._mod(MOD_LATENT, b, 1, device),
            self.state_proj(batch["qpos"]).unsqueeze(1) + self._mod(MOD_STATE, b, 1, device),
        ]
        lang = None
        if self.use_language:
            if self.cfg.conditioning == "taskid":
                lang = self.lang_proj(self.task_emb(batch["task_id"]))
            else:
                lang = self.lang_proj(batch["language_embedding"])
            if self.use_lang_token:
                tokens.append(lang.unsqueeze(1) + self._mod(MOD_LANG, b, 1, device))

        for cam, mid in zip(self.cameras, (MOD_SCENE, MOD_WRIST)):
            # FiLM modulates the backbone itself; language is then not optional.
            feat, h, w = (self.backbones[cam](batch[cam], lang) if self.use_film
                          else self.backbones[cam](batch[cam]))   # (B, h*w, D)
            feat = feat + self._visual_pos(h, w, device) + self._mod(mid, b, feat.shape[1], device)
            tokens.append(feat)
        return torch.cat(tokens, dim=1)

    # -------------------------------------------------------------- forward
    def forward(self, batch: dict, sample_latent: bool | None = None) -> dict:
        """Training uses z ~ q(z|A,q); inference uses z = 0 (ACT's convention)."""
        training = self.training if sample_latent is None else sample_latent
        b = batch["qpos"].shape[0]
        device = batch["qpos"].device

        if training and "action_chunk" in batch:
            mu, logvar = self.cvae(
                batch["action_chunk"], batch["qpos"], batch["action_chunk_mask"]
            )
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            mu = logvar = None
            z = torch.zeros(b, self.cfg.latent_dim, device=device)

        memory = self.encoder(self.build_context(batch, z))
        q = self.query_emb.weight.unsqueeze(0).expand(b, -1, -1)
        out = self.decoder(q, memory)
        return {"actions": self.action_head(out), "mu": mu, "logvar": logvar}


# ---------------------------------------------------------------------- loss
def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL( q(z|.) || N(0,I) ), averaged over the batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def act_loss(pred: dict, batch: dict, kl_weight: float) -> dict[str, Tensor]:
    """Masked L1 reconstruction + beta * KL.

    Padded chunk entries contribute exactly zero: the per-element L1 is
    multiplied by the mask before any reduction, and the denominator counts only
    valid elements.
    """
    a_hat = pred["actions"]
    target = batch["action_chunk"]
    mask = batch["action_chunk_mask"].unsqueeze(-1).to(a_hat.dtype)  # (B,k,1)

    l1 = (F.l1_loss(a_hat, target, reduction="none") * mask).sum()
    denom = mask.sum() * a_hat.shape[-1]
    action_loss = l1 / denom.clamp(min=1.0)

    if pred["mu"] is not None:
        kl = kl_divergence(pred["mu"], pred["logvar"])
    else:
        kl = torch.zeros((), device=a_hat.device)

    return {"loss": action_loss + kl_weight * kl, "action_loss": action_loss, "kl": kl}


def build_model(cfg) -> ACT:
    return ACT(cfg)
