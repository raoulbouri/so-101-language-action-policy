"""Instruction text and its frozen embedding.

`language_instruction` is the raw command string; `language_embedding` is a
fixed-size vector from a *frozen* pre-trained text encoder (CLIP ViT-B/32 by
default, 512-d). The encoder is never fine-tuned here -- ACT consumes the vector
as a conditioning token, so it has to be a stable function of the string across
the whole dataset.

If torch/transformers are unavailable the pipeline still runs, using a
deterministic hashing encoder. That fallback is *not* semantically meaningful;
the encoder name is written into the dataset so a consumer can tell the two
apart instead of silently training on noise.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
EMBED_DIM = 512


class TextEncoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingTextEncoder:
    """Deterministic bag-of-words fallback. Reproducible, not semantic."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.name = "hashing-fallback"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class ClipTextEncoder:
    """Frozen CLIP text tower. Embeddings are L2-normalised."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None):
        import torch
        import transformers
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

        # Loading only the text tower from a full CLIP checkpoint makes
        # transformers print a long "UNEXPECTED" report for every vision weight.
        # It is expected here, and the noise buries the collection progress.
        transformers.logging.set_verbosity_error()

        self._torch = torch
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        # CLIPTextModelWithProjection rather than CLIPModel.get_text_features:
        # transformers 5.x changed the latter to return a model-output object
        # instead of a tensor, so the projection is taken explicitly here.
        self._model = (
            CLIPTextModelWithProjection.from_pretrained(model_name).to(self.device).eval()
        )
        for param in self._model.parameters():
            param.requires_grad_(False)
        self.dim = int(self._model.config.projection_dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        batch = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            out = self._model(**batch)
            feats = out.text_embeds
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def build_text_encoder(prefer_clip: bool = True, model_name: str = DEFAULT_CLIP_MODEL):
    """Return the best available encoder, falling back loudly rather than silently."""
    if prefer_clip:
        try:
            return ClipTextEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 - any import/download failure
            print(
                f"[language] CLIP encoder unavailable ({type(exc).__name__}: {exc}); "
                "falling back to the deterministic hashing encoder. Install the "
                "'text' extra for real embeddings.",
            )
    return HashingTextEncoder()


class CachedEncoder:
    """Instructions repeat heavily across episodes; encode each string once."""

    def __init__(self, encoder):
        self._encoder = encoder
        self._cache: dict[str, np.ndarray] = {}
        self.name = encoder.name
        self.dim = encoder.dim

    def encode_one(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = self._encoder.encode([text])[0]
        return self._cache[text]
