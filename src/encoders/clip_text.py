"""CLIP text encoder (OpenAI ViT-B/32 by default).

CLIP (Contrastive Language-Image Pretraining) trains a dual-encoder model
to align image and text representations via a **contrastive loss** over
400 M image-caption pairs.  The text side is a Transformer with a causal
self-attention mask; the image side is a ViT.

H3 role
-------
CLIP-text sits on the **contrastive pretraining axis** of the H3
experiment:

  Axis           | Encoder         | Objective
  ---------------+-----------------+-----------
  MLM            | SBERT           | NLI fine-tuned BERT
  Contrastive    | CLIP text       | Image-text alignment ← this file
  Retrieval      | Qwen3           | Query-document retrieval

The CLIP text encoder is also the shared text backbone for the H2
multimodal experiments: image embeddings come from CLIPVisionModel,
and both live in the same latent space after projection, enabling
late-fusion by concatenation or cosine similarity.

Implementation
--------------
We use ``CLIPTextModelWithProjection`` from HuggingFace ``transformers``
rather than the raw ``CLIPTextModel``, so the output passes through the
learned linear projection that maps the transformer's ``[EOS]`` token
representation into the shared 512-D CLIP embedding space (same space
as CLIP image embeddings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai/clip-vit-b-32"
_CLIP_DIM = 512          # output of the text projection head for ViT-B/32
_CLIP_MAX_TOKENS = 77    # CLIP tokeniser hard limit


class CLIPTextEncoder(BaseEncoder):
    """CLIP text encoder (``CLIPTextModelWithProjection``).

    Encodes product titles and descriptions into the 512-D joint CLIP
    embedding space so that the resulting product vectors are directly
    comparable with CLIP image embeddings (useful for H2 multimodal
    fusion experiments).

    The backbone is always **frozen**: ``eval()`` + ``no_grad()``.

    Args:
        cache_dir: Directory for cached ``.pt`` files.
        device: Inference device.
        model_name: HuggingFace CLIP model identifier.
        batch_size: Texts per forward pass.  Halved automatically on OOM.
        normalize_embeddings: L2-normalise output (default True).
            CLIP embeddings are conventionally normalised so that
            similarity reduces to cosine distance.

    Raises:
        ImportError: If ``transformers`` is not installed.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/embeddings",
        device: str | torch.device = "auto",
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__(cache_dir=cache_dir, device=device)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading CLIP text model '%s' on %s …", model_name, self.device)
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            ) from exc

        self._tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self._model = CLIPTextModelWithProjection.from_pretrained(model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad_(False)

        # Resolve projection dimension from model config.
        self._dim: int = self._model.config.projection_dim  # type: ignore[attr-defined]
        logger.info(
            "CLIPTextEncoder ready: model='%s' dim=%d device=%s",
            model_name, self._dim, self.device,
        )

    # ------------------------------------------------------------------
    # BaseEncoder interface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """CLIP projection dimension (512 for ViT-B/32)."""
        return self._dim

    def encode(self, inputs: list[str] | list[Any]) -> torch.Tensor:
        """Encode a list of texts into CLIP embedding vectors.

        Long texts are truncated to CLIP's 77-token limit; very short
        texts are padded.  Embeddings are optionally L2-normalised.

        Args:
            inputs: List of product strings.

        Returns:
            Float32 tensor of shape ``(N, embedding_dim)``.
        """
        if not inputs:
            return torch.empty(0, self.embedding_dim, dtype=torch.float32)

        import torch.nn.functional as F

        batch_size = self.batch_size
        all_embeds: list[torch.Tensor] = []

        i = 0
        while i < len(inputs):
            batch = inputs[i: i + batch_size]
            while True:
                try:
                    tokens = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=_CLIP_MAX_TOKENS,
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        out = self._model(**tokens)
                    # text_embeds is the projected [EOS] representation.
                    emb = out.text_embeds.cpu().float()
                    if self.normalize_embeddings:
                        emb = F.normalize(emb, p=2, dim=-1)
                    all_embeds.append(emb)
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and batch_size > 1:
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        logger.warning(
                            "CUDA OOM — retrying CLIP batch with batch_size=%d",
                            batch_size,
                        )
                    else:
                        raise
            i += batch_size

        return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    try:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast  # noqa: F401
    except ImportError:
        print("transformers not installed — skipping CLIP smoke test.")
        sys.exit(0)

    tmp = Path(tempfile.mkdtemp(prefix="clip_smoke_"))
    enc = CLIPTextEncoder(cache_dir=tmp, device="cpu", batch_size=4)
    texts = [
        "red leather camera bag with shoulder strap",
        "wireless Bluetooth over-ear headphones",
        "compact mirrorless camera body",
        "sport in-ear earbuds with mic",
    ]

    x = enc.encode(texts)
    assert x.shape == (4, enc.embedding_dim), f"Shape mismatch: {x.shape}"
    assert x.dtype == torch.float32
    norms = x.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Not unit-norm: {norms}"

    x2 = enc.encode_cached(texts, "clip_smoke")
    assert torch.allclose(x, x2)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"clip_text.py smoke test OK — dim={enc.embedding_dim}")
