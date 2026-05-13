"""Sentence-BERT text encoder.

Wraps ``sentence-transformers`` (which itself wraps HuggingFace Transformers)
to produce fixed-size sentence embeddings for every product title+description.

The default model (``all-MiniLM-L6-v2``) is a 22 M-parameter BERT variant
fine-tuned on NLI / semantic-similarity data using a contrastive loss.
Its pretraining *objective* is closer to BERT (MLM) with a supervised
NLI fine-tuning stage, which places it on the "discriminative / MLM"
axis of the H3 pretraining-objective comparison (vs. CLIP contrastive and
Qwen3 retrieval).

Encoding strategy
-----------------
- The underlying transformer is always run in ``eval()`` mode with
  ``torch.no_grad()`` (frozen weights — no gradient computation).
- Embeddings are normalised to unit length by default so that cosine
  similarity equals dot product; this is the convention for all
  sentence-transformers models.
- Batch size is configurable.  If a ``RuntimeError: CUDA out of memory``
  occurs the batch size is halved and the call is retried until it
  succeeds or drops to 1.

Relation to other encoders (H3 comparison axes)
------------------------------------------------
Pretraining objective  : NLI fine-tune (discriminative)
Model family           : BERT encoder-only
Parameter count        : ~22 M (all-MiniLM-L6-v2)

Compare with:
  - BoW / TF-IDF      : statistical, no pretraining
  - Qwen3-Embedding   : retrieval (decoder-only, much larger)
  - CLIP text encoder : image-text contrastive (ViT-B/32)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

# Canonical model IDs for this encoder tier.
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Known embedding dimensions for common sentence-transformers models so
# ``embedding_dim`` can be answered without loading the model at import time.
_KNOWN_DIMS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "bert-base-nli-mean-tokens": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}


class SentenceBERTEncoder(BaseEncoder):
    """Sentence-BERT text encoder using the ``sentence-transformers`` library.

    Uses mean pooling over the final-layer token representations, normalised
    to unit L2 norm.  The transformer backbone is kept **frozen** during
    encoding: ``model.eval()`` + ``torch.no_grad()`` are enforced.

    Args:
        cache_dir: Directory for cached embedding ``.pt`` files.
        device: Inference device (``"auto"`` picks CUDA when available).
        model_name: HuggingFace / sentence-transformers model identifier.
            Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"``.
        batch_size: Initial batch size.  Reduced automatically on OOM.
        normalize_embeddings: Whether to L2-normalise the output
            embeddings.  Defaults to ``True`` (conventional for SBERT).

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
            Install with ``pip install sentence-transformers``.
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
        self._dim: int | None = _KNOWN_DIMS.get(model_name)

        logger.info(
            "Loading sentence-transformers model '%s' on %s …",
            model_name, self.device,
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name, device=str(self.device))
        # Always freeze: this encoder is used only for feature extraction.
        self._model.eval()
        # Resolve dimension from the model if not in the lookup table.
        if self._dim is None:
            self._dim = self._model.get_sentence_embedding_dimension()

        logger.info(
            "SentenceBERTEncoder ready: model='%s' dim=%d device=%s",
            model_name, self._dim, self.device,
        )

    # ------------------------------------------------------------------
    # BaseEncoder interface
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the loaded model."""
        return self._dim  # type: ignore[return-value]

    def encode(self, inputs: list[str] | list[Any]) -> torch.Tensor:
        """Encode a list of texts into SBERT embeddings.

        Runs ``sentence_transformers.SentenceTransformer.encode`` in
        eval mode.  Automatically reduces ``batch_size`` and retries if a
        CUDA out-of-memory error occurs.

        Args:
            inputs: List of strings to encode.  Length N.

        Returns:
            Float32 tensor of shape ``(N, embedding_dim)``.

        Raises:
            RuntimeError: If encoding fails for a reason other than OOM.
        """
        if not inputs:
            return torch.empty(0, self.embedding_dim, dtype=torch.float32)

        batch_size = self.batch_size
        while True:
            try:
                embs = self._model.encode(
                    inputs,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=len(inputs) > 200,
                )
                return embs.cpu().float()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        "CUDA OOM during SBERT encode — retrying with batch_size=%d",
                        batch_size,
                    )
                else:
                    raise


# ---------------------------------------------------------------------------
# Smoke test (skipped automatically when sentence-transformers is absent)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print("sentence-transformers not installed — skipping SBERT smoke test.")
        sys.exit(0)

    tmp = Path(tempfile.mkdtemp(prefix="sbert_smoke_"))
    enc = SentenceBERTEncoder(
        cache_dir=tmp, device="cpu", batch_size=4, normalize_embeddings=True
    )
    texts = [
        "red leather camera bag",
        "wireless Bluetooth headphones",
        "mirrorless camera body",
        "noise-cancelling earbuds",
    ]

    x = enc.encode(texts)
    assert x.shape == (4, enc.embedding_dim), f"Shape mismatch: {x.shape}"
    assert x.dtype == torch.float32
    # Unit-norm check (normalised embeddings).
    norms = x.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"Not unit-norm: {norms}"

    # Round-trip cache test.
    x_cached = enc.encode_cached(texts, "sbert_smoke")
    assert torch.allclose(x, x_cached, atol=1e-6)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"sbert.py smoke test OK — embedding_dim={enc.embedding_dim}")
