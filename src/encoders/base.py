"""Abstract base class that every encoder in the project must implement.

All encoders — text (BoW, TF-IDF, SBERT, Qwen3, CLIP-text) and vision
(ResNet-50, CLIP-image, DINOv2) — inherit from ``BaseEncoder`` so
experiment scripts can treat them uniformly.

Design decisions
----------------
* ``encode(inputs)`` is the only abstract method.  It always receives the
  full list of inputs for a given call (no internal state about the dataset).
  Pre-trained / frozen encoders (SBERT, Qwen3, CLIP, DINOv2) can call it
  directly on all N product texts.  Statistical encoders (BoW, TF-IDF) must
  call ``fit(train_inputs)`` first to learn the vocabulary / IDF weights on
  training data only, preventing label leakage.
* ``fit(train_inputs)`` is a no-op in the base class.  Frozen encoders
  inherit this default.  Statistical encoders override it.
* ``encode_cached(inputs, cache_key)`` wraps ``encode``: on first call it
  computes embeddings and saves them to ``{cache_dir}/{cache_key}.pt``; on
  subsequent calls it loads and returns the cached tensor.  The caller is
  responsible for supplying a unique ``cache_key`` (e.g.,
  ``"tfidf_electronics_seed0"``).
* ``embedding_dim`` is an abstract property so callers can ask for the
  output width before running a full encode pass.

PIL import
----------
Vision encoders accept ``list[PIL.Image.Image]`` as input.  Pillow is an
optional dependency at the module level; it is imported lazily inside
methods that need it so text-only workflows do not require it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import torch

from src.utils.device import resolve_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PIL import (used only by vision encoders)
# ---------------------------------------------------------------------------
try:
    from PIL.Image import Image as PILImage  # noqa: F401 — re-exported for subclasses
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class BaseEncoder(ABC):
    """Abstract text or vision encoder.

    All encoders produce fixed-size ``float32`` embeddings and cache them
    to disk after first computation.

    Args:
        cache_dir: Directory where ``*.pt`` embedding tensors are stored.
            Created automatically if it does not exist.
        device: Torch device for inference.  ``"auto"`` selects CUDA when
            available, otherwise CPU.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/embeddings",
        device: str | torch.device = "auto",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, inputs: list[str] | list[Any]) -> torch.Tensor:
        """Encode a list of inputs into a float32 embedding matrix.

        All subclasses must implement this method.  It must be pure
        (no side effects) and idempotent (calling twice with the same
        inputs returns identical results).

        For statistical encoders (BoW, TF-IDF) ``fit()`` must be called
        with the training texts before the first call to ``encode``.

        Args:
            inputs: List of strings (text encoders) or ``PIL.Image``
                objects (vision encoders).  Length N.

        Returns:
            ``float32`` tensor of shape ``(N, embedding_dim)``.

        Raises:
            RuntimeError: If the encoder has not been fitted yet
                (statistical encoders only).
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output embedding width.

        For frozen pre-trained encoders this is fixed (e.g. 384 for
        all-MiniLM-L6-v2).  For statistical encoders it equals
        ``max_features`` before fitting and the actual vocabulary size
        after fitting.

        Returns:
            Integer embedding dimension.
        """
        ...

    # ------------------------------------------------------------------
    # Optional fit hook (no-op for frozen models)
    # ------------------------------------------------------------------

    def fit(self, train_inputs: list[str] | list[Any]) -> None:
        """Fit the encoder on training data.

        This is a no-op for frozen pre-trained encoders.  Statistical
        encoders (BoW, TF-IDF) override it to learn a vocabulary /
        IDF table from ``train_inputs`` only, preventing data leakage
        to val / test sets.

        Args:
            train_inputs: Training texts or images.  Length N_train.
        """
        # Default: frozen encoder, nothing to do.

    # ------------------------------------------------------------------
    # Caching helper
    # ------------------------------------------------------------------

    def encode_cached(
        self,
        inputs: list[str] | list[Any],
        cache_key: str,
    ) -> torch.Tensor:
        """Encode ``inputs`` and persist the result, or load from cache.

        The cache is stored at ``{cache_dir}/{cache_key}.pt`` as a
        serialised ``torch.Tensor``.  If the file exists it is loaded
        directly, skipping the (possibly expensive) encode call.

        The caller is responsible for supplying a ``cache_key`` that
        uniquely identifies the (encoder, dataset, split) combination.
        A good convention is ``"{encoder_name}_{dataset}_{split_seed}"``.

        Args:
            inputs: Texts or images to encode.
            cache_key: Unique string identifier used as the filename
                stem inside ``cache_dir``.

        Returns:
            ``float32`` tensor of shape ``(N, embedding_dim)``.
        """
        cache_path = self.cache_dir / f"{cache_key}.pt"
        if cache_path.exists():
            try:
                x = torch.load(cache_path, weights_only=True)
                logger.info(
                    "Loaded cached embeddings: '%s' shape=%s",
                    cache_key, tuple(x.shape),
                )
                return x
            except Exception as exc:  # corrupt / old format file
                logger.warning(
                    "Failed to load cache '%s' (%s) — re-encoding.", cache_key, exc
                )

        logger.info(
            "Encoding %d inputs (cache_key='%s') …", len(inputs), cache_key
        )
        x = self.encode(inputs).float()
        torch.save(x, cache_path)
        logger.info(
            "Cached embeddings: '%s' → %s  shape=%s",
            cache_key, cache_path, tuple(x.shape),
        )
        return x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"{type(self).__name__}("
            f"embedding_dim={self.embedding_dim}, "
            f"cache_dir={self.cache_dir}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Smoke test — uses a trivial concrete subclass, no model download needed
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")

    class _IdentityEncoder(BaseEncoder):
        """Maps each string to its UTF-8 byte counts in a fixed 8-D vector."""

        _DIM = 8

        def encode(self, inputs: list[str]) -> torch.Tensor:
            rows = []
            for s in inputs:
                enc = s.encode("utf-8")[:self._DIM]
                padded = list(enc) + [0] * (self._DIM - len(enc))
                rows.append(padded)
            return torch.tensor(rows, dtype=torch.float32)

        @property
        def embedding_dim(self) -> int:
            return self._DIM

    tmp = Path(tempfile.mkdtemp(prefix="base_encoder_smoke_"))
    enc = _IdentityEncoder(cache_dir=tmp)
    texts = ["hello", "world", "graph", "neural"]

    # First call: encode + save.
    x1 = enc.encode_cached(texts, "smoke_test")
    assert x1.shape == (4, 8)
    assert x1.dtype == torch.float32
    assert (tmp / "smoke_test.pt").exists()

    # Second call: load from cache (encode should NOT be called again).
    x2 = enc.encode_cached(texts, "smoke_test")
    assert torch.allclose(x1, x2)

    # fit() is a no-op on base class.
    enc.fit(texts)

    shutil.rmtree(tmp, ignore_errors=True)
    print("base.py smoke test OK")
