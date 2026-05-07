"""Tier-0 sparse text encoders (Bag-of-Words and TF-IDF).

Both wrappers convert a list of product titles+descriptions into a dense
``torch.float32`` tensor of shape ``(N, D)`` with ``D = max_features``.
They serve as the H3 baselines: any modern encoder we benchmark must beat
these.

Design notes
------------
* These encoders are *fit on the training split only* and *transformed*
  on val and test, just like any classical text feature extractor — so
  the API exposes ``fit_transform`` and ``transform`` separately.
* The output is intentionally dense, not sparse, because downstream
  consumers (the GCN/GAT layers and the MLP baseline) all assume a dense
  ``x`` tensor. With ``max_features=10_000`` and ~100k nodes the dense
  matrix fits comfortably in RAM (~4 GB float32 worst case).
* Cache files: ``data/embeddings/bow_{split_seed}.pt`` and
  ``data/embeddings/tfidf_{split_seed}.pt``. Each cache stores a dict
  with the dense tensor plus the fitted vocabulary so the same vector
  space can be re-used at inference time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)


def _to_dense_float32(matrix) -> torch.Tensor:
    """Convert a scipy sparse matrix into a dense ``torch.float32`` tensor."""
    arr = matrix.toarray().astype(np.float32, copy=False)
    return torch.from_numpy(arr)


@dataclass
class _BaseSparseEncoder:
    """Common scaffolding for the BoW / TF-IDF encoders.

    Subclasses must populate ``self._vec`` with a fitted sklearn
    vectorizer in ``fit_transform``.
    """

    max_features: int = 10_000
    lowercase: bool = True
    stop_words: str | None = "english"
    ngram_range: tuple[int, int] = (1, 1)
    min_df: int = 2

    def __post_init__(self) -> None:
        self._vec = None  # populated in fit_transform

    def transform(self, texts: Sequence[str]) -> torch.Tensor:
        """Apply the already-fitted vectorizer to new documents.

        Args:
            texts: List of strings of length ``M``.

        Returns:
            Dense float tensor of shape ``(M, max_features)``.

        Raises:
            RuntimeError: If called before ``fit_transform``.
        """
        if self._vec is None:
            raise RuntimeError(
                f"{type(self).__name__}.transform called before fit_transform."
            )
        return _to_dense_float32(self._vec.transform(texts))

    @property
    def vocabulary_(self) -> dict[str, int]:
        """Return the fitted vocabulary (token -> column index)."""
        if self._vec is None:
            raise RuntimeError("Vectorizer not fitted yet.")
        return self._vec.vocabulary_


class BagOfWords(_BaseSparseEncoder):
    """Bag-of-Words encoder (raw token counts) backed by sklearn ``CountVectorizer``.

    The output is a dense ``(N, max_features)`` ``float32`` tensor. With
    default args, English stop-words are removed and unigrams appearing
    in fewer than ``min_df`` documents are dropped.
    """

    def fit_transform(self, texts: Sequence[str]) -> torch.Tensor:
        """Fit the vectorizer on ``texts`` and return their dense BoW vectors.

        Args:
            texts: List of strings of length ``N``.

        Returns:
            Dense float tensor of shape ``(N, max_features)``.
        """
        self._vec = CountVectorizer(
            max_features=self.max_features,
            lowercase=self.lowercase,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            dtype=np.float32,
        )
        return _to_dense_float32(self._vec.fit_transform(texts))


class TFIDF(_BaseSparseEncoder):
    """TF-IDF encoder backed by sklearn ``TfidfVectorizer`` (sublinear TF, L2 norm)."""

    def fit_transform(self, texts: Sequence[str]) -> torch.Tensor:
        """Fit the vectorizer on ``texts`` and return their dense TF-IDF vectors.

        Args:
            texts: List of strings of length ``N``.

        Returns:
            Dense float tensor of shape ``(N, max_features)``.
        """
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            lowercase=self.lowercase,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        return _to_dense_float32(self._vec.fit_transform(texts))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def cache_path(kind: str, split_seed: int, root: str | Path = "data/embeddings") -> Path:
    """Return the canonical cache filename for an encoder output.

    Args:
        kind: ``"bow"`` or ``"tfidf"``.
        split_seed: The split seed the embeddings were computed against.
        root: Output directory. Defaults to ``data/embeddings``.

    Returns:
        ``Path`` pointing at ``{root}/{kind}_{split_seed}.pt``.
    """
    if kind not in {"bow", "tfidf"}:
        raise ValueError(f"kind must be 'bow' or 'tfidf', got {kind!r}")
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{kind}_{split_seed}.pt"


def save_embeddings(
    tensor: torch.Tensor,
    vocabulary: dict[str, int],
    path: str | Path,
) -> None:
    """Persist an embedding tensor and its vocabulary to a ``.pt`` file.

    Args:
        tensor: Dense ``(N, D)`` float tensor.
        vocabulary: Token -> column index mapping (the fitted vocab).
        path: Destination file path.
    """
    torch.save({"x": tensor, "vocab": vocabulary}, path)


def load_embeddings(path: str | Path) -> tuple[torch.Tensor, dict[str, int]]:
    """Load a previously cached encoder output.

    Args:
        path: File written by ``save_embeddings``.

    Returns:
        ``(tensor, vocabulary)`` tuple.
    """
    blob = torch.load(path, weights_only=False)
    return blob["x"], blob["vocab"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    train = [
        "red leather camera bag with strap",
        "wireless bluetooth headphones over-ear",
        "vintage 35mm film camera body",
        "noise cancelling earbuds with mic",
    ]
    test = [
        "compact mirrorless camera",
        "in-ear sport headphones",
    ]

    for cls in (BagOfWords, TFIDF):
        enc = cls(max_features=64, min_df=1)
        X_train = enc.fit_transform(train)
        X_test = enc.transform(test)
        assert X_train.shape == (4, len(enc.vocabulary_))
        assert X_test.shape == (2, len(enc.vocabulary_))
        assert X_train.dtype == torch.float32
        assert X_test.dtype == torch.float32
        # Round-trip save/load.
        path = cache_path("bow" if cls is BagOfWords else "tfidf", 0, root=".tmp_embeddings")
        save_embeddings(X_train, enc.vocabulary_, path)
        x2, vocab2 = load_embeddings(path)
        assert torch.allclose(x2, X_train)
        assert vocab2 == enc.vocabulary_
        path.unlink()
        path.parent.rmdir()
        print(f"{cls.__name__} OK -> shape {tuple(X_train.shape)}, vocab={len(vocab2)}")
    print("bow_tfidf.py smoke test OK")
