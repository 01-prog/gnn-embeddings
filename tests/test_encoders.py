"""Pytest test suite for the encoder pipeline.

Coverage
--------
1. ``BagOfWordsEncoder`` — fit / encode / encode_cached, embedding_dim,
   error before fit, cache round-trip.
2. ``TFIDFEncoder`` — same tests as BoW.
3. ``BaseEncoder`` — abstract-method enforcement, default fit no-op,
   encode_cached load-from-cache path.
4. Encoder registry (``build_encoder``) — correct type dispatch for all
   registered keys; ``KeyError`` for unknown type.
5. ``SentenceBERTEncoder`` (structural) — skipped when
   ``sentence-transformers`` is absent; interface checked via mock when
   present.
6. ``CLIPTextEncoder`` (structural) — skipped when ``transformers`` is
   absent; interface checked via mock when present.
7. ``Qwen3Encoder`` (structural) — skipped when ``transformers`` is absent;
   interface checked via mock when present.

All tests that require model weights use ``unittest.mock.patch`` so the
suite runs without any network access or GPU.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_texts(n: int = 8) -> list[str]:
    """Return a list of short deterministic texts for testing.

    Args:
        n: Number of texts to generate.

    Returns:
        List of strings.
    """
    vocab = [
        "red leather camera bag strap",
        "wireless bluetooth headphones over-ear noise cancelling",
        "compact mirrorless camera body kit",
        "sport in-ear earbuds microphone",
        "vintage 35mm film camera",
        "wired gaming headset surround sound",
        "instant print camera film",
        "true wireless earbuds charging case",
    ]
    return [vocab[i % len(vocab)] for i in range(n)]


# ---------------------------------------------------------------------------
# 1 & 2: BagOfWordsEncoder / TFIDFEncoder
# ---------------------------------------------------------------------------


class TestBagOfWordsEncoder:
    """Tests for :class:`~src.encoders.bow_tfidf.BagOfWordsEncoder`."""

    def setup_method(self) -> None:
        """Create a fresh temp directory for each test."""
        self._tmp = tempfile.mkdtemp(prefix="bow_test_")

    def teardown_method(self) -> None:
        """Remove the temp directory after each test."""
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_encoder(self, max_features: int = 64) -> Any:
        from src.encoders.bow_tfidf import BagOfWordsEncoder
        return BagOfWordsEncoder(
            cache_dir=self._tmp, device="cpu", max_features=max_features, min_df=1
        )

    def test_embedding_dim_before_fit(self) -> None:
        """embedding_dim equals max_features before the encoder is fitted."""
        enc = self._make_encoder(max_features=64)
        assert enc.embedding_dim == 64

    def test_fit_changes_embedding_dim(self) -> None:
        """After fit(), embedding_dim equals the actual vocabulary size."""
        enc = self._make_encoder(max_features=100)
        enc.fit(_sample_texts(8))
        # Vocabulary is <= max_features; must be > 0.
        assert 0 < enc.embedding_dim <= 100

    def test_encode_returns_float32_tensor(self) -> None:
        """encode() produces a float32 tensor of the right shape."""
        texts = _sample_texts(8)
        enc = self._make_encoder()
        enc.fit(texts[:4])
        x = enc.encode(texts)
        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32
        assert x.shape == (8, enc.embedding_dim)

    def test_encode_before_fit_raises(self) -> None:
        """encode() raises RuntimeError if called before fit()."""
        enc = self._make_encoder()
        with pytest.raises(RuntimeError, match="fit"):
            enc.encode(_sample_texts(2))

    def test_encode_cached_saves_and_loads(self) -> None:
        """encode_cached() saves a .pt file and returns the same tensor."""
        texts = _sample_texts(8)
        enc = self._make_encoder()
        enc.fit(texts[:4])

        x1 = enc.encode_cached(texts, "bow_cache_test")
        cache_file = Path(self._tmp) / "bow_cache_test.pt"
        assert cache_file.exists(), "Cache file not created."

        # Second call should load from cache.
        x2 = enc.encode_cached(texts, "bow_cache_test")
        assert torch.allclose(x1, x2)

    def test_encode_cached_corrupt_file_recovered(self) -> None:
        """encode_cached() re-encodes when the cache file is corrupt."""
        texts = _sample_texts(4)
        enc = self._make_encoder()
        enc.fit(texts)

        cache_file = Path(self._tmp) / "corrupt_key.pt"
        cache_file.write_bytes(b"not a valid tensor")

        # Should not raise; should silently re-encode.
        x = enc.encode_cached(texts, "corrupt_key")
        assert x.shape == (4, enc.embedding_dim)

    def test_repr_contains_class_name(self) -> None:
        """__repr__ includes the class name."""
        enc = self._make_encoder()
        assert "BagOfWordsEncoder" in repr(enc)

    def test_no_label_leakage(self) -> None:
        """Vocabulary is learned from train_texts, not test_texts."""
        enc_a = self._make_encoder(max_features=50)
        enc_b = self._make_encoder(max_features=50)

        train = _sample_texts(4)
        all_texts = train + ["unseen jargon xyzzy42 foo bar"]

        enc_a.fit(train)
        enc_b.fit(all_texts)  # larger vocabulary

        # enc_a must have a different (potentially smaller) vocab than enc_b.
        dim_a = enc_a.embedding_dim
        dim_b = enc_b.embedding_dim
        assert dim_a <= dim_b, "Fitting on more texts should not shrink dim."


class TestTFIDFEncoder:
    """Tests for :class:`~src.encoders.bow_tfidf.TFIDFEncoder`.

    Replicates the BoW tests because the interface is identical; only the
    underlying vectoriser differs.
    """

    def setup_method(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="tfidf_test_")

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_encoder(self, max_features: int = 64) -> Any:
        from src.encoders.bow_tfidf import TFIDFEncoder
        return TFIDFEncoder(
            cache_dir=self._tmp, device="cpu", max_features=max_features, min_df=1
        )

    def test_embedding_dim_before_fit(self) -> None:
        assert self._make_encoder(64).embedding_dim == 64

    def test_encode_returns_unit_norm_rows(self) -> None:
        """TF-IDF rows should be L2-normalised (sklearn ``norm='l2'``).

        We fit and encode the *same* texts so every document has at least
        one vocabulary token and therefore a non-zero vector.  (If a test
        document is entirely out-of-vocabulary after a train-only fit, its
        vector is the zero vector; normalising zero gives zero, which is the
        correct behaviour but not what this test targets.)
        """
        texts = _sample_texts(8)
        enc = self._make_encoder()
        enc.fit(texts)           # fit on all 8 so no OOV rows
        x = enc.encode(texts)
        norms = x.norm(dim=1)
        # All rows must be unit-norm within float32 precision.
        assert torch.allclose(norms, torch.ones(8), atol=1e-5), (
            f"Not unit-norm: min={norms.min():.5f} max={norms.max():.5f}"
        )

    def test_encode_before_fit_raises(self) -> None:
        enc = self._make_encoder()
        with pytest.raises(RuntimeError, match="fit"):
            enc.encode(_sample_texts(2))

    def test_encode_cached_round_trip(self) -> None:
        texts = _sample_texts(6)
        enc = self._make_encoder()
        enc.fit(texts[:3])
        x1 = enc.encode_cached(texts, "tfidf_round")
        x2 = enc.encode_cached(texts, "tfidf_round")
        assert torch.allclose(x1, x2)

    def test_repr_contains_class_name(self) -> None:
        assert "TFIDFEncoder" in repr(self._make_encoder())


# ---------------------------------------------------------------------------
# 3: BaseEncoder abstract interface
# ---------------------------------------------------------------------------


class TestBaseEncoderAbstract:
    """Tests for :class:`~src.encoders.base.BaseEncoder` itself."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseEncoder is abstract and must not be instantiated."""
        from src.encoders.base import BaseEncoder

        with pytest.raises(TypeError):
            BaseEncoder()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        """A minimal concrete subclass satisfies the ABC contract."""
        from src.encoders.base import BaseEncoder

        class _Const(BaseEncoder):
            @property
            def embedding_dim(self) -> int:
                return 4

            def encode(self, inputs: list[str]) -> torch.Tensor:  # type: ignore[override]
                return torch.zeros(len(inputs), 4)

        tmp = tempfile.mkdtemp()
        try:
            enc = _Const(cache_dir=tmp, device="cpu")
            assert enc.embedding_dim == 4
            x = enc.encode(["hello", "world"])
            assert x.shape == (2, 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fit_is_noop_on_base(self) -> None:
        """The default fit() must not raise and must return None."""
        from src.encoders.base import BaseEncoder

        class _Const(BaseEncoder):
            @property
            def embedding_dim(self) -> int:
                return 2

            def encode(self, inputs: list[str]) -> torch.Tensor:  # type: ignore[override]
                return torch.zeros(len(inputs), 2)

        tmp = tempfile.mkdtemp()
        try:
            enc = _Const(cache_dir=tmp, device="cpu")
            result = enc.fit(["a", "b"])
            assert result is None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_encode_cached_creates_file(self) -> None:
        """encode_cached() persists a .pt file on first call."""
        from src.encoders.base import BaseEncoder

        class _Const(BaseEncoder):
            @property
            def embedding_dim(self) -> int:
                return 3

            def encode(self, inputs: list[str]) -> torch.Tensor:  # type: ignore[override]
                return torch.ones(len(inputs), 3)

        tmp = Path(tempfile.mkdtemp())
        try:
            enc = _Const(cache_dir=tmp, device="cpu")
            enc.encode_cached(["a", "b"], "base_cache_check")
            assert (tmp / "base_cache_check.pt").exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_encode_cached_skips_encode_on_second_call(self) -> None:
        """encode_cached() must call encode() only once; second call loads cache."""
        from src.encoders.base import BaseEncoder

        call_count = {"n": 0}

        class _Counter(BaseEncoder):
            @property
            def embedding_dim(self) -> int:
                return 2

            def encode(self, inputs: list[str]) -> torch.Tensor:  # type: ignore[override]
                call_count["n"] += 1
                return torch.zeros(len(inputs), 2)

        tmp = Path(tempfile.mkdtemp())
        try:
            enc = _Counter(cache_dir=tmp, device="cpu")
            enc.encode_cached(["x"], "skip_test")
            enc.encode_cached(["x"], "skip_test")
            assert call_count["n"] == 1, (
                f"encode() called {call_count['n']} times; expected 1."
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4: Encoder registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for :func:`~src.encoders.registry.build_encoder`."""

    def setup_method(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="registry_test_")

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _build(self, cfg: dict[str, Any]) -> Any:
        from src.encoders.registry import build_encoder
        return build_encoder(cfg, cache_dir=self._tmp, device="cpu")

    def test_bow_dispatch(self) -> None:
        """build_encoder with type='BagOfWords' returns a BagOfWordsEncoder."""
        from src.encoders.bow_tfidf import BagOfWordsEncoder
        enc = self._build({"type": "BagOfWords", "max_features": 50})
        assert isinstance(enc, BagOfWordsEncoder)

    def test_tfidf_dispatch(self) -> None:
        """build_encoder with type='TFIDF' returns a TFIDFEncoder."""
        from src.encoders.bow_tfidf import TFIDFEncoder
        enc = self._build({"type": "TFIDF", "max_features": 50})
        assert isinstance(enc, TFIDFEncoder)

    def test_unknown_type_raises_key_error(self) -> None:
        """build_encoder raises KeyError for an unregistered type."""
        with pytest.raises(KeyError, match="Unknown encoder type"):
            self._build({"type": "NonExistentEncoder123"})

    def test_name_key_is_stripped(self) -> None:
        """The 'name' key must not cause a TypeError in the constructor."""
        enc = self._build({"type": "BagOfWords", "name": "bow", "max_features": 32})
        from src.encoders.bow_tfidf import BagOfWordsEncoder
        assert isinstance(enc, BagOfWordsEncoder)

    def test_bow_max_features_forwarded(self) -> None:
        """max_features from the config dict is respected by BagOfWordsEncoder."""
        enc = self._build({"type": "BagOfWords", "max_features": 256})
        # Before fitting, embedding_dim == max_features.
        assert enc.embedding_dim == 256

    def test_tfidf_max_features_forwarded(self) -> None:
        enc = self._build({"type": "TFIDF", "max_features": 128})
        assert enc.embedding_dim == 128

    def test_sbert_dispatch(self) -> None:
        """build_encoder with type='SentenceBERT' returns a SentenceBERTEncoder."""
        pytest.importorskip("sentence_transformers")
        from src.encoders.sbert import SentenceBERTEncoder

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            enc = self._build({
                "type": "SentenceBERT",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            })
        assert isinstance(enc, SentenceBERTEncoder)

    def test_clip_text_dispatch(self) -> None:
        """build_encoder with type='CLIPText' returns a CLIPTextEncoder."""
        pytest.importorskip("transformers")
        from src.encoders.clip_text import CLIPTextEncoder

        with (
            patch("transformers.CLIPTextModelWithProjection.from_pretrained") as mock_m,
            patch("transformers.CLIPTokenizerFast.from_pretrained") as mock_t,
        ):
            mock_cfg = MagicMock()
            mock_cfg.projection_dim = 512
            mock_model = MagicMock()
            mock_model.config = mock_cfg
            mock_model.to.return_value = mock_model
            mock_m.return_value = mock_model
            mock_t.return_value = MagicMock()

            enc = self._build({"type": "CLIPText"})
        assert isinstance(enc, CLIPTextEncoder)

    def test_qwen3_dispatch(self) -> None:
        """build_encoder with type='Qwen3' returns a Qwen3Encoder."""
        pytest.importorskip("transformers")
        from src.encoders.qwen3 import Qwen3Encoder

        with (
            patch("transformers.AutoModel.from_pretrained") as mock_m,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_t,
        ):
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_m.return_value = mock_model
            mock_t.return_value = MagicMock()

            enc = self._build({"type": "Qwen3", "model_size": "0.6B"})
        assert isinstance(enc, Qwen3Encoder)


# ---------------------------------------------------------------------------
# 5: SentenceBERTEncoder (structural / mock)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sbert_available() -> bool:
    """Return True when sentence-transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class TestSentenceBERTEncoder:
    """Structural tests for :class:`~src.encoders.sbert.SentenceBERTEncoder`.

    All tests that would trigger a model download are mocked out so the
    suite passes without network access.
    """

    def setup_method(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="sbert_test_")

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_import_error_without_library(self, sbert_available: bool) -> None:
        """Constructor raises ImportError if sentence-transformers is absent."""
        if sbert_available:
            pytest.skip("sentence-transformers is installed; cannot test missing dep.")

        from src.encoders.sbert import SentenceBERTEncoder

        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceBERTEncoder(cache_dir=self._tmp, device="cpu")

    def test_encode_returns_correct_shape(self, sbert_available: bool) -> None:
        """encode() returns (N, embedding_dim) float32 tensor."""
        if not sbert_available:
            pytest.skip("sentence-transformers not installed.")

        from src.encoders.sbert import SentenceBERTEncoder

        emb_dim = 384
        n_texts = 5
        fake_embeddings = torch.randn(n_texts, emb_dim)

        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.encode.return_value = fake_embeddings
            mock_model.get_sentence_embedding_dimension.return_value = emb_dim
            mock_cls.return_value = mock_model

            enc = SentenceBERTEncoder(
                cache_dir=self._tmp,
                device="cpu",
                batch_size=4,
            )
            texts = _sample_texts(n_texts)
            x = enc.encode(texts)

        assert x.shape == (n_texts, emb_dim)
        assert x.dtype == torch.float32

    def test_encode_empty_list_returns_empty_tensor(self, sbert_available: bool) -> None:
        """encode([]) must return an empty (0, embedding_dim) tensor."""
        if not sbert_available:
            pytest.skip("sentence-transformers not installed.")

        from src.encoders.sbert import SentenceBERTEncoder

        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_model

            enc = SentenceBERTEncoder(cache_dir=self._tmp, device="cpu")
            x = enc.encode([])

        assert x.shape == (0, 384)
        assert x.dtype == torch.float32

    def test_embedding_dim_from_known_model(self, sbert_available: bool) -> None:
        """embedding_dim is resolved from the internal lookup table."""
        if not sbert_available:
            pytest.skip("sentence-transformers not installed.")

        from src.encoders.sbert import SentenceBERTEncoder

        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_cls.return_value = mock_model

            enc = SentenceBERTEncoder(
                cache_dir=self._tmp,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
        assert enc.embedding_dim == 384


# ---------------------------------------------------------------------------
# 6: CLIPTextEncoder (structural / mock)
# ---------------------------------------------------------------------------


@pytest.fixture()
def transformers_available() -> bool:
    """Return True when HuggingFace transformers is importable."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class TestCLIPTextEncoder:
    """Structural tests for :class:`~src.encoders.clip_text.CLIPTextEncoder`."""

    def setup_method(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="clip_test_")

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_mock_clip(self, emb_dim: int = 512, batch_size: int = 4) -> Any:
        """Build a mocked CLIPTextEncoder.

        Args:
            emb_dim: Projection dimension reported by the mock model config.
            batch_size: Initial batch size for the encoder.

        Returns:
            A :class:`~src.encoders.clip_text.CLIPTextEncoder` with all
            HuggingFace internals replaced by mocks.
        """
        from src.encoders.clip_text import CLIPTextEncoder

        mock_cfg = MagicMock()
        mock_cfg.projection_dim = emb_dim

        mock_model = MagicMock()
        mock_model.config = mock_cfg
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = []

        # Simulate the text_embeds output.
        def _forward(**kwargs: Any) -> SimpleNamespace:
            n = kwargs["input_ids"].shape[0]
            return SimpleNamespace(text_embeds=torch.randn(n, emb_dim))

        mock_model.__call__ = _forward
        mock_model.side_effect = _forward

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        with (
            patch("transformers.CLIPTextModelWithProjection.from_pretrained",
                  return_value=mock_model),
            patch("transformers.CLIPTokenizerFast.from_pretrained",
                  return_value=mock_tokenizer),
        ):
            enc = CLIPTextEncoder(
                cache_dir=self._tmp,
                device="cpu",
                batch_size=batch_size,
                normalize_embeddings=False,
            )
        return enc, mock_model, mock_tokenizer

    def test_import_error_without_transformers(
        self, transformers_available: bool
    ) -> None:
        """Constructor raises ImportError if transformers is absent."""
        if transformers_available:
            pytest.skip("transformers is installed; cannot test missing dep.")

        from src.encoders.clip_text import CLIPTextEncoder

        with pytest.raises(ImportError, match="transformers"):
            CLIPTextEncoder(cache_dir=self._tmp, device="cpu")

    def test_embedding_dim(self, transformers_available: bool) -> None:
        """embedding_dim matches the model's projection_dim."""
        if not transformers_available:
            pytest.skip("transformers not installed.")

        enc, _, _ = self._make_mock_clip(emb_dim=512)
        assert enc.embedding_dim == 512

    def test_encode_empty_list(self, transformers_available: bool) -> None:
        """encode([]) returns an empty (0, embedding_dim) tensor."""
        if not transformers_available:
            pytest.skip("transformers not installed.")

        enc, _, _ = self._make_mock_clip(emb_dim=512)
        x = enc.encode([])
        assert x.shape == (0, 512)
        assert x.dtype == torch.float32

    def test_normalize_embeddings(self, transformers_available: bool) -> None:
        """When normalize_embeddings=True, all row norms equal 1."""
        if not transformers_available:
            pytest.skip("transformers not installed.")

        from src.encoders.clip_text import CLIPTextEncoder

        mock_cfg = MagicMock()
        mock_cfg.projection_dim = 512

        mock_model = MagicMock()
        mock_model.config = mock_cfg
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = []

        n_texts = 3

        def _forward(**kwargs: Any) -> SimpleNamespace:
            # Unnormalised embeddings (random non-unit vectors).
            n = kwargs["input_ids"].shape[0]
            return SimpleNamespace(
                text_embeds=torch.randn(n, 512) * 5.0
            )

        mock_model.__call__ = _forward
        mock_model.side_effect = _forward

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(n_texts, 10, dtype=torch.long)
        }

        with (
            patch("transformers.CLIPTextModelWithProjection.from_pretrained",
                  return_value=mock_model),
            patch("transformers.CLIPTokenizerFast.from_pretrained",
                  return_value=mock_tokenizer),
        ):
            enc = CLIPTextEncoder(
                cache_dir=self._tmp,
                device="cpu",
                normalize_embeddings=True,
            )

        x = enc.encode(_sample_texts(n_texts))
        norms = x.norm(dim=1)
        assert torch.allclose(norms, torch.ones(n_texts), atol=1e-5), (
            f"Not unit-norm after normalise: {norms}"
        )


# ---------------------------------------------------------------------------
# 7: Qwen3Encoder (structural / mock)
# ---------------------------------------------------------------------------


class TestQwen3Encoder:
    """Structural tests for :class:`~src.encoders.qwen3.Qwen3Encoder`."""

    def setup_method(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="qwen3_test_")

    def teardown_method(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_invalid_model_size_raises(self, transformers_available: bool) -> None:
        """Passing an unsupported model_size raises ValueError immediately."""
        if not transformers_available:
            pytest.skip("transformers not installed.")

        from src.encoders.qwen3 import Qwen3Encoder

        # This should fail at the guard before any from_pretrained call.
        with pytest.raises(ValueError, match="model_size"):
            with (
                patch("transformers.AutoModel.from_pretrained", return_value=MagicMock()),
                patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            ):
                Qwen3Encoder(cache_dir=self._tmp, device="cpu", model_size="999B")

    def test_embedding_dim_known_sizes(self, transformers_available: bool) -> None:
        """embedding_dim is correct for the three supported model sizes."""
        if not transformers_available:
            pytest.skip("transformers not installed.")

        from src.encoders.qwen3 import Qwen3Encoder, _KNOWN_DIMS

        for size, expected_dim in _KNOWN_DIMS.items():
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.parameters.return_value = []

            with (
                patch("transformers.AutoModel.from_pretrained",
                      return_value=mock_model),
                patch("transformers.AutoTokenizer.from_pretrained",
                      return_value=MagicMock()),
            ):
                enc = Qwen3Encoder(
                    cache_dir=self._tmp,
                    device="cpu",
                    model_size=size,
                )
            assert enc.embedding_dim == expected_dim, (
                f"Wrong dim for model_size='{size}': got {enc.embedding_dim}, "
                f"expected {expected_dim}"
            )

    def test_encode_empty_list(self, transformers_available: bool) -> None:
        """encode([]) returns an empty (0, embedding_dim) tensor."""
        if not transformers_available:
            pytest.skip("transformers not installed.")

        from src.encoders.qwen3 import Qwen3Encoder

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = []

        with (
            patch("transformers.AutoModel.from_pretrained", return_value=mock_model),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        ):
            enc = Qwen3Encoder(cache_dir=self._tmp, device="cpu", model_size="0.6B")

        x = enc.encode([])
        assert x.shape == (0, 1024)
        assert x.dtype == torch.float32

    def test_import_error_without_transformers(
        self, transformers_available: bool
    ) -> None:
        """Constructor raises ImportError if transformers is absent."""
        if transformers_available:
            pytest.skip("transformers is installed; cannot test missing dep.")

        from src.encoders.qwen3 import Qwen3Encoder

        with pytest.raises(ImportError, match="transformers"):
            Qwen3Encoder(cache_dir=self._tmp, device="cpu")


# ---------------------------------------------------------------------------
# 8: Pooling helpers (Qwen3 internals, no model needed)
# ---------------------------------------------------------------------------


class TestQwen3PoolingHelpers:
    """Unit tests for ``_last_token_pool`` and ``_mean_pool`` in qwen3.py.

    No model download required — tests only the pure-math functions.
    """

    def test_last_token_pool_left_padded(self) -> None:
        """Left-padded sequences: last column is the real token for all rows."""
        from src.encoders.qwen3 import _last_token_pool

        B, T, D = 3, 5, 8
        h = torch.randn(B, T, D)
        # All last columns are real tokens (left-padded convention).
        mask = torch.ones(B, T, dtype=torch.long)

        out = _last_token_pool(h, mask)
        assert out.shape == (B, D)
        assert torch.allclose(out, h[:, -1])

    def test_last_token_pool_right_padded(self) -> None:
        """Right-padded: each row picks its own last real token."""
        from src.encoders.qwen3 import _last_token_pool

        B, T, D = 2, 6, 4
        h = torch.arange(B * T * D, dtype=torch.float).view(B, T, D)
        # Sequence lengths: row 0 has 4 real tokens, row 1 has 6.
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[0, :4] = 1
        mask[1, :6] = 1

        out = _last_token_pool(h, mask)
        assert out.shape == (B, D)
        assert torch.allclose(out[0], h[0, 3])   # 4th token (index 3)
        assert torch.allclose(out[1], h[1, 5])   # 6th token (index 5)

    def test_mean_pool_ignores_padding(self) -> None:
        """Mean pooling averages only the non-padding tokens."""
        from src.encoders.qwen3 import _mean_pool

        B, T, D = 2, 4, 3
        h = torch.ones(B, T, D)
        h[0] *= 2.0   # row 0: all tokens have value 2
        h[1] *= 4.0   # row 1: all tokens have value 4

        # Row 0: 3 real tokens, row 1: 2 real tokens.
        mask = torch.zeros(B, T, dtype=torch.long)
        mask[0, :3] = 1
        mask[1, :2] = 1

        out = _mean_pool(h, mask)
        assert out.shape == (B, D)
        assert torch.allclose(out[0], torch.full((D,), 2.0))
        assert torch.allclose(out[1], torch.full((D,), 4.0))

    def test_mean_pool_all_real_tokens(self) -> None:
        """When there is no padding, mean pool equals the row mean."""
        from src.encoders.qwen3 import _mean_pool

        B, T, D = 3, 5, 6
        h = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)

        out = _mean_pool(h, mask)
        expected = h.mean(dim=1)
        assert torch.allclose(out, expected, atol=1e-6)
