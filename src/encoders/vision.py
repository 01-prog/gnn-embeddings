"""Vision encoders for the H2 multimodal experiment.

Three encoder tiers — each frozen, each representing a different pretraining
paradigm — are compared to quantify the *vision* axis of H2:

Tier 0 (baseline)     ResNetEncoder     ResNet-50, supervised ImageNet
Tier 1 (contrastive)  CLIPVisionEncoder CLIP ViT-B/32, image-text contrastive
Tier 2 (self-supervised) DINOv2Encoder  ViT-B/14 or ViT-L/14, DINO v2

H2 pretraining-axis comparison (vision side):
  Axis           | Encoder         | Objective
  ---------------+-----------------+------------------------------------------
  Supervised     | ResNet-50       | ImageNet top-1 classification
  Contrastive    | CLIP ViT-B/32   | Image-text alignment (same space as CLIPText)
  Self-supervised| DINOv2 ViT-B/14 | DINO v2 self-distillation (no labels)

Interface
---------
All three implement ``BaseEncoder`` with:

  ``fit(inputs)``          → no-op (frozen weights)
  ``encode(inputs)``       → ``list[Path | None]`` → ``(N, D)`` float32 tensor
  ``encode_cached(...)``   → inherited from BaseEncoder
  ``embedding_dim``        → int property

The ``inputs`` for vision encoders are **image paths** (``pathlib.Path``),
not strings.  A ``None`` entry means the image is unavailable; the encoder
fills that row with a **zero vector** of shape ``(1, embedding_dim)``.

Zero-fill rationale
-------------------
Replacing missing images with zero vectors is the simplest option that
preserves the transductive GNN's requirement for a fixed-size feature matrix.
The zero vector is out-of-distribution for all three encoders (real images
produce non-zero embeddings after normalisation), which may suppress the
image contribution for those nodes.  The H2 runner flags this as a potential
confounder by comparing accuracy on image-available vs. image-unavailable
test nodes separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.encoders.base import BaseEncoder

logger = logging.getLogger(__name__)

# ── ImageNet normalization constants (used by ResNet) ─────────────────────
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared batch helper
# ---------------------------------------------------------------------------


def _load_pil_images(paths: list[Path]) -> list[Any]:
    """Load a list of image paths as PIL RGB images.

    Corrupt / unreadable files are returned as ``None`` (the caller
    replaces them with zero vectors).

    Args:
        paths: Non-None image paths.

    Returns:
        List of ``PIL.Image.Image`` objects (or ``None`` on load failure).
    """
    from PIL import Image as PILImage

    images = []
    for p in paths:
        try:
            images.append(PILImage.open(p).convert("RGB"))
        except Exception as exc:
            logger.warning("Failed to load image '%s': %s", p, exc)
            images.append(None)
    return images


def _encode_with_paths(
    encoder: "BaseVisionEncoder",
    inputs: list[Path | None],
) -> torch.Tensor:
    """Shared encode logic: batch non-None paths, fill None with zeros.

    Processes the valid images in mini-batches of ``encoder.batch_size``,
    automatically halving on CUDA OOM.  Invalid / missing images receive a
    zero vector in the output.

    Args:
        encoder: A ``BaseVisionEncoder`` with a ``_encode_pil`` method.
        inputs: List of ``Path | None`` of length ``N``.

    Returns:
        Float32 tensor of shape ``(N, encoder.embedding_dim)``.
    """
    n = len(inputs)
    if n == 0:
        return torch.empty(0, encoder.embedding_dim, dtype=torch.float32)

    result = torch.zeros(n, encoder.embedding_dim, dtype=torch.float32)

    # Separate valid from missing.
    valid = [(i, p) for i, p in enumerate(inputs) if p is not None]
    if not valid:
        return result

    indices, paths = zip(*valid)
    paths_list = list(paths)
    indices_list = list(indices)

    batch_size = encoder.batch_size
    pos = 0
    while pos < len(paths_list):
        batch_paths = paths_list[pos: pos + batch_size]
        batch_indices = indices_list[pos: pos + batch_size]
        while True:
            try:
                pil_imgs = _load_pil_images(batch_paths)
                # Separate successfully-loaded from corrupt.
                good: list[tuple[int, Any]] = [
                    (batch_indices[j], img)
                    for j, img in enumerate(pil_imgs)
                    if img is not None
                ]
                if good:
                    good_indices, good_imgs = zip(*good)
                    embs = encoder._encode_pil(list(good_imgs))  # (k, D)
                    embs_cpu = embs.cpu().float()
                    for orig_idx, emb in zip(good_indices, embs_cpu):
                        result[orig_idx] = emb
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning(
                        "CUDA OOM — retrying vision batch with batch_size=%d",
                        batch_size,
                    )
                else:
                    raise
        pos += batch_size

    return result


class BaseVisionEncoder(BaseEncoder):
    """Abstract base for the three vision encoders.

    Adds a ``batch_size`` attribute and a ``_encode_pil`` abstract method
    that subclasses implement.  The public ``encode`` method is provided
    here via :func:`_encode_with_paths`.

    Args:
        cache_dir: Directory for ``.pt`` cached embedding files.
        device: Inference device.
        batch_size: Number of images per forward pass.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/embeddings",
        device: str | torch.device = "auto",
        batch_size: int = 32,
    ) -> None:
        super().__init__(cache_dir=cache_dir, device=device)
        self.batch_size = batch_size

    def encode(self, inputs: list[Path | None] | list[Any]) -> torch.Tensor:
        """Encode a list of image paths into vision embeddings.

        ``None`` entries produce zero vectors; corrupt files are treated as
        missing (also zero).

        Args:
            inputs: List of ``pathlib.Path | None`` of length ``N``.

        Returns:
            Float32 tensor of shape ``(N, embedding_dim)``.
        """
        return _encode_with_paths(self, inputs)  # type: ignore[arg-type]

    def _encode_pil(self, images: list[Any]) -> torch.Tensor:
        """Encode a batch of PIL images.

        Subclasses must implement this.  The input list contains only
        non-None, successfully-loaded PIL ``Image`` objects.

        Args:
            images: List of ``PIL.Image.Image`` of length ``B``.

        Returns:
            Float tensor of shape ``(B, embedding_dim)`` on ``self.device``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. ResNetEncoder — supervised ImageNet baseline
# ---------------------------------------------------------------------------


class ResNetEncoder(BaseVisionEncoder):
    """ResNet-50 vision encoder (supervised ImageNet baseline).

    Uses the **avg-pooled penultimate layer** (after ``layer4``, before the
    fully-connected classifier) to extract 2048-dimensional feature vectors.

    Pretraining objective: supervised top-1 classification on ImageNet-1K.
    This is the simplest vision feature extractor and serves as the
    tier-0 vision baseline in H2.

    The backbone weights are fully **frozen** (``eval()`` + ``no_grad()``).

    Args:
        cache_dir: Directory for cached ``.pt`` files.
        device: Inference device.
        batch_size: Images per forward pass.
        normalize_embeddings: L2-normalise the 2048-dim output (default False
            for ResNet as the features are not conventionally normalised).

    Raises:
        ImportError: If ``torchvision`` is not installed.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/embeddings",
        device: str | torch.device = "auto",
        batch_size: int = 32,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__(cache_dir=cache_dir, device=device, batch_size=batch_size)
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading ResNet-50 (ImageNet weights) on %s …", self.device)
        try:
            import torchvision.models as tv_models
            from torchvision.models import ResNet50_Weights
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "torchvision is not installed. Run: pip install torchvision"
            ) from exc

        # Remove the final fully-connected layer.  After the global average
        # pool (AdaptiveAvgPool2d → flatten), the output is (B, 2048).
        backbone = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self._model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self._model = self._model.to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        self._dim = 2048
        logger.info("ResNetEncoder ready: dim=%d device=%s", self._dim, self.device)

    @property
    def embedding_dim(self) -> int:
        """Feature dimension: 2048 for ResNet-50."""
        return self._dim

    def _encode_pil(self, images: list[Any]) -> torch.Tensor:
        """Forward a batch of PIL images through ResNet-50.

        Args:
            images: List of ``PIL.Image.Image`` of length ``B``.

        Returns:
            Float tensor of shape ``(B, 2048)`` on ``self.device``.
        """
        import torch.nn.functional as F

        tensors = torch.stack(
            [self._transforms(img) for img in images]
        ).to(self.device)                              # (B, 3, 224, 224)

        with torch.no_grad():
            out = self._model(tensors)                 # (B, 2048, 1, 1)
        emb = out.flatten(1)                           # (B, 2048)

        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb


# ---------------------------------------------------------------------------
# 2. CLIPVisionEncoder — image-text contrastive (same space as CLIPTextEncoder)
# ---------------------------------------------------------------------------


class CLIPVisionEncoder(BaseVisionEncoder):
    """CLIP ViT-B/32 vision encoder (contrastive image-text pretraining).

    Uses ``CLIPVisionModelWithProjection`` from HuggingFace so the output
    lies in the **same 512-D CLIP embedding space** as
    :class:`~src.encoders.clip_text.CLIPTextEncoder`.  This enables
    direct comparison (and late-fusion by addition or cosine similarity)
    between text and image CLIP embeddings.

    Pretraining objective: contrastive image-text alignment on 400 M
    image-caption pairs (same CLIP training as the text encoder).

    Args:
        cache_dir: Directory for cached ``.pt`` files.
        device: Inference device.
        model_name: HuggingFace CLIP model identifier.
        batch_size: Images per forward pass.
        normalize_embeddings: L2-normalise output (default True —
            CLIP embeddings are conventionally unit-norm).

    Raises:
        ImportError: If ``transformers`` or ``Pillow`` is not installed.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/embeddings",
        device: str | torch.device = "auto",
        model_name: str = "openai/clip-vit-b-32",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__(cache_dir=cache_dir, device=device, batch_size=batch_size)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading CLIP vision model '%s' on %s …", model_name, self.device)
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            ) from exc

        self._processor = CLIPImageProcessor.from_pretrained(model_name)
        self._model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self._model = self._model.to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._dim: int = self._model.config.projection_dim  # 512 for ViT-B/32
        logger.info(
            "CLIPVisionEncoder ready: model='%s' dim=%d device=%s",
            model_name, self._dim, self.device,
        )

    @property
    def embedding_dim(self) -> int:
        """CLIP projection dimension (512 for ViT-B/32)."""
        return self._dim

    def _encode_pil(self, images: list[Any]) -> torch.Tensor:
        """Forward a batch through CLIP vision + projection head.

        Args:
            images: List of ``PIL.Image.Image`` of length ``B``.

        Returns:
            Float tensor of shape ``(B, 512)`` on ``self.device``.
        """
        import torch.nn.functional as F

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        emb = out.image_embeds                         # (B, 512), already projected

        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# ---------------------------------------------------------------------------
# 3. DINOv2Encoder — self-supervised ViT baseline
# ---------------------------------------------------------------------------

# Known CLS-token dimensions for each DINOv2 variant.
_DINOV2_DIMS: dict[str, int] = {
    "B": 768,    # ViT-B/14  (~86 M parameters)
    "L": 1024,   # ViT-L/14  (~307 M parameters)
}

_DINOV2_HF_IDS: dict[str, str] = {
    "B": "facebook/dinov2-base",
    "L": "facebook/dinov2-large",
}


class DINOv2Encoder(BaseVisionEncoder):
    """DINOv2 vision encoder (self-supervised ViT pretraining).

    Uses the ``[CLS]`` token from the final Transformer block as the image
    representation.  Unlike ResNet (supervised) and CLIP (contrastive),
    DINOv2 was trained without *any* labels, making it the pure
    self-supervised point on the H2 pretraining-objective axis.

    Supported variants:

    ======== ============= ==========
    model_size  HF model   dim
    ======== ============= ==========
    ``"B"``  dinov2-base   768
    ``"L"``  dinov2-large  1024
    ======== ============= ==========

    Args:
        cache_dir: Directory for cached ``.pt`` files.
        device: Inference device.
        model_size: ``"B"`` (86 M, default) or ``"L"`` (307 M, stretch goal).
        batch_size: Images per forward pass.
        normalize_embeddings: L2-normalise CLS token (default True).

    Raises:
        ValueError: If ``model_size`` is not ``"B"`` or ``"L"``.
        ImportError: If ``transformers`` is not installed.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/embeddings",
        device: str | torch.device = "auto",
        model_size: str = "B",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if model_size not in _DINOV2_DIMS:
            raise ValueError(
                f"model_size must be one of {list(_DINOV2_DIMS)}, got '{model_size}'"
            )
        super().__init__(cache_dir=cache_dir, device=device, batch_size=batch_size)
        self.model_size = model_size
        self.normalize_embeddings = normalize_embeddings
        self._dim = _DINOV2_DIMS[model_size]
        model_id = _DINOV2_HF_IDS[model_size]

        logger.info(
            "Loading DINOv2-%s ('%s') on %s …", model_size, model_id, self.device
        )
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run: pip install transformers"
            ) from exc

        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model = self._model.to(self.device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        logger.info(
            "DINOv2Encoder ready: size=%s dim=%d device=%s",
            model_size, self._dim, self.device,
        )

    @property
    def embedding_dim(self) -> int:
        """CLS-token dimension (768 for ViT-B/14, 1024 for ViT-L/14)."""
        return self._dim

    def _encode_pil(self, images: list[Any]) -> torch.Tensor:
        """Forward a batch through DINOv2 and extract the CLS token.

        Args:
            images: List of ``PIL.Image.Image`` of length ``B``.

        Returns:
            Float tensor of shape ``(B, embedding_dim)`` on ``self.device``.
        """
        import torch.nn.functional as F

        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        # CLS token is the first token of last_hidden_state.
        emb = out.last_hidden_state[:, 0]              # (B, dim)

        if self.normalize_embeddings:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    # ── Test zero-fill for None inputs ──────────────────────────────────────
    # We do NOT download any model; we just verify the _encode_with_paths
    # helper produces the right zero tensors when all inputs are None.

    class _FakeVision(BaseVisionEncoder):
        """Minimal concrete vision encoder for testing _encode_with_paths."""

        @property
        def embedding_dim(self) -> int:
            return 16

        def _encode_pil(self, images: list[Any]) -> torch.Tensor:
            return torch.randn(len(images), 16)

    tmp = Path(tempfile.mkdtemp(prefix="vision_smoke_"))
    try:
        enc = _FakeVision(cache_dir=tmp, device="cpu", batch_size=4)

        # All None → all zeros.
        x = enc.encode([None, None, None])
        assert x.shape == (3, 16)
        assert x.dtype == torch.float32
        assert (x == 0).all(), "Expected zeros for all-None inputs."

        # Mix of valid paths (None here, but FakeVision handles None gracefully
        # after filtering) and None.
        x2 = enc.encode([])
        assert x2.shape == (0, 16)

        # encode_cached round-trip.
        x3 = enc.encode_cached([None, None], "vision_smoke_cache")
        x4 = enc.encode_cached([None, None], "vision_smoke_cache")
        assert torch.allclose(x3, x4)
        assert (tmp / "vision_smoke_cache.pt").exists()

        logger.info("BaseVisionEncoder / _encode_with_paths smoke test OK.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("vision.py smoke test OK (zero-fill and cache round-trip verified).")
    print("Full ResNet/CLIP/DINOv2 tests require model downloads.")
