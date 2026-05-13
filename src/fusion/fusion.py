"""Multimodal fusion modules for combining text and image embeddings.

Three strategies are implemented, from parameter-free to fully learnable:

ConcatFusion (0 parameters)
    Concatenates text and image embeddings along the feature dimension.
    The downstream GNN sees a single vector of width ``text_dim + image_dim``.
    No learned weights — pure architectural choice.

WeightedFusion (2 parameters: raw logits → softmax weights)
    Projects each modality to a shared ``output_dim``, then computes a
    **softmax-weighted sum**.  Because the weights are learned jointly with
    the GNN, the network can down-weight nodes whose image is a zero vector
    (unavailable images).  This "soft attention over modalities" also allows
    dataset-level modality preference to emerge from data.

GatedFusion (MLP, ~(text_dim+image_dim) * output_dim parameters)
    Implements Gated Multimodal Units (Arevalo et al., ICLR 2017):

        h_t = tanh(W_t · text)
        h_i = tanh(W_i · image)
        z   = sigmoid(gate_MLP([text ; image]))     ← element-wise gate
        out = z ⊙ h_t + (1 − z) ⊙ h_i

    The gate is conditioned on *both* modalities, so it learns an
    instance-level fusion weight that can suppress missing-image nodes
    more precisely than WeightedFusion's global scalars.

FusedModel wrapper
    Wraps a fusion module + backbone GNN into a single ``nn.Module`` that
    satisfies the project's ``forward(x, edge_index)`` contract.
    ``x`` is expected to be the *concatenation* of text and image embeddings
    (produced before training in the H2 runner).  FusedModel splits ``x``
    internally, applies the fusion, and passes the result to the backbone.

    This design keeps ``train_model`` (which calls
    ``model(data.x, data.edge_index)``) completely unchanged.

FusedModelFactory
    A callable used by ``hyperparameter_search`` (via its optional
    ``model_factory`` argument) to build a fresh FusedModel for each
    hyperparameter trial without exposing the construction details to
    ``src/train.py``.

build_fusion
    Factory function: ``str × dims → BaseFusion``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseFusion(nn.Module, ABC):
    """Abstract base for all fusion modules.

    All subclasses must implement :meth:`forward` and expose
    :attr:`output_dim` so downstream components (e.g. the GNN backbone)
    can determine their ``in_channels`` without inspecting the internals.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimension of the fused output tensor.

        Returns:
            Integer width of ``forward(text_x, image_x)``'s output.
        """
        ...

    @abstractmethod
    def forward(self, text_x: torch.Tensor, image_x: torch.Tensor) -> torch.Tensor:
        """Fuse text and image embeddings.

        Args:
            text_x:  Text embedding matrix of shape ``(N, text_dim)``.
            image_x: Image embedding matrix of shape ``(N, image_dim)``.
                     Rows corresponding to missing images are **zero
                     vectors** (filled by the vision encoder).

        Returns:
            Fused tensor of shape ``(N, output_dim)``.
        """
        ...


# ---------------------------------------------------------------------------
# 1. ConcatFusion
# ---------------------------------------------------------------------------


class ConcatFusion(BaseFusion):
    """Parameter-free fusion: concatenate text and image embeddings.

    The simplest possible fusion strategy — no learned weights, no
    projection.  The downstream GNN receives both modalities as a single
    wider feature vector of dimension ``text_dim + image_dim``.

    Missing-image nodes contribute a zero sub-vector in the image half of
    the concatenation.  Whether the GNN can learn to ignore these is an
    empirical question addressed in the H2 experiment.

    Args:
        text_dim: Width of the text embedding vectors.
        image_dim: Width of the image embedding vectors.
    """

    def __init__(self, text_dim: int, image_dim: int) -> None:
        super().__init__()
        self._text_dim  = text_dim
        self._image_dim = image_dim

    @property
    def output_dim(self) -> int:
        """``text_dim + image_dim``."""
        return self._text_dim + self._image_dim

    def forward(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate along the last dimension.

        Args:
            text_x:  ``(N, text_dim)``
            image_x: ``(N, image_dim)``

        Returns:
            ``(N, text_dim + image_dim)``
        """
        return torch.cat([text_x, image_x], dim=-1)

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"ConcatFusion(text_dim={self._text_dim}, "
            f"image_dim={self._image_dim}, "
            f"output_dim={self.output_dim})"
        )


# ---------------------------------------------------------------------------
# 2. WeightedFusion
# ---------------------------------------------------------------------------


class WeightedFusion(BaseFusion):
    """Learnable softmax-weighted sum of the two modalities.

    Each modality is first projected to a common ``output_dim``.
    Two raw scalar logits (``raw_weights``) are trained via
    back-propagation; their softmax gives the per-modality contribution:

        w = softmax([w_text, w_image])
        out = w[0] · proj_text(text_x)  +  w[1] · proj_image(image_x)

    Because the weights are global (shared across all nodes), the network
    learns a **dataset-level** preference between modalities.  For datasets
    with many missing images, ``w[1]`` will converge toward zero, making
    this a natural soft fall-back to text-only.

    If ``text_dim == output_dim`` the text projection is an identity (no
    parameters); similarly for the image side.

    Args:
        text_dim: Width of the text embedding vectors.
        image_dim: Width of the image embedding vectors.
        output_dim: Dimension of the fused output.  Defaults to
            ``max(text_dim, image_dim)``.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim or max(text_dim, image_dim)

        self.text_proj = (
            nn.Linear(text_dim, self._output_dim)
            if text_dim != self._output_dim
            else nn.Identity()
        )
        self.image_proj = (
            nn.Linear(image_dim, self._output_dim)
            if image_dim != self._output_dim
            else nn.Identity()
        )
        # Initialise logits at zero → softmax gives (0.5, 0.5) at start.
        self.raw_weights = nn.Parameter(torch.zeros(2))

    @property
    def output_dim(self) -> int:
        """Dimension of the weighted-sum output."""
        return self._output_dim

    def forward(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute softmax-weighted sum of projected modalities.

        Args:
            text_x:  ``(N, text_dim)``
            image_x: ``(N, image_dim)``

        Returns:
            ``(N, output_dim)``
        """
        w = torch.softmax(self.raw_weights, dim=0)   # shape (2,)
        t = self.text_proj(text_x)                   # (N, output_dim)
        i = self.image_proj(image_x)                 # (N, output_dim)
        return w[0] * t + w[1] * i                   # (N, output_dim)

    def modality_weights(self) -> tuple[float, float]:
        """Return the current text and image softmax weights (for logging).

        Returns:
            ``(w_text, w_image)`` as Python floats.
        """
        with torch.no_grad():
            w = torch.softmax(self.raw_weights, dim=0)
        return float(w[0]), float(w[1])

    def __repr__(self) -> str:  # noqa: D401
        wt, wi = self.modality_weights()
        return (
            f"WeightedFusion(output_dim={self._output_dim}, "
            f"w_text={wt:.3f}, w_image={wi:.3f})"
        )


# ---------------------------------------------------------------------------
# 3. GatedFusion
# ---------------------------------------------------------------------------


class GatedFusion(BaseFusion):
    """Instance-level gating network for multimodal fusion.

    Implements *Gated Multimodal Units* (Arevalo et al., ICLR 2017) with a
    two-layer gate MLP:

        h_t = tanh(W_t · text_x)
        h_i = tanh(W_i · image_x)
        z   = σ(MLP([text_x ; image_x]))   ∈ (0, 1)^output_dim
        out = z ⊙ h_t + (1 − z) ⊙ h_i

    The gate ``z`` is computed from the **concatenation** of both modalities,
    so it produces a *per-node, per-dimension* gating decision rather than
    a single global weight.  This is strictly more expressive than
    WeightedFusion: for a node with a missing image (zero ``image_x``), the
    gate network sees this and can set ``z ≈ 1`` (pure text) node-by-node.

    Args:
        text_dim: Width of the text embedding vectors.
        image_dim: Width of the image embedding vectors.
        output_dim: Fusion output dimension.  Defaults to
            ``max(text_dim, image_dim)``.
        gate_hidden_dim: Hidden layer width in the gate MLP.  Defaults to
            ``max(32, (text_dim + image_dim) // 4)``.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        output_dim: int | None = None,
        gate_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self._output_dim = output_dim or max(text_dim, image_dim)
        gate_hidden = gate_hidden_dim or max(32, (text_dim + image_dim) // 4)

        # Modality projection heads.
        self.text_proj  = nn.Linear(text_dim,  self._output_dim)
        self.image_proj = nn.Linear(image_dim, self._output_dim)

        # Gate network: shallow MLP on concatenated input → element-wise gate.
        self.gate_net = nn.Sequential(
            nn.Linear(text_dim + image_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, self._output_dim),
            nn.Sigmoid(),
        )

    @property
    def output_dim(self) -> int:
        """Dimension of the gated fusion output."""
        return self._output_dim

    def forward(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gated multimodal fusion.

        Args:
            text_x:  ``(N, text_dim)``
            image_x: ``(N, image_dim)``

        Returns:
            ``(N, output_dim)``
        """
        h_t = torch.tanh(self.text_proj(text_x))           # (N, output_dim)
        h_i = torch.tanh(self.image_proj(image_x))         # (N, output_dim)
        concat = torch.cat([text_x, image_x], dim=-1)      # (N, text+image)
        z = self.gate_net(concat)                           # (N, output_dim)
        return z * h_t + (1.0 - z) * h_i                   # (N, output_dim)

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"GatedFusion(output_dim={self._output_dim}, "
            f"gate_net={self.gate_net})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_fusion(
    name: str,
    text_dim: int,
    image_dim: int,
    output_dim: int | None = None,
) -> BaseFusion:
    """Instantiate a fusion module by name.

    Args:
        name: One of ``"concat"``, ``"weighted"``, ``"gated"``
            (case-insensitive).
        text_dim: Text embedding dimension.
        image_dim: Image embedding dimension.
        output_dim: Output dimension for WeightedFusion and GatedFusion.
            Ignored for ConcatFusion (output = text_dim + image_dim).
            Defaults to ``max(text_dim, image_dim)`` when ``None``.

    Returns:
        An initialised :class:`BaseFusion` subclass instance.

    Raises:
        KeyError: If ``name`` is not a known fusion strategy.
    """
    key = name.lower().strip()
    if key == "concat":
        return ConcatFusion(text_dim, image_dim)
    if key == "weighted":
        return WeightedFusion(text_dim, image_dim, output_dim)
    if key == "gated":
        return GatedFusion(text_dim, image_dim, output_dim)
    raise KeyError(
        f"Unknown fusion strategy '{name}'. "
        f"Registered: concat, weighted, gated."
    )


# ---------------------------------------------------------------------------
# FusedModel — wraps fusion + backbone into forward(x, edge_index)
# ---------------------------------------------------------------------------


class FusedModel(nn.Module):
    """Fusion module + GNN backbone in a single ``forward(x, edge_index)`` model.

    The H2 runner precomputes ``data.x = cat([text_emb, image_emb], dim=-1)``
    and passes it through the standard training loop.  FusedModel receives
    this concatenated tensor, splits it back into the two modalities, applies
    the fusion, and feeds the result to the backbone GNN.

    This design means ``train_model`` (which calls
    ``model(data.x, data.edge_index)``) requires **no changes** for the
    multimodal case.

    Args:
        fusion: An initialised :class:`BaseFusion` (can have learnable params).
        backbone: Any ``nn.Module`` with ``forward(x, edge_index)`` — typically
            a GCN / GraphSAGE / GAT with ``in_channels == fusion.output_dim``.
        text_dim: Width of the text half of the concatenated input.
        image_dim: Width of the image half of the concatenated input.
    """

    def __init__(
        self,
        fusion: BaseFusion,
        backbone: nn.Module,
        text_dim: int,
        image_dim: int,
    ) -> None:
        super().__init__()
        self.fusion   = fusion
        self.backbone = backbone
        self.text_dim  = text_dim
        self.image_dim = image_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Split → fuse → backbone forward.

        Args:
            x: ``(N, text_dim + image_dim)`` — concatenated node features.
            edge_index: ``(2, E)`` edge tensor.

        Returns:
            ``(N, out_channels)`` class logits.
        """
        text_x  = x[:, : self.text_dim]
        image_x = x[:, self.text_dim :]
        fused   = self.fusion(text_x, image_x)      # (N, fusion.output_dim)
        return self.backbone(fused, edge_index)

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"FusedModel(text_dim={self.text_dim}, image_dim={self.image_dim}, "
            f"fusion={self.fusion.__class__.__name__}, "
            f"backbone={self.backbone.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# FusedModelFactory — callable for hyperparameter_search
# ---------------------------------------------------------------------------


class FusedModelFactory:
    """Callable that builds a fresh :class:`FusedModel` for each HP trial.

    Pass an instance as the ``model_factory`` argument of
    :func:`src.train.hyperparameter_search` so the training loop can build
    multimodal models without knowing about fusion internals.

    Args:
        gnn_cls: GNN class (e.g. :class:`~src.models.gnns.GCN`).
        fusion_name: One of ``"concat"``, ``"weighted"``, ``"gated"``.
        text_dim: Text embedding dimension.
        image_dim: Image embedding dimension.
        fusion_dim: Output dimension for Weighted/Gated fusion.  Defaults
            to ``max(text_dim, image_dim)``.
        extra_gnn_kwargs: Extra kwargs forwarded to ``gnn_cls.__init__``
            (e.g. ``{"num_heads": 2}`` for GAT).
    """

    def __init__(
        self,
        gnn_cls: Type[nn.Module],
        fusion_name: str,
        text_dim: int,
        image_dim: int,
        fusion_dim: int | None = None,
        extra_gnn_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.gnn_cls         = gnn_cls
        self.fusion_name     = fusion_name
        self.text_dim        = text_dim
        self.image_dim       = image_dim
        self.fusion_dim      = fusion_dim
        self.extra_gnn_kwargs = extra_gnn_kwargs or {}

    def __call__(self, config: Any) -> FusedModel:
        """Build a :class:`FusedModel` from a :class:`~src.train.TrainConfig`.

        Args:
            config: A :class:`~src.train.TrainConfig` with at least
                ``hidden_channels``, ``out_channels``, ``dropout``,
                ``num_layers`` filled in.

        Returns:
            Freshly initialised :class:`FusedModel`.
        """
        from dataclasses import replace as dc_replace
        from src.train import _build_model_for_config

        fusion = build_fusion(
            self.fusion_name, self.text_dim, self.image_dim, self.fusion_dim
        )
        # Backbone in_channels = fusion output dim (NOT the raw concat width).
        backbone_config = dc_replace(config, in_channels=fusion.output_dim)
        backbone = _build_model_for_config(self.gnn_cls, backbone_config)
        return FusedModel(fusion, backbone, self.text_dim, self.image_dim)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    torch.manual_seed(0)
    N, T, V, C = 16, 32, 64, 4     # nodes, text_dim, image_dim, classes

    text_x  = torch.randn(N, T)
    image_x = torch.randn(N, V)

    # ----- ConcatFusion -----
    cf = ConcatFusion(T, V)
    out = cf(text_x, image_x)
    assert out.shape == (N, T + V), f"ConcatFusion shape: {out.shape}"
    assert cf.output_dim == T + V

    # ----- WeightedFusion -----
    wf = WeightedFusion(T, V, output_dim=64)
    out = wf(text_x, image_x)
    assert out.shape == (N, 64), f"WeightedFusion shape: {out.shape}"
    wt, wi = wf.modality_weights()
    assert abs(wt + wi - 1.0) < 1e-5, f"Weights don't sum to 1: {wt}+{wi}"

    # ----- GatedFusion -----
    gf = GatedFusion(T, V, output_dim=48)
    out = gf(text_x, image_x)
    assert out.shape == (N, 48), f"GatedFusion shape: {out.shape}"
    # Gate values must be in (0, 1).
    with torch.no_grad():
        cat_in = torch.cat([text_x, image_x], dim=-1)
        z = gf.gate_net(cat_in)
    assert (z >= 0).all() and (z <= 1).all(), "Gate values out of [0,1]"

    # ----- build_fusion factory -----
    for name in ("concat", "weighted", "gated"):
        f = build_fusion(name, T, V)
        assert isinstance(f, BaseFusion)
    try:
        build_fusion("unknown", T, V)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # ----- FusedModel with a toy backbone -----
    class _ToyBackbone(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.lin = nn.Linear(in_ch, out_ch)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    for name in ("concat", "weighted", "gated"):
        fusion = build_fusion(name, T, V)
        backbone = _ToyBackbone(fusion.output_dim, C)
        model = FusedModel(fusion, backbone, T, V)

        x_full = torch.cat([text_x, image_x], dim=-1)
        ei = torch.empty(2, 0, dtype=torch.long)
        logits = model(x_full, ei)
        assert logits.shape == (N, C), f"{name} FusedModel output shape: {logits.shape}"

    logger.info("All smoke tests passed.")
    print("fusion.py smoke test OK")
