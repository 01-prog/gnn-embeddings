"""pytest tests for multimodal fusion modules (src/fusion/fusion.py).

Coverage
--------
- ConcatFusion  : output shape, value correctness, parameter count.
- WeightedFusion: output shape, weight normalisation, trainability.
- GatedFusion   : output shape, gate-value bounds, gradient flow.
- build_fusion  : factory dispatch and KeyError for unknown names.
- FusedModel    : split-fuse-backbone contract, all fusion types.
- FusedModelFactory: builds FusedModel with correct in_channels.

All tests are CPU-only and require only torch (no PyG, no GPU).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.fusion.fusion import (
    BaseFusion,
    ConcatFusion,
    FusedModel,
    FusedModelFactory,
    GatedFusion,
    WeightedFusion,
    build_fusion,
)

# ── Shared dimensions ────────────────────────────────────────────────────────
TEXT_DIM  = 32
IMAGE_DIM = 64
OUT_DIM   = 48
N         = 16     # number of nodes in every test batch


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def text_x() -> torch.Tensor:
    """Random text embedding matrix, shape (N, TEXT_DIM)."""
    torch.manual_seed(0)
    return torch.randn(N, TEXT_DIM)


@pytest.fixture
def image_x() -> torch.Tensor:
    """Random image embedding matrix, shape (N, IMAGE_DIM)."""
    torch.manual_seed(1)
    return torch.randn(N, IMAGE_DIM)


@pytest.fixture
def zero_image_x() -> torch.Tensor:
    """All-zero image matrix (simulates fully-missing images)."""
    return torch.zeros(N, IMAGE_DIM)


@pytest.fixture
def empty_edge_index() -> torch.Tensor:
    """Empty edge_index tensor (shape (2, 0)) for graph-free tests."""
    return torch.empty(2, 0, dtype=torch.long)


# ---------------------------------------------------------------------------
# ConcatFusion
# ---------------------------------------------------------------------------

class TestConcatFusion:
    """Tests for the parameter-free concatenation fusion strategy."""

    def test_output_dim_property(self) -> None:
        """output_dim == text_dim + image_dim."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        assert f.output_dim == TEXT_DIM + IMAGE_DIM

    def test_forward_output_shape(self, text_x: torch.Tensor, image_x: torch.Tensor) -> None:
        """Forward pass returns (N, text_dim + image_dim) tensor."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        out = f(text_x, image_x)
        assert out.shape == (N, TEXT_DIM + IMAGE_DIM)

    def test_forward_values_are_exact_concatenation(
        self, text_x: torch.Tensor, image_x: torch.Tensor
    ) -> None:
        """The first text_dim columns must be text_x; the rest must be image_x."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        out = f(text_x, image_x)
        assert torch.allclose(out[:, :TEXT_DIM], text_x)
        assert torch.allclose(out[:, TEXT_DIM:], image_x)

    def test_no_learnable_parameters(self) -> None:
        """ConcatFusion is parameter-free."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        assert sum(p.numel() for p in f.parameters()) == 0

    def test_repr_contains_class_name(self) -> None:
        """repr() mentions class name and dimensions."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        r = repr(f)
        assert "ConcatFusion" in r
        assert str(TEXT_DIM) in r
        assert str(IMAGE_DIM) in r

    def test_batch_size_one(self) -> None:
        """Works correctly on a single-node 'batch'."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        out = f(torch.randn(1, TEXT_DIM), torch.randn(1, IMAGE_DIM))
        assert out.shape == (1, TEXT_DIM + IMAGE_DIM)

    def test_is_base_fusion_subclass(self) -> None:
        """ConcatFusion must implement the BaseFusion interface."""
        f = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        assert isinstance(f, BaseFusion)
        assert isinstance(f, nn.Module)


# ---------------------------------------------------------------------------
# WeightedFusion
# ---------------------------------------------------------------------------

class TestWeightedFusion:
    """Tests for the softmax-weighted-sum fusion strategy."""

    def test_default_output_dim(self) -> None:
        """Default output_dim = max(text_dim, image_dim)."""
        f = WeightedFusion(TEXT_DIM, IMAGE_DIM)
        assert f.output_dim == max(TEXT_DIM, IMAGE_DIM)

    def test_custom_output_dim(self) -> None:
        """Custom output_dim is respected."""
        f = WeightedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        assert f.output_dim == OUT_DIM

    def test_forward_output_shape(self, text_x: torch.Tensor, image_x: torch.Tensor) -> None:
        """Forward returns (N, output_dim) tensor."""
        f = WeightedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        out = f(text_x, image_x)
        assert out.shape == (N, OUT_DIM)

    def test_initial_weights_are_equal(self) -> None:
        """raw_weights initialised to zeros → softmax → (0.5, 0.5)."""
        f = WeightedFusion(TEXT_DIM, IMAGE_DIM)
        wt, wi = f.modality_weights()
        assert abs(wt - 0.5) < 1e-5, f"Expected w_text=0.5, got {wt}"
        assert abs(wi - 0.5) < 1e-5, f"Expected w_image=0.5, got {wi}"

    def test_weights_always_sum_to_one(self) -> None:
        """softmax(raw_weights) sums to 1.0 regardless of parameter values."""
        f = WeightedFusion(TEXT_DIM, IMAGE_DIM)
        with torch.no_grad():
            f.raw_weights.data = torch.tensor([2.0, -1.0])
        wt, wi = f.modality_weights()
        assert abs(wt + wi - 1.0) < 1e-5, f"w_text + w_image = {wt + wi}"

    def test_weights_are_trainable_parameters(self) -> None:
        """raw_weights must be an nn.Parameter with requires_grad=True."""
        f = WeightedFusion(TEXT_DIM, IMAGE_DIM)
        assert isinstance(f.raw_weights, nn.Parameter)
        assert f.raw_weights.requires_grad

    def test_identity_projection_when_dims_match(self) -> None:
        """When text_dim == output_dim, text_proj should be nn.Identity."""
        common = IMAGE_DIM   # IMAGE_DIM > TEXT_DIM, use it as the common dim
        f = WeightedFusion(common, common, output_dim=common)
        assert isinstance(f.text_proj,  nn.Identity)
        assert isinstance(f.image_proj, nn.Identity)


# ---------------------------------------------------------------------------
# GatedFusion
# ---------------------------------------------------------------------------

class TestGatedFusion:
    """Tests for the Gated Multimodal Units (GMU) fusion strategy."""

    def test_default_output_dim(self) -> None:
        """Default output_dim = max(text_dim, image_dim)."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM)
        assert f.output_dim == max(TEXT_DIM, IMAGE_DIM)

    def test_custom_output_dim(self) -> None:
        """Custom output_dim is respected."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        assert f.output_dim == OUT_DIM

    def test_forward_output_shape(self, text_x: torch.Tensor, image_x: torch.Tensor) -> None:
        """Forward returns (N, output_dim) tensor."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        out = f(text_x, image_x)
        assert out.shape == (N, OUT_DIM)

    def test_gate_values_are_in_unit_interval(
        self, text_x: torch.Tensor, image_x: torch.Tensor
    ) -> None:
        """Gate network uses Sigmoid, so all gate values must be in [0, 1]."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        with torch.no_grad():
            cat_in = torch.cat([text_x, image_x], dim=-1)
            z = f.gate_net(cat_in)
        assert (z >= 0).all(), "Gate value below 0"
        assert (z <= 1).all(), "Gate value above 1"

    def test_has_learnable_parameters(self) -> None:
        """GatedFusion must have non-zero learnable parameter count."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        total = sum(p.numel() for p in f.parameters())
        assert total > 0

    def test_zero_image_does_not_produce_nan(
        self, text_x: torch.Tensor, zero_image_x: torch.Tensor
    ) -> None:
        """All-zero image input (missing images) must not produce NaN output."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        out = f(text_x, zero_image_x)
        assert not torch.isnan(out).any(), "NaN detected for zero image input"

    def test_gradient_flows_through_gate(
        self, text_x: torch.Tensor, image_x: torch.Tensor
    ) -> None:
        """Loss.backward() must succeed (gradient flows through gate + projections)."""
        f = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        t = text_x.clone().requires_grad_(True)
        i = image_x.clone().requires_grad_(True)
        out = f(t, i)
        out.sum().backward()
        assert t.grad is not None
        assert i.grad is not None


# ---------------------------------------------------------------------------
# build_fusion factory
# ---------------------------------------------------------------------------

class TestBuildFusion:
    """Tests for the build_fusion dispatcher."""

    def test_concat_returns_concat_fusion(self) -> None:
        assert isinstance(build_fusion("concat", TEXT_DIM, IMAGE_DIM), ConcatFusion)

    def test_weighted_returns_weighted_fusion(self) -> None:
        assert isinstance(build_fusion("weighted", TEXT_DIM, IMAGE_DIM), WeightedFusion)

    def test_gated_returns_gated_fusion(self) -> None:
        assert isinstance(build_fusion("gated", TEXT_DIM, IMAGE_DIM), GatedFusion)

    def test_case_insensitive_dispatch(self) -> None:
        """Names should be matched case-insensitively."""
        assert isinstance(build_fusion("Concat",  TEXT_DIM, IMAGE_DIM), ConcatFusion)
        assert isinstance(build_fusion("WEIGHTED", TEXT_DIM, IMAGE_DIM), WeightedFusion)
        assert isinstance(build_fusion("Gated",   TEXT_DIM, IMAGE_DIM), GatedFusion)

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown fusion"):
            build_fusion("unknown", TEXT_DIM, IMAGE_DIM)

    def test_output_dim_forwarded_to_weighted(self) -> None:
        f = build_fusion("weighted", TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        assert f.output_dim == OUT_DIM

    def test_output_dim_forwarded_to_gated(self) -> None:
        f = build_fusion("gated", TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        assert f.output_dim == OUT_DIM

    def test_output_dim_ignored_for_concat(self) -> None:
        """ConcatFusion output_dim is always text_dim + image_dim."""
        f = build_fusion("concat", TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        assert f.output_dim == TEXT_DIM + IMAGE_DIM


# ---------------------------------------------------------------------------
# FusedModel
# ---------------------------------------------------------------------------

class _RecordingBackbone(nn.Module):
    """Minimal backbone that records the fused tensor it receives.

    Args:
        in_ch: Expected input (fused) feature dimension.
        out_ch: Number of output classes.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
        self.last_x: torch.Tensor | None = None
        self.in_ch = in_ch

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Record ``x`` and apply a linear classifier."""
        self.last_x = x
        return self.lin(x)


class TestFusedModel:
    """Tests for the FusedModel wrapper."""

    @pytest.fixture(autouse=True)
    def _build(
        self,
        text_x: torch.Tensor,
        image_x: torch.Tensor,
        empty_edge_index: torch.Tensor,
    ) -> None:
        """Set up a default ConcatFusion + RecordingBackbone FusedModel."""
        fusion   = ConcatFusion(TEXT_DIM, IMAGE_DIM)
        backbone = _RecordingBackbone(fusion.output_dim, 4)
        self.model      = FusedModel(fusion, backbone, TEXT_DIM, IMAGE_DIM)
        self.text_x     = text_x
        self.image_x    = image_x
        self.x_full     = torch.cat([text_x, image_x], dim=-1)
        self.edge_index = empty_edge_index

    def test_forward_output_shape(self) -> None:
        """FusedModel.forward() returns (N, out_channels) logits."""
        out = self.model(self.x_full, self.edge_index)
        assert out.shape == (N, 4)

    def test_backbone_receives_fused_tensor(self) -> None:
        """Backbone's last_x should have width == fusion.output_dim."""
        self.model(self.x_full, self.edge_index)
        assert self.model.backbone.last_x is not None
        assert self.model.backbone.last_x.shape[-1] == self.model.fusion.output_dim

    def test_text_image_split_is_correct(self) -> None:
        """FusedModel must use x[:, :text_dim] as text and x[:, text_dim:] as image."""
        # For ConcatFusion, fused = cat([text_x, image_x]) = x_full.
        # So backbone.last_x[:, :TEXT_DIM] should equal text_x.
        self.model(self.x_full, self.edge_index)
        bx = self.model.backbone.last_x
        assert bx is not None
        assert torch.allclose(bx[:, :TEXT_DIM], self.text_x)
        assert torch.allclose(bx[:, TEXT_DIM:], self.image_x)

    def test_repr_is_informative(self) -> None:
        """repr() contains class name, fusion type, and backbone type."""
        r = repr(self.model)
        assert "FusedModel" in r
        assert "ConcatFusion" in r

    def test_all_fusion_types_produce_valid_output(
        self, text_x: torch.Tensor, image_x: torch.Tensor
    ) -> None:
        """Every fusion strategy passes a forward call without errors."""
        x_full = torch.cat([text_x, image_x], dim=-1)
        ei     = torch.empty(2, 0, dtype=torch.long)
        for name in ("concat", "weighted", "gated"):
            fusion   = build_fusion(name, TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
            backbone = _RecordingBackbone(fusion.output_dim, 4)
            model    = FusedModel(fusion, backbone, TEXT_DIM, IMAGE_DIM)
            out      = model(x_full, ei)
            assert out.shape == (N, 4), f"Wrong output shape for fusion='{name}'"

    def test_gradient_flows_end_to_end(self) -> None:
        """Backward through GatedFusion + linear backbone must succeed."""
        fusion   = GatedFusion(TEXT_DIM, IMAGE_DIM, output_dim=OUT_DIM)
        backbone = _RecordingBackbone(OUT_DIM, 4)
        model    = FusedModel(fusion, backbone, TEXT_DIM, IMAGE_DIM)

        x = self.x_full.clone().detach().requires_grad_(True)
        out  = model(x, self.edge_index)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed back to input"


# ---------------------------------------------------------------------------
# FusedModelFactory
# ---------------------------------------------------------------------------

class TestFusedModelFactory:
    """Tests for FusedModelFactory — the callable injected into hyperparameter_search."""

    def _make_cfg(self) -> "TrainConfig":
        """Return a minimal TrainConfig for factory tests (no PyG dependency)."""
        from src.train import TrainConfig
        return TrainConfig(
            in_channels=TEXT_DIM + IMAGE_DIM,   # will be overridden by factory
            hidden_channels=32,
            out_channels=4,
            dropout=0.0,
            num_layers=2,
        )

    def test_returns_fused_model_instance(self) -> None:
        """__call__ returns a FusedModel, not a raw GNN."""
        from src.models.baselines import MLPClassifier
        factory = FusedModelFactory(
            gnn_cls=MLPClassifier,
            fusion_name="concat",
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
        )
        model = factory(self._make_cfg())
        assert isinstance(model, FusedModel)

    def test_concat_backbone_in_channels(self) -> None:
        """ConcatFusion: backbone in_channels = TEXT_DIM + IMAGE_DIM."""
        from src.models.baselines import MLPClassifier
        factory = FusedModelFactory(
            gnn_cls=MLPClassifier,
            fusion_name="concat",
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
        )
        model = factory(self._make_cfg())
        assert model.fusion.output_dim == TEXT_DIM + IMAGE_DIM

    def test_weighted_gated_backbone_in_channels(self) -> None:
        """Weighted/Gated: backbone in_channels = fusion_dim (not raw concat)."""
        from src.models.baselines import MLPClassifier
        for fusion_name in ("weighted", "gated"):
            factory = FusedModelFactory(
                gnn_cls=MLPClassifier,
                fusion_name=fusion_name,
                text_dim=TEXT_DIM,
                image_dim=IMAGE_DIM,
                fusion_dim=OUT_DIM,
            )
            model = factory(self._make_cfg())
            assert model.fusion.output_dim == OUT_DIM, (
                f"fusion_name={fusion_name}: expected output_dim={OUT_DIM}, "
                f"got {model.fusion.output_dim}"
            )

    def test_forward_works_after_factory_build(self) -> None:
        """Model built by factory must accept the correct input and run forward."""
        from src.models.baselines import MLPClassifier
        factory = FusedModelFactory(
            gnn_cls=MLPClassifier,
            fusion_name="weighted",
            text_dim=TEXT_DIM,
            image_dim=IMAGE_DIM,
            fusion_dim=OUT_DIM,
        )
        model = factory(self._make_cfg())
        x  = torch.randn(N, TEXT_DIM + IMAGE_DIM)
        ei = torch.empty(2, 0, dtype=torch.long)
        out = model(x, ei)
        assert out.shape == (N, 4)

    def test_all_fusion_strategies_produce_valid_models(self) -> None:
        """Factory must succeed for all three fusion strategies."""
        from src.models.baselines import MLPClassifier
        x  = torch.randn(N, TEXT_DIM + IMAGE_DIM)
        ei = torch.empty(2, 0, dtype=torch.long)
        for name in ("concat", "weighted", "gated"):
            factory = FusedModelFactory(
                gnn_cls=MLPClassifier,
                fusion_name=name,
                text_dim=TEXT_DIM,
                image_dim=IMAGE_DIM,
            )
            model = factory(self._make_cfg())
            out = model(x, ei)
            assert out.shape == (N, 4), f"Forward failed for fusion='{name}'"
