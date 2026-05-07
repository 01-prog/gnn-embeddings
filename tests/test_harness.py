"""Unit tests for ``src.eval.harness``."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.eval.harness import (
    _to_predictions,
    aggregate_metrics,
    evaluate,
    run_seed_sweep,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class ConstantModel(nn.Module):
    """Returns a fixed tensor regardless of input. Useful for evaluator tests."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.logits


def _fake_data(y: torch.Tensor) -> SimpleNamespace:
    """Build a minimal stand-in for a ``torch_geometric.data.Data`` object.

    Provides ``.x``, ``.edge_index``, and ``.y`` — the three attributes
    that ``evaluate`` reads from ``data``.
    """
    n = y.shape[0]
    return SimpleNamespace(
        y=y,
        x=torch.zeros(n, 4),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_perfect_predictions() -> None:
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.eye(3)[y]
    data = _fake_data(y)
    mask = torch.ones(6, dtype=torch.bool)
    out = evaluate(ConstantModel(logits), data, mask)
    assert out["accuracy"] == pytest.approx(1.0)
    assert out["f1_macro"] == pytest.approx(1.0)
    assert out["f1_weighted"] == pytest.approx(1.0)


def test_evaluate_all_wrong() -> None:
    # Always predicts class 0; ground truth never is.
    y = torch.tensor([1, 1, 2, 2, 1, 2])
    logits = torch.zeros(6, 3)
    logits[:, 0] = 1.0
    data = _fake_data(y)
    mask = torch.ones(6, dtype=torch.bool)
    out = evaluate(ConstantModel(logits), data, mask)
    assert out["accuracy"] == pytest.approx(0.0)
    assert out["f1_macro"] == pytest.approx(0.0)
    assert out["f1_weighted"] == pytest.approx(0.0)


def test_evaluate_respects_mask() -> None:
    # All-zero predictions; only the first three nodes are class 0, so masking
    # to those should give perfect accuracy on that subset.
    y = torch.tensor([0, 0, 0, 1, 1, 2])
    logits = torch.zeros(6, 3)
    logits[:, 0] = 1.0
    data = _fake_data(y)
    mask = torch.tensor([True, True, True, False, False, False])
    out = evaluate(ConstantModel(logits), data, mask)
    assert out["accuracy"] == pytest.approx(1.0)


def test_evaluate_accepts_1d_predictions() -> None:
    # Already-reduced predictions (e.g. from a sklearn-wrapping module).
    y = torch.tensor([0, 1, 0, 1])
    preds_1d = torch.tensor([0, 1, 1, 1])
    data = _fake_data(y)
    mask = torch.ones(4, dtype=torch.bool)
    out = evaluate(ConstantModel(preds_1d), data, mask)
    assert out["accuracy"] == pytest.approx(0.75)


def test_evaluate_rejects_non_bool_mask() -> None:
    y = torch.tensor([0, 1])
    logits = torch.eye(2)[y]
    with pytest.raises(ValueError, match="BoolTensor"):
        evaluate(ConstantModel(logits), _fake_data(y), torch.tensor([1, 1]))


def test_evaluate_rejects_2d_mask() -> None:
    y = torch.tensor([0, 1])
    logits = torch.eye(2)[y]
    with pytest.raises(ValueError, match="1-D"):
        evaluate(ConstantModel(logits), _fake_data(y), torch.tensor([[True], [True]]))


def test_evaluate_restores_train_mode() -> None:
    y = torch.tensor([0, 1])
    logits = torch.eye(2)[y]
    model = ConstantModel(logits)
    model.train()
    evaluate(model, _fake_data(y), torch.ones(2, dtype=torch.bool))
    assert model.training is True


# ---------------------------------------------------------------------------
# _to_predictions()
# ---------------------------------------------------------------------------


def test_to_predictions_argmaxes_2d() -> None:
    out = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    assert torch.equal(_to_predictions(out), torch.tensor([1, 0, 1]))


def test_to_predictions_passes_through_1d() -> None:
    out = torch.tensor([2, 0, 1])
    assert torch.equal(_to_predictions(out), out.long())


def test_to_predictions_rejects_3d() -> None:
    with pytest.raises(ValueError):
        _to_predictions(torch.zeros(2, 2, 2))


# ---------------------------------------------------------------------------
# aggregate_metrics()
# ---------------------------------------------------------------------------


def test_aggregate_metrics_basic() -> None:
    runs = [
        {"accuracy": 0.80, "f1_macro": 0.78},
        {"accuracy": 0.81, "f1_macro": 0.79},
        {"accuracy": 0.82, "f1_macro": 0.80},
        {"accuracy": 0.83, "f1_macro": 0.81},
        {"accuracy": 0.84, "f1_macro": 0.82},
    ]
    s = aggregate_metrics(runs)
    assert s["n_runs"] == 5
    assert s["accuracy_mean"] == pytest.approx(0.82)
    # Sample std (ddof=1) of (0.80..0.84 step 0.01) is sqrt(0.0001 * 2.5) ~= 0.01581
    assert s["accuracy_std"] == pytest.approx(np.std(np.arange(80, 85), ddof=1) / 100, rel=1e-6)
    half_ci = 1.96 * s["accuracy_std"] / math.sqrt(5)
    assert s["accuracy_ci95_lower"] == pytest.approx(0.82 - half_ci)
    assert s["accuracy_ci95_upper"] == pytest.approx(0.82 + half_ci)


def test_aggregate_metrics_empty() -> None:
    assert aggregate_metrics([]) == {"n_runs": 0}


def test_aggregate_metrics_single_run_zero_std() -> None:
    s = aggregate_metrics([{"accuracy": 0.7}])
    assert s["accuracy_mean"] == pytest.approx(0.7)
    assert s["accuracy_std"] == pytest.approx(0.0)
    assert s["accuracy_ci95_lower"] == pytest.approx(0.7)
    assert s["accuracy_ci95_upper"] == pytest.approx(0.7)


def test_aggregate_metrics_ignores_non_numeric() -> None:
    runs = [
        {"accuracy": 0.7, "encoder": "tfidf", "ok": True},
        {"accuracy": 0.8, "encoder": "tfidf", "ok": False},
    ]
    s = aggregate_metrics(runs)
    assert "accuracy_mean" in s
    assert "encoder_mean" not in s
    # Booleans must NOT be aggregated even though bool subclasses int.
    assert "ok_mean" not in s


# ---------------------------------------------------------------------------
# run_seed_sweep()
# ---------------------------------------------------------------------------


def test_run_seed_sweep_passes_consecutive_seeds() -> None:
    seen: list[int] = []

    def train(seed: int) -> dict[str, float]:
        seen.append(seed)
        return {"accuracy": 0.5 + 0.01 * seed}

    runs, _ = run_seed_sweep(train, n_seeds=4, base_seed=10)
    assert seen == [10, 11, 12, 13]
    assert [r["seed"] for r in runs] == [10, 11, 12, 13]


def test_run_seed_sweep_summary_matches_aggregate() -> None:
    def train(seed: int) -> dict[str, float]:
        return {"accuracy": 0.80 + 0.01 * seed, "f1_macro": 0.78 + 0.01 * seed}

    runs, summary = run_seed_sweep(train, n_seeds=5, base_seed=0)
    expected = aggregate_metrics(runs)
    for k, v in expected.items():
        assert summary[k] == pytest.approx(v)


def test_run_seed_sweep_rejects_zero() -> None:
    with pytest.raises(ValueError):
        run_seed_sweep(lambda s: {"accuracy": 0.5}, n_seeds=0)


def test_run_seed_sweep_does_not_overwrite_user_seed() -> None:
    # If train_fn already set a "seed" key, run_seed_sweep should keep it.
    def train(seed: int) -> dict[str, float | int]:
        return {"accuracy": 0.5, "seed": seed * 100}

    runs, _ = run_seed_sweep(train, n_seeds=3, base_seed=0)
    assert [r["seed"] for r in runs] == [0, 100, 200]
