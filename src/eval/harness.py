"""Shared evaluation harness used across all encoder/model pairings.

Two responsibilities:

- ``evaluate(model, data, mask)``: compute accuracy and macro/weighted F1
  on a single fold (train/val/test) for either a GNN or a flat classifier.
- ``run_seed_sweep(train_fn, n_seeds, base_seed)``: rerun a training
  function ``n_seeds`` times with consecutive seeds and aggregate the
  resulting metrics into a mean / std / 95 % CI summary.

Why a single shared harness? The supervisor's experiment design rules
require every (encoder, model) pair to be evaluated identically. By
funneling every run through these two entry points we guarantee that
splits, metrics, and statistical reporting are consistent across the
benchmark table.

Model contract
--------------
``evaluate`` calls the model in ``eval`` mode under ``torch.no_grad()``.
The model must be callable on the PyG ``Data`` object directly:

    out = model(data)        # shape (N, C) logits, or (N,) class ids

GNN modules naturally take a ``Data`` (or unpack ``data.x`` /
``data.edge_index`` internally). Flat classifiers (MLP / LogReg) can be
wrapped in a thin ``nn.Module`` that ignores ``data.edge_index`` and
forwards ``data.x``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------


def _to_predictions(out: torch.Tensor) -> torch.Tensor:
    """Reduce a model output to a 1-D ``LongTensor`` of class indices.

    Accepts either ``(N, C)`` logits/probabilities (argmax over the last
    axis) or already-reduced ``(N,)`` predictions.

    Args:
        out: Model output tensor.

    Returns:
        ``LongTensor`` of shape ``(N,)``.

    Raises:
        ValueError: If ``out`` is not 1-D or 2-D.
    """
    if out.dim() == 2:
        return out.argmax(dim=-1)
    if out.dim() == 1:
        return out.long()
    raise ValueError(f"Unexpected model output shape: {tuple(out.shape)}")


def evaluate(
    model: nn.Module,
    data: Any,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute accuracy and macro / weighted F1 on a fold.

    Args:
        model: A ``torch.nn.Module`` with signature
            ``forward(x, edge_index) -> Tensor`` returning either
            ``(N, C)`` logits or ``(N,)`` class ids.  The harness calls
            ``model(data.x, data.edge_index)`` so every model — GNN or
            flat classifier — must accept both tensors (baselines may
            simply ignore ``edge_index``).  Will be set to ``eval()``
            mode for the call.
        data: Any object with attributes ``x``, ``edge_index``, and
            ``y`` of shapes ``(N, D)``, ``(2, E)``, and ``(N,)``
            respectively.  A PyG ``Data`` object satisfies this.
        mask: Boolean ``(N,)`` mask selecting the rows to score
            (typically ``data.train_mask`` / ``val_mask`` / ``test_mask``).

    Returns:
        Dict with keys ``"accuracy"``, ``"f1_macro"``, ``"f1_weighted"``
        as Python floats.

    Raises:
        ValueError: If ``mask`` is not boolean or has wrong shape.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be a BoolTensor, got dtype={mask.dtype}")
    if mask.dim() != 1:
        raise ValueError(f"mask must be 1-D, got shape {tuple(mask.shape)}")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        preds = _to_predictions(out)
    finally:
        if was_training:
            model.train()

    y_true = data.y[mask].detach().cpu().numpy()
    y_pred = preds[mask].detach().cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


# ---------------------------------------------------------------------------
# Seed sweep
# ---------------------------------------------------------------------------


# A "run record" returned by ``train_fn`` is any flat mapping. Numeric
# values are aggregated; everything else is kept verbatim in the per-seed
# list but ignored by the summary.
RunRecord = Mapping[str, Any]


def aggregate_metrics(runs: Sequence[RunRecord]) -> dict[str, float | int]:
    """Aggregate a list of per-seed run dicts.

    For every numeric key present in at least one run we compute mean,
    sample standard deviation (ddof=1), and a 95 % normal CI as
    ``mean ± 1.96 * std / sqrt(n)``.

    Args:
        runs: List of run records produced by ``train_fn(seed)``.

    Returns:
        Summary dict with ``n_runs`` plus, for every numeric key ``k``:
        ``f"{k}_mean"``, ``f"{k}_std"``, ``f"{k}_ci95_lower"``,
        ``f"{k}_ci95_upper"``.
    """
    summary: dict[str, float | int] = {"n_runs": len(runs)}
    if not runs:
        return summary

    # ``seed`` is metadata, not a metric — skip it.
    skip = {"seed"}

    numeric_keys: set[str] = set()
    for r in runs:
        for k, v in r.items():
            if k in skip:
                continue
            if isinstance(v, bool):
                # ``isinstance(True, int)`` is True in Python — exclude.
                continue
            if isinstance(v, (int, float)):
                numeric_keys.add(k)

    for k in sorted(numeric_keys):
        vals = np.array(
            [r[k] for r in runs if k in r and isinstance(r[k], (int, float))],
            dtype=float,
        )
        n = len(vals)
        if n == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        ci_half = float(1.96 * std / np.sqrt(n)) if n > 0 else 0.0
        summary[f"{k}_mean"] = mean
        summary[f"{k}_std"] = std
        summary[f"{k}_ci95_lower"] = float(mean - ci_half)
        summary[f"{k}_ci95_upper"] = float(mean + ci_half)
    return summary


def run_seed_sweep(
    train_fn: Callable[[int], RunRecord],
    n_seeds: int,
    base_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Run ``train_fn`` ``n_seeds`` times and aggregate the results.

    Seeds passed to ``train_fn`` are ``base_seed, base_seed + 1, ...``,
    so two sweeps with the same ``(n_seeds, base_seed)`` are bit-for-bit
    reproducible (assuming ``train_fn`` itself is).

    Args:
        train_fn: Callable taking an integer seed and returning a flat
            mapping with at least one numeric metric (e.g.
            ``{"accuracy": 0.81, "f1_macro": 0.79, "runtime_s": 12.3}``).
            The returned mapping is augmented with ``"seed"`` before
            being added to the per-seed list.
        n_seeds: Number of repetitions; must be ``>= 1``. The supervisor
            requires ``>= 5`` for a publishable result.
        base_seed: First seed value. Defaults to ``0``.

    Returns:
        Tuple ``(per_seed, summary)`` where ``per_seed`` is a list of
        dicts (one per run, with ``seed`` added) and ``summary`` is the
        output of :func:`aggregate_metrics`.

    Raises:
        ValueError: If ``n_seeds < 1``.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds}")

    per_seed: list[dict[str, Any]] = []
    for i in range(n_seeds):
        seed = base_seed + i
        record = dict(train_fn(seed))
        record.setdefault("seed", seed)
        per_seed.append(record)
        logger.info("seed=%d -> %s", seed, {
            k: round(v, 4) for k, v in record.items()
            if isinstance(v, float)
        })

    summary = aggregate_metrics(per_seed)
    return per_seed, summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    # Toy "model" that returns hand-crafted logits and toy data.
    class _ConstModel(nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.logits = logits

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return self.logits

    from types import SimpleNamespace

    y = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = torch.eye(3)[y]  # perfect predictions
    x_dummy = torch.zeros(6, 2)
    ei_dummy = torch.empty(2, 0, dtype=torch.long)
    data = SimpleNamespace(y=y, x=x_dummy, edge_index=ei_dummy)
    full_mask = torch.ones(6, dtype=torch.bool)
    out = evaluate(_ConstModel(logits), data, full_mask)
    assert out["accuracy"] == 1.0
    assert out["f1_macro"] == 1.0

    # Sweep with a fake train_fn linear in seed.
    def fake_train(seed: int) -> dict[str, float]:
        return {"accuracy": 0.80 + 0.01 * seed, "f1_macro": 0.78 + 0.01 * seed}

    runs, summary = run_seed_sweep(fake_train, n_seeds=5, base_seed=0)
    assert len(runs) == 5
    assert summary["n_runs"] == 5
    assert abs(summary["accuracy_mean"] - 0.82) < 1e-9
    assert summary["accuracy_ci95_lower"] < summary["accuracy_mean"] < summary["accuracy_ci95_upper"]
    print("harness.py smoke test OK -- summary:", summary)
