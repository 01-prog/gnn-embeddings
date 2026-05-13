"""Unified training loop and hyperparameter search for all models.

This module is the central engine that the experiment scripts call.  It
handles three concerns:

1. **PyTorch training** (``_train_pytorch``): Adam optimiser, cosine LR
   schedule with linear warmup, early stopping on validation accuracy,
   and checkpoint saving.  Used for GCN, GraphSAGE, GAT, and MLP.

2. **Sklearn training** (``_train_logreg``): single ``fit`` call on the
   training split; no LR schedule.  Used for ``LogRegClassifier``.

3. **Hyperparameter search** (``hyperparameter_search``): random search
   over a user-supplied grid.  Each trial logs a JSONL record to
   ``results/hparam_search_{encoder}_{model}.jsonl`` so the full search
   trace is reproducible.

The public entry point ``train_model`` dispatches between the two training
paths based on whether the model is an instance of ``LogRegClassifier``.

TrainConfig
-----------
All hyperparameters are captured in a ``TrainConfig`` dataclass so
callers never pass loose keyword arguments that could be silently ignored
or mismatched.  The dataclass is JSON-serialisable (all fields are plain
Python scalars) so it can be embedded directly in result files.

Transductive training convention
---------------------------------
All GNN models operate on the full graph during the forward pass and
compute the classification loss **only on the training-mask nodes**.
This is the standard transductive node-classification setup used in the
Cora/CiteSeer/Amazon benchmarks from the PyG literature.
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.eval.harness import evaluate
from src.models.baselines import LogRegClassifier
from src.utils.device import resolve_device
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All hyperparameters and metadata for a single training run.

    Args:
        in_channels: Input feature dimension (set by caller after embedding).
        hidden_channels: Hidden layer width.
        out_channels: Number of output classes (set by caller from data).
        dropout: Dropout probability.
        num_layers: Number of message-passing / hidden layers.
        lr: Peak learning rate for Adam.
        weight_decay: L2 weight decay coefficient.
        max_epochs: Hard upper bound on training epochs.
        patience: Early-stopping patience (epochs without val improvement).
        warmup_epochs: Number of linear warmup epochs before cosine decay.
        min_lr: Minimum LR reached at the end of cosine schedule.
        seed: Random seed for this run.
        encoder_name: Name of the encoder that produced ``data.x``.
        model_name: Name of the model class (used for checkpoint naming).
        checkpoint_dir: Directory where ``*.pt`` checkpoints are written.
        device: ``"auto"`` (GPU if available) or ``"cpu"`` / ``"cuda"``.
        num_heads: GAT-only: number of attention heads.
        C: LogReg-only: inverse regularisation strength.
        logreg_max_iter: LogReg-only: max sklearn solver iterations.
    """

    # Architecture
    in_channels: int = 0
    hidden_channels: int = 128
    out_channels: int = 0
    dropout: float = 0.3
    num_layers: int = 2

    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    warmup_epochs: int = 10
    min_lr: float = 1e-5

    # Identity
    seed: int = 0
    encoder_name: str = "unknown"
    model_name: str = "unknown"
    checkpoint_dir: str = "results/checkpoints"

    # Hardware
    device: str = "auto"

    # Model-specific extras
    num_heads: int = 2        # GAT
    C: float = 1.0            # LogReg
    logreg_max_iter: int = 1000  # LogReg

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        """Construct from a (possibly partial) dict; unknown keys are ignored.

        Args:
            d: Dictionary of config values.

        Returns:
            A ``TrainConfig`` with fields populated from ``d`` and
            defaults elsewhere.
        """
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    """Resolve the ``device`` string from ``TrainConfig``.

    Delegates to :func:`src.utils.device.resolve_device`.

    Args:
        device_str: ``"auto"``, ``"cpu"``, ``"cuda"``, etc.

    Returns:
        A ``torch.device``.
    """
    return resolve_device(device_str)


def _move_data(data: Any, device: torch.device) -> Any:
    """Move tensor attributes of ``data`` to ``device``.

    Prefers ``data.to(device)`` if the object supports it (PyG ``Data``),
    then falls back to attribute-by-attribute movement for SimpleNamespace
    and similar stand-ins.

    Args:
        data: A PyG ``Data`` object or duck-typed equivalent with tensor
            attributes ``x``, ``edge_index``, ``y``, ``*_mask``.
        device: Target device.

    Returns:
        The same object with all tensor attributes moved.
    """
    if hasattr(data, "to") and callable(data.to):
        return data.to(device)
    for attr in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask"):
        val = getattr(data, attr, None)
        if isinstance(val, torch.Tensor):
            setattr(data, attr, val.to(device))
    return data


# ---------------------------------------------------------------------------
# PyTorch training path
# ---------------------------------------------------------------------------


def _train_pytorch(
    model: nn.Module,
    data: Any,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train a PyTorch model with Adam + warmup-cosine schedule + early stop.

    Args:
        model: An ``nn.Module`` with ``forward(x, edge_index)`` signature.
        data: Graph data with ``x``, ``edge_index``, ``y``, and split
            masks.
        config: Full ``TrainConfig`` for this run.

    Returns:
        Result dict with keys: ``encoder_name``, ``model_name``, ``seed``,
        ``best_epoch``, ``val_metric``, ``test_accuracy``,
        ``test_f1_macro``, ``test_f1_weighted``, ``runtime_seconds``,
        ``hyperparams``.
    """
    set_seed(config.seed)
    device = _resolve_device(config.device)
    model = model.to(device)
    data = _move_data(data, device)

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (
        ckpt_dir / f"{config.encoder_name}_{config.model_name}_seed{config.seed}.pt"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Warmup + cosine schedule.
    # LinearLR: LR grows from lr * 1e-3 → lr over warmup_epochs steps.
    # CosineAnnealingLR: LR decays from lr → min_lr over the remaining epochs.
    n_cosine = max(1, config.max_epochs - config.warmup_epochs)
    warmup = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=config.warmup_epochs
    )
    cosine = CosineAnnealingLR(optimizer, T_max=n_cosine, eta_min=config.min_lr)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[config.warmup_epochs]
    )

    best_val_acc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    t0 = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        # ----- train step -----
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ----- validation -----
        val_metrics = evaluate(model, data, data.val_mask)
        val_acc = val_metrics["accuracy"]

        current_lr = scheduler.get_last_lr()[0]
        logger.debug(
            json.dumps({
                "epoch": epoch,
                "train_loss": round(float(loss), 5),
                "val_acc": round(val_acc, 5),
                "lr": round(current_lr, 8),
            })
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info(
                "epoch %d | val_acc=%.4f (best) | loss=%.4f",
                epoch, val_acc, float(loss),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                logger.info(
                    "Early stop at epoch %d (no improvement for %d epochs)",
                    epoch, config.patience,
                )
                break

    runtime = time.perf_counter() - t0

    # Load best checkpoint and evaluate on test set.
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    test_metrics = evaluate(model, data, data.test_mask)

    return {
        "encoder_name": config.encoder_name,
        "model_name": config.model_name,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "val_metric": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "runtime_seconds": round(runtime, 2),
        "hyperparams": {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "hidden_channels": config.hidden_channels,
            "dropout": config.dropout,
            "num_layers": config.num_layers,
        },
    }


# ---------------------------------------------------------------------------
# Sklearn (LogReg) training path
# ---------------------------------------------------------------------------


def _train_logreg(
    model: LogRegClassifier,
    data: Any,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train a ``LogRegClassifier`` via sklearn's ``fit`` on the CPU.

    Args:
        model: An initialised (but unfitted) ``LogRegClassifier``.
        data: Graph data object (only ``x``, ``y``, and split masks used).
        config: ``TrainConfig`` for identity / metadata fields.

    Returns:
        Same result dict schema as ``_train_pytorch``.
    """
    t0 = time.perf_counter()

    # sklearn must run on CPU.
    x_cpu = data.x.detach().cpu()
    y_cpu = data.y.detach().cpu()

    x_train = x_cpu[data.train_mask.cpu()]
    y_train = y_cpu[data.train_mask.cpu()]

    logger.info(
        "Fitting LogReg (C=%.4g) on %d training nodes …", config.C, int(y_train.shape[0])
    )
    model.fit(x_train, y_train)

    # Evaluate on CPU data (forward() always moves to CPU inside).
    data_cpu = _move_data(deepcopy(data), torch.device("cpu"))
    val_metrics = evaluate(model, data_cpu, data_cpu.val_mask.cpu())
    test_metrics = evaluate(model, data_cpu, data_cpu.test_mask.cpu())
    runtime = time.perf_counter() - t0

    return {
        "encoder_name": config.encoder_name,
        "model_name": config.model_name,
        "seed": config.seed,
        "best_epoch": 1,
        "val_metric": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "runtime_seconds": round(runtime, 2),
        "hyperparams": {"C": config.C},
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    data: Any,
    config: TrainConfig,
) -> dict[str, Any]:
    """Train a model and return test metrics at the best validation epoch.

    Dispatches to ``_train_pytorch`` or ``_train_logreg`` based on the
    model's type.

    Args:
        model: A ``BaseGNN``, ``MLPClassifier``, or ``LogRegClassifier``.
        data: PyG ``Data`` (or duck-typed equivalent) with ``x``,
            ``edge_index``, ``y``, and ``*_mask`` attributes.
        config: All hyperparameters and metadata for this run.

    Returns:
        Dict with keys ``encoder_name``, ``model_name``, ``seed``,
        ``best_epoch``, ``val_metric``, ``test_accuracy``,
        ``test_f1_macro``, ``test_f1_weighted``, ``runtime_seconds``,
        ``hyperparams``.
    """
    logger.info(
        "train_model: encoder=%s model=%s seed=%d device=%s",
        config.encoder_name, config.model_name, config.seed, config.device,
    )
    if isinstance(model, LogRegClassifier):
        return _train_logreg(model, data, config)
    return _train_pytorch(model, data, config)


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------


def _sample_params(
    search_space: dict[str, list[Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Draw one random configuration from ``search_space``.

    Args:
        search_space: Maps each hyperparameter name to a list of
            candidate values.
        rng: NumPy random generator for reproducibility.

    Returns:
        Dict mapping each name to a sampled value.
    """
    return {k: rng.choice(v).item() for k, v in search_space.items()}


def _build_model_for_config(
    model_cls: Type[nn.Module],
    config: TrainConfig,
) -> nn.Module:
    """Instantiate the correct model from a ``TrainConfig``.

    Args:
        model_cls: One of ``GCN``, ``GraphSAGE``, ``GAT``,
            ``MLPClassifier``, or ``LogRegClassifier``.
        config: Provides all constructor arguments.

    Returns:
        An initialised (untrained) model.
    """
    if model_cls is LogRegClassifier:
        return LogRegClassifier(
            config.in_channels, config.out_channels,
            C=config.C, max_iter=config.logreg_max_iter,
        )

    # All GNNs and MLP share the same constructor signature.
    kwargs: dict[str, Any] = {}
    # GAT needs num_heads; pass it only if the class accepts it.
    import inspect
    sig = inspect.signature(model_cls.__init__)
    if "num_heads" in sig.parameters:
        kwargs["num_heads"] = config.num_heads

    return model_cls(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=config.out_channels,
        dropout=config.dropout,
        num_layers=config.num_layers,
        **kwargs,
    )


def hyperparameter_search(
    model_cls: Type[nn.Module],
    data: Any,
    search_space: dict[str, list[Any]],
    n_trials: int,
    base_config: TrainConfig,
    results_dir: str | Path = "results",
    warm_start_params: dict[str, Any] | None = None,
    model_factory: Callable[["TrainConfig"], nn.Module] | None = None,
) -> dict[str, Any]:
    """Random search over ``search_space`` for the best val accuracy.

    For each trial the function:

    1. Samples hyperparameters from ``search_space``.
    2. Builds and trains the model with ``train_model``.
    3. Appends a JSONL record to
       ``results/hparam_search_{encoder}_{model}.jsonl``.

    If ``warm_start_params`` is provided it is always run as trial 0,
    which is useful when re-using a previously found best configuration
    (e.g., the H1 best GNN hyperparams as a starting point for H3).
    The remaining ``n_trials - 1`` slots are filled with random samples.

    If ``model_factory`` is provided it is called with the trial config
    to build the model, completely bypassing ``model_cls`` and
    ``_build_model_for_config``.  This is used by the H2 multimodal
    runner to build :class:`~src.fusion.fusion.FusedModel` instances
    without modifying the training loop.

    Args:
        model_cls: Model class to instantiate for each trial.  Ignored
            when ``model_factory`` is not ``None``.
        data: Graph data object.
        search_space: Dict mapping hyperparameter names to candidate
            value lists.  GNN search spaces typically include ``lr``,
            ``weight_decay``, ``hidden_channels``, ``dropout``.
            LogReg search spaces typically include only ``C``.
        n_trials: Total number of trials (including the warm-start trial
            when ``warm_start_params`` is not ``None``).
        base_config: Template config; per-trial values override matching
            fields.
        results_dir: Directory for JSONL logs and checkpoints.
        warm_start_params: Optional dict of hyperparameter values to
            evaluate as trial 0 before the random search begins.
        model_factory: Optional callable ``(TrainConfig) → nn.Module``.
            When provided, replaces the default
            ``_build_model_for_config(model_cls, config)`` call so
            callers can inject custom architectures (e.g. FusedModel).

    Returns:
        Dict with keys ``"best_params"`` (dict) and
        ``"best_val_metric"`` (float).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        results_dir
        / f"hparam_search_{base_config.encoder_name}_{base_config.model_name}.jsonl"
    )
    rng = np.random.default_rng(base_config.seed)

    # Build full list of param dicts: optional warm-start first, then random.
    all_trial_params: list[dict[str, Any]] = []
    if warm_start_params is not None:
        all_trial_params.append(dict(warm_start_params))
    n_random = n_trials - len(all_trial_params)
    for _ in range(n_random):
        all_trial_params.append(_sample_params(search_space, rng))

    best_val = -1.0
    best_params: dict[str, Any] = {}

    logger.info(
        "Hyperparameter search: model=%s encoder=%s n_trials=%d%s",
        base_config.model_name, base_config.encoder_name, n_trials,
        " (warm-start included)" if warm_start_params else "",
    )

    for trial_idx, params in enumerate(all_trial_params):
        label = "warm-start" if trial_idx == 0 and warm_start_params else "random"
        logger.info("Trial %d/%d [%s] | params=%s", trial_idx + 1, n_trials, label, params)

        # Build config for this trial (use trial_idx as seed for determinism).
        trial_config = replace(
            base_config,
            seed=base_config.seed + trial_idx,
            lr=float(params.get("lr", base_config.lr)),
            weight_decay=float(params.get("weight_decay", base_config.weight_decay)),
            hidden_channels=int(params.get("hidden_channels", base_config.hidden_channels)),
            dropout=float(params.get("dropout", base_config.dropout)),
            C=float(params.get("C", base_config.C)),
        )

        if model_factory is not None:
            model = model_factory(trial_config)
        else:
            model = _build_model_for_config(model_cls, trial_config)
        result = train_model(model, data, trial_config)

        record = {
            "trial": trial_idx,
            "params": params,
            "val_metric": result["val_metric"],
            "test_accuracy": result["test_accuracy"],
            "runtime_seconds": result["runtime_seconds"],
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        if result["val_metric"] > best_val:
            best_val = result["val_metric"]
            best_params = params
            logger.info(
                "  → new best val_acc=%.4f with params=%s", best_val, params
            )

    logger.info(
        "Hparam search done. best_val=%.4f best_params=%s", best_val, best_params
    )
    return {"best_params": best_params, "best_val_metric": best_val}


# ---------------------------------------------------------------------------
# Smoke test  (no PyG needed: uses SimpleNamespace + MLPClassifier + LogReg)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from types import SimpleNamespace

    from src.models.baselines import MLPClassifier

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")

    torch.manual_seed(0)
    N, D, C = 120, 16, 3

    # --- synthetic dataset ---
    x = torch.randn(N, D)
    y = torch.randint(0, C, (N,))
    ei = torch.empty(2, 0, dtype=torch.long)
    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:72]] = True
    val_mask[perm[72:96]] = True
    test_mask[perm[96:]] = True
    data = SimpleNamespace(
        x=x, edge_index=ei, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
    )

    cfg = TrainConfig(
        in_channels=D, hidden_channels=32, out_channels=C,
        dropout=0.0, num_layers=2, lr=1e-2, weight_decay=0.0,
        max_epochs=20, patience=10, warmup_epochs=3, min_lr=1e-5,
        seed=0, encoder_name="smoke", model_name="MLP",
        checkpoint_dir="results/.smoke_checkpoints", device="cpu",
    )

    # MLP training
    mlp = MLPClassifier(D, 32, C, dropout=0.0, num_layers=2)
    res = train_model(mlp, data, cfg)
    assert "test_accuracy" in res and 0 <= res["test_accuracy"] <= 1
    logger.info("MLP result: test_acc=%.4f runtime=%.2fs", res["test_accuracy"], res["runtime_seconds"])

    # LogReg training
    lr_cfg = replace(cfg, model_name="LogReg", C=1.0)
    lr_model = LogRegClassifier(D, C, C=1.0)
    res_lr = train_model(lr_model, data, lr_cfg)
    assert "test_accuracy" in res_lr
    logger.info("LogReg result: test_acc=%.4f", res_lr["test_accuracy"])

    # Hparam search (2 trials only for speed)
    search_space: dict[str, list] = {
        "lr": [0.01, 0.001],
        "hidden_channels": [16, 32],
        "dropout": [0.0],
    }
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        hparam_result = hyperparameter_search(
            model_cls=MLPClassifier,
            data=data,
            search_space=search_space,
            n_trials=2,
            base_config=replace(cfg, model_name="MLP", checkpoint_dir=tmp),
            results_dir=tmp,
        )
    assert "best_params" in hparam_result
    assert 0 <= hparam_result["best_val_metric"] <= 1
    logger.info("Hparam search result: %s", hparam_result)

    # Clean up smoke checkpoints
    import shutil
    shutil.rmtree("results/.smoke_checkpoints", ignore_errors=True)

    print("train.py smoke test OK")
