"""Non-graph baseline classifiers: MLPClassifier and LogRegClassifier.

Both implement the same ``forward(x, edge_index) -> Tensor`` interface as
the GNNs in ``src.models.gnns`` so the evaluation harness and training
loop can treat all models uniformly.

MLPClassifier
-------------
A standard feed-forward network that ignores ``edge_index`` entirely.
It serves as the "no-graph" baseline in the H1 hypothesis test: if the
MLP with the same features outperforms the GNNs, the co-purchase edges
carry no useful signal.

LogRegClassifier
----------------
A thin wrapper around ``sklearn.linear_model.LogisticRegression``.
Logistic regression has no iteration-level training loop, so the
training protocol is different (single ``fit`` call, no LR schedule).
The wrapper exposes:

- ``fit(x, y)`` — trains the sklearn model on the given tensors (moved
  to CPU and converted to numpy internally).
- ``forward(x, edge_index)`` — returns log-probabilities as a float
  tensor compatible with ``argmax`` for predictions.  Because log is
  monotonic, ``argmax(log_proba) == argmax(proba)``.

``LogRegClassifier`` uses ``model.eval()`` / ``model.training`` as no-ops
(inherited from ``nn.Module``) so it is drop-in safe with ``evaluate()``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLPClassifier(nn.Module):
    """Multi-layer perceptron node classifier.

    Architecture: ``[Linear → BN → ReLU → Dropout] × (num_layers-1)``
    followed by a final ``Linear(hidden_channels, out_channels)`` with no
    activation (raw logits output).

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Width of all hidden layers.
        out_channels: Number of output classes.
        dropout: Dropout probability.
        num_layers: Total depth (including the output layer). Must be
            ``>= 1``.  When ``num_layers == 1`` the model degenerates to
            a single linear layer (identical to logistic regression
            without regularisation).

    Raises:
        ValueError: If ``num_layers < 1``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.3,
        num_layers: int = 2,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = float(dropout)
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        for i in range(num_layers - 1):
            in_ch = in_channels if i == 0 else hidden_channels
            layers += [
                nn.Linear(in_ch, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
        # Final output layer — no activation, no dropout.
        in_last = in_channels if num_layers == 1 else hidden_channels
        layers.append(nn.Linear(in_last, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Classify each node using only its feature vector.

        Args:
            x: Node features of shape ``(N, in_channels)``.
            edge_index: Ignored — included only for interface parity
                with GNN models.

        Returns:
            Raw class logits of shape ``(N, out_channels)``.
        """
        return self.net(x)

    def reset_parameters(self) -> None:
        """Re-initialise all linear and batch-norm weights."""
        for m in self.net.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


# ---------------------------------------------------------------------------
# LogReg (sklearn wrapper)
# ---------------------------------------------------------------------------


class LogRegClassifier(nn.Module):
    """Logistic regression classifier backed by sklearn.

    This wrapper satisfies the ``forward(x, edge_index)`` interface
    expected by the evaluation harness, but training is performed
    in a single call to ``fit()`` (not via a PyTorch optimiser loop).

    The ``src.train`` module detects this class with
    ``isinstance(model, LogRegClassifier)`` and routes it to a dedicated
    training path.

    Args:
        in_channels: Input feature dimension.
        out_channels: Number of output classes.
        C: Inverse regularisation strength (``C = 1 / weight_decay``).
            Larger ``C`` → less regularisation.
        max_iter: Maximum solver iterations passed to sklearn.
        solver: sklearn solver name (``"lbfgs"`` by default; scales well
            for high-dimensional sparse data).
        random_state: sklearn random seed for reproducibility.

    Attributes:
        _fitted: ``True`` once ``fit()`` has been called.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        from sklearn.linear_model import LogisticRegression

        self._logreg = LogisticRegression(
            C=float(C),
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "LogRegClassifier":
        """Train the underlying sklearn model.

        Args:
            x: Training features of shape ``(N_train, in_channels)``.
            y: Training labels of shape ``(N_train,)``.

        Returns:
            ``self`` (for method chaining).
        """
        X = x.detach().cpu().numpy()
        Y = y.detach().cpu().numpy()
        self._logreg.fit(X, Y)
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities for each node.

        ``argmax(log_proba)`` equals ``argmax(proba)`` so predictions are
        identical to using ``predict_proba``.

        Args:
            x: Node features of shape ``(N, in_channels)``.
            edge_index: Ignored — included for interface parity with GNNs.

        Returns:
            Float tensor of shape ``(N, out_channels)`` containing
            log-probabilities.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError(
                "LogRegClassifier.forward() called before fit(). "
                "Call fit(x_train, y_train) first."
            )
        X = x.detach().cpu().numpy()
        log_proba = self._logreg.predict_log_proba(X)
        return torch.from_numpy(log_proba).float()

    def reset_parameters(self) -> None:
        """Reset the sklearn model to an unfitted state.

        Resets ``C`` and other constructor arguments to their current
        values so the object can be re-used for a new trial.
        """
        from sklearn.linear_model import LogisticRegression

        self._logreg = LogisticRegression(
            C=self._logreg.C,
            max_iter=self._logreg.max_iter,
            solver=self._logreg.solver,
            multi_class="auto",
            random_state=self._logreg.random_state,
            n_jobs=-1,
        )
        self._fitted = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BASELINE_REGISTRY: dict[str, type[nn.Module]] = {
    "MLP": MLPClassifier,
    "LogReg": LogRegClassifier,
}


def build_baseline(
    name: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int = 128,
    dropout: float = 0.3,
    num_layers: int = 2,
    **kwargs: object,
) -> nn.Module:
    """Instantiate a baseline classifier by name.

    Args:
        name: ``"MLP"`` or ``"LogReg"``.
        in_channels: Input feature dimension.
        out_channels: Number of output classes.
        hidden_channels: Hidden width (MLP only, ignored for LogReg).
        dropout: Dropout probability (MLP only).
        num_layers: Depth (MLP only).
        **kwargs: Forwarded to the model constructor (e.g. ``C`` for
            ``LogRegClassifier``).

    Returns:
        An initialised baseline model.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _BASELINE_REGISTRY:
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {sorted(_BASELINE_REGISTRY)}"
        )
    if name == "MLP":
        return MLPClassifier(
            in_channels, hidden_channels, out_channels, dropout, num_layers, **kwargs
        )
    return LogRegClassifier(in_channels, out_channels, **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")

    N, D, C = 60, 32, 4
    x = torch.randn(N, D)
    y = torch.randint(0, C, (N,))
    ei = torch.empty(2, 0, dtype=torch.long)  # no edges (ignored by both)

    # --- MLP ---
    for num_layers in (1, 2, 3):
        mlp = MLPClassifier(D, 64, C, dropout=0.0, num_layers=num_layers)
        out = mlp(x, ei)
        assert out.shape == (N, C), f"MLP-{num_layers}: {out.shape}"
        mlp.reset_parameters()
        logger.info("MLP num_layers=%d OK | params=%d", num_layers, sum(p.numel() for p in mlp.parameters()))

    # --- LogReg ---
    logreg = LogRegClassifier(D, C, C=1.0)
    try:
        logreg(x, ei)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    logreg.fit(x[:40], y[:40])
    out = logreg(x, ei)
    assert out.shape == (N, C)
    assert out.dtype == torch.float32
    # Log-probabilities should sum to 0 in log-space (i.e. exp sums to 1).
    assert torch.allclose(out.exp().sum(dim=-1), torch.ones(N), atol=1e-5), \
        "log_proba rows should sum to 0 in log space"
    logger.info("LogReg OK | out=%s", tuple(out.shape))

    logreg.reset_parameters()
    assert not logreg._fitted

    # Factory
    mlp2 = build_baseline("MLP", D, C, hidden_channels=32)
    assert isinstance(mlp2, MLPClassifier)
    lr2 = build_baseline("LogReg", D, C, C=0.5)
    assert isinstance(lr2, LogRegClassifier)
    logger.info("build_baseline factory OK")
    print("baselines.py smoke test OK")
