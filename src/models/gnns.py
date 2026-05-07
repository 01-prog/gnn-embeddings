"""GNN node-classification architectures: GCN, GraphSAGE, and GAT.

All three inherit from ``BaseGNN`` which supplies:

- A ``forward(x, edge_index)`` entry point that the evaluation harness and
  training loop always call (baselines follow the same signature so they
  are drop-in replaceable).
- A ``reset_parameters()`` method used by the hyperparameter search to
  re-initialise weights between trials without rebuilding the module.
- A template-method ``_build_layers()`` that each subclass overrides to
  populate ``self.convs`` (the list of message-passing layers) and
  ``self.bns`` (matching ``BatchNorm1d`` layers).

Forward pass pattern (shared by all three, applies BN + ReLU + dropout
after every message-passing layer):

    x → [conv → BN → ReLU → dropout] × num_layers → classifier → logits

The ``classifier`` is a plain ``nn.Linear(hidden_channels, out_channels)``
that is shared across all architectures so the comparison between GNNs
and the MLP baseline is fair (same number of parameters in the head).

GAT note
--------
Intermediate GAT layers use ``concat=True`` so the output is
``hidden_channels * num_heads`` wide; the last layer uses ``concat=False``
to produce exactly ``hidden_channels``.  Batch normalisation dimensions
are adjusted accordingly.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseGNN(nn.Module):
    """Abstract base class for GNN node classifiers.

    Subclasses must implement ``_build_layers()`` to populate
    ``self.convs`` and ``self.bns`` with matching ``nn.ModuleList``
    entries.

    Args:
        in_channels: Input feature dimension (= embedding dimension).
        hidden_channels: Width of each hidden layer.
        out_channels: Number of output classes.
        dropout: Dropout probability applied after each conv layer.
        num_layers: Number of message-passing layers (``>= 1``).

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

        self.convs: nn.ModuleList = nn.ModuleList()
        self.bns: nn.ModuleList = nn.ModuleList()
        self._build_layers()

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def _build_layers(self) -> None:
        """Populate ``self.convs`` and ``self.bns``.

        Must be overridden by every subclass.  Called from ``__init__``
        before ``self.classifier`` is created.

        Raises:
            NotImplementedError: Always (this is an abstract method).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_layers()"
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run message passing then classify each node.

        Args:
            x: Node feature matrix of shape ``(N, in_channels)``.
            edge_index: Graph connectivity of shape ``(2, E)``.

        Returns:
            Raw (unnormalised) class logits of shape ``(N, out_channels)``.
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def reset_parameters(self) -> None:
        """Re-initialise all learnable weights to their defaults.

        Called by the hyperparameter search at the start of each trial
        so every trial sees a fresh initialisation.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------


class GCN(BaseGNN):
    """Graph Convolutional Network (Kipf & Welling, 2017).

    Uses symmetric normalised ``GCNConv`` layers. Each layer maps the
    previous hidden representation through the normalised adjacency.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Number of output classes.
        dropout: Dropout probability.
        num_layers: Number of ``GCNConv`` layers.
    """

    def _build_layers(self) -> None:
        from torch_geometric.nn import GCNConv

        for i in range(self.num_layers):
            in_ch = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(GCNConv(in_ch, self.hidden_channels, cached=False))
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))


# ---------------------------------------------------------------------------
# GraphSAGE
# ---------------------------------------------------------------------------


class GraphSAGE(BaseGNN):
    """GraphSAGE (Hamilton et al., 2017) with mean aggregation.

    At each layer the central node's representation is concatenated with
    the mean of its neighbours' representations and linearly projected.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Number of output classes.
        dropout: Dropout probability.
        num_layers: Number of ``SAGEConv`` layers.
    """

    def _build_layers(self) -> None:
        from torch_geometric.nn import SAGEConv

        for i in range(self.num_layers):
            in_ch = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(SAGEConv(in_ch, self.hidden_channels, aggr="mean"))
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------


class GAT(BaseGNN):
    """Graph Attention Network (Veličković et al., 2018).

    Intermediate layers concatenate ``num_heads`` attention heads (output
    width = ``hidden_channels * num_heads``); the final message-passing
    layer averages heads (output width = ``hidden_channels``) so the
    shared linear classifier always sees ``hidden_channels`` features.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Per-head hidden dimension.  The actual hidden
            width after intermediate layers is ``hidden_channels *
            num_heads``; only the last layer outputs ``hidden_channels``.
        out_channels: Number of output classes.
        dropout: Dropout probability (also passed to ``GATConv`` for
            attention coefficient dropout).
        num_layers: Total number of ``GATConv`` layers.
        num_heads: Number of attention heads in every layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.3,
        num_layers: int = 2,
        num_heads: int = 2,
    ) -> None:
        self.num_heads = num_heads
        super().__init__(in_channels, hidden_channels, out_channels, dropout, num_layers)

    def _build_layers(self) -> None:
        from torch_geometric.nn import GATConv

        for i in range(self.num_layers):
            in_ch = self.in_channels if i == 0 else self.hidden_channels * self.num_heads
            is_last = i == self.num_layers - 1
            if is_last:
                # Average heads → output is hidden_channels
                self.convs.append(
                    GATConv(in_ch, self.hidden_channels, heads=self.num_heads,
                            concat=False, dropout=self.dropout)
                )
                self.bns.append(nn.BatchNorm1d(self.hidden_channels))
            else:
                # Concatenate heads → output is hidden_channels * num_heads
                self.convs.append(
                    GATConv(in_ch, self.hidden_channels, heads=self.num_heads,
                            concat=True, dropout=self.dropout)
                )
                self.bns.append(nn.BatchNorm1d(self.hidden_channels * self.num_heads))

    def reset_parameters(self) -> None:
        """Re-initialise; calls the base class then resets head count."""
        super().reset_parameters()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GNN_REGISTRY: dict[str, type[BaseGNN]] = {
    "GCN": GCN,
    "GraphSAGE": GraphSAGE,
    "GAT": GAT,
}


def build_gnn(
    name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float = 0.3,
    num_layers: int = 2,
    **kwargs: object,
) -> BaseGNN:
    """Instantiate a GNN by name.

    Args:
        name: One of ``"GCN"``, ``"GraphSAGE"``, ``"GAT"``.
        in_channels: Input feature dimension.
        hidden_channels: Hidden layer width.
        out_channels: Number of classes.
        dropout: Dropout probability.
        num_layers: Number of message-passing layers.
        **kwargs: Extra keyword arguments forwarded to the model
            constructor (e.g. ``num_heads`` for GAT).

    Returns:
        An initialised ``BaseGNN`` sub-instance.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _GNN_REGISTRY:
        raise KeyError(
            f"Unknown GNN '{name}'. Available: {sorted(_GNN_REGISTRY)}"
        )
    return _GNN_REGISTRY[name](
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
        num_layers=num_layers,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Smoke test (no real data required — uses synthetic tensors)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")

    try:
        from torch_geometric.utils import erdos_renyi_graph
        N, D, C, E = 50, 32, 4, 100
        edge_index = erdos_renyi_graph(N, edge_prob=0.1, directed=False)
    except ImportError:
        logger.error("torch_geometric not installed — run `make install` first.")
        sys.exit(1)

    x = torch.randn(N, D)
    torch.manual_seed(0)

    for name, cls in _GNN_REGISTRY.items():
        kwargs = {"num_heads": 2} if name == "GAT" else {}
        model = cls(D, 64, C, dropout=0.1, num_layers=2, **kwargs)
        out = model(x, edge_index)
        assert out.shape == (N, C), f"{name}: expected ({N},{C}), got {out.shape}"
        model.reset_parameters()
        out2 = model(x, edge_index)
        assert out2.shape == (N, C)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("%-12s ok | params=%d | out=%s", name, n_params, tuple(out.shape))

    # Factory
    gcn = build_gnn("GCN", D, 64, C)
    assert isinstance(gcn, GCN)
    logger.info("build_gnn factory OK")
    print("gnns.py smoke test OK")
