"""Pluggable GNN encoders for the PolyJEPA engine.

All encoders map node features ``(N, in_dim)`` plus an ``edge_index`` to node
embeddings ``(N, latent_dim)``. Following BGRL (Thakoor et al., ICLR 2022) every
layer applies BatchNorm and a PReLU activation, including the final layer;
BatchNorm doubles as a mild collapse deterrent (it centers the embeddings).

The default backbone is a 2-layer GCN (Kipf and Welling, 2017): two layers keep
the receptive field shallow and avoid over-smoothing. ``MLPEncoder`` is a
structure-free control that ignores ``edge_index`` entirely, useful for
separating how much of a probe's signal comes from topology versus features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class Encoder(nn.Module, ABC):
    """Base class for node encoders. Subclasses set ``self.latent_dim``."""

    latent_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings of shape ``(num_nodes, latent_dim)``."""


def _norm_act(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.BatchNorm1d(dim), nn.PReLU())


class _ConvStack(Encoder):
    """Shared stack: a list of conv layers each followed by BatchNorm + PReLU.

    Subclasses supply a ``_make_conv(in_dim, out_dim, is_last)`` factory.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.latent_dim = latent_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [latent_dim]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            is_last = i == num_layers - 1
            self.convs.append(self._make_conv(dims[i], dims[i + 1], is_last))
            self.norms.append(_norm_act(dims[i + 1]))

    def _make_conv(self, in_dim: int, out_dim: int, is_last: bool) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = conv(h, edge_index)
            h = norm(h)
        return h


class GCNEncoder(_ConvStack):
    """Graph Convolutional Network encoder (default backbone)."""

    def _make_conv(self, in_dim: int, out_dim: int, is_last: bool) -> nn.Module:
        return GCNConv(in_dim, out_dim)


class SAGEEncoder(_ConvStack):
    """GraphSAGE (mean aggregator) encoder."""

    def _make_conv(self, in_dim: int, out_dim: int, is_last: bool) -> nn.Module:
        return SAGEConv(in_dim, out_dim, aggr="mean")


class GATEncoder(_ConvStack):
    """Graph Attention Network encoder.

    Intermediate layers use ``heads`` attention heads (concatenated); the final
    layer uses a single head so the output width equals ``latent_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        heads: int = 4,
    ) -> None:
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads for GAT")
        self.heads = heads
        super().__init__(in_dim, hidden_dim, latent_dim, num_layers)

    def _make_conv(self, in_dim: int, out_dim: int, is_last: bool) -> nn.Module:
        if is_last:
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        return GATConv(in_dim, out_dim // self.heads, heads=self.heads, concat=True)


class MLPEncoder(Encoder):
    """Structure-free encoder: ignores ``edge_index`` (topology control)."""

    def __init__(
        self, in_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.latent_dim = latent_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [latent_dim]
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.PReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_BACKBONES: dict[str, type[Encoder]] = {
    "gcn": GCNEncoder,
    "sage": SAGEEncoder,
    "gat": GATEncoder,
    "mlp": MLPEncoder,
}


def build_backbone(
    name: str,
    in_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_layers: int = 2,
    **kwargs,
) -> Encoder:
    """Construct an encoder by name (``gcn`` / ``sage`` / ``gat`` / ``mlp``)."""
    key = name.lower()
    if key not in _BACKBONES:
        raise ValueError(
            f"unknown backbone {name!r}; choose from {sorted(_BACKBONES)}"
        )
    return _BACKBONES[key](in_dim, hidden_dim, latent_dim, num_layers, **kwargs)
