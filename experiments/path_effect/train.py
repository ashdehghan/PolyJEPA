"""GCN node classification on Cora under a per-node loss schedule.

The only thing that varies between arms is ``W`` — when each node's gradient mass
lands. Initialisation, data, split, optimiser and epoch count are identical, and every
schedule carries the same total mass per node (see ``schedules.py``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """Kipf & Welling's Cora reference architecture."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    return float((pred == y[mask]).float().mean())


def train_once(
    data: Data,
    weights: np.ndarray,
    seed: int,
    hidden: int = 16,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> dict[str, float]:
    """Train under schedule ``weights`` (T, n_train). Returns final val/test accuracy.

    The loss at epoch t is ``sum_i W[t, i] * CE_i / n_train``. Under the flat schedule
    (``W == 1``) this is exactly the standard mean cross-entropy, so the baseline arm
    reproduces the reference setup rather than a variant of it.
    """
    torch.manual_seed(seed)
    epochs, n_train = weights.shape
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    assert train_idx.numel() == n_train

    model = GCN(data.num_features, hidden, int(data.y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = torch.from_numpy(weights).float()

    for t in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        ce = F.cross_entropy(logits[train_idx], data.y[train_idx], reduction="none")
        loss = (w[t] * ce).sum() / n_train
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return {
        "val_acc": _accuracy(logits, data.y, data.val_mask),
        "test_acc": _accuracy(logits, data.y, data.test_mask),
    }
