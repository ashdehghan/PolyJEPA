"""Synthetic graphs with planted ground truth for validating the probes.

Each generator returns a PyG ``Data`` with a known indicator attached, so a
probe's residual can be correlated against the property it claims to measure:

- :func:`planted_sbm` plants community structure plus ``bridge_mask`` nodes that
  connect across communities (targets for probes C and D).
- :func:`inject_feature_outliers` plants ``outlier_mask`` nodes whose features
  are off-distribution (targets for probes A and B).

All randomness is seeded for reproducibility. These generators are intentionally
small and dependency-free (no ``graphcl`` import) so the package stays
self-contained.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def planted_sbm(
    num_communities: int = 3,
    nodes_per_comm: int = 40,
    p_in: float = 0.30,
    p_out: float = 0.02,
    num_features: int = 32,
    feature_sep: float = 2.0,
    num_bridges: int = 6,
    bridge_degree: int = 5,
    seed: int = 0,
) -> Data:
    """A stochastic block model with planted cross-community bridge nodes.

    Node ``y`` is the community label. ``bridge_mask`` marks nodes wired to other
    communities; their local neighborhoods are heterogeneous, so neighbor- and
    ring-based probes should score them as less predictable.
    """
    g = torch.Generator().manual_seed(seed)
    n = num_communities * nodes_per_comm
    comm = torch.arange(n) // nodes_per_comm

    means = torch.randn(num_communities, num_features, generator=g) * feature_sep
    x = means[comm] + torch.randn(n, num_features, generator=g)

    iu = torch.triu_indices(n, n, offset=1)
    same = comm[iu[0]] == comm[iu[1]]
    probs = same.float() * p_in + (~same).float() * p_out
    keep = torch.rand(iu.shape[1], generator=g) < probs
    e = iu[:, keep]

    bridge_mask = torch.zeros(n, dtype=torch.bool)
    bridge_nodes = torch.randperm(n, generator=g)[:num_bridges]
    extra: list[list[int]] = []
    for b in bridge_nodes.tolist():
        bridge_mask[b] = True
        others = (comm != comm[b]).nonzero().flatten()
        sel = others[torch.randperm(others.numel(), generator=g)[:bridge_degree]]
        for o in sel.tolist():
            extra.append([b, o])
    if extra:
        e = torch.cat([e, torch.tensor(extra, dtype=torch.long).t()], dim=1)

    edge_index = torch.cat([e, e.flip(0)], dim=1)
    data = Data(x=x, edge_index=edge_index, y=comm)
    data.bridge_mask = bridge_mask
    return data


def inject_feature_outliers(
    data: Data, frac: float = 0.1, scale: float = 6.0, seed: int = 0
) -> Data:
    """Replace a fraction of nodes' features with off-distribution noise.

    Adds ``outlier_mask``. Preserves topology, labels, and any existing
    ``bridge_mask``. Outliers' features disagree with their neighborhoods, so
    focal-recovery (A) and neighborhood (B) probes should score them as less
    predictable.
    """
    g = torch.Generator().manual_seed(seed)
    n = int(data.num_nodes)
    k = max(1, int(round(frac * n)))
    idx = torch.randperm(n, generator=g)[:k]
    x = data.x.clone()
    x[idx] = torch.randn(k, x.shape[1], generator=g) * scale
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True

    out = Data(x=x, edge_index=data.edge_index, y=getattr(data, "y", None))
    out.outlier_mask = mask
    if hasattr(data, "bridge_mask"):
        out.bridge_mask = data.bridge_mask
    return out
