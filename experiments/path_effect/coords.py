"""Hand-crafted node coordinates.

Five one-line statistics we can verify by inspection. These stand in for a learned
coordinate system: if *no* coordinate system predicts which schedules win, a learned
one will not either, and the premise behind a probe bank is false.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

COORD_NAMES = ("degree", "clustering", "pagerank", "homophily", "feat_norm")


def node_coords(data: Data) -> np.ndarray:
    """Return an ``(N, 5)`` matrix of per-node coordinates, z-scored per column.

    ``homophily`` is *feature* homophily (cosine similarity between a node's features
    and its neighbourhood mean), not label homophily: the coordinates must stay
    label-free, exactly as a self-supervised probe bank would be.
    """
    g = to_networkx(data, to_undirected=True)
    g.remove_edges_from(nx.selfloop_edges(g))
    n = data.num_nodes

    degree = np.array([g.degree(i) for i in range(n)], dtype=np.float64)
    clustering = np.array(list(nx.clustering(g).values()), dtype=np.float64)
    pagerank = np.array(list(nx.pagerank(g).values()), dtype=np.float64)

    x = data.x.float()
    x_norm = torch.nn.functional.normalize(x, dim=1)
    nbr_mean = torch.zeros_like(x)
    for i in range(n):
        nbrs = list(g.neighbors(i))
        if nbrs:
            nbr_mean[i] = x[nbrs].mean(dim=0)
    homophily = (x_norm * torch.nn.functional.normalize(nbr_mean, dim=1)).sum(1).numpy()

    feat_norm = x.norm(dim=1).numpy()

    coords = np.stack([degree, clustering, pagerank, homophily, feat_norm], axis=1)
    mu, sd = coords.mean(0, keepdims=True), coords.std(0, keepdims=True)
    return (coords - mu) / np.where(sd == 0.0, 1.0, sd)
