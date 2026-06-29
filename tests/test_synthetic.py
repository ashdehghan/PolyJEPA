"""L1 unit tests for the planted synthetic generators."""

from __future__ import annotations

import torch

from polyjepa.synthetic import inject_feature_outliers, planted_sbm


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(s), int(d)) for s, d in edge_index.t().tolist()}


def test_planted_sbm_shapes_and_labels():
    data = planted_sbm(num_communities=3, nodes_per_comm=10, num_features=8, seed=0)
    assert data.num_nodes == 30
    assert data.x.shape == (30, 8)
    assert int(data.y.min()) == 0 and int(data.y.max()) == 2


def test_planted_sbm_bridge_mask_count():
    data = planted_sbm(num_communities=3, nodes_per_comm=10, num_bridges=5, seed=0)
    assert int(data.bridge_mask.sum()) == 5


def test_planted_sbm_edges_symmetric():
    data = planted_sbm(num_communities=2, nodes_per_comm=8, seed=1)
    es = _edge_set(data.edge_index)
    assert all((d, s) in es for (s, d) in es)


def test_planted_sbm_is_reproducible():
    a = planted_sbm(num_communities=2, nodes_per_comm=8, seed=3)
    b = planted_sbm(num_communities=2, nodes_per_comm=8, seed=3)
    assert torch.allclose(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)


def test_inject_feature_outliers_mask_and_topology():
    base = planted_sbm(num_communities=2, nodes_per_comm=10, num_bridges=2, seed=0)
    out = inject_feature_outliers(base, frac=0.2, scale=8.0, seed=1)
    assert int(out.outlier_mask.sum()) == round(0.2 * out.num_nodes)
    # topology preserved
    assert torch.equal(out.edge_index, base.edge_index)
    # outlier features changed; non-outliers untouched
    om = out.outlier_mask
    assert not torch.allclose(out.x[om], base.x[om])
    assert torch.allclose(out.x[~om], base.x[~om])
    # planted bridge_mask is carried through
    assert torch.equal(out.bridge_mask, base.bridge_mask)
