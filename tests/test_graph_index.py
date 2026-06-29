"""L1 unit tests for GraphIndex (neighborhoods, rings, communities)."""

from __future__ import annotations

import torch

from polyjepa.graph_index import GraphIndex


def _path4() -> GraphIndex:
    # Directed input 0->1->2->3; GraphIndex symmetrizes to an undirected path.
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    return GraphIndex(edge_index, num_nodes=4)


def test_neighbors_undirected_and_excludes_self():
    gi = _path4()
    assert gi.neighbors(1).tolist() == [0, 2]
    assert gi.neighbors(0).tolist() == [1]
    assert 1 not in gi.neighbors(1).tolist()


def test_ring2_is_exactly_two_hops():
    gi = _path4()
    assert gi.ring2(0).tolist() == [2]  # 0-1-2
    assert gi.ring2(1).tolist() == [3]  # 1-2-3
    assert 1 not in gi.ring2(1).tolist()  # never includes self
    assert 0 not in gi.ring2(0).tolist()


def _two_pentagons_bridged() -> GraphIndex:
    # Two 5-cycles {0..4} and {5..9} joined by a single bridge edge (4,5).
    a = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    b = [(5, 6), (6, 7), (7, 8), (8, 9), (9, 5)]
    edges = a + b + [(4, 5)]
    src = [s for s, _ in edges]
    dst = [d for _, d in edges]
    return GraphIndex(torch.tensor([src, dst]), num_nodes=10)


def test_community_is_nondegenerate_and_contains_self():
    gi = _two_pentagons_bridged()
    comm0 = set(gi.community_of(0).tolist())
    assert 0 in comm0
    # A real community: not a singleton, not the whole graph. (The bridge node
    # may join the adjacent community, so we do not require all neighbors.)
    assert 1 < len(comm0) < gi.num_nodes
    assert 1 in comm0  # an interior neighbor stays with the node


def test_sample_sibling_is_non_neighbor_same_community():
    gi = _two_pentagons_bridged()
    gen = torch.Generator().manual_seed(0)
    sib = gi.sample_sibling(0, gen)
    assert sib is not None
    assert sib != 0
    assert sib not in gi.neighbors(0).tolist()
    assert sib in gi.community_of(0).tolist()


def test_sample_sibling_none_when_no_candidate():
    # A triangle: node 0's community is {0,1,2} and its neighbors are {1,2}, so
    # there is no non-neighbor sibling and sample_sibling returns None.
    gi = GraphIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]), num_nodes=3)
    assert gi.sample_sibling(0, torch.Generator().manual_seed(0)) is None
