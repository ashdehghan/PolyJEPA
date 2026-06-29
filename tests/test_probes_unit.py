"""L1/L2 unit tests for probe construction and pair-building branches.

Covers the parts of the probes that the property tests in test_probes.py do not
exercise: constructor validation, the neighbor-masking path of RecoverFocal, and
the "no valid focal" branch where make_pair returns None.
"""

from __future__ import annotations

import pytest
import torch
from tests.conftest import fast_kwargs

from polyjepa import (
    AugmentedView,
    CommunitySibling,
    JEPAEngine,
    PooledNeighborhood,
    RecoverFocal,
    SampledNeighbor,
    TwoHopRing,
)
from polyjepa.graph_index import GraphIndex
from polyjepa.pair_design import PairContext


def _ctx(data, focal: list[int]) -> PairContext:
    gi = GraphIndex(data.edge_index, data.num_nodes)
    x = data.x.float()
    return PairContext(
        x=x,
        edge_index=data.edge_index,
        focal_idx=torch.tensor(focal, dtype=torch.long),
        mask_token=torch.zeros(x.shape[1]),
        index=gi,
        generator=torch.Generator().manual_seed(0),
        num_nodes=data.num_nodes,
        device=torch.device("cpu"),
    )


def test_recoverfocal_rejects_no_mask_source():
    with pytest.raises(ValueError, match="mask source"):
        RecoverFocal(mask_focal=False, neighbor_mask_ratio=0.0)


def test_recoverfocal_rejects_bad_ratio_and_hops():
    with pytest.raises(ValueError, match="neighbor_mask_ratio"):
        RecoverFocal(neighbor_mask_ratio=1.5)
    with pytest.raises(ValueError, match="neighbor_hops"):
        RecoverFocal(neighbor_mask_ratio=0.5, neighbor_hops=3)


def test_recoverfocal_neighbor_masking_runs(sbm_bridges):
    probe = RecoverFocal(mask_focal=True, neighbor_mask_ratio=0.5, neighbor_hops=2)
    scores = JEPAEngine(probe, **fast_kwargs(epochs=30)).fit(sbm_bridges).score()
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    "probe",
    [SampledNeighbor(), PooledNeighborhood(), TwoHopRing(), CommunitySibling()],
)
def test_masked_probe_returns_none_on_isolated_focal(clustered_with_isolated, probe):
    # An isolated focal node has no neighbors/ring/community sibling, so there is
    # no valid target and make_pair returns None.
    ctx = _ctx(clustered_with_isolated, [clustered_with_isolated.isolated_idx])
    assert probe.make_pair(ctx) is None


def test_recoverfocal_and_augmented_return_none_on_empty_focal(clustered_with_isolated):
    ctx = _ctx(clustered_with_isolated, [])
    assert RecoverFocal().make_pair(ctx) is None
    assert AugmentedView().make_pair(ctx) is None
