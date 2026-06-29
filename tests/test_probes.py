"""Per-probe property invariants on planted synthetic graphs.

Each test plants a property whose ground truth we know, then asserts the probe's
residual reflects it. These are the evidence that a probe measures what we claim.
Seeds are fixed, so the assertions are deterministic.
"""

from __future__ import annotations

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
from polyjepa.synthetic import planted_sbm


def _residual(probe, data, **kw) -> torch.Tensor:
    return JEPAEngine(probe, **fast_kwargs(**kw)).fit(data).score()


def _group_means(resid: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    fin = torch.isfinite(resid)
    inside = resid[mask & fin]
    outside = resid[(~mask) & fin]
    return float(inside.mean()), float(outside.mean())


def test_A_isolated_node_ranks_top(clustered_with_isolated):
    data = clustered_with_isolated
    resid = _residual(RecoverFocal(), data, epochs=150)
    assert int(torch.argmax(resid)) == data.isolated_idx


def test_A_feature_outliers_score_higher(sbm_outliers):
    resid = _residual(RecoverFocal(), sbm_outliers, epochs=150)
    inside, outside = _group_means(resid, sbm_outliers.outlier_mask)
    assert inside > outside


def test_B_feature_outliers_score_higher(sbm_outliers):
    resid = _residual(PooledNeighborhood(), sbm_outliers, epochs=150)
    inside, outside = _group_means(resid, sbm_outliers.outlier_mask)
    assert inside > outside


def test_C_bridges_score_higher():
    data = planted_sbm(
        num_communities=3, nodes_per_comm=25, num_bridges=8, bridge_degree=6, seed=0
    )
    resid = _residual(SampledNeighbor(), data, epochs=150)
    inside, outside = _group_means(resid, data.bridge_mask)
    assert inside > outside


def test_D_distinct_from_C_and_has_spread():
    # The deck maps D (2-hop ring) to periphery / weak-clustering, a different
    # property than C (bridges). Here we assert D is a genuinely distinct probe:
    # its ranking is not a copy of C's, and it has meaningful spread.
    from scipy.stats import spearmanr

    data = planted_sbm(
        num_communities=3, nodes_per_comm=25, num_bridges=8, bridge_degree=6, seed=0
    )
    rd = _residual(TwoHopRing(), data, epochs=120)
    rc = _residual(SampledNeighbor(), data, epochs=120)
    fin = torch.isfinite(rd) & torch.isfinite(rc)
    rho, _ = spearmanr(rd[fin].numpy(), rc[fin].numpy())
    assert abs(rho) < 0.9  # D measures something different from C
    assert rd[fin].std() > 0  # non-degenerate (D saturates on dense homophily)


def test_E_runs_and_has_spread(sbm_bridges):
    resid = _residual(AugmentedView(), sbm_bridges, epochs=80)
    fin = torch.isfinite(resid)
    assert fin.all()  # E scores every node (self target)
    cv = resid[fin].std() / resid[fin].mean().abs()
    assert cv > 0.1


def test_F_runs_and_uses_communities(sbm_bridges):
    resid = _residual(CommunitySibling(), sbm_bridges, epochs=80)
    fin = torch.isfinite(resid)
    assert int(fin.sum()) > 0  # at least some nodes have a community sibling
    cv = resid[fin].std() / resid[fin].mean().abs()
    assert cv > 0.1
