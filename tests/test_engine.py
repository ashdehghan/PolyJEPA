"""Engine-level invariants: determinism, anti-collapse, schedules, backbones."""

from __future__ import annotations

import math

import pytest
import torch
from tests.conftest import fast_kwargs, make_deterministic
from torch import nn

from polyjepa import JEPAEngine, PooledNeighborhood, RecoverFocal
from polyjepa.backbones import build_backbone
from polyjepa.engine import _Predictor, _TargetEMA


def test_score_shape_and_finite(sbm_bridges):
    eng = JEPAEngine(RecoverFocal(), **fast_kwargs()).fit(sbm_bridges)
    s = eng.score()
    assert s.shape == (sbm_bridges.num_nodes,)
    assert torch.isfinite(s).all()  # probe A scores every node


def test_deterministic_with_seed(sbm_bridges):
    s1 = JEPAEngine(RecoverFocal(), **fast_kwargs(seed=7)).fit(sbm_bridges).score()
    s2 = JEPAEngine(RecoverFocal(), **fast_kwargs(seed=7)).fit(sbm_bridges).score()
    assert torch.allclose(s1, s2)


def test_different_seed_differs(sbm_bridges):
    s1 = JEPAEngine(RecoverFocal(), **fast_kwargs(seed=1)).fit(sbm_bridges).score()
    s2 = JEPAEngine(RecoverFocal(), **fast_kwargs(seed=2)).fit(sbm_bridges).score()
    assert not torch.allclose(s1, s2)


def test_scores_do_not_collapse(sbm_bridges):
    eng = JEPAEngine(RecoverFocal(), **fast_kwargs()).fit(sbm_bridges)
    s = eng.score()
    cv = s.std() / s.mean().abs()
    assert cv > 0.1
    assert eng.diagnostics.healthy()


def test_loss_improves(sbm_bridges):
    eng = JEPAEngine(RecoverFocal(), **fast_kwargs(epochs=200)).fit(sbm_bridges)
    losses = eng.diagnostics.losses
    assert min(losses) < losses[0]  # training reduced the loss at some point


def test_embedding_shape(sbm_bridges):
    eng = JEPAEngine(RecoverFocal(), **fast_kwargs(latent_dim=48)).fit(sbm_bridges)
    emb = eng.embeddings()
    assert emb.shape == (sbm_bridges.num_nodes, 48)
    assert torch.isfinite(emb).all()


def test_ema_schedule_endpoints():
    eng = JEPAEngine(RecoverFocal(), ema_base=0.99, epochs=1000)
    assert math.isclose(eng._ema_tau(0), 0.99, abs_tol=1e-6)
    assert math.isclose(eng._ema_tau(1000), 1.0, abs_tol=1e-6)
    # monotonically non-decreasing toward 1.0
    assert eng._ema_tau(500) > eng._ema_tau(0)
    assert eng._ema_tau(1000) > eng._ema_tau(500)


def test_lr_warmup_then_decay():
    eng = JEPAEngine(RecoverFocal(), lr=5e-4, epochs=100, warmup_steps=10)
    assert eng._lr_factor(0) < eng._lr_factor(9)  # warming up
    assert math.isclose(eng._lr_factor(9), 1.0, abs_tol=1e-6)  # peak at warmup end
    assert eng._lr_factor(99) < eng._lr_factor(10)  # decaying after


@pytest.mark.parametrize("backbone", ["gcn", "sage", "gat", "mlp"])
def test_backbone_swap(sbm_bridges, backbone):
    eng = JEPAEngine(
        RecoverFocal(), backbone=backbone, **fast_kwargs(epochs=40)
    ).fit(sbm_bridges)
    s = eng.score()
    assert s.shape == (sbm_bridges.num_nodes,)
    assert torch.isfinite(s).all()


# -- L2 deep-learning-pattern tests --------------------------------------------


def test_gradient_flows_to_all_params():
    """One JEPA step gives every trainable parameter (encoder, predictor, and the
    mask token) a finite gradient, and a non-zero one somewhere. Catches dead
    sub-graphs and a mask token that never participates.
    """
    make_deterministic(0)
    encoder = build_backbone("gcn", in_dim=4, hidden_dim=16, latent_dim=16)
    predictor = _Predictor(16, 16, 2)
    mask_token = torch.zeros(4, requires_grad=True)

    x = torch.randn(8, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    x_masked = x.clone()
    x_masked[torch.tensor([0, 1])] = mask_token  # makes the loss depend on the token

    with torch.no_grad():
        target = encoder(x, edge_index)
    pred = predictor(encoder(x_masked, edge_index))
    loss = (pred - target).pow(2).sum(dim=-1).mean()
    loss.backward()

    params = list(encoder.parameters()) + list(predictor.parameters())
    for p in params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    assert any(p.grad.abs().sum() > 0 for p in params)
    assert mask_token.grad is not None and mask_token.grad.abs().sum() > 0


def test_ema_update_matches_formula():
    src = nn.Linear(3, 3)
    target = _TargetEMA(src)
    old = [p.clone() for p in target.encoder.parameters()]
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)  # move the source away from the target
    momentum = 0.9
    target.update(src, momentum)
    for o, tp, sp in zip(
        old, target.encoder.parameters(), src.parameters(), strict=True
    ):
        expected = momentum * o + (1.0 - momentum) * sp
        assert torch.allclose(tp, expected)


def test_training_strongly_reduces_loss(sbm_bridges):
    """Bootstrap SSL losses plateau rather than hitting zero, but training must
    still cut the loss substantially (measured: ~10x). Proves end-to-end learning.
    """
    eng = JEPAEngine(RecoverFocal(), **fast_kwargs(epochs=300)).fit(sbm_bridges)
    losses = eng.diagnostics.losses
    assert min(losses) < 0.5 * losses[0]


def test_unscored_node_is_nan(clustered_with_isolated):
    """Probe B has no target for an isolated node, so its residual is NaN, not a
    misleading zero.
    """
    data = clustered_with_isolated
    resid = JEPAEngine(PooledNeighborhood(), **fast_kwargs(epochs=20)).fit(data).score()
    assert torch.isnan(resid[data.isolated_idx])


# -- engine guard rails / construction branches --------------------------------


def test_predictor_depth_one_is_linear():
    pred = _Predictor(latent_dim=8, hidden_dim=16, depth=1)
    assert pred(torch.randn(3, 8)).shape == (3, 8)


def test_focal_ratio_must_be_valid():
    with pytest.raises(ValueError, match="focal_ratio"):
        JEPAEngine(RecoverFocal(), focal_ratio=0.0)


def test_fit_requires_node_features(sbm_bridges):
    from torch_geometric.data import Data

    no_x = Data(edge_index=sbm_bridges.edge_index, num_nodes=sbm_bridges.num_nodes)
    with pytest.raises(ValueError, match="requires data.x"):
        JEPAEngine(RecoverFocal(), **fast_kwargs(epochs=2)).fit(no_x)


def test_runs_without_seed(sbm_bridges):
    eng = JEPAEngine(
        RecoverFocal(), epochs=10, hidden_dim=32, latent_dim=32, seed=None
    ).fit(sbm_bridges)
    assert eng.score().shape == (sbm_bridges.num_nodes,)


def test_score_triggers_fit_if_needed(sbm_bridges):
    eng = JEPAEngine(RecoverFocal(), **fast_kwargs(epochs=10))
    scores = eng.score(sbm_bridges)  # not fitted yet
    assert torch.isfinite(scores).all()
