"""Gradient Influence Score (GIS) — dynamic alternative to static probe features.

Tests whether per-node early-training gradient influence can predict β (the
compass coefficient that encodes when each node should be trained).

β is uncorrelated with all static node properties (degree, clustering, probes).
The hypothesis here: β measures gradient dynamics, so a direct early-epoch
measurement of gradient influence should correlate with β even if structure can't.

Three GIS variants are computed for each training node i:

  GIS_loss_i    = Σ_t exp(-t/τ) · ℓ_i(t)
                  Exponentially-weighted sum of per-node CE loss across E epochs.
                  τ=3 weights epoch 0 at 1.0, epoch 9 at exp(-3) ≈ 0.05.

  GIS_slope_i   = ℓ_i(E//2) - ℓ_i(0)
                  Late-minus-early loss slope: fast fallers may be "easy" nodes
                  whose early training hurts (they saturate and pollute gradients).

  GIS_early_i   = mean(ℓ_i(0..2))
                  Mean loss in the first 3 epochs — how hard is this node at the start?

All three are computed across S=5 independent seeds and averaged to reduce variance.

Correlations r(GIS_*, β) and Spearman ρ(GIS_*, β) are reported on training nodes only.

Usage:
    cd workspace/PolyJEPA
    .venv/bin/python -m experiments.path_effect.gis --datasets Cora
    .venv/bin/python -m experiments.path_effect.gis --datasets Cora CiteSeer PubMed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch_geometric.datasets import Planetoid

from experiments.path_effect.train import GCN

ROOT = Path(__file__).resolve().parent

# Number of epochs for GIS measurement (far fewer than full training; captures early dynamics)
GIS_EPOCHS = 15
# Number of seeds to average over (reduces noise)
GIS_SEEDS = 5
# Decay constant for exponential early-epoch weight (in epochs)
GIS_TAU = 3.0


def _compute_gis_one_seed(
    data,
    seed: int,
    epochs: int = GIS_EPOCHS,
    hidden: int = 16,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> np.ndarray:
    """Run flat-schedule training for `epochs` steps, record per-node CE.

    Returns loss_matrix of shape (epochs, n_train), where loss_matrix[t, i]
    is the cross-entropy of training node i at epoch t.
    """
    torch.manual_seed(seed)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    n_train = train_idx.numel()

    model = GCN(data.num_features, hidden, int(data.y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_matrix = np.zeros((epochs, n_train), dtype=np.float32)

    for t in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        ce = F.cross_entropy(logits[train_idx], data.y[train_idx], reduction="none")
        # Record before taking the gradient step
        loss_matrix[t] = ce.detach().cpu().numpy()
        # Flat schedule: uniform weight for all nodes
        loss = ce.mean()
        loss.backward()
        opt.step()

    return loss_matrix  # (epochs, n_train)


def _gis_scores(loss_matrix: np.ndarray, tau: float = GIS_TAU) -> dict[str, np.ndarray]:
    """Compute GIS variants from a (epochs, n_train) loss matrix."""
    E = loss_matrix.shape[0]
    # Exponentially-decaying early-epoch weights
    t_idx = np.arange(E, dtype=np.float32)
    weights = np.exp(-t_idx / tau)
    weights /= weights.sum()

    gis_loss = (weights[:, None] * loss_matrix).sum(axis=0)   # (n_train,)
    gis_slope = loss_matrix[E // 2] - loss_matrix[0]          # (n_train,)
    gis_early = loss_matrix[:3].mean(axis=0)                   # (n_train,)

    return {"gis_loss": gis_loss, "gis_slope": gis_slope, "gis_early": gis_early}


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (pearson_r, pearson_p, spearman_rho, spearman_p)."""
    r, pr = stats.pearsonr(a, b)
    rho, ps = stats.spearmanr(a, b)
    return float(r), float(pr), float(rho), float(ps)


def process_dataset(ds: str, epochs: int, seeds: int) -> None:
    npz_path = ROOT / f"replicate_{ds.lower()}.npz"
    if not npz_path.exists():
        print(f"  {ds}: no replicate npz — run replicate.py first, skipping")
        return

    print(f"\n{'='*60}")
    print(f"{ds}")
    print(f"{'='*60}")

    d = np.load(npz_path)
    beta = d["beta"].astype(np.float64)   # (n_train,) — compass coefficients
    n_train = beta.shape[0]
    print(f"  n_train={n_train}, beta range [{beta.min():.4f}, {beta.max():.4f}]")

    root = f"/tmp/claude-1000/{ds.lower()}"
    data = Planetoid(root=root, name=ds)[0]

    # Compute GIS across multiple seeds and average
    print(f"  running {seeds} seeds × {epochs} epochs …")
    all_matrices = []
    for seed in range(seeds):
        mat = _compute_gis_one_seed(data, seed=seed, epochs=epochs)
        all_matrices.append(mat)
        print(f"    seed {seed}: mean loss epoch 0 = {mat[0].mean():.4f}, epoch {epochs-1} = {mat[-1].mean():.4f}")

    mean_matrix = np.stack(all_matrices).mean(axis=0)  # (epochs, n_train)
    scores = _gis_scores(mean_matrix)

    print(f"\n  {'GIS variant':20s} {'r(β)':>8} {'p_r':>8} {'ρ(β)':>8} {'p_ρ':>8}")
    print(f"  {'-'*60}")

    results = {}
    for name, gis in scores.items():
        r, pr, rho, ps = _corr(gis, beta)
        sign = "**" if pr < 0.05 else ("*" if pr < 0.1 else "  ")
        print(f"  {name:20s} {r:+8.4f}{sign} {pr:8.4f} {rho:+8.4f}  {ps:8.4f}")
        results[name] = {"r": r, "p_r": pr, "rho": rho, "p_rho": ps,
                         "gis": gis.tolist()}

    # Also check: absolute value (magnitude matters regardless of direction?)
    for name, gis in scores.items():
        r_abs, pr_abs, _, _ = _corr(np.abs(gis), np.abs(beta))
        results[name]["r_abs"] = r_abs
        results[name]["p_r_abs"] = pr_abs

    print(f"\n  (** p<0.05, * p<0.1)")
    print(f"\n  β summary: mean {beta.mean():.4f}, std {beta.std():.4f}")

    # Show top-5 and bottom-5 β nodes and their GIS values
    sort_idx = beta.argsort()
    gis_loss = scores["gis_loss"]
    print(f"\n  Top-5 β (train late):")
    for i in sort_idx[-5:][::-1]:
        print(f"    node[{i}]: β={beta[i]:+.4f}  gis_loss={gis_loss[i]:.4f}")
    print(f"  Bottom-5 β (train early):")
    for i in sort_idx[:5]:
        print(f"    node[{i}]: β={beta[i]:+.4f}  gis_loss={gis_loss[i]:.4f}")

    out = {
        "dataset": ds,
        "n_train": int(n_train),
        "gis_epochs": epochs,
        "gis_seeds": seeds,
        "gis_tau": GIS_TAU,
        "beta": beta.tolist(),
        "results": results,
    }
    out_path = ROOT / f"gis_{ds.lower()}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora"])
    ap.add_argument("--epochs", type=int, default=GIS_EPOCHS,
                    help="GIS measurement epochs (default 15)")
    ap.add_argument("--seeds", type=int, default=GIS_SEEDS,
                    help="Seeds to average over (default 5)")
    a = ap.parse_args()

    torch.set_num_threads(1)
    for ds in a.datasets:
        process_dataset(ds, a.epochs, a.seeds)


if __name__ == "__main__":
    main()
