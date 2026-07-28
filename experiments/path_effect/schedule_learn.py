"""E5 — learn the schedule by gradient descent instead of searching for it.

The compass (replicate.py) explores random Sinkhorn-projected schedules and exploits
them once through a linear surrogate. This experiment asks what a proper optimizer
finds. The schedule is parameterized as W = Sinkhorn(exp(Z)) with unconstrained
logits Z, so both marginals hold exactly for every Z and the projection is
differentiable. The whole 60-step training run is unrolled, and the validation loss
at the end is backpropagated through training and through the projection into Z.

Learning happens against a simplified deterministic inner loop: a functional GCN
with no dropout, trained by a functional Adam, at the two fixed search seeds. The
learned schedule is then evaluated under the unchanged standard pipeline
(train.train_once, dropout on) at the ten held-out seeds, exactly like every other
arm in the paper. Held-out data is never touched during learning.

Pre-registered criteria (Appendix "Learning the schedule"):
  primary  — held-out test gain over flat exceeds the compass's +1.9 pts on Cora;
  secondary — exceeds the held-out gain of the best validation-selected random
              schedule (the winner's-curse yardstick).

Usage:
    .venv/bin/python -m experiments.path_effect.schedule_learn --outer-steps 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, degree

from experiments.path_effect.schedules import arrival_time, sinkhorn
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
T = 60
SEARCH_SEEDS = (100, 101)
HOLD_SEEDS = tuple(range(200, 210))
INNER_LR = 0.01
INNER_WD = 5e-4
HIDDEN = 16


# ---------------------------------------------------------------------------
# Differentiable Sinkhorn projection (torch; the numpy version is not autograd-able)
# ---------------------------------------------------------------------------

def sinkhorn_torch(M: torch.Tensor, iters: int = 15) -> torch.Tensor:
    """Project a positive (T, N) matrix onto rows summing to N and columns to T."""
    t, n = M.shape
    W = M
    for _ in range(iters):
        W = W * (n / W.sum(dim=1, keepdim=True))
        W = W * (t / W.sum(dim=0, keepdim=True))
    return W


# ---------------------------------------------------------------------------
# Deterministic functional GCN + functional Adam (differentiable inner loop)
# ---------------------------------------------------------------------------

def _norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Symmetrically normalized adjacency with self-loops, as a sparse tensor."""
    ei, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = ei
    deg = degree(row, num_nodes, dtype=torch.float32)
    dinv = deg.pow(-0.5)
    vals = dinv[row] * dinv[col]
    return torch.sparse_coo_tensor(ei, vals, (num_nodes, num_nodes)).coalesce()


def _init_params(seed: int, in_dim: int, out_dim: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def glorot(shape):
        bound = (6.0 / (shape[0] + shape[1])) ** 0.5
        return (torch.rand(shape, generator=g) * 2 - 1) * bound

    return [glorot((in_dim, HIDDEN)), torch.zeros(HIDDEN),
            glorot((HIDDEN, out_dim)), torch.zeros(out_dim)]


def _forward(adj, ax, params):
    w1, b1, w2, b2 = params
    h = torch.relu(ax @ w1 + b1)
    return torch.sparse.mm(adj, h) @ w2 + b2


def _inner_train(W, adj, ax, y, train_idx, val_idx, seed, in_dim, out_dim):
    """Unrolled 60-step deterministic training under schedule W. Differentiable in W."""
    params = [p.clone().requires_grad_(True) for p in _init_params(seed, in_dim, out_dim)]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    n_train = train_idx.numel()
    for step in range(T):
        logits = _forward(adj, ax, params)
        ce = F.cross_entropy(logits[train_idx], y[train_idx], reduction="none")
        loss = (W[step] * ce).sum() / n_train
        grads = torch.autograd.grad(loss, params, create_graph=True)
        new_params, new_m, new_v = [], [], []
        for p, g_, m_, v_ in zip(params, grads, m, v):
            g_ = g_ + INNER_WD * p
            m_ = b1 * m_ + (1 - b1) * g_
            v_ = b2 * v_ + (1 - b2) * g_ * g_
            mhat = m_ / (1 - b1 ** (step + 1))
            vhat = v_ / (1 - b2 ** (step + 1))
            new_params.append(p - INNER_LR * mhat / (vhat.sqrt() + eps))
            new_m.append(m_)
            new_v.append(v_)
        params, m, v = new_params, new_m, new_v
    logits = _forward(adj, ax, params)
    val_ce = F.cross_entropy(logits[val_idx], y[val_idx])
    with torch.no_grad():
        val_acc = float((logits[val_idx].argmax(1) == y[val_idx]).float().mean())
    return val_ce, val_acc


def _search_objective(Z, adj, ax, y, train_idx, val_idx, in_dim, out_dim):
    W = sinkhorn_torch(torch.exp(Z))
    ces, accs = [], []
    for seed in SEARCH_SEEDS:
        ce, acc = _inner_train(W, adj, ax, y, train_idx, val_idx, seed, in_dim, out_dim)
        ces.append(ce)
        accs.append(acc)
    return torch.stack(ces).mean(), float(np.mean(accs)), W


# ---------------------------------------------------------------------------
# The winner's-curse yardstick: rebuild the best validation-selected random schedule
# ---------------------------------------------------------------------------

def _best_random_schedule(n: int, n_candidates: int, best_idx: int) -> np.ndarray:
    """Regenerate replicate.py's candidate stream and return candidate best_idx."""
    rng = np.random.default_rng(0)
    sigmas = rng.uniform(0.1, 2.0, size=n_candidates)
    cand = None
    for k in range(best_idx + 1):
        cand = sigmas[k] * rng.normal(size=(T, n))
    return sinkhorn(np.exp(cand))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--outer-steps", type=int, default=300)
    ap.add_argument("--outer-lr", type=float, default=0.05)
    ap.add_argument("--restarts", type=int, default=3)
    a = ap.parse_args()

    torch.set_num_threads(4)
    data = Planetoid(root=f"/tmp/claude-1000/{a.dataset.lower()}", name=a.dataset)[0]
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    n = train_idx.numel()
    in_dim, out_dim = data.num_features, int(data.y.max()) + 1
    adj = _norm_adj(data.edge_index, data.num_nodes)
    ax = torch.sparse.mm(adj, data.x)          # A_hat X, precomputed once
    print(f"{a.dataset}: n_train={n}, unrolling {T} steps x {len(SEARCH_SEEDS)} seeds")

    t0 = time.time()
    best = {"val_ce": float("inf"), "Z": None, "restart": -1, "step": -1}
    trajectories = []
    for r in range(a.restarts):
        g = torch.Generator().manual_seed(1000 + r)
        Z = (torch.zeros(T, n) if r == 0
             else 0.3 * torch.randn(T, n, generator=g)).requires_grad_(True)
        opt = torch.optim.Adam([Z], lr=a.outer_lr)
        traj = []
        for step in range(a.outer_steps):
            opt.zero_grad()
            val_ce, val_acc, _ = _search_objective(
                Z, adj, ax, data.y, train_idx, val_idx, in_dim, out_dim)
            val_ce.backward()
            opt.step()
            traj.append({"step": step, "val_ce": float(val_ce), "val_acc": val_acc})
            if float(val_ce) < best["val_ce"]:
                best = {"val_ce": float(val_ce), "Z": Z.detach().clone(),
                        "restart": r, "step": step}
            if step % 25 == 0:
                print(f"  restart {r} step {step:3d}: search val_ce={float(val_ce):.4f} "
                      f"val_acc={val_acc:.4f} ({time.time()-t0:.0f}s)")
        trajectories.append(traj)

    W_learned = sinkhorn_torch(torch.exp(best["Z"])).numpy()
    print(f"best: restart {best['restart']} step {best['step']} "
          f"val_ce={best['val_ce']:.4f} ({time.time()-t0:.0f}s)")

    # ---------------- standard-pipeline evaluation (identical to every other arm)
    def score(W):
        return np.array([train_once(data, W, seed=s)["test_acc"] for s in HOLD_SEEDS])

    flat = score(np.ones((T, n)))
    learned = score(W_learned)

    rep = np.load(ROOT / f"replicate_{a.dataset.lower()}.npz")
    val_arr = rep["val"]
    best_idx = int(val_arr.argmax())
    W_bestrand = _best_random_schedule(n, len(val_arr), best_idx)
    bestrand = score(W_bestrand)

    beta = rep["beta"].astype(np.float64)
    arr_learned = arrival_time(W_learned)
    corr_beta = float(np.corrcoef(arr_learned, beta)[0, 1])
    path_mag = float(np.abs(W_learned - 1.0).mean())

    gains = {
        "learned": float((learned - flat).mean()),
        "learned_wins": int((learned > flat).sum()),
        "best_random_val_selected": float((bestrand - flat).mean()),
        "best_random_wins": int((bestrand > flat).sum()),
    }
    print(f"\nflat: {flat.mean():.4f}")
    print(f"learned:      {learned.mean():.4f}  gain {gains['learned']*100:+.2f} pts "
          f"({gains['learned_wins']}/10)")
    print(f"best random (val-selected): {bestrand.mean():.4f}  "
          f"gain {gains['best_random_val_selected']*100:+.2f} pts")
    print(f"corr(arrival(W_learned), beta) = {corr_beta:+.3f}   "
          f"mean |W-1| = {path_mag:.3f}")

    out = {
        "dataset": a.dataset,
        "config": {"outer_steps": a.outer_steps, "outer_lr": a.outer_lr,
                   "restarts": a.restarts, "search_seeds": list(SEARCH_SEEDS),
                   "hold_seeds": list(HOLD_SEEDS), "inner": "functional Adam, no dropout"},
        "best_search": {"restart": best["restart"], "step": best["step"],
                        "val_ce": best["val_ce"]},
        "flat_test": flat.tolist(),
        "learned_test": learned.tolist(),
        "best_random_test": bestrand.tolist(),
        "gains": gains,
        "corr_arrival_beta": corr_beta,
        "path_magnitude": path_mag,
        "trajectories": trajectories,
    }
    (ROOT / f"schedule_learn_{a.dataset.lower()}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(ROOT / f"schedule_learn_{a.dataset.lower()}.npz",
                        W_learned=W_learned, Z=best["Z"].numpy())
    print(f"\nwrote schedule_learn_{a.dataset.lower()}.json + .npz "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
