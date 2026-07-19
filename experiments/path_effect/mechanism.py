"""Why does the compass schedule work? Path dependence, or just recency?

The compass (a ridge fit on 8000 random schedules, trained on VALIDATION accuracy only)
orders the 140 training nodes by when they should receive their gradient. Sweeping in that
order beats flat by +1.3 pts; sweeping in reverse LOSES 12.8 pts. Two stories fit:

  RECENCY   the spotlight lets nodes go dark, so whichever nodes are under the light at the
            END dominate the final weights. The compass merely learned "don't finish on the
            harmful nodes." Then the effect is a last-mile artifact, not path dependence.
  PATH      the ordering matters throughout training; the trajectory is what counts.

TEST 1 (discriminator): force the last K epochs FLAT in both directions (marginals still
pinned). If the gain and the collapse both vanish as K grows, it is recency.

TEST 2 (artifact check): a sweep concentrates mass in a narrow window. If the compass order
correlates with node CLASS, the final window trains on almost one class -- which would explain
a -12.8 collapse for a boring reason. Measure class purity of the final window directly.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch_geometric.datasets import Planetoid

from experiments.path_effect.schedules import _spotlight, sinkhorn
from experiments.path_effect.train import train_once

torch.set_num_threads(4)
ROOT = __file__.rsplit("/", 1)[0]
T, HOLD = 60, range(200, 210)

d = np.load(f"{ROOT}/capture.npz")
data = Planetoid(root="/tmp/claude-1000/cora", name="Cora")[0]
n = int(data.train_mask.sum())
train_idx = data.train_mask.nonzero(as_tuple=True)[0].numpy()
y = data.y.numpy()[train_idx]

beta = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(d["arrival"].astype(np.float64), d["val"]).coef_
rank = beta.argsort().argsort() / (n - 1)


def score(W):
    return np.array([train_once(data, W, seed=s)["test_acc"] for s in HOLD])


def sweep_with_flat_tail(order, k):
    """Spotlight for the first T-k epochs, flat for the last k. Marginals stay pinned."""
    raw = _spotlight(order, T, 0.15)
    if k:
        raw[T - k:] = raw.mean()  # neutral ending: every node equal for the last k epochs
    return sinkhorn(raw)


flat = score(np.ones((T, n)))
print(f"flat baseline: {flat.mean():.4f}\n")
print("TEST 1 — neutralise the ending (last K epochs flat, marginals still pinned)")
print(f"{'K':>3} {'forward':>9} {'gain':>8} | {'reversed':>9} {'gain':>9}")
for k in (0, 5, 10, 20, 30):
    f = score(sweep_with_flat_tail(rank, k))
    r = score(sweep_with_flat_tail(1 - rank, k))
    print(f"{k:3d} {f.mean():9.4f} {f.mean() - flat.mean():+8.4f} | "
          f"{r.mean():9.4f} {r.mean() - flat.mean():+9.4f}")

print("\nTEST 2 — is the final window class-pure? (a boring explanation for the collapse)")
W_f, W_r = sinkhorn(_spotlight(rank, T, 0.15)), sinkhorn(_spotlight(1 - rank, T, 0.15))


def final_window_stats(W, label):
    w = W[-5:].mean(0)                      # mass each node gets in the last 5 epochs
    p = w / w.sum()
    cls = np.array([p[y == c].sum() for c in range(int(y.max()) + 1)])
    ent = -(cls[cls > 0] * np.log(cls[cls > 0])).sum() / np.log(len(cls))
    print(f"  {label:22s} class mass {np.round(cls, 2)}  normalised entropy {ent:.3f} "
          f"(1.0 = balanced)")
    return ent


e_flat = final_window_stats(np.ones((T, n)), "flat")
e_f = final_window_stats(W_f, "compass forward")
e_r = final_window_stats(W_r, "compass reversed")
print(f"\ncorr(beta, class one-hot): max |r| over classes = "
      f"{max(abs(np.corrcoef(beta, (y == c).astype(float))[0, 1]) for c in range(int(y.max()) + 1)):.3f}")
