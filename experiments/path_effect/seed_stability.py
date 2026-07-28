"""E0 seed stability — the hard gate on the fingerprint.

The fingerprint R is produced by training six JEPA probes, and training starts from
seeded random weights. If the geometry of R moves when nothing changes but the seed,
the fingerprint measures optimization noise rather than the graph, and everything
built on it (the anatomy maps, the probe compass, probe-space selection) inherits
that noise. The design (manuscript, experiment E0) asks for geometric stability,
not numerical identity: per-column rank correlation across seeds, and k-NN
neighborhood overlap in the z-scored probe space.

Protocol: recompute the full fingerprint at fresh seeds with everything else fixed
(same 200 epochs per probe as the cached seed-0 reference), then compare each new
fingerprint to the reference on training rows:

  - per-column Spearman rank correlation (does each probe order nodes the same way?)
  - k-NN neighborhood overlap at k=10 in the 6-D z-scored space (does each node
    keep the same neighbors?), with the expected overlap under random ranking
    reported beside it (k / (n_train - 1)).

Stochastic probes C and F average 4 scoring passes whose randomness also follows
the seed, so their columns carry both sources of seed variance.

Usage:
    .venv/bin/python -m experiments.path_effect.seed_stability --seeds 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.datasets import Planetoid

from polyjepa import fingerprint as compute_fingerprint

ROOT = Path(__file__).resolve().parent
K_NN = 10


def _train_rows(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Training rows of a residual matrix, NaN-filled and column z-scored.

    Mirrors _load_or_compute_fingerprint in probe_compass.py so the geometry
    tested here is the geometry every downstream experiment actually used.
    """
    R_train = R[mask].copy()
    for col in range(R_train.shape[1]):
        vals = R_train[:, col]
        finite = vals[np.isfinite(vals)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        R_train[np.isnan(vals), col] = fill
    mu = R_train.mean(axis=0)
    sigma = R_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (R_train - mu) / sigma


def _knn_sets(X: np.ndarray, k: int) -> list[set[int]]:
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    order = np.argsort(d, axis=1)[:, :k]
    return [set(row) for row in order]


def _compare(Ra: np.ndarray, Rb: np.ndarray, probe_names: list[str]) -> dict:
    per_col = {}
    for j, name in enumerate(probe_names):
        rho = spearmanr(Ra[:, j], Rb[:, j]).statistic
        per_col[name] = float(rho)
    na, nb = _knn_sets(Ra, K_NN), _knn_sets(Rb, K_NN)
    overlap = float(np.mean([len(a & b) / K_NN for a, b in zip(na, nb)]))
    return {"spearman_per_probe": per_col,
            "spearman_mean": float(np.mean(list(per_col.values()))),
            "knn_overlap_k10": overlap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()

    torch.set_num_threads(4)
    data = Planetoid(root=f"/tmp/claude-1000/{a.dataset.lower()}", name=a.dataset)[0]
    mask = data.train_mask.numpy().astype(bool)
    n_train = int(mask.sum())

    ref_cache = ROOT / f"fingerprint_{a.dataset.lower()}.npz"
    ref = np.load(ref_cache, allow_pickle=True)
    R_ref = _train_rows(ref["residuals"], mask)
    probe_names = [str(p) for p in ref["probe_names"]]
    print(f"{a.dataset}: reference seed 0 loaded ({n_train} train rows, "
          f"probes {probe_names})")

    t0 = time.time()
    fingerprints = {0: R_ref}
    for seed in a.seeds:
        cache = ROOT / f"fingerprint_{a.dataset.lower()}_seed{seed}.npz"
        if cache.exists():
            R = np.load(cache)["residuals"]
            print(f"  seed {seed}: loaded from cache")
        else:
            print(f"  seed {seed}: computing ({a.epochs} epochs per probe)…")
            fp = compute_fingerprint(data, epochs=a.epochs, seed=seed)
            R = fp.residuals.numpy()
            np.savez_compressed(cache, residuals=R.astype(np.float32),
                                probe_names=np.array(fp.probe_names))
            print(f"  seed {seed}: done ({time.time() - t0:.0f}s)")
        fingerprints[seed] = _train_rows(R, mask)

    pairs = {}
    seeds = sorted(fingerprints)
    for i, sa in enumerate(seeds):
        for sb in seeds[i + 1:]:
            key = f"{sa}v{sb}"
            pairs[key] = _compare(fingerprints[sa], fingerprints[sb], probe_names)
            p = pairs[key]
            print(f"  {key}: mean Spearman {p['spearman_mean']:+.3f}, "
                  f"kNN overlap {p['knn_overlap_k10']:.3f}")

    out = {
        "dataset": a.dataset,
        "config": {"epochs": a.epochs, "seeds": seeds, "k": K_NN,
                   "n_train": n_train,
                   "chance_knn_overlap": K_NN / (n_train - 1)},
        "pairs": pairs,
    }
    path = ROOT / f"seed_stability_{a.dataset.lower()}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.name} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
