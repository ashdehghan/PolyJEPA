"""The full protocol on one dataset: does the compass reproduce, and does it beat hand-crafted?

Protocol (identical for every dataset):
  1. Sample N random marginal-matched schedules; score each on VALIDATION (2 search seeds).
  2. Fit a ridge compass: arrival profile -> validation accuracy. Test is never seen.
  3. Read the compass's node ordering; build a sweep from it; force the last K epochs flat so
     the recency channel is closed (mechanism.py showed the raw reversal number is recency,
     while the GAIN survives -- and grows -- once the ending is neutralised).
  4. Score on TEST with 10 initialisation seeds the search never saw. Compare against:
       flat  |  reversed compass  |  random-order sweeps  |  each hand-crafted coordinate
     The hand-crafted arms are the paper's central comparison: a learned coordinate system
     against the statistics a person would reach for.

Usage: python -m experiments.path_effect.replicate --dataset Cora --candidates 2000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch_geometric.datasets import Planetoid

from experiments.path_effect.coords import COORD_NAMES, node_coords
from experiments.path_effect.schedules import _spotlight, arrival_time, sinkhorn
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
T = 60
SEARCH_SEEDS = (100, 101)
HOLD_SEEDS = tuple(range(200, 210))
_DATA = None


def _init(root: str, name: str) -> None:
    global _DATA
    torch.set_num_threads(1)
    _DATA = Planetoid(root=root, name=name)[0]


def _eval(job):
    z, seeds = job
    w = sinkhorn(np.exp(z))
    out = [train_once(_DATA, w, seed=s) for s in seeds]
    return (float(np.mean([o["val_acc"] for o in out])),
            float(np.mean([o["test_acc"] for o in out])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Cora")          # Cora | CiteSeer | PubMed
    ap.add_argument("--candidates", type=int, default=2000)
    ap.add_argument("--flat-tail", type=int, default=30)  # K: last K epochs flat
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    root = f"/tmp/claude-1000/{a.dataset.lower()}"
    data = Planetoid(root=root, name=a.dataset)[0]
    n = int(data.train_mask.sum())
    print(f"{a.dataset}: {data.num_nodes} nodes, {n} train, {int(data.y.max()) + 1} classes")

    pool = mp.Pool(a.workers, initializer=_init, initargs=(root, a.dataset))
    t0 = time.time()

    # 1. random schedules, scored on validation
    rng = np.random.default_rng(0)
    cand = [s * rng.normal(size=(T, n)) for s in rng.uniform(0.1, 2.0, size=a.candidates)]
    res = pool.map(_eval, [(c, SEARCH_SEEDS) for c in cand], chunksize=4)
    val = np.array([r[0] for r in res])
    arr = np.stack([arrival_time(sinkhorn(np.exp(c))) for c in cand])
    print(f"scored {a.candidates} schedules ({time.time() - t0:.0f}s)")

    # 2. the compass: validation only, test never seen
    beta = RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(arr, val).coef_
    rank = beta.argsort().argsort() / (n - 1)

    # 3/4. designed schedules, scored on test with unseen seeds
    def tail(order):
        raw = _spotlight(order, T, 0.15)
        if a.flat_tail:
            raw[T - a.flat_tail:] = raw.mean()
        return sinkhorn(raw)

    def score(W):
        return np.array([train_once(data, W, seed=s)["test_acc"] for s in HOLD_SEEDS])

    flat = score(np.ones((T, n)))
    arms: dict[str, np.ndarray] = {
        "compass (learned)": tail(rank),
        "compass REVERSED": tail(1 - rank),
    }
    co = node_coords(data)[data.train_mask.numpy()]
    for i, nm in enumerate(COORD_NAMES):
        arms[f"hand-crafted: {nm}"] = tail(co[:, i].argsort().argsort() / (n - 1))
    r2 = np.random.default_rng(7)
    for k in range(3):
        arms[f"random order #{k + 1}"] = tail(r2.permutation(n) / (n - 1))

    print(f"\n{a.dataset}  (flat tail K={a.flat_tail}; 10 unseen init seeds; test set)")
    print(f"{'arm':26s} {'test':>7} {'vs flat':>9} {'sem':>7} {'wins':>6}")
    print(f"{'flat':26s} {flat.mean():7.4f} {'--':>9} {'':>7} {'':>6}")
    out = {"dataset": a.dataset, "config": vars(a), "flat": float(flat.mean()),
           "flat_per_seed": flat.tolist(), "coord_names": list(COORD_NAMES), "arms": {}}
    per_seed = {"flat": flat}
    for nm, W in arms.items():
        s = score(W)
        d = s - flat
        sem = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{nm:26s} {s.mean():7.4f} {d.mean():+9.4f} {sem:7.4f} {int((d > 0).sum()):4d}/10")
        out["arms"][nm] = {"test": float(s.mean()), "gain": float(d.mean()), "sem": float(sem),
                           "wins": int((d > 0).sum()), "per_seed": s.tolist(),
                           "paired_t": float(d.mean() / sem)}
        per_seed[nm] = s
    pool.close()
    pool.join()

    # Everything, so no analysis or plot ever needs this to be re-run.
    (ROOT / f"replicate_{a.dataset.lower()}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(
        ROOT / f"replicate_{a.dataset.lower()}.npz",
        val=val,                                   # candidate validation scores
        test_search=np.array([r[1] for r in res]),  # candidate test scores (search seeds)
        arrival=arr.astype(np.float32),            # candidate arrival profiles
        beta=beta,                                 # the compass: per-node early/late coefficient
        rank=rank,                                 # the node ordering it implies
        coords=co,                                 # hand-crafted node coordinates
        labels=data.y.numpy()[data.train_mask.numpy()],
        arm_names=np.array(list(per_seed)),
        arm_per_seed=np.stack([per_seed[k] for k in per_seed]),
        hold_seeds=np.array(HOLD_SEEDS),
    )
    print(f"\nwrote replicate_{a.dataset.lower()}.json + .npz  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
