"""Score hand-crafted coordinates in the REVERSED direction.

replicate.py ran the 5 hand-crafted arms forward only (easy-to-hard node ordering).
The compass was run both forward and reversed, creating an asymmetry in the comparison.
This script loads saved replicate_{dataset}.npz files, builds the reversed hand-crafted
sweeps, scores them on the same 10 held-out seeds, and writes

    replicate_{dataset}_reverse_hc.json

with the symmetric comparison table.

Usage:
    python -m experiments.path_effect.reverse_hc --datasets Cora CiteSeer PubMed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from experiments.path_effect.coords import COORD_NAMES
from experiments.path_effect.schedules import _spotlight, sinkhorn
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
T = 60
HOLD_SEEDS = tuple(range(200, 210))


def score_arm(data, W: np.ndarray) -> np.ndarray:
    return np.array([train_once(data, W, seed=s)["test_acc"] for s in HOLD_SEEDS])


def make_sweep(rank_01: np.ndarray, flat_tail: int = 30) -> np.ndarray:
    raw = _spotlight(rank_01, T, 0.15)
    if flat_tail:
        raw[T - flat_tail:] = raw.mean()
    return sinkhorn(raw)


def process_dataset(ds: str) -> None:
    npz_path = ROOT / f"replicate_{ds.lower()}.npz"
    json_path = ROOT / f"replicate_{ds.lower()}.json"
    if not npz_path.exists():
        print(f"  {ds}: no replicate npz found, skipping")
        return

    d = np.load(npz_path)
    orig = json.loads(json_path.read_text())
    flat_val = orig["flat"]
    flat_tail = orig["config"].get("flat_tail", 30)

    coords = d["coords"]          # (n_train, 5), z-scored hand-crafted
    n = coords.shape[0]

    root = f"/tmp/claude-1000/{ds.lower()}"
    data = Planetoid(root=root, name=ds)[0]

    flat_scores = score_arm(data, np.ones((T, n)))
    flat_mean = float(flat_scores.mean())

    print(f"\n{ds}  (flat tail K={flat_tail}; 10 unseen init seeds; test set)")
    print(f"  flat: {flat_mean:.4f}  (orig stored: {flat_val:.4f})")

    rows = {}
    for i, nm in enumerate(COORD_NAMES):
        fwd_rank = coords[:, i].argsort().argsort() / (n - 1)
        rev_rank = 1.0 - fwd_rank

        fwd_scores = score_arm(data, make_sweep(fwd_rank, flat_tail))
        rev_scores = score_arm(data, make_sweep(rev_rank, flat_tail))

        for tag, scores in [("forward", fwd_scores), ("reversed", rev_scores)]:
            gain = scores - flat_scores
            sem = gain.std(ddof=1) / np.sqrt(len(gain))
            wins = int((gain > 0).sum())
            key = f"hand-crafted: {nm} ({tag})"
            rows[key] = {
                "coord": nm, "direction": tag,
                "test": float(scores.mean()),
                "gain": float(gain.mean()),
                "sem": float(sem),
                "wins": wins,
                "paired_t": float(gain.mean() / sem) if sem > 0 else 0.0,
                "per_seed": scores.tolist(),
            }
            sign = "+" if gain.mean() >= 0 else ""
            print(f"  {key:45s}  {scores.mean():.4f}  {sign}{gain.mean():.4f}  "
                  f"sem {sem:.4f}  {wins}/10")

        fwd_rev_gap = float((fwd_scores - rev_scores).mean())
        print(f"    gap (fwd - rev): {fwd_rev_gap:+.4f}")

    out = {
        "dataset": ds,
        "flat": flat_mean,
        "flat_per_seed": flat_scores.tolist(),
        "arms": rows,
        "flat_tail": flat_tail,
        "hold_seeds": list(HOLD_SEEDS),
    }
    out_path = ROOT / f"replicate_{ds.lower()}_reverse_hc.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"  wrote {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    a = ap.parse_args()
    torch.set_num_threads(1)
    for ds in a.datasets:
        process_dataset(ds)


if __name__ == "__main__":
    main()
