"""Sweep: train Cora GCN under every schedule x every seed, write one results table.

Usage:  python -m experiments.path_effect.run [--epochs 60] [--seeds 5] [--random 40]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from experiments.path_effect.coords import COORD_NAMES, node_coords
from experiments.path_effect.schedules import build_schedules, descriptors
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--random", type=int, default=40)
    p.add_argument("--data-root", default="/tmp/claude-1000/cora")
    args = p.parse_args()

    torch.set_num_threads(4)
    data = Planetoid(root=args.data_root, name="Cora")[0]
    coords_all = node_coords(data)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    coords = coords_all[train_idx]  # schedules only weight training nodes
    print(f"Cora: {data.num_nodes} nodes, {int(data.train_mask.sum())} train")

    scheds = build_schedules(coords, t=args.epochs, n_random=args.random, seed=0)
    print(f"{len(scheds)} schedules x {args.seeds} seeds = {len(scheds) * args.seeds} runs")

    rows = []
    t0 = time.time()
    for i, s in enumerate(scheds):
        for seed in range(args.seeds):
            m = train_once(data, s.weights, seed=seed)
            rows.append(
                {"schedule": s.name, "family": s.family, "seed": seed, **m}
            )
        accs = [r["test_acc"] for r in rows[-args.seeds :]]
        print(
            f"[{i + 1:3d}/{len(scheds)}] {s.name:34s} "
            f"test {np.mean(accs):.4f} +- {np.std(accs):.4f}  ({time.time() - t0:.0f}s)"
        )

    desc = {s.name: descriptors(s.weights, coords).tolist() for s in scheds}
    OUT.write_text(
        json.dumps(
            {
                "config": vars(args),
                "coord_names": list(COORD_NAMES),
                "rows": rows,
                "descriptors": desc,
                # the scrambled-coordinate control: descriptors recomputed with node
                # coordinates permuted, so the schedule is identical but the coupling
                # to node identity is destroyed. Anything the real descriptors predict
                # that these do not is genuine coordinate signal.
                "control_descriptors": [
                    {
                        s.name: descriptors(
                            s.weights, coords[np.random.default_rng(100 + k).permutation(len(coords))]
                        ).tolist()
                        for s in scheds
                    }
                    for k in range(20)
                ],
            },
            indent=1,
        )
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
