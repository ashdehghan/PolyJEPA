"""End-to-end proof of concept: fingerprint the Cora citation graph.

Runs all six probes on Cora and prints a summary of the resulting fingerprint:
per-probe coverage and residual statistics, collapse diagnostics, the cross-probe
Spearman matrix (to confirm the probes are differentiated, not redundant), and
embedding-bank shapes.

This is a single-dataset PoC, not an experiment campaign. Usage::

    PYTHONPATH=src python examples/fingerprint_cora.py --epochs 200

The Planetoid root defaults to ``datasets`` under this project; pass ``--root``
to point at an existing download and avoid network access.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from polyjepa import fingerprint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets", help="Planetoid root dir")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Planetoid(root=args.root, name="Cora")[0]
    print(
        f"Cora: N={data.num_nodes} E={data.edge_index.shape[1]} "
        f"F={data.x.shape[1]}\n"
    )

    fp = fingerprint(
        data,
        epochs=args.epochs,
        hidden_dim=args.hidden,
        latent_dim=args.latent,
        seed=args.seed,
    )

    print("Per-probe summary")
    print(f"  {'probe':<6}{'scored':>9}{'resid mean':>14}{'resid std':>12}"
          f"{'emb std':>10}{'healthy':>9}")
    for name in fp.probe_names:
        r = fp.residual(name)
        fin = torch.isfinite(r)
        d = fp.diagnostics[name]
        print(
            f"  {name:<6}{int(fin.sum()):>6}/{data.num_nodes:<3}"
            f"{r[fin].mean():>13.3f}{r[fin].std():>12.3f}"
            f"{d.embed_std[-1]:>10.3f}{str(d.healthy()):>9}"
        )

    print(f"\n  embedding bank: 6 probes x ({data.num_nodes}, {args.latent})")
    print(f"  all diagnostics healthy: {fp.healthy()}")

    np.set_printoptions(precision=2, suppress=True)
    print("\nCross-probe Spearman (residual rankings)")
    print("  order:", fp.probe_names)
    print(fp.cross_probe.numpy())

    offdiag = fp.cross_probe.clone()
    offdiag.fill_diagonal_(float("nan"))
    vals = offdiag.numpy()
    vals = vals[np.isfinite(vals)]
    print(
        f"\n  off-diagonal |rho|: max={np.abs(vals).max():.2f} "
        f"mean={np.abs(vals).mean():.2f}  "
        f"(low values => probes measure different properties)"
    )


if __name__ == "__main__":
    main()
