"""Generate the documentation figures from real PolyJEPA runs.

Deterministic (seeded). Each probe's residual is correlated against interpretable,
computable node statistics, so the figures show *what each probe measures* and
that the probes are distinct, rather than just decorating the page.

Produces, into ``docs/assets/generated/``:
  - ``signature_synth.png`` / ``signature_cora.png``: the probe x property
    signature heatmap (rows = probes, columns = node statistics, cells =
    Spearman correlation between the probe's residual and the statistic);
  - ``scatter_<P>.png``: per probe, residual vs the statistic it tracks most
    strongly, with the Spearman value;
  - ``cross_probe.png`` / ``cross_probe_cora.png``: the cross-probe Spearman
    matrix (distinctness among probes).

A high residual means a node is *unpredictable / surprising* under that probe's
objective. Figures use the colorblind-safe ``cividis`` colormap (also legible in
grayscale); the signature/correlation maps add the numeric value in every cell so
the sign is unambiguous.

Run from the project root::

    PYTHONPATH=src python docs/scripts/generate_figures.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from torch_geometric.datasets import Planetoid

from polyjepa import fingerprint
from polyjepa.synthetic import inject_feature_outliers, planted_sbm

OUT = pathlib.Path(__file__).resolve().parents[1] / "assets" / "generated"
SEQ = "cividis"
DIVERGING = "coolwarm"
SEED = 0
EPOCHS = 150

PROBE_TITLES = {
    "A": "A · recover-focal",
    "B": "B · pooled-neighborhood",
    "C": "C · sampled-neighbor",
    "D": "D · two-hop-ring",
    "E": "E · augmented-view",
    "F": "F · community-sibling",
}
STAT_LABELS = {
    "degree": "degree",
    "betweenness": "betweenness",
    "clustering": "clustering",
    "kcore": "k-core",
    "homophily": "label homophily",
    "is_bridge": "planted bridge",
    "is_outlier": "planted outlier",
}


def to_nx(data) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(int(data.num_nodes)))
    g.add_edges_from(data.edge_index.t().tolist())
    return g


def node_stats(data, g: nx.Graph, planted: bool) -> dict[str, np.ndarray]:
    n = int(data.num_nodes)
    deg = np.array([g.degree(i) for i in range(n)], dtype=float)
    k = None if n <= 300 else 500
    bet = nx.betweenness_centrality(g, k=k, seed=SEED, normalized=True)
    clus = nx.clustering(g)
    core = nx.core_number(g)
    y = data.y.cpu().numpy() if getattr(data, "y", None) is not None else None
    homo = np.full(n, np.nan)
    if y is not None:
        for v in range(n):
            nbrs = list(g.neighbors(v))
            if nbrs:
                homo[v] = np.mean([1.0 if y[u] == y[v] else 0.0 for u in nbrs])
    stats = {
        "degree": deg,
        "betweenness": np.array([bet[i] for i in range(n)]),
        "clustering": np.array([clus[i] for i in range(n)]),
        "kcore": np.array([core[i] for i in range(n)], dtype=float),
        "homophily": homo,
    }
    if planted:
        stats["is_bridge"] = data.bridge_mask.cpu().numpy().astype(float)
        stats["is_outlier"] = data.outlier_mask.cpu().numpy().astype(float)
    return stats


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.unique(b[ok]).size < 2 or np.unique(a[ok]).size < 2:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def signature_heatmap(residuals, names, stats, title, outname):
    cols = list(stats.keys())
    mat = np.full((len(names), len(cols)), np.nan)
    for i in range(len(names)):
        r = residuals[:, i].numpy()
        for j, c in enumerate(cols):
            mat[i, j] = _spearman(r, stats[c])

    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 1.6, 0.55 * len(names) + 1.6))
    im = ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([STAT_LABELS[c] for c in cols], rotation=30, ha="right",
                       fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([PROBE_TITLES[n] for n in names], fontsize=8)
    for i in range(len(names)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(v) > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman(residual, statistic)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"What each probe tracks ({title})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / outname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cols, mat


CONTINUOUS = ("degree", "betweenness", "clustering", "kcore", "homophily")


def per_probe_scatter(name, residual, stats):
    r = residual.numpy()
    # pick the continuous statistic this probe tracks most strongly (binary
    # planted columns are shown in the signature heatmap, not as a scatter)
    best, best_rho = None, 0.0
    for c in CONTINUOUS:
        rho = _spearman(r, stats[c])
        if np.isfinite(rho) and abs(rho) > abs(best_rho):
            best, best_rho = c, rho
    if best is None:
        best, best_rho = "degree", _spearman(r, stats["degree"])

    s = stats[best]
    ok = np.isfinite(r) & np.isfinite(s)
    # Plot the residual percentile, not the raw value: only ranks are meaningful,
    # and a single extreme residual would otherwise squash a linear axis.
    pct = rankdata(r[ok]) / ok.sum()
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    sc = ax.scatter(s[ok], pct, s=26, c=pct, cmap=SEQ, edgecolors="white",
                    linewidths=0.3, alpha=0.9)
    ax.set_xlabel(STAT_LABELS[best], fontsize=9)
    ax.set_ylabel("residual percentile (higher = less predictable)", fontsize=8)
    ax.set_title(f"{PROBE_TITLES[name]}\nresidual vs {STAT_LABELS[best]} "
                 f"(Spearman {best_rho:+.2f})", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"scatter_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best, best_rho


def cross_probe_heatmap(matrix, names, outname, title):
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    m = matrix.numpy()
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = m[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / outname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(data, planted, tag, per_probe):
    g = to_nx(data)
    stats = node_stats(data, g, planted=planted)
    fp = fingerprint(data, epochs=EPOCHS, hidden_dim=64, latent_dim=64, seed=SEED)
    signature_heatmap(fp.residuals, fp.probe_names, stats, tag,
                      f"signature_{tag}.png")
    cross_probe_heatmap(fp.cross_probe, fp.probe_names,
                        f"cross_probe{'' if tag == 'synth' else '_cora'}.png",
                        f"Cross-probe correlation ({tag})")
    if per_probe:
        for k, name in enumerate(fp.probe_names):
            best, rho = per_probe_scatter(name, fp.residuals[:, k], stats)
            print(f"  {name}: strongest vs {best} (rho {rho:+.2f})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    print("synthetic graph (planted ground truth)...")
    data = inject_feature_outliers(
        planted_sbm(num_communities=3, nodes_per_comm=26, p_in=0.18, p_out=0.015,
                    num_bridges=6, bridge_degree=5, num_features=48, seed=SEED),
        frac=0.08, scale=8.0, seed=SEED + 1,
    )
    run(data, planted=True, tag="synth", per_probe=True)

    print("Cora (real graph, computed statistics)...")
    cora = Planetoid(root="datasets", name="Cora")[0]
    run(cora, planted=False, tag="cora", per_probe=False)

    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
