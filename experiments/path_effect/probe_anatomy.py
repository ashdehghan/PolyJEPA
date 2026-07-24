"""Probe-space anatomy: is the timing compass ANY function of probe position?

Every probe-vs-beta test so far was linear (ridge, Pearson). A cluster-shaped or
lobed relationship would produce exactly those nulls while still being real. This
script takes the nonparametric look:

  1. The anatomy map — PCA of the train-node fingerprint R^6 with probe-loading
     arrows, colored by beta / |beta| / GIS difficulty / degree, plus an overlay
     of which nodes each selection strategy picked (FPS rim vs ProbCover core),
     and a k-means panel with the eta^2(beta | clusters) statistic.
  2. The measurement matrix — three spaces (probe R^6, hand-crafted stats, SGC
     features) x three targets (beta, |beta|, sign(beta)), each tested with
     leave-one-out kNN prediction against a permutation null. This answers
     "is beta any function of this space?" — linear or not.

Everything runs from cached artifacts (fingerprint/replicate/gis/selection files);
no training. PubMed is skipped (no fingerprint cache).

Usage:
    .venv/bin/python -m experiments.path_effect.probe_anatomy --datasets Cora CiteSeer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import KMeans

from experiments.path_effect.coords import COORD_NAMES
from experiments.path_effect.selection import _load_R_train, _load_data, _sgc

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 10,
})
INK, GAIN, LOSS, MUT = "#22303b", "#2a7d5f", "#b23b3b", "#9aa7b0"

N_PERM = 500
KNN_K = 10
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Nonparametric predictability: LOO kNN against a permutation null
# ---------------------------------------------------------------------------

def _loo_knn_predict(D: np.ndarray, y: np.ndarray, k: int, classify: bool) -> np.ndarray:
    """Leave-one-out kNN prediction from a precomputed distance matrix."""
    n = len(y)
    Dx = D.copy()
    np.fill_diagonal(Dx, np.inf)
    nn = np.argsort(Dx, axis=1)[:, :k]                     # (n, k) neighbor ids
    if not classify:
        return y[nn].mean(axis=1)
    pred = np.empty(n)
    for i in range(n):
        vals, counts = np.unique(y[nn[i]], return_counts=True)
        pred[i] = vals[counts.argmax()]
    return pred


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def knn_test(X: np.ndarray, y: np.ndarray, classify: bool = False,
             k: int = KNN_K, n_perm: int = N_PERM) -> dict:
    """LOO kNN predictability of y from X, with a permutation null.

    Returns the observed score (R^2, or accuracy minus base rate for
    classification), the null's 95th percentile, and a one-sided p-value.
    """
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)

    def score(yy: np.ndarray) -> float:
        pred = _loo_knn_predict(D, yy, k, classify)
        if classify:
            base = np.bincount(yy.astype(int) + 1).max() / len(yy)
            return float((pred == yy).mean() - base)
        return _r2(yy, pred)

    obs = score(y)
    null = np.array([score(RNG.permutation(y)) for _ in range(n_perm)])
    return {"observed": obs, "null_p95": float(np.quantile(null, 0.95)),
            "p_value": float((null >= obs).mean()), "k": k, "n_perm": n_perm}


def eta2_test(labels: np.ndarray, y: np.ndarray, n_perm: int = N_PERM) -> dict:
    """Fraction of y's variance explained by cluster membership, permutation-tested."""
    def eta2(yy: np.ndarray) -> float:
        grand = yy.mean()
        ss_between = sum(len(yy[labels == c]) * (yy[labels == c].mean() - grand) ** 2
                         for c in np.unique(labels))
        return float(ss_between / ((yy - grand) ** 2).sum())

    obs = eta2(y)
    null = np.array([eta2(RNG.permutation(y)) for _ in range(n_perm)])
    return {"eta2": obs, "null_p95": float(np.quantile(null, 0.95)),
            "p_value": float((null >= obs).mean())}


# ---------------------------------------------------------------------------
# Per-dataset anatomy
# ---------------------------------------------------------------------------

def _load_all(ds: str):
    data = _load_data(ds)
    train_global = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    R_z = _load_R_train(data, ds)
    rep = np.load(ROOT / f"replicate_{ds.lower()}.npz")
    beta = rep["beta"].astype(np.float64)
    coords = rep["coords"].astype(np.float64)              # (n_train, 5), z-scored
    gis = np.array(json.loads((ROOT / f"gis_{ds.lower()}.json").read_text())
                   ["results"]["gis_loss"]["gis"])
    E_train = _sgc(data)[train_global]
    sel = json.loads((ROOT / f"selection_{ds.lower()}.json").read_text())["arms"]
    g2l = {g: i for i, g in enumerate(train_global.tolist())}
    picks = {arm: np.array([g2l[g] for g in sel[arm]["10"]["selected_global_ids"]])
             for arm in ("fps_probe", "ccs_pc1", "probcover_r1") if arm in sel}
    return data, R_z, beta, coords, gis, E_train, picks


def _scatter(ax, Z, c, cmap, title, norm=None):
    s = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap, norm=norm, s=22,
                   edgecolors="white", linewidths=0.4)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(s, ax=ax, fraction=0.046, pad=0.03)


def fig_anatomy(ds: str, R_z, beta, gis, coords, picks, probe_names) -> dict:
    # PCA layout of probe space, shared across all panels
    Rc = R_z - R_z.mean(0)
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    Z = Rc @ Vt[:2].T
    evr = (S ** 2 / (S ** 2).sum())[:2]

    km = KMeans(n_clusters=6, random_state=0, n_init=10).fit(R_z)
    eta_beta = eta2_test(km.labels_, beta)
    eta_abs = eta2_test(km.labels_, np.abs(beta))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    b = beta * 1000                                        # per-mil scale for legibility
    lim = np.abs(b).max()
    _scatter(axes[0, 0], Z, b, "RdBu_r", r"$\beta$ (train late $\to$ red)  [$\times 10^3$]",
             norm=TwoSlopeNorm(0.0, -lim, lim))
    # probe-loading arrows on the beta panel, scaled to stay inside the cloud
    loading_norms = np.linalg.norm(Vt[:2], axis=0)
    scale = 0.55 * min(np.abs(Z[:, 0]).max(), np.abs(Z[:, 1]).max()) / loading_norms.max()
    for j, name in enumerate(probe_names):
        ax, ay = Vt[0, j] * scale, Vt[1, j] * scale
        axes[0, 0].annotate("", xy=(ax, ay), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color=INK, lw=0.9, alpha=0.6))
        axes[0, 0].text(ax * 1.15, ay * 1.15, name[0], fontsize=7.5, color=INK,
                        ha="center", va="center", clip_on=True)
    _scatter(axes[0, 1], Z, np.abs(b), "viridis", r"$|\beta|$ — timing sensitivity")
    _scatter(axes[0, 2], Z, gis, "viridis", "GIS early loss (difficulty)")
    _scatter(axes[1, 0], Z, coords[:, 0], "viridis", "degree (z)")

    ax = axes[1, 1]
    ax.scatter(Z[:, 0], Z[:, 1], c=MUT, s=16, alpha=0.5, edgecolors="none",
               label="train pool")
    style = {"fps_probe": (LOSS, "s", "FPS probe"),
             "probcover_r1": (GAIN, "o", "ProbCover $q_{25}$"),
             "ccs_pc1": (INK, "D", "CCS PC1")}
    for arm, idx in picks.items():
        c, m, lbl = style[arm]
        ax.scatter(Z[idx, 0], Z[idx, 1], facecolors="none", edgecolors=c, marker=m,
                   s=90, linewidths=1.6, label=lbl)
    ax.set_title("who selects whom (10% budget)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, loc="best", handletextpad=0.2)

    ax = axes[1, 2]
    ax.scatter(Z[:, 0], Z[:, 1], c=km.labels_, cmap="tab10", s=22,
               edgecolors="white", linewidths=0.4)
    for c in range(6):
        zc = Z[km.labels_ == c].mean(axis=0)
        ax.text(zc[0], zc[1], f"{beta[km.labels_ == c].mean() * 1000:+.1f}",
                fontsize=8, ha="center", va="center", color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(rf"k-means (k=6), cluster-mean $\beta \times 10^3$"
                 + "\n" + rf"$\eta^2$={eta_beta['eta2']:.2f} (p={eta_beta['p_value']:.2f})")
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{ds}: probe-space anatomy  (PC1 {evr[0]:.0%}, PC2 {evr[1]:.0%} var)",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_anatomy_{ds.lower()}.pdf")
    fig.savefig(OUT / f"fig_anatomy_{ds.lower()}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote fig_anatomy_{ds.lower()}")
    return {"eta2_beta_k6": eta_beta, "eta2_absbeta_k6": eta_abs,
            "pca_explained_var": evr.tolist()}


def fig_stats(all_stats: dict) -> None:
    """The measurement matrix: 3 spaces x 3 targets x datasets, LOO-kNN vs null."""
    targets = [("beta", r"$\beta$  (LOO kNN $R^2$)"),
               ("absbeta", r"$|\beta|$  (LOO kNN $R^2$)"),
               ("sign", r"sign($\beta$)  (acc $-$ base)")]
    spaces = [("probe", "probe $\\R^6$".replace("\\R", "\\mathbb{R}")),
              ("handcrafted", "hand-crafted"), ("sgc", "SGC")]
    ds_marks = {"Cora": "o", "CiteSeer": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    ypos = np.arange(len(spaces))[::-1]
    for ax, (tkey, tlabel) in zip(axes, targets):
        for yi, (skey, slabel) in zip(ypos, spaces):
            for ds, stats in all_stats.items():
                r = stats["knn"][skey][tkey]
                off = 0.16 if ds == "Cora" else -0.16
                ax.plot([r["null_p95"]], [yi + off], "|", color=MUT, ms=14, mew=1.6)
                sig = r["p_value"] < 0.05
                ax.plot([r["observed"]], [yi + off], ds_marks[ds], ms=6.5,
                        color=(GAIN if sig else INK),
                        mfc=(GAIN if sig else "white"), mew=1.3)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_title(tlabel, fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([s for _, s in spaces])
    handles = [plt.Line2D([], [], marker=m, ls="", color=INK, mfc="white", label=ds)
               for ds, m in ds_marks.items()]
    handles += [plt.Line2D([], [], marker="|", ls="", color=MUT, ms=12,
                           label="perm. null 95%"),
                plt.Line2D([], [], marker="o", ls="", color=GAIN, label="p<0.05")]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT / "fig_anatomy_stats.pdf")
    fig.savefig(OUT / "fig_anatomy_stats.png", dpi=150)
    plt.close(fig)
    print("  wrote fig_anatomy_stats")


def process_dataset(ds: str) -> dict | None:
    if not (ROOT / f"fingerprint_{ds.lower()}.npz").exists():
        print(f"{ds}: no fingerprint cache — skipped")
        return None
    print(f"\n{'=' * 60}\n{ds}\n{'=' * 60}")
    data, R_z, beta, coords, gis, E_train, picks = _load_all(ds)
    probe_names = [str(x) for x in
                   np.load(ROOT / f"fingerprint_{ds.lower()}.npz",
                           allow_pickle=True)["probe_names"]]

    spaces = {"probe": R_z, "handcrafted": coords, "sgc": E_train}
    targets = {"beta": (beta, False), "absbeta": (np.abs(beta), False),
               "sign": (np.sign(beta), True)}
    knn = {}
    print(f"  {'space':13s} {'target':8s} {'observed':>9} {'null95':>8} {'p':>7}")
    for skey, X in spaces.items():
        knn[skey] = {}
        for tkey, (y, classify) in targets.items():
            r = knn_test(X, y, classify=classify)
            knn[skey][tkey] = r
            star = " *" if r["p_value"] < 0.05 else ""
            print(f"  {skey:13s} {tkey:8s} {r['observed']:+9.3f} "
                  f"{r['null_p95']:8.3f} {r['p_value']:7.3f}{star}")

    # |beta| vs classical stats — the "hubs are timing-sensitive" hint, tested
    from scipy import stats as sps
    absbeta_corr = {}
    for j, name in enumerate(COORD_NAMES):
        r, pr = sps.pearsonr(np.abs(beta), coords[:, j])
        rho, ps = sps.spearmanr(np.abs(beta), coords[:, j])
        absbeta_corr[name] = {"pearson_r": float(r), "p": float(pr),
                              "spearman_rho": float(rho), "p_rho": float(ps)}
        if pr < 0.1:
            print(f"  |beta| vs {name}: r={r:+.3f} (p={pr:.3f}), "
                  f"rho={rho:+.3f} (p={ps:.3f})")

    fig_extra = fig_anatomy(ds, R_z, beta, gis, coords, picks, probe_names)
    out = {"dataset": ds, "n_train": len(beta), "knn": knn,
           "absbeta_vs_coords": absbeta_corr, **fig_extra,
           "config": {"knn_k": KNN_K, "n_perm": N_PERM, "kmeans_k": 6}}
    (ROOT / f"probe_anatomy_{ds.lower()}.json").write_text(json.dumps(out, indent=1))
    print(f"  wrote probe_anatomy_{ds.lower()}.json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    a = ap.parse_args()
    all_stats = {}
    for ds in a.datasets:
        r = process_dataset(ds)
        if r is not None:
            all_stats[ds] = r
    if all_stats:
        fig_stats(all_stats)


if __name__ == "__main__":
    main()
