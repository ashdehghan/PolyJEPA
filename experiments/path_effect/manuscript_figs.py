"""Figures for the manuscript results section. All numbers come from captured data;
nothing is recomputed by training. Writes PDF+PNG into the manuscript figures/ staging dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 10,
})
INK, GAIN, LOSS, MUT = "#22303b", "#2a7d5f", "#b23b3b", "#9aa7b0"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1: the landscape
# 8000 marginal-matched schedules on Cora: timing alone moves accuracy, and val picks test.
def fig_landscape():
    d = np.load(HERE / "capture.npz")
    val, test = d["val"], d["test"]
    flat = json.loads((HERE / "capture.json").read_text()).get("flat_test")
    if flat is None:
        flat = 0.7894  # measured flat baseline, 10 seeds
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax[0].hist(test, bins=60, color=MUT, alpha=0.85)
    ax[0].axvline(flat, color=INK, lw=1.6, ls="--", label=f"i.i.d. baseline {flat:.3f}")
    ax[0].axvline(test.max(), color=GAIN, lw=1.6, label=f"best schedule {test.max():.3f}")
    ax[0].set_xlabel("test accuracy"); ax[0].set_ylabel("# schedules")
    ax[0].set_title("(a) 8,000 schedules, identical per-node budget")
    ax[0].legend(frameon=False, fontsize=7.5)

    r = np.corrcoef(val, test)[0, 1]
    ax[1].scatter(val, test, s=4, alpha=0.25, color=INK, edgecolors="none")
    b, a = np.polyfit(val, test, 1)
    xs = np.array([val.min(), val.max()])
    ax[1].plot(xs, a + b * xs, color=GAIN, lw=1.6)
    ax[1].set_xlabel("validation accuracy"); ax[1].set_ylabel("test accuracy")
    ax[1].set_title(f"(b) selection transfers  (r = {r:.2f})")
    save(fig, "fig_landscape")


# ------------------------------------------------- fig 2: replication across three datasets
# The honest headline. Forward vs reversed vs best hand-crafted, as gain over i.i.d.
def fig_replication():
    sets = ["cora", "citeseer", "pubmed"]
    names = ["Cora", "CiteSeer", "PubMed"]
    arms = ["compass (learned)", "compass REVERSED"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(sets)); w = 0.26
    fwd, rev, fsem, rsem, hc, hcname = [], [], [], [], [], []
    for s in sets:
        j = json.loads((HERE / f"replicate_{s}.json").read_text())["arms"]
        fwd.append(j[arms[0]]["gain"] * 100); fsem.append(j[arms[0]]["sem"] * 100)
        rev.append(j[arms[1]]["gain"] * 100); rsem.append(j[arms[1]]["sem"] * 100)
        best = max((k for k in j if k.startswith("hand-crafted")),
                   key=lambda k: j[k]["gain"])
        hc.append(j[best]["gain"] * 100); hcname.append(best.split(": ")[1])
    ax.bar(x - w, fwd, w, yerr=fsem, color=GAIN, label="compass forward", capsize=2)
    ax.bar(x, rev, w, yerr=rsem, color=LOSS, label="compass reversed", capsize=2)
    ax.bar(x + w, hc, w, color=MUT, label="best hand-crafted coord")
    for xi, (h, nm) in enumerate(zip(hc, hcname)):
        ax.text(xi + w, h + (0.1 if h >= 0 else -0.35), nm, ha="center", fontsize=6.5, color=INK)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("accuracy gain over i.i.d. (points)")
    ax.set_title("Path effect replicates in direction; exploitability decays with graph size")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    save(fig, "fig_replication")


# --------------------------------------------------- fig 3: the forward-minus-reverse gap
# The one thing robust across all three: order carries signed information everywhere.
def fig_gap():
    sets = ["cora", "citeseer", "pubmed"]; names = ["Cora", "CiteSeer", "PubMed"]
    gaps = []
    for s in sets:
        j = json.loads((HERE / f"replicate_{s}.json").read_text())["arms"]
        gaps.append((j["compass (learned)"]["gain"] - j["compass REVERSED"]["gain"]) * 100)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.bar(names, gaps, color=INK, width=0.6)
    for i, g in enumerate(gaps):
        ax.text(i, g + 0.1, f"{g:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("forward $-$ reversed (points)")
    ax.set_title("Signed path effect,\nsame direction on every dataset")
    save(fig, "fig_gap")


# -------------------------------------- fig 4: mechanism — the gain is not recency
# Neutralise the last K epochs. Reversal collapse is recency; forward gain survives and grows.
def fig_mechanism():
    K = [0, 5, 10, 20, 30]
    fwd = [1.31, 3.29, 1.82, 2.99, 3.05]      # from mechanism.py, Cora, gain over flat (points)
    rev = [-12.80, -5.80, -2.93, -3.04, -1.70]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.plot(K, fwd, "o-", color=GAIN, label="compass forward")
    ax.plot(K, rev, "s-", color=LOSS, label="compass reversed")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xlabel("last $K$ of 60 epochs forced flat")
    ax.set_ylabel("gain over i.i.d. (points)")
    ax.set_title("Closing the recency channel (Cora)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_mechanism")


# ------------------------------------------ fig 5: E3 selection — coverage transfers
# Gain over the random arm at each label budget. Identity is carried by direct labels,
# markers and linestyle, not color alone (the green/red pair is CVD-tight).
def fig_selection():
    sets = ["cora", "citeseer", "pubmed"]
    names = ["Cora", "CiteSeer", "PubMed"]
    # Computers is excluded from the main figure: its flat-100% baseline (0.368) fails
    # the pre-registered sanity gate (published GCN ~0.80), so arm contrasts there are
    # optimization noise. Its table lives in the appendix with that caveat.
    for extra, label in (("photo", "Amazon Photo"),):
        if (HERE / f"selection_{extra}.json").exists():
            sets.append(extra)
            names.append(label)
    arms = [  # key, label, color, linestyle, marker
        ("probcover_r1", "ProbCover ($q_{25}$)", GAIN, "-", "o"),
        ("easy_strat", "easy-first (strat.)", GAIN, "--", "D"),
        ("fps_probe", "FPS probe space", INK, "-", "s"),
        ("ccs_pc1", "CCS on PC1", MUT, "-", "^"),
        ("fps_sgc", "FPS SGC space", MUT, "--", "v"),
        ("hard_strat", "hard-first (strat.)", LOSS, "--", "d"),
    ]
    ncols = 3
    nrows = (len(sets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.9 * nrows),
                             sharey=True, squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes[len(sets):]:
        ax.set_visible(False)
    for ax, s, nm in zip(flat_axes, sets, names):
        j = json.loads((HERE / f"selection_{s}.json").read_text())["arms"]
        ks = sorted(int(k) for k in j["random"])
        for key, label, color, ls, mk in arms:
            if key not in j:
                continue
            g = [j[key][str(k)]["gain_vs_random"] * 100 for k in ks]
            ax.plot(ks, g, ls, marker=mk, ms=3.5, lw=1.4, color=color,
                    label=label if s == "cora" else None)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xscale("log")
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("label budget (%)")
        ax.set_title(nm)
        if s == "pubmed":
            ax.text(0.97, 0.04, "fingerprint arms deferred", transform=ax.transAxes,
                    ha="right", fontsize=6.5, color=MUT, style="italic")
    for row in axes:
        row[0].set_ylabel("gain over random\nselection (points)")
    fig.legend(frameon=False, fontsize=7, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(hspace=0.55)
    save(fig, "fig_selection")


# ------------------------------- fig 6: JEPA architecture -------------------------------
def fig_jepa():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 11); ax.set_ylim(-0.4, 6); ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f4f5", ec=INK, fs=8.5, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, style="-", color=INK, text=None, ty=0.25):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, ls=style))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha="center", fontsize=7.5,
                    color=color)

    box(0.2, 2.3, 1.5, 1.4, "graph\n$(X, E)$", fc="white", bold=True)
    # context branch (top)
    box(2.4, 4.1, 1.9, 1.2, "context view\n(targets masked)")
    box(4.9, 4.1, 1.6, 1.2, "encoder\n$f_\\theta$")
    box(7.0, 4.1, 1.5, 1.2, "predictor\n$g_\\phi$")
    # target branch (bottom)
    box(2.4, 0.6, 1.9, 1.2, "target view\n(clean graph)")
    box(4.9, 0.6, 1.6, 1.2, "target encoder\n$f_{\\bar\\theta}$")
    box(7.0, 0.6, 1.5, 1.2, "mean-pool\ntargets")
    box(8.9, 2.3, 1.0, 1.4, "$r_i$", fc="#eae6f7", bold=True)

    arrow(1.7, 3.4, 2.4, 4.5); arrow(1.7, 2.6, 2.4, 1.4)
    arrow(4.3, 4.7, 4.9, 4.7); arrow(6.5, 4.7, 7.0, 4.7)
    ax.text(6.75, 5.55, "read at focal $i$", ha="center", fontsize=7.5, color=INK)
    arrow(4.3, 1.2, 4.9, 1.2); arrow(6.5, 1.2, 7.0, 1.2)
    arrow(8.5, 4.5, 9.1, 3.7)
    arrow(8.5, 1.3, 9.1, 2.3)
    # EMA + stop-grad annotations
    arrow(5.7, 4.1, 5.7, 1.8, style="--", color=MUT)
    ax.text(5.95, 2.9, "EMA copy\n(no gradients)", fontsize=7.5, color=MUT)
    ax.text(9.4, -0.15, "$r_i$: squared distance between\nprediction and pooled target",
            ha="center", fontsize=7.5, color=INK)
    ax.set_title("One probe evaluation: the pair design chooses the views and the targets;"
                 " everything else is fixed", fontsize=9)
    save(fig, "fig_jepa")


# ------------------------------- fig 7: the six probes -------------------------------
def fig_probes():
    pos = {0: (0.0, 0.5), 1: (0.9, 1.2), 2: (0.9, -0.2), 3: (1.0, 0.5),
           4: (1.9, 1.3), 5: (1.9, -0.3), 6: (3.0, 0.5), 7: (3.8, 1.1),
           8: (3.8, -0.1), 9: (4.6, 0.9), 10: (4.9, 0.2), 11: (4.3, -0.6)}
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 5), (3, 6), (4, 5),
             (6, 7), (6, 8), (7, 9), (8, 9), (9, 10), (8, 11), (10, 11)]
    panels = [
        ("A: recover the focal node", {0}, "the node itself, from its surroundings"),
        ("B: pooled 1-hop neighborhood", {1, 2, 3}, "the average of all neighbors"),
        ("C: one sampled neighbor", {2}, "a single neighbor, drawn per pass"),
        ("D: pooled two-hop ring", {4, 5, 6}, "the average of nodes at distance 2"),
        ("E: across two augmentations", set(), "the node itself, across two noisy views"),
        ("F: community sibling", {5}, "a same-community non-neighbor"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0))
    for ax, (title, targets, sub) in zip(axes.ravel(), panels):
        dashed = {(1, 2), (2, 5), (8, 9)} if title.startswith("E") else set()
        for u, v in edges:
            ls = "--" if (u, v) in dashed else "-"
            col = LOSS if (u, v) in dashed else MUT
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], ls, color=col,
                    lw=1.1, zorder=1, alpha=0.45 if ls == "--" else 0.9)
        if title.startswith("F"):
            for cx, cy, w, h in [(1.1, 0.5, 3.1, 2.8), (3.9, 0.2, 3.0, 2.6)]:
                ax.add_patch(plt.matplotlib.patches.Ellipse(
                    (cx, cy), w, h, fill=False, ls=":", ec=INK, lw=0.9, alpha=0.5))
        for n, (x, y) in pos.items():
            if n == 0:
                fc, ec_, lw = "white", INK, 2.2
            elif n in targets:
                fc, ec_, lw = GAIN, LOSS, 1.6
            else:
                fc, ec_, lw = "#d6dbde", MUT, 0.8
            ax.scatter([x], [y], s=210, facecolor=fc, edgecolor=ec_, linewidth=lw,
                       zorder=3, linestyle="--" if n in targets else "-")
        ax.set_title(title, fontsize=9.5)
        ax.text(2.45, -1.35, sub, ha="center", fontsize=8, color=INK, style="italic")
        ax.set_xlim(-0.6, 5.5); ax.set_ylim(-1.7, 1.9)
        ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=2.0, ms=10,
                   label="focal node $i$ (prediction read here)"),
        plt.Line2D([], [], marker="o", ls="", mfc=GAIN, mec=LOSS, mew=1.5, ms=10,
                   label="target (masked in the context view)"),
        plt.Line2D([], [], marker="o", ls="", mfc="#d6dbde", mec=MUT, ms=10,
                   label="visible context"),
        plt.Line2D([], [], ls="--", color=LOSS, label="dropped edge (E only)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.05))
    save(fig, "fig_probes")


if __name__ == "__main__":
    fig_landscape()
    fig_replication()
    fig_gap()
    fig_mechanism()
    fig_selection()
    print("all figures in", OUT)


# ------------------------------------------------------------- fig 9: learned schedule
# E5: the gradient-learned Cora schedule as a heatmap, plus the outer-loop trajectory.
def fig_learned():
    d = json.loads((HERE / "schedule_learn_cora.json").read_text())
    W = np.load(HERE / "schedule_learn_cora.npz")["W_learned"]   # (60, 140)
    T = W.shape[0]
    tt = np.linspace(0.0, 1.0, T)[:, None]
    arrival = (W * tt).sum(axis=0) / W.sum(axis=0)
    order = np.argsort(arrival)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9),
                           gridspec_kw={"width_ratios": [1.35, 1.0]})
    im = ax[0].imshow(np.log1p(W[:, order]), aspect="auto", origin="lower",
                      cmap="Greys", interpolation="nearest")
    ax[0].set_xlabel("training node (sorted by learned arrival time)")
    ax[0].set_ylabel("epoch")
    ax[0].set_title("(a) learned schedule $W$ (log scale)")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label(r"$\log(1+W_{t,i})$", fontsize=7, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(wspace=0.42)

    for r, traj in enumerate(d["trajectories"]):
        steps = [p["step"] for p in traj]
        acc = [p["val_acc"] * 100 for p in traj]
        ax[1].plot(steps, acc, color=(INK if r == 0 else GAIN), lw=1.4,
                   label=f"restart {r}")
    ax[1].axhline(d["trajectories"][0][0]["val_acc"] * 100, color=MUT, lw=1.0,
                  ls="--", label="flat start")
    ax[1].set_xlabel("outer optimization step")
    ax[1].set_ylabel("search validation accuracy (%)")
    ax[1].set_title("(b) outer-loop trajectory")
    ax[1].legend(frameon=False, fontsize=7)
    save(fig, "fig_learned")


# --------------------------------------------- fig 10/11: schedule matrix + compass pipeline
# Appendix visuals: the T x N schedule as a picture, and how the compass becomes a schedule.
def fig_schedule_matrix():
    from experiments.path_effect.schedules import _spotlight, sinkhorn
    T_, N_ = 12, 8
    order = np.linspace(0.0, 1.0, N_)
    flat = np.ones((T_, N_))
    fwd = sinkhorn(_spotlight(order, T_, 0.18))
    rev = fwd[::-1].copy()

    def arrivals(W):
        tt = np.linspace(0.0, 1.0, T_)[:, None]
        return (W * tt).sum(axis=0) / W.sum(axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    vmax = fwd.max()
    for a, W, title in zip(ax, [flat, fwd, rev],
                           ["(a) flat (i.i.d. training)", "(b) a timed schedule",
                            "(c) its time reverse"]):
        a.imshow(W, aspect="auto", origin="lower", cmap="Greys",
                 vmin=0, vmax=vmax, interpolation="nearest")
        a.scatter(np.arange(N_), arrivals(W) * (T_ - 1), s=22, color=GAIN,
                  zorder=3, label=r"arrival time $\tau_i$")
        a.set_title(title, fontsize=9)
        a.set_xlabel("training node $i$")
        a.set_xticks(range(0, N_, 2))
    ax[0].set_ylabel("epoch $t$")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "fig_schedule_matrix")


def fig_compass_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 11); ax.set_ylim(-0.3, 6); ax.axis("off")

    def box(x, y, w, h, text, fc="#f2f4f5", ec=INK, fs=8, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, color=INK, text=None, ty=0.25, fs=7):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha="center",
                    fontsize=fs, color=color)

    # fitting row (top)
    box(0.2, 4.2, 2.3, 1.4, "2,000 random\nschedules $W$\n(Sinkhorn-projected)", fc="white")
    box(3.3, 4.2, 2.3, 1.4, "arrival profiles\n$\\tau \\in \\mathbb{R}^{N}$\n(one time per node)")
    box(6.4, 4.2, 1.9, 1.4, "ridge\nregression")
    box(9.1, 4.2, 1.7, 1.4, "compass\n$\\beta \\in \\mathbb{R}^{N}$", fc="#eae6f7", bold=True)
    arrow(2.5, 4.9, 3.3, 4.9)
    arrow(5.6, 4.9, 6.4, 4.9)
    arrow(8.3, 4.9, 9.1, 4.9)
    ax.text(5.6, 3.35, "validation accuracy\nper schedule\n(2 search seeds)",
            ha="center", fontsize=7, color=MUT)
    arrow(6.1, 3.75, 6.9, 4.15, color=MUT)

    # building row (bottom)
    box(0.2, 0.6, 2.0, 1.4, "rank $\\beta$\nin $[0,1]$\n(ordering only)", fs=7.5)
    box(2.8, 0.6, 2.4, 1.4, "Gaussian spotlight\nsweeps the ordering", fs=7.5)
    box(5.8, 0.6, 1.8, 1.4, "flatten last\n30 epochs", fs=7.5)
    box(8.2, 0.6, 2.6, 1.4, "Sinkhorn $\\to$ compass\nschedule (scored on\n10 unseen seeds)",
        fc="#e7f0ec", fs=7.5, bold=True)
    arrow(9.95, 4.2, 9.95, 2.7)
    arrow(9.95, 2.7, 1.3, 2.05)
    arrow(2.2, 1.3, 2.8, 1.3)
    arrow(5.2, 1.3, 5.8, 1.3)
    arrow(7.6, 1.3, 8.2, 1.3)
    ax.set_title("The compass: fit on validation arrivals (top), then rebuilt into a schedule "
                 "(bottom)", fontsize=9)
    save(fig, "fig_compass_pipeline")
