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


if __name__ == "__main__":
    fig_landscape()
    fig_replication()
    fig_gap()
    fig_mechanism()
    fig_selection()
    print("all figures in", OUT)
