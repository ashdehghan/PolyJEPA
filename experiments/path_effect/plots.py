"""Figures for the oracle schedule search."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 160, "axes.spines.top": False,
                     "axes.spines.right": False})

d = np.load(ROOT / "capture.npz")
j = json.loads((ROOT / "capture.json").read_text())
val, test, arr, scale = d["val"], d["test"], d["arrival"].astype(np.float64), d["scale"]
flat = np.mean(j["flat_holdout"]["holdout_test_per_seed"])
top20 = j["top20_holdout"]


def cv_r2(X, y):
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 4, 30)), X, y,
                          cv=KFold(5, shuffle=True, random_state=0))
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# --- fig 1: distribution ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(test * 100, bins=60, color="#4C78A8", alpha=0.85)
ax.axvline(flat * 100, color="#E45756", lw=1.6, label=f"flat training ({flat * 100:.1f}%)")
ax.axvline(test.max() * 100, color="#54A24B", lw=1.6, ls="--",
           label=f"best schedule ({test.max() * 100:.1f}%)")
ax.set_xlabel("test accuracy (%)"), ax.set_ylabel("schedules")
ax.set_title("8,000 marginal-matched schedules on Cora", loc="left")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(), fig.savefig(FIG / "fig1_distribution.png"), plt.close(fig)

# --- fig 2: val vs test (is the selection real?) ---------------------------------------
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.scatter(val * 100, test * 100, s=3, alpha=0.25, color="#4C78A8", edgecolors="none")
r = np.corrcoef(val, test)[0, 1]
ax.set_xlabel("validation accuracy (%)"), ax.set_ylabel("test accuracy (%)")
ax.set_title(f"selection transfers:  r = {r:.2f}", loc="left")
fig.tight_layout(), fig.savefig(FIG / "fig2_val_vs_test.png"), plt.close(fig)

# --- fig 3: is the winner's advantage structure, or shape? ------------------------------
rng = np.random.default_rng(0)
bars = {
    "arrival profile\n(which node, when)": cv_r2(arr, test),
    "node identity\nshuffled (null)": float(np.mean(
        [cv_r2(np.stack([rng.permutation(row) for row in arr]), test) for _ in range(3)])),
    "schedule scale\nalone (nuisance)": cv_r2(scale[:, None], test),
}
fig, ax = plt.subplots(figsize=(4.4, 3.2))
cols = ["#54A24B", "#BAB0AC", "#BAB0AC"]
ax.bar(range(3), list(bars.values()), color=cols, width=0.6)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(3)), ax.set_xticklabels(bars.keys(), fontsize=7.5)
ax.set_ylabel("cross-validated $R^2$")
ax.set_title("what predicts a schedule's accuracy", loc="left")
for i, v in enumerate(bars.values()):
    ax.text(i, v + 0.004, f"{v:+.3f}", ha="center", fontsize=8)
fig.tight_layout(), fig.savefig(FIG / "fig3_predictability.png"), plt.close(fig)

# --- fig 4: held-out gains, top-20 + winner paired ---------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
g = [(h["holdout_test_mean"] - flat) * 100 for h in top20]
a1.bar(range(1, 21), g, color="#4C78A8", width=0.75)
a1.axhline(0, color="#E45756", lw=1.2)
a1.set_xlabel("schedule rank (by validation)"), a1.set_ylabel("gain over flat (pts)")
a1.set_title("top-20 on unseen initialisations", loc="left")

w = np.array(top20[0]["holdout_test_per_seed"]) * 100
f = np.array(j["flat_holdout"]["holdout_test_per_seed"]) * 100
x = np.arange(len(w))
a2.plot(x, f, "o-", color="#E45756", label="flat", ms=4)
a2.plot(x, w, "o-", color="#54A24B", label="best schedule", ms=4)
a2.set_xlabel("held-out init seed"), a2.set_ylabel("test accuracy (%)")
a2.set_title(f"paired: wins {int((w > f).sum())}/{len(w)} seeds", loc="left")
a2.legend(frameon=False, fontsize=8)
fig.tight_layout(), fig.savefig(FIG / "fig4_holdout.png"), plt.close(fig)

print("wrote:", *[p.name for p in sorted(FIG.glob("*.png"))])
print(f"flat {flat:.4f} | winner {top20[0]['holdout_test_mean']:.4f} | "
      f"top20 mean {np.mean([h['holdout_test_mean'] for h in top20]):.4f}")
print({k: round(v, 4) for k, v in bars.items()})
