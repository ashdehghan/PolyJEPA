"""Two readouts.

1. IS THERE A PRIZE? Spread in accuracy across schedules vs spread across seeds at a
   fixed schedule. If path-spread <= seed-noise there is no headroom for any schedule.
   Held-out seeds decide whether the best schedule is real or just the luckiest draw.
2. ARE THE WINNERS PREDICTABLE? Cross-validated R^2 of a ridge regression from schedule
   descriptors (how the schedule couples arrival time to node coordinates) to accuracy,
   against a null of the same regression on scrambled-coordinate descriptors.

Readout 2 is the number this experiment exists to produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent


def r2_cv(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Out-of-fold R^2. Negative means worse than predicting the mean."""
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    pred = cross_val_predict(model, x, y, cv=cv)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main() -> None:
    blob = json.loads((ROOT / "results.json").read_text())
    rows = blob["rows"]
    names = sorted({r["schedule"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    acc = {
        (r["schedule"], r["seed"]): r["test_acc"] for r in rows
    }
    a = np.array([[acc[(n, s)] for s in seeds] for n in names])  # (S, seeds)
    fam = {r["schedule"]: r["family"] for r in rows}
    flat_i = names.index("flat")

    # --- readout 1: is there a prize? -------------------------------------------------
    per_sched = a.mean(axis=1)
    path_spread = float(per_sched.std())
    seed_noise = float(a.std(axis=1).mean())
    # standard error of a schedule's mean: the resolution of our instrument
    resolution = seed_noise / np.sqrt(len(seeds))

    print("=" * 78)
    print("READOUT 1 — IS THERE A PRIZE?")
    print("=" * 78)
    print(f"schedules {len(names)}   seeds {len(seeds)}   epochs {blob['config']['epochs']}")
    print(f"flat baseline test acc      {per_sched[flat_i]:.4f}  (std over seeds {a[flat_i].std():.4f})")
    print(f"spread ACROSS schedules     {path_spread:.4f}   <-- the path effect")
    print(f"spread ACROSS seeds (mean)  {seed_noise:.4f}   <-- the noise floor")
    print(f"ratio path/seed             {path_spread / seed_noise:.2f}")
    print(f"resolution (sem of a mean)  {resolution:.4f}")
    print()
    print(f"best  {names[int(per_sched.argmax())]:34s} {per_sched.max():.4f}")
    print(f"worst {names[int(per_sched.argmin())]:34s} {per_sched.min():.4f}")
    print(f"range {per_sched.max() - per_sched.min():.4f}")
    print()
    for f in sorted(set(fam.values())):
        idx = [i for i, n in enumerate(names) if fam[n] == f]
        print(f"  family {f:12s} n={len(idx):3d}  mean {per_sched[idx].mean():.4f}  "
              f"best {per_sched[idx].max():.4f}  worst {per_sched[idx].min():.4f}")

    # Winner's curse guard: choose on some seeds, judge on the others.
    if len(seeds) >= 4:
        k = len(seeds) // 2
        sel, conf = a[:, :k].mean(1), a[:, k:].mean(1)
        pick = int(sel.argmax())
        print()
        print(f"selection on seeds {seeds[:k]}, confirmation on seeds {seeds[k:]}:")
        print(f"  picked {names[pick]}  -> confirm {conf[pick]:.4f} vs flat {conf[flat_i]:.4f} "
              f"(delta {conf[pick] - conf[flat_i]:+.4f})")
        print(f"  in-sample delta was {sel[pick] - sel[flat_i]:+.4f} — the gap between these "
              f"two numbers IS the winner's curse")

    # --- readout 2: are the winners predictable? --------------------------------------
    desc = blob["descriptors"]
    x = np.array([desc[n] for n in names])
    y = per_sched
    real = r2_cv(x, y)
    null = np.array(
        [r2_cv(np.array([ctrl[n] for n in names]), y) for ctrl in blob["control_descriptors"]]
    )

    print()
    print("=" * 78)
    print("READOUT 2 — ARE THE WINNERS PREDICTABLE?  (the number we came for)")
    print("=" * 78)
    print(f"cross-validated R^2, real coordinates      {real:+.4f}")
    print(f"scrambled-coordinate control (n={len(null)})     "
          f"mean {null.mean():+.4f}   95th pct {np.percentile(null, 95):+.4f}   "
          f"max {null.max():+.4f}")
    beats = int((null >= real).sum())
    print(f"controls matching or beating the real fit  {beats}/{len(null)}  "
          f"(p ~= {(beats + 1) / (len(null) + 1):.3f})")
    print()
    if real <= 0:
        verdict = ("NO SIGNAL. The descriptor cannot predict accuracy at all — it is worse "
                   "than guessing the mean. Node properties do not tell you which schedule wins.")
    elif beats > len(null) * 0.05:
        verdict = ("NO SIGNAL ABOVE CHANCE. The real fit sits inside the scrambled null: "
                   "whatever it explains, it explains without knowing anything about the nodes.")
    else:
        verdict = (f"SIGNAL. Real R^2 = {real:.3f} beats every scrambled control. The winning "
                   f"schedules ARE predictable from node coordinates — the premise holds.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
