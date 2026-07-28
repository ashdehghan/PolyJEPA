"""Post-hoc anatomy of the E5 gradient-learned Cora schedule. No training.

Every number the manuscript's learned-schedule chapter quotes about the structure of
W_learned comes from this script, so each has a reproducible source. Inputs are the
committed/cached artifacts only:

  schedule_learn_cora.npz   W_learned and the winning logits Z (best restart only)
  schedule_learn_cora.json  per-seed held-out accuracies + both outer trajectories
  replicate_cora.npz        the 2,000-schedule compass pool: beta, rank, coords, labels
  capture.npz               the 8,000-schedule landscape; kept raw logits for the
                            top/bottom-60 by validation (reference distribution)
  fingerprint_cora.npz      six-probe residuals for all nodes (train rows used)

Analyses (mirrors the chapter plan): A1 staircase quantification (+ the same metrics on
the top-20 kept random schedules as a reference), A2 does the projection preserve the
logits' timing, A3 restart evidence from trajectories alone, A4 predictors of the
learned arrival order (12 tests, permutation p-values), A5 sensitivity of the beta
correlation + schedule-level cosine to the compass arm, A6 entry anatomy, A7 outcome
arithmetic on the ten held-out seeds. The search-vs-replicate validation comparison is
deliberately absent: replicate.py scored schedules under the stochastic pipeline while
the E5 search objective is the deterministic surrogate, so the two numbers are not
comparable.

Usage:
    .venv/bin/python -m experiments.path_effect.learned_anatomy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from torch_geometric.datasets import Planetoid

from experiments.path_effect.schedules import _spotlight, arrival_time, sinkhorn

ROOT = Path(__file__).resolve().parent
T = 60
FLAT_TAIL = 30          # replicate.py's compass arm: last K epochs flattened
N_PERM = 10_000
N_BOOT = 10_000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Staircase metrics for a single schedule
# ---------------------------------------------------------------------------

def staircase_metrics(W: np.ndarray) -> dict:
    """Per-node timing statistics of a (T, N) schedule with column sums T."""
    t_axis = np.arange(W.shape[0])
    p = W / W.sum(axis=0, keepdims=True)               # per-node time distribution
    arrival = (p * t_axis[:, None]).sum(axis=0)        # mean epoch, in epochs

    cdf = np.cumsum(p, axis=0)
    q25 = np.array([np.searchsorted(cdf[:, i], 0.25) for i in range(W.shape[1])])
    q75 = np.array([np.searchsorted(cdf[:, i], 0.75) for i in range(W.shape[1])])
    iqr = q75 - q25

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.where(p > 0, np.log(p), 0.0)).sum(axis=0)
    support = np.exp(ent)                              # effective #epochs per node

    near = np.abs(t_axis[:, None] - arrival[None, :]) <= 3
    mass_pm3 = (p * near).sum(axis=0)

    order = np.argsort(arrival)
    sorted_arr = arrival[order]
    grid = np.linspace(sorted_arr[0], sorted_arr[-1], len(sorted_arr))
    return {
        "arrival_min": float(sorted_arr[0]),
        "arrival_max": float(sorted_arr[-1]),
        "arrival_span": float(sorted_arr[-1] - sorted_arr[0]),
        "arrival_vs_uniform_grid_r": float(np.corrcoef(sorted_arr, grid)[0, 1]),
        "max_interarrival_gap": float(np.diff(sorted_arr).max()),
        "iqr_median": float(np.median(iqr)),
        "iqr_q25": float(np.quantile(iqr, 0.25)),
        "iqr_q75": float(np.quantile(iqr, 0.75)),
        "effective_support_median": float(np.median(support)),
        "mass_within_3_epochs_median": float(np.median(mass_pm3)),
    }


def perm_p(x: np.ndarray, y: np.ndarray, observed: float, stat) -> float:
    """Two-sided permutation p-value for stat(x, y) under shuffles of y."""
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        null[k] = stat(x, RNG.permutation(y))
    return float((np.abs(null) >= abs(observed)).mean())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def eta_squared(groups: np.ndarray, x: np.ndarray) -> float:
    grand = x.mean()
    ss_tot = ((x - grand) ** 2).sum()
    ss_between = sum(
        len(x[groups == g]) * (x[groups == g].mean() - grand) ** 2
        for g in np.unique(groups)
    )
    return float(ss_between / ss_tot)


def main() -> None:
    W = np.load(ROOT / "schedule_learn_cora.npz")["W_learned"].astype(np.float64)
    Z = np.load(ROOT / "schedule_learn_cora.npz")["Z"].astype(np.float64)
    run = json.loads((ROOT / "schedule_learn_cora.json").read_text())
    rep = np.load(ROOT / "replicate_cora.npz", allow_pickle=True)
    cap = np.load(ROOT / "capture.npz")
    fp = np.load(ROOT / "fingerprint_cora.npz", allow_pickle=True)

    t_steps, n = W.shape
    assert (t_steps, n) == (T, 140), (t_steps, n)
    # schedule_learn.py runs 15 Sinkhorn iterations ending on a column rescale, so the
    # column budgets are exact and the row budgets are approximate. Record the residual
    # rather than hiding it; the chapter states it.
    row_dev = float(np.abs(W.sum(axis=1) - n).max())
    col_dev = float(np.abs(W.sum(axis=0) - T).max())
    assert col_dev < 1e-3, "columns must sum to T"
    assert row_dev < 1.0, f"row budgets off by more than 1: {row_dev}"

    # Reproduce the three published numbers before trusting anything downstream.
    flat_test = np.array(run["flat_test"])
    learned_test = np.array(run["learned_test"])
    bestrand_test = np.array(run["best_random_test"])
    gain = float((learned_test - flat_test).mean())
    assert abs(gain - run["gains"]["learned"]) < 1e-9
    beta = rep["beta"].astype(np.float64)
    arrival_w = arrival_time(W)                       # normalized [0, 1] scale
    corr_beta = float(np.corrcoef(arrival_w, beta)[0, 1])
    assert abs(corr_beta - run["corr_arrival_beta"]) < 1e-6
    path_mag = float(np.abs(W - 1.0).mean())
    assert abs(path_mag - run["path_magnitude"]) < 1e-6

    out: dict = {"dataset": "Cora", "T": T, "n_train": n,
                 "row_budget_max_abs_dev": row_dev,
                 "col_budget_max_abs_dev": col_dev}

    # ---- A1: staircase quantification, with flat + good-random references ----------
    a1 = {"learned": staircase_metrics(W)}
    a1["flat_reference"] = staircase_metrics(np.ones((T, n)))
    # Top-20-by-validation random schedules from the 8,000-schedule landscape: the kept
    # raw logits are the candidates themselves, schedule = sinkhorn(exp(z)).
    kept_idx, kept_z = cap["kept_idx"], cap["kept_z"]
    top_kept = np.argsort(cap["val"][kept_idx])[-20:]
    rand_metrics = [staircase_metrics(sinkhorn(np.exp(kept_z[i].astype(np.float64))))
                    for i in top_kept]
    a1["top20_random_reference"] = {
        k: float(np.median([m[k] for m in rand_metrics])) for k in rand_metrics[0]
    }
    out["A1_staircase"] = a1

    # ---- A2: the projection preserves the logits' timing ---------------------------
    p_z = np.exp(Z - Z.max(axis=0, keepdims=True))
    p_z /= p_z.sum(axis=0, keepdims=True)             # column softmax of Z
    arrival_z = (p_z * np.arange(T)[:, None]).sum(axis=0)
    arrival_w_epochs = arrival_w * (T - 1)
    out["A2_z_vs_w"] = {
        "arrival_pearson": float(np.corrcoef(arrival_z, arrival_w_epochs)[0, 1]),
        "cell_pearson_expz_w": float(np.corrcoef(np.exp(Z).ravel(), W.ravel())[0, 1]),
        "cell_spearman_expz_w": spearman(np.exp(Z).ravel(), W.ravel()),
    }

    # ---- A3: restart evidence, trajectories only -----------------------------------
    a3 = []
    for r, traj in enumerate(run["trajectories"]):
        accs = np.array([p["val_acc"] for p in traj])
        start = accs[0]
        above = np.nonzero(accs > start)[0]
        a3.append({
            "restart": r,
            "start_val_acc": float(start),
            "best_val_acc": float(accs.max()),
            "final_val_acc": float(accs[-1]),
            "first_step_above_start": int(above[0]) if len(above) else None,
        })
    out["A3_restarts"] = {
        "note": "only the winning restart's Z was saved; schedule-level agreement "
                "between restarts is not measurable from the artifacts",
        "trajectories": a3,
    }

    # ---- A4: predictors of the learned arrival order (12 tests) --------------------
    data = Planetoid(root="/tmp/claude-1000/cora", name="Cora")[0]
    train_mask = data.train_mask.numpy()
    assert train_mask.sum() == n
    resid = fp["residuals"][train_mask]
    probe_names = [str(s) for s in fp["probe_names"]]
    coords = rep["coords"].astype(np.float64)
    coord_names = ["degree", "clustering", "pagerank", "homophily", "feat_norm"]
    labels = rep["labels"].astype(int)
    assert np.array_equal(labels, data.y.numpy()[train_mask])

    tests = []
    for name, col in list(zip(coord_names, coords.T)) + list(zip(probe_names, resid.T)):
        ok = ~np.isnan(col)
        r = spearman(arrival_w[ok], col[ok])
        tests.append({
            "predictor": name, "n": int(ok.sum()), "spearman": r,
            "perm_p": perm_p(arrival_w[ok], col[ok], r, spearman),
        })
    e2 = eta_squared(labels, arrival_w)
    null_e2 = np.array([eta_squared(RNG.permutation(labels), arrival_w)
                        for _ in range(N_PERM)])
    tests.append({"predictor": "label (eta^2)", "n": n, "eta_squared": e2,
                  "perm_p": float((null_e2 >= e2).mean()),
                  "eta_squared_null_mean": float(null_e2.mean())})
    out["A4_predictors"] = {
        "n_tests": len(tests), "bonferroni_alpha": 0.05 / len(tests),
        "two_sigma_null_band_r": float(2 / np.sqrt(n)), "tests": tests,
    }

    # ---- A5: sensitivity of the beta correlation -----------------------------------
    rank = rep["rank"].astype(np.float64)
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        boot[k] = np.corrcoef(arrival_w[idx], beta[idx])[0, 1]
    raw_compass = _spotlight(rank, T, 0.15)
    raw_compass[T - FLAT_TAIL:] = raw_compass.mean()
    w_compass = sinkhorn(raw_compass)
    dl, dc = (W - 1.0).ravel(), (w_compass - 1.0).ravel()
    out["A5_beta_sensitivity"] = {
        "pearson": corr_beta,
        "spearman": spearman(arrival_w, beta),
        "pearson_vs_rank_beta": float(np.corrcoef(arrival_w, rank)[0, 1]),
        "pearson_bootstrap_ci95": [float(np.quantile(boot, q)) for q in (0.025, 0.975)],
        "pearson_perm_p": perm_p(arrival_w, beta, corr_beta,
                                 lambda x, y: float(np.corrcoef(x, y)[0, 1])),
        "cosine_pathdev_learned_vs_compass_arm": float(
            dl @ dc / (np.linalg.norm(dl) * np.linalg.norm(dc))),
    }

    # ---- A6: entry anatomy ----------------------------------------------------------
    flat_entries = np.sort(W.ravel())[::-1]
    top1pct = int(np.ceil(0.01 * flat_entries.size))
    tmax, imax = np.unravel_index(W.argmax(), W.shape)
    ent_t = np.array([stats.entropy(W[t] / W[t].sum()) for t in range(T)])
    out["A6_entries"] = {
        "mean_abs_dev_from_1": path_mag,
        "max_entry": float(W.max()),
        "max_entry_epoch": int(tmax),
        "max_entry_node_arrival_rank": int(np.argsort(np.argsort(arrival_w))[imax]),
        "min_entry": float(W.min()),
        "share_entries_below_0.1": float((W < 0.1).mean()),
        "top1pct_cells_mass_share": float(flat_entries[:top1pct].sum() / W.sum()),
        "per_epoch_node_entropy_first10_mean": float(ent_t[:10].mean()),
        "per_epoch_node_entropy_last10_mean": float(ent_t[-10:].mean()),
        "per_epoch_node_entropy_flat": float(np.log(n)),
    }

    # ---- A7: outcome arithmetic on the ten held-out seeds --------------------------
    d = learned_test - flat_test
    tt = stats.ttest_rel(learned_test, flat_test)
    wc = stats.wilcoxon(learned_test, flat_test)
    out["A7_outcome"] = {
        "gain_mean": gain, "gain_sem": float(d.std(ddof=1) / np.sqrt(len(d))),
        "wins": int((d > 0).sum()), "n_seeds": len(d),
        "sign_test_two_sided_p": float(2 * 0.5 ** len(d)),
        "paired_t": float(tt.statistic), "paired_t_p": float(tt.pvalue),
        "wilcoxon_p": float(wc.pvalue),
        "best_random_gain_mean": float((bestrand_test - flat_test).mean()),
        "best_random_wins": int((bestrand_test > flat_test).sum()),
        "yardstick_pool": "2,000-schedule replicate (compass) pool, "
                          "regenerated stream, val-argmax",
    }

    (ROOT / "learned_anatomy_cora.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])
    print("\nwrote learned_anatomy_cora.json")


if __name__ == "__main__":
    main()
