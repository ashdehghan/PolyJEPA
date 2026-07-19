"""Probe compass: connect the PolyJEPA fingerprint to the schedule experiment.

The arrival compass in replicate.py fits:
    val_s ≈ tau_s · β    (ridge over n_train parameters, one per node)
and uses β to rank nodes. The rank is the arrow the compass points.

This script tests whether the probe fingerprint R ∈ R^{N×P} (from PolyJEPA)
can supply the same arrow, in two ways:

  Arm 1 — Arrival compass (baseline from replicate.py, re-confirmed here)
     Features per schedule: tau_s ∈ R^{n_train}
     Fit: val_s ≈ tau_s · β  →  rank by β

  Arm 2 — Probe compass
     Compress: x_s = (tau_s @ R_z) / n  ∈ R^P  (schedule × probe coupling)
       x_s[p] = correlation between schedule s's timing and probe p's residuals
     Fit: val_s ≈ x_s · γ  (ridge over P=6 parameters)
     Rank: R_z @ γ  (per-node predicted benefit of early training)
     Hypothesis: P=6 parameters generalise better than n_train=140.

  Arm 3 — Probe-β prediction
     Fit: β ≈ R_z @ δ + ε  (can probe features predict the compass coefficients?)
     Rank: R_z @ δ
     This answers: does the fingerprint COMPRESS the per-node compass information?
     A good fit means we could get the ordering without 2000 random schedules on
     future datasets — we could transfer the probe compass cross-domain.

All three arms share the same 2000 schedule evaluations from replicate_{ds}.npz.
The fingerprint is computed once and cached to fingerprint_{ds}.npz.

Usage:
    cd workspace/PolyJEPA
    .venv/bin/python -m experiments.path_effect.probe_compass --datasets Cora
    .venv/bin/python -m experiments.path_effect.probe_compass --datasets Cora CiteSeer PubMed
    .venv/bin/python -m experiments.path_effect.probe_compass --quick   # 50 fp epochs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch_geometric.datasets import Planetoid

from polyjepa import fingerprint as compute_fingerprint
from experiments.path_effect.schedules import _spotlight, arrival_time, sinkhorn
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
T = 60
HOLD_SEEDS = tuple(range(200, 210))


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def _load_or_compute_fingerprint(
    data, ds: str, epochs: int, seed: int
) -> np.ndarray:
    """Return column-z-scored residuals for training nodes, shape (n_train, 6).

    Cache the raw residuals to fingerprint_{ds}.npz beside the replicate files.
    """
    cache = ROOT / f"fingerprint_{ds.lower()}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  loaded fingerprint from {cache.name}")
        R = d["residuals"]   # (N_total, 6)
    else:
        print(f"  computing fingerprint ({epochs} epochs per probe, seed {seed})…")
        fp = compute_fingerprint(data, epochs=epochs, seed=seed)
        R = fp.residuals.numpy()   # (N_total, 6)
        np.savez_compressed(cache, residuals=R.astype(np.float32),
                            probe_names=np.array(fp.probe_names))
        print(f"  wrote {cache.name}")

    # Extract training nodes
    mask = data.train_mask.numpy().astype(bool)
    R_train = R[mask]   # (n_train, 6)

    # Fill NaN (unscored nodes) with column median
    for col in range(R_train.shape[1]):
        col_vals = R_train[:, col]
        finite = col_vals[np.isfinite(col_vals)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        R_train[np.isnan(col_vals), col] = fill

    # Column z-score (avoid dividing by zero for constant columns)
    mu = R_train.mean(axis=0)
    sigma = R_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (R_train - mu) / sigma


# ---------------------------------------------------------------------------
# Compass building
# ---------------------------------------------------------------------------

def _make_sweep(rank_01: np.ndarray, flat_tail: int = 30) -> np.ndarray:
    raw = _spotlight(rank_01, T, 0.15)
    if flat_tail:
        raw[T - flat_tail:] = raw.mean()
    return sinkhorn(raw)


def _score_arm(data, W: np.ndarray) -> np.ndarray:
    return np.array([train_once(data, W, seed=s)["test_acc"] for s in HOLD_SEEDS])


def _ridge(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return RidgeCV(alphas=np.logspace(-3, 4, 30)).fit(X, y).coef_


def _coef_is_degenerate(coef: np.ndarray, tol: float = 1e-6) -> bool:
    """True if ridge collapsed to all-zero (no signal case)."""
    return float(np.abs(coef).max()) < tol


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------

def process_dataset(ds: str, fp_epochs: int, fp_seed: int) -> None:
    npz_path = ROOT / f"replicate_{ds.lower()}.npz"
    json_path = ROOT / f"replicate_{ds.lower()}.json"
    if not npz_path.exists():
        print(f"  {ds}: no replicate npz — run replicate.py first, skipping")
        return

    print(f"\n{'='*60}")
    print(f"{ds}")
    print(f"{'='*60}")

    # Load replicate data
    d = np.load(npz_path)
    orig = json.loads(json_path.read_text())
    flat_tail = orig["config"].get("flat_tail", 30)

    arrival = d["arrival"].astype(np.float64)   # (n_candidates, n_train)
    val = d["val"].astype(np.float64)           # (n_candidates,)
    beta_raw = d["beta"].astype(np.float64)     # (n_train,) — existing compass coefs
    n = arrival.shape[1]

    print(f"  {arrival.shape[0]} candidate schedules, n_train={n}, flat_tail={flat_tail}")

    # Load data and fingerprint
    root = f"/tmp/claude-1000/{ds.lower()}"
    data = Planetoid(root=root, name=ds)[0]
    R_z = _load_or_compute_fingerprint(data, ds, fp_epochs, fp_seed)
    # R_z: (n_train, 6) column-z-scored

    # Flat baseline (re-score for paired comparison)
    flat_scores = _score_arm(data, np.ones((T, n)))
    flat_mean = float(flat_scores.mean())
    flat_stored = orig["flat"]
    print(f"  flat: {flat_mean:.4f}  (replicate stored: {flat_stored:.4f})")

    # ----- Arm 1: arrival compass (reconfirm with replicate's β) --------
    rank_arrival = beta_raw.argsort().argsort() / (n - 1)
    W_arrival = _make_sweep(rank_arrival, flat_tail)
    scores_arrival = _score_arm(data, W_arrival)

    # ----- Arm 2: probe compass -----------------------------------------
    # x_s = (arrival[s] @ R_z) / n   — shape (n_candidates, 6)
    X_probe = (arrival @ R_z) / n
    gamma = _ridge(X_probe, val)
    probe_scores_raw = R_z @ gamma   # (n_train,) predicted benefit of early training
    if _coef_is_degenerate(gamma):
        # No signal: use uniform random ordering (not index order, which is not neutral)
        rng = np.random.default_rng(42)
        rank_probe = rng.permutation(n).argsort() / (n - 1)
        gamma_note = "DEGENERATE (random fallback)"
    else:
        rank_probe = probe_scores_raw.argsort().argsort() / (n - 1)
        gamma_note = ""
    W_probe = _make_sweep(rank_probe, flat_tail)
    scores_probe = _score_arm(data, W_probe)

    # ----- Arm 3: probe-β prediction ------------------------------------
    # β ≈ R_z @ δ  — can probe features predict per-node compass coefficients?
    delta = _ridge(R_z, beta_raw)
    beta_pred = R_z @ delta
    if _coef_is_degenerate(delta):
        rng2 = np.random.default_rng(43)
        rank_beta_pred = rng2.permutation(n).argsort() / (n - 1)
        delta_note = "DEGENERATE (random fallback)"
    else:
        rank_beta_pred = beta_pred.argsort().argsort() / (n - 1)
        delta_note = ""
    W_beta_pred = _make_sweep(rank_beta_pred, flat_tail)
    scores_beta_pred = _score_arm(data, W_beta_pred)

    # Measure probe-β prediction quality (how well can we compress β?)
    beta_r2 = 1 - np.sum((beta_raw - beta_pred) ** 2) / np.sum((beta_raw - beta_raw.mean()) ** 2)

    # ----- Print summary ------------------------------------------------
    print(f"\n  {'arm':40s} {'test':>7} {'gain':>8} {'sem':>7} {'wins':>6}")
    print(f"  {'flat':40s} {flat_mean:7.4f} {'--':>8}")

    rows = {}
    for label, scores in [
        ("arrival compass (replicate β)", scores_arrival),
        ("probe compass (γ from R_z·arr coupling)", scores_probe),
        ("probe-β prediction (δ predicts β)", scores_beta_pred),
    ]:
        gain = scores - flat_scores
        sem = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        wins = int((gain > 0).sum())
        sign = "+" if gain.mean() >= 0 else ""
        print(f"  {label:40s} {scores.mean():7.4f} {sign}{gain.mean():7.4f} {sem:7.4f} {wins:4d}/10")
        rows[label] = {
            "test": float(scores.mean()),
            "gain": float(gain.mean()),
            "sem": sem,
            "wins": wins,
            "paired_t": float(gain.mean() / sem) if sem > 0 else 0.0,
            "per_seed": scores.tolist(),
        }

    print(f"\n  probe-β prediction R² (β ≈ R_z @ δ): {beta_r2:.4f}")
    print(f"  gamma (probe compass coefs):  {np.round(gamma, 3)} {gamma_note}")
    print(f"  delta (probe-β pred coefs):   {np.round(delta[:6], 3)} {delta_note}")

    out = {
        "dataset": ds,
        "flat": flat_mean,
        "flat_per_seed": flat_scores.tolist(),
        "arms": rows,
        "probe_beta_r2": float(beta_r2),
        "gamma": gamma.tolist(),
        "delta": delta.tolist(),
        "flat_tail": flat_tail,
        "hold_seeds": list(HOLD_SEEDS),
        "fp_epochs": fp_epochs,
        "fp_seed": fp_seed,
    }
    out_path = ROOT / f"probe_compass_{ds.lower()}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\n  wrote {out_path.name}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora"])
    ap.add_argument("--fp-epochs", type=int, default=200,
                    help="JEPA training epochs per probe (reduce for quick test)")
    ap.add_argument("--fp-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="Alias for --fp-epochs 50 (fast sanity check)")
    a = ap.parse_args()

    if a.quick:
        a.fp_epochs = 50

    torch.set_num_threads(1)
    for ds in a.datasets:
        process_dataset(ds, a.fp_epochs, a.fp_seed)


if __name__ == "__main__":
    main()
