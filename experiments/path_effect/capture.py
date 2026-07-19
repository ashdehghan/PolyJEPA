"""Re-run the random-search phase of the oracle, capturing EVERYTHING needed to plot.

The RNG stream is replayed exactly as in ``oracle.py`` (same seed, same draws in the same
order), so this reproduces the identical 8000 candidates -- including the winner whose
matrix the original run failed to save.

What lands on disk (``capture.npz`` + ``capture.json``):

  per candidate (8000 of them)
    scale          the sampled Z scale
    val            accuracy on the val set   (search seeds 100,101) -- the selection signal
    test           accuracy on the test set  (search seeds 100,101) -- the honesty check
    arrival        (140,) mean arrival epoch of each node's gradient mass, in [0,1]
                   -> this is the schedule's *shape*, and it is what any structure must live in
    zsum/zstd      cheap summary stats of the raw Z

  full Z matrices for the top-50 and bottom-50 by val, plus 100 random ones
    -> enough to reconstruct or inspect any interesting schedule without keeping 270MB

  held-out re-scoring (init seeds 200-209, NEVER seen by the search)
    the winner, flat, and the top-20 by val, each with per-seed test accuracy
    -> separates "the schedule really is better" from "we got lucky selecting on val"

Plots this answers:
  1. distribution of test accuracy over 8000 schedules, with flat marked
  2. val vs test scatter -> how much of the selection gain is real vs winner's curse
  3. arrival-time profiles of winners vs losers -> is there any structure at all
  4. held-out gain of the top-20 -> does val-selection transfer to unseen inits
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import Planetoid

from experiments.path_effect.schedules import arrival_time, sinkhorn
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
T, POP, GENS, LR = 60, 20, 200, 0.15
SEARCH_SEEDS = (100, 101)
HOLDOUT_SEEDS = tuple(range(200, 210))

_DATA = None


def _init_worker(root: str) -> None:
    global _DATA
    torch.set_num_threads(1)
    _DATA = Planetoid(root=root, name="Cora")[0]


def _eval(job):
    z, seeds = job
    w = sinkhorn(np.exp(z))
    out = [train_once(_DATA, w, seed=s) for s in seeds]
    return (
        float(np.mean([o["val_acc"] for o in out])),
        float(np.mean([o["test_acc"] for o in out])),
        [float(o["test_acc"]) for o in out],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-root", default="/tmp/claude-1000/cora")
    args = ap.parse_args()

    data = Planetoid(root=args.data_root, name="Cora")[0]
    n = int(data.train_mask.sum())
    rng = np.random.default_rng(0)  # identical stream to oracle.py

    # Burn the ES phase's draws so the control sees the same numbers it saw before.
    for _ in range(GENS):
        rng.normal(size=(POP, T, n))

    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(args.data_root,))
    t0 = time.time()

    scales, vals, tests, arrivals, zsum, zstd = [], [], [], [], [], []
    keep_z: dict[int, np.ndarray] = {}
    idx = 0
    for g in range(GENS):
        s = rng.uniform(0.1, 2.0, size=2 * POP)
        cand = [si * rng.normal(size=(T, n)) for si in s]
        res = pool.map(_eval, [(c, SEARCH_SEEDS) for c in cand], chunksize=1)
        for si, c, (v, te, _) in zip(s, cand, res):
            scales.append(float(si))
            vals.append(v)
            tests.append(te)
            arrivals.append(arrival_time(sinkhorn(np.exp(c))))
            zsum.append(float(c.sum()))
            zstd.append(float(c.std()))
            # Keeping all 8000 Z matrices costs 270MB of RAM and killed the last run.
            # Keep only what any analysis could want: a fixed 1-in-80 sample, plus a
            # rolling top/bottom 60 by val (the winners are what we actually inspect).
            if idx % 80 == 0:
                keep_z[idx] = c.astype(np.float32)
            ranked = sorted(range(len(vals)), key=lambda j: vals[j])
            elite = set(ranked[-60:]) | set(ranked[:60])
            if idx in elite:
                keep_z[idx] = c.astype(np.float32)
            for j in [k for k in keep_z if k not in elite and k % 80 != 0]:
                del keep_z[j]
            idx += 1
        if g % 10 == 0:
            # Checkpoint. A killed run must never cost more than a few minutes: everything
            # up to here is on disk and the analysis works on a partial capture.
            np.savez_compressed(
                ROOT / "capture_partial.npz",
                gens_done=np.array([g + 1]),
                scale=np.array(scales), val=np.array(vals), test=np.array(tests),
                arrival=np.array(arrivals, dtype=np.float32),
                zsum=np.array(zsum), zstd=np.array(zstd),
                kept_z=np.stack([keep_z[i] for i in sorted(keep_z)]).astype(np.float32),
            )
            print(f"[{g:3d}/{GENS}] best val {max(vals):.4f}  ({time.time() - t0:.0f}s) "
                  f"[ckpt {len(vals)} cands]", flush=True)

    vals_a = np.array(vals)
    tests_a = np.array(tests)
    order = vals_a.argsort()
    keep_idx = sorted(keep_z)

    # --- held-out re-scoring: winner, flat, and the top-20 by val ---------------------
    print("\nre-scoring on held-out init seeds 200-209...", flush=True)
    top20 = order[-20:][::-1]
    jobs = [(keep_z[int(i)], HOLDOUT_SEEDS) for i in top20]
    jobs.append((np.zeros((T, n)), HOLDOUT_SEEDS))  # flat
    res = pool.map(_eval, jobs, chunksize=1)
    pool.close()
    pool.join()

    hold_top = [{"rank": r, "idx": int(i), "val": float(vals_a[i]),
                 "holdout_test_mean": res[r][1], "holdout_test_per_seed": res[r][2]}
                for r, i in enumerate(top20)]
    flat_hold = {"holdout_test_mean": res[-1][1], "holdout_test_per_seed": res[-1][2]}

    win = hold_top[0]
    d = np.array(win["holdout_test_per_seed"]) - np.array(flat_hold["holdout_test_per_seed"])
    sem = d.std(ddof=1) / np.sqrt(len(d))

    print("\n" + "=" * 74)
    print(f"WINNER (best val, idx {win['idx']}): held-out test {win['holdout_test_mean']:.4f}")
    print(f"FLAT                              : held-out test {flat_hold['holdout_test_mean']:.4f}")
    print(f"paired diff {d.mean():+.4f} +- {sem:.4f}   wins {int((d > 0).sum())}/{len(d)} seeds"
          f"   t = {d.mean() / sem:.2f}")
    print(f"val-vs-test correlation over all {len(vals)} candidates (search seeds): "
          f"r = {np.corrcoef(vals_a, tests_a)[0, 1]:.3f}")
    print(f"  ^ near zero => the selection gain is winner's curse, not signal")
    print(f"top-20 held-out gain over flat: mean "
          f"{np.mean([h['holdout_test_mean'] for h in hold_top]) - flat_hold['holdout_test_mean']:+.4f}")

    np.savez_compressed(
        ROOT / "capture.npz",
        scale=np.array(scales), val=vals_a, test=tests_a,
        arrival=np.array(arrivals, dtype=np.float32),
        zsum=np.array(zsum), zstd=np.array(zstd),
        kept_idx=np.array(keep_idx),
        kept_z=np.stack([keep_z[i] for i in keep_idx]),
    )
    (ROOT / "capture.json").write_text(json.dumps({
        "search_seeds": list(SEARCH_SEEDS), "holdout_seeds": list(HOLDOUT_SEEDS),
        "n_candidates": len(vals), "flat_holdout": flat_hold, "top20_holdout": hold_top,
        "val_test_corr": float(np.corrcoef(vals_a, tests_a)[0, 1]),
    }, indent=1))
    print(f"\nwrote {ROOT / 'capture.npz'} and capture.json")


if __name__ == "__main__":
    main()
