"""Oracle schedule search: is there a prize AT ALL?

The schedule matrix itself is the free variable. ``W = sinkhorn(exp(Z))`` with ``Z``
unconstrained, so the per-node marginal is pinned by construction: the search can never
win by spending more gradient on some nodes, only by spending it at different TIMES.
``Z = 0`` is exactly flat training.

An evolution strategy optimises ``Z`` directly against VALIDATION accuracy. It is allowed
to cheat -- it sees the val set -- because we want the CEILING of what any schedule can
buy, not a deployable method. No node features, no coordinates, no hand-crafted ordering:
nothing in this file encodes a prior about which nodes matter.

Two guards make the answer honest:

- **Fresh-seed re-scoring.** The winner is re-trained on init seeds never used during the
  search and scored on TEST. Search-time gains that were really seed overfitting die here.
- **Random-search control.** The same number of candidates, drawn at random. If the ES
  cannot beat best-of-random, the search found no structure and neither did we.

KILL CRITERION (declared before running, per the plan):
    held-out test gain over flat < +1.5 points, OR ES does not beat best-of-random
    => no prize exists, the premise of a probe bank is dead, we write the null.
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

from experiments.path_effect.schedules import sinkhorn
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "oracle_results.json"

_DATA = None


def _init_worker(data_root: str) -> None:
    global _DATA
    torch.set_num_threads(1)
    _DATA = Planetoid(root=data_root, name="Cora")[0]


def _evaluate(job: tuple[np.ndarray, tuple[int, ...]]) -> tuple[float, float]:
    """Mean (val, test) accuracy of schedule ``Z`` over the given init seeds."""
    z, seeds = job
    w = sinkhorn(np.exp(z))
    out = [train_once(_DATA, w, seed=s) for s in seeds]
    return (
        float(np.mean([o["val_acc"] for o in out])),
        float(np.mean([o["test_acc"] for o in out])),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--pop", type=int, default=20)  # antithetic pairs -> 2*pop candidates
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--data-root", default="/tmp/claude-1000/cora")
    args = p.parse_args()

    search_seeds = (100, 101)  # the ES may overfit these; that is what the guards catch
    holdout_seeds = tuple(range(200, 210))  # never seen during search

    data = Planetoid(root=args.data_root, name="Cora")[0]
    n = int(data.train_mask.sum())
    t = args.epochs
    rng = np.random.default_rng(0)
    n_candidates = 2 * args.pop * args.gens

    print(f"Z is {t}x{n} = {t * n} free parameters")
    print(f"ES: {args.gens} gens x {2 * args.pop} candidates = {n_candidates} candidates")
    print(f"random control: {n_candidates} candidates (equal budget)")

    pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(args.data_root,))
    t0 = time.time()

    # --- baseline ---------------------------------------------------------------------
    flat_val, flat_test = _evaluate_local(pool, np.zeros((t, n)), search_seeds)
    flat_hold_val, flat_hold_test = _evaluate_local(pool, np.zeros((t, n)), holdout_seeds)
    print(f"flat: search-seed val {flat_val:.4f} | held-out test {flat_hold_test:.4f}")

    # --- evolution strategy -----------------------------------------------------------
    z = np.zeros((t, n))
    history = []
    for g in range(args.gens):
        eps = rng.normal(size=(args.pop, t, n))
        cand = [z + args.sigma * e for e in eps] + [z - args.sigma * e for e in eps]
        res = pool.map(_evaluate, [(c, search_seeds) for c in cand], chunksize=1)
        f = np.array([r[0] for r in res])
        fp, fm = f[: args.pop], f[args.pop :]

        adv = fp - fm
        if adv.std() > 1e-9:
            adv = (adv - adv.mean()) / adv.std()
        grad = np.einsum("p,pij->ij", adv, eps) / (2 * args.pop * args.sigma)
        z = z + args.lr * grad

        best = float(f.max())
        history.append({"gen": g, "best_val": best, "mean_val": float(f.mean())})
        if g % 10 == 0 or g == args.gens - 1:
            cur_val, _ = _evaluate_local(pool, z, search_seeds)
            print(
                f"[gen {g:3d}] centre val {cur_val:.4f}  pop best {best:.4f}  "
                f"({time.time() - t0:.0f}s)"
            )

    es_val, es_test = _evaluate_local(pool, z, search_seeds)
    es_hold_val, es_hold_test = _evaluate_local(pool, z, holdout_seeds)

    # --- random-search control (equal candidate budget) -------------------------------
    print("\nrandom-search control...")
    best_rand_val, best_rand_z = -1.0, None
    batch = 2 * args.pop
    for g in range(args.gens):
        scales = rng.uniform(0.1, 2.0, size=batch)
        cand = [s * rng.normal(size=(t, n)) for s in scales]
        res = pool.map(_evaluate, [(c, search_seeds) for c in cand], chunksize=1)
        for c, (v, _) in zip(cand, res):
            if v > best_rand_val:
                best_rand_val, best_rand_z = v, c
        if g % 25 == 0:
            print(f"[rand {g:3d}] best val {best_rand_val:.4f}  ({time.time() - t0:.0f}s)")

    rand_hold_val, rand_hold_test = _evaluate_local(pool, best_rand_z, holdout_seeds)
    pool.close()
    pool.join()

    # --- verdict ----------------------------------------------------------------------
    gain = es_hold_test - flat_hold_test
    beats_random = es_hold_test > rand_hold_test
    alive = gain >= 0.015 and beats_random

    print("\n" + "=" * 78)
    print("ORACLE SEARCH — DOES A GOOD SCHEDULE EXIST AT ALL?")
    print("=" * 78)
    print(f"                        search-seed val   held-out test (seeds 200-209)")
    print(f"flat                        {flat_val:.4f}            {flat_hold_test:.4f}")
    print(f"ES-optimised schedule       {es_val:.4f}            {es_hold_test:.4f}")
    print(f"best-of-random ({n_candidates})     {best_rand_val:.4f}            {rand_hold_test:.4f}")
    print()
    print(f"held-out test gain over flat: {gain:+.4f}  (kill line: +0.0150)")
    print(f"ES beats best-of-random:      {beats_random}")
    print(f"search-time val gain:         {es_val - flat_val:+.4f}  "
          f"(the gap to held-out gain is search overfitting)")
    print()
    print("VERDICT:", "PRIZE EXISTS — the premise survives." if alive else
          "NO PRIZE. Even an oracle allowed to see the val set cannot buy 1.5 points by "
          "reordering alone. No probe, feature, or coordinate system can beat an oracle. "
          "The premise of PolyJEPA is dead. Write the null.")

    OUT.write_text(json.dumps({
        "config": vars(args),
        "flat": {"val": flat_val, "test": flat_test,
                 "holdout_val": flat_hold_val, "holdout_test": flat_hold_test},
        "es": {"val": es_val, "test": es_test,
               "holdout_val": es_hold_val, "holdout_test": es_hold_test},
        "random": {"val": best_rand_val,
                   "holdout_val": rand_hold_val, "holdout_test": rand_hold_test},
        "gain": gain, "beats_random": bool(beats_random), "alive": bool(alive),
        "history": history,
        "z": z.tolist(),
    }, indent=1))
    print(f"\nwrote {OUT}")


def _evaluate_local(pool, z: np.ndarray, seeds: tuple[int, ...]) -> tuple[float, float]:
    return pool.apply(_evaluate, ((z, seeds),))


if __name__ == "__main__":
    main()
