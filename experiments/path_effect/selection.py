"""E3 — selection: can label-free coverage in probe space pick the right k% of labels?

The timing experiments showed the compass coefficient beta is gradient-dynamic and
dataset-local — no static probe predicts it. Selection is a different question:
"which k% of the labeled pool, if supervised on, lets the GCN generalize best?"
Representativeness is plausibly structural, so the probe fingerprint gets a second
chance here, against honest baselines.

Transductive setting: selecting k% means only those nodes contribute to the loss;
every node still flows through message passing. The selection mask is delivered
through ``train_once``'s weight matrix — columns of unselected nodes are zero and
selected columns carry ``n_train / n_selected`` so the loss equals the MEAN cross-
entropy over the selected subset. That keeps the per-step gradient scale comparable
across budgets, and k=100% reduces exactly to the flat baseline.

Arms (selectors see no labels unless tagged label-aware):
  random      k nodes uniformly, 10 draws                      (baseline)
  fps_probe   farthest-point sampling in fingerprint space R^6 (the hypothesis)
  fps_sgc     farthest-point sampling in SGC feature space A^2X (embedding coverage)
  easy/hard   k easiest / hardest by GIS early loss — LABEL-AWARE, the CLNode-style
              context arms (easy-vs-hard gap at 5% is the sanity gate)
  ccs_pc1     stratified sampling along PC1 of the fingerprint — the honest scalar
              ablation ("is this just CCS with more axes?")
  probcover   greedy max-coverage with radius-r balls in SGC space, covering ALL
              nodes; r from a label-free k-means purity heuristic + a small sweep

Usage:
    .venv/bin/python -m experiments.path_effect.selection --datasets Cora CiteSeer PubMed
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy import stats
from sklearn.cluster import KMeans
from torch_geometric.datasets import Planetoid

from experiments.path_effect.gis import GIS_SEEDS, _compute_gis_one_seed, _gis_scores
from experiments.path_effect.train import train_once

ROOT = Path(__file__).resolve().parent
T = 60
HOLD_SEEDS = tuple(range(200, 210))
BUDGETS = (5, 10, 20, 50)
N_RANDOM_DRAWS = 10
PROBCOVER_PURITY = 0.95
_DATA = None


# ---------------------------------------------------------------------------
# Data and feature spaces
# ---------------------------------------------------------------------------

AMAZON = {"photo": "Photo", "computers": "Computers"}
SPLIT_SEED = 0


def _amazon_masks(data, seed: int = SPLIT_SEED) -> None:
    """Shchur/CLNode split convention for datasets without a standard split:
    20 labeled nodes per class train, 500 val, 1000 test (seeded, deterministic)."""
    rng = np.random.default_rng(seed)
    y = data.y.numpy()
    n = data.num_nodes
    train = []
    for c in np.unique(y):
        train.extend(rng.choice(np.where(y == c)[0], size=20, replace=False))
    rest = np.setdiff1d(np.arange(n), np.array(train))
    rest = rng.permutation(rest)
    val, test = rest[:500], rest[500:1500]
    for name, idx in (("train_mask", train), ("val_mask", val), ("test_mask", test)):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[np.asarray(idx)] = True
        setattr(data, name, mask)


def _load_data(ds: str):
    if ds.lower() in AMAZON:
        from torch_geometric.datasets import Amazon

        data = Amazon(root=f"/tmp/claude-1000/amazon-{ds.lower()}", name=AMAZON[ds.lower()])[0]
        _amazon_masks(data)
        return data
    return Planetoid(root=f"/tmp/claude-1000/{ds.lower()}", name=ds)[0]


def _load_R_train(data, ds: str) -> np.ndarray | None:
    """Column-z-scored fingerprint rows for training nodes, (n_train, 6).

    Load-only: E3 never computes fingerprints (PubMed's is deferred — per-focal
    scoring at N=19,717 is O(N^2)). Same NaN-fill + z-score as probe_compass.
    """
    cache = ROOT / f"fingerprint_{ds.lower()}.npz"
    if not cache.exists():
        return None
    R = np.load(cache)["residuals"]                       # (N_total, 6)
    R_train = R[data.train_mask.numpy().astype(bool)].copy()
    for col in range(R_train.shape[1]):
        vals = R_train[:, col]
        finite = vals[np.isfinite(vals)]
        R_train[np.isnan(vals), col] = float(np.median(finite)) if len(finite) else 0.0
    mu, sigma = R_train.mean(axis=0), R_train.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (R_train - mu) / sigma


def _sgc(data, hops: int = 2) -> np.ndarray:
    """Row-normalized SGC features A_hat^hops X — the label-free embedding space."""
    n = data.num_nodes
    src, dst = data.edge_index.numpy()
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = A + sp.eye(n, format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    Dinv = sp.diags(1.0 / np.sqrt(d))
    A_hat = Dinv @ A @ Dinv
    X = data.x.numpy().astype(np.float32)
    for _ in range(hops):
        X = A_hat @ X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (X / norms).astype(np.float32)


def _gis_early(data) -> np.ndarray:
    """Mean early-training CE per training node (label-aware difficulty)."""
    mats = [_compute_gis_one_seed(data, seed=s) for s in range(GIS_SEEDS)]
    return _gis_scores(np.stack(mats).mean(axis=0))["gis_early"]


def _clnode_difficulty(data, seed: int = 0, epochs: int = 200) -> np.ndarray:
    """CLNode's multi-perspective difficulty (Wei et al., WSDM 2023), alpha=1.

    Train f1 on the full labeled set; pseudo-label everything (true labels kept
    on train nodes); D_local = label entropy of the closed neighborhood (Eq. 5-6),
    D_global = 1 - similarity to the label-class prototype (Eq. 7-10),
    D = D_local + D_global (Eq. 11 with their fixed alpha=1). Label-aware.
    """
    import torch.nn.functional as F

    from experiments.path_effect.train import GCN

    torch.manual_seed(seed)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    model = GCN(data.num_features, 16, int(data.y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        F.cross_entropy(logits[train_idx], data.y[train_idx]).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        h = F.relu(model.conv1(data.x, data.edge_index))       # (N, 16)
        logits = model.conv2(h, data.edge_index)
    y_tilde = logits.argmax(dim=1)
    y_tilde[train_idx] = data.y[train_idx]
    h, y_tilde = h.numpy(), y_tilde.numpy()
    n_classes = int(data.y.max()) + 1

    neigh = [[v] for v in range(data.num_nodes)]               # closed neighborhoods
    for s, d in zip(*data.edge_index.numpy()):
        neigh[s].append(int(d))

    protos = np.stack([h[y_tilde == c].mean(axis=0) for c in range(n_classes)])
    D = np.zeros(len(train_idx))
    for j, u in enumerate(train_idx.tolist()):
        p = np.bincount(y_tilde[neigh[u]], minlength=n_classes) / len(neigh[u])
        d_local = -(p[p > 0] * np.log(p[p > 0])).sum()
        proto_logits = h[u] @ protos.T
        sims = np.exp(proto_logits - proto_logits.max())   # overflow-safe; ratios unchanged
        d_global = 1.0 - sims[y_tilde[u]] / sims.max()
        D[j] = d_local + d_global
    return D


def _stratified(scores: np.ndarray, y_train: np.ndarray, k: int,
                hardest: bool = False) -> np.ndarray:
    """Class-stratified easy/hard: round-robin over classes, picking each class's
    next easiest (or hardest) node — closes the class-collapse confound."""
    order = np.argsort(scores) if not hardest else np.argsort(-scores)
    by_class = {c: [i for i in order if y_train[i] == c] for c in np.unique(y_train)}
    sel: list[int] = []
    while len(sel) < k:
        for c in by_class:
            if by_class[c] and len(sel) < k:
                sel.append(by_class[c].pop(0))
    return np.asarray(sel)


# ---------------------------------------------------------------------------
# Selectors — all return LOCAL indices into the train pool
# ---------------------------------------------------------------------------

def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared distances (len(A), len(B)) without the (a, b, p) broadcast."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)


def _fps(X: np.ndarray, k: int, start: list[int] | None = None) -> np.ndarray:
    """Farthest-point sampling, deterministic medoid start."""
    sel = list(start) if start else [int(((X - X.mean(0)) ** 2).sum(1).argmin())]
    d = np.full(len(X), np.inf)
    for i in sel:
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    while len(sel) < k:
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    return np.asarray(sel[:k])


def _ccs_pc1(R_z: np.ndarray, k: int) -> np.ndarray:
    """CCS degenerated honestly for tiny k: quantile strata along PC1 of the
    fingerprint, median-rank representative per stratum (no tail pruning)."""
    _, _, Vt = np.linalg.svd(R_z - R_z.mean(0), full_matrices=False)
    order = np.argsort(R_z @ Vt[0])
    return np.array([c[len(c) // 2] for c in np.array_split(order, k)])


def _choose_radius(E: np.ndarray, n_classes: int, purity_floor: float = PROBCOVER_PURITY,
                   n_anchors: int = 500, seed: int = 0) -> tuple[float, float]:
    """Label-free ProbCover radius: largest r whose balls are >= purity_floor pure
    under k-means pseudo-labels (k = n_classes). Returns (r_star, purity_at_r_star)."""
    labels = KMeans(n_clusters=n_classes, random_state=seed, n_init=10).fit_predict(E)
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(E), size=min(n_anchors, len(E)), replace=False)
    D = np.sqrt(_sq_dists(E[anchors], E))
    radii = np.quantile(D[D > 0], np.geomspace(1e-4, 0.5, 25))
    same = labels[anchors][:, None] == labels[None, :]

    def _purity(r: float) -> float:
        ball = D <= r
        counts = ball.sum(1)
        return float(((ball & same).sum(1)[counts > 0] / counts[counts > 0]).mean())

    best = (float(radii[0]), _purity(float(radii[0])))
    for r in radii:
        purity = _purity(float(r))
        if purity >= purity_floor:
            best = (float(r), purity)
        else:
            break
    return best


def _probcover(E: np.ndarray, cand_global: np.ndarray, k: int, r: float) -> np.ndarray:
    """Greedy max coverage: candidates are the train pool, the universe is ALL nodes
    (the transductive objective is covering the whole graph). Falls back to FPS
    completion if every node is covered before k picks."""
    cov = _sq_dists(E[cand_global], E) <= r * r        # (n_train, N)
    uncovered = np.ones(E.shape[0], dtype=bool)
    sel: list[int] = []
    for _ in range(k):
        gains = (cov & uncovered).sum(1)
        gains[sel] = -1
        c = int(gains.argmax())
        if gains[c] <= 0:
            return _fps(E[cand_global], k, start=sel)
        sel.append(c)
        uncovered &= ~cov[c]
    return np.asarray(sel)


# ---------------------------------------------------------------------------
# E0-style diagnostic on the fixed fingerprint (gates probes G/H)
# ---------------------------------------------------------------------------

def _e0_diagnostics(ds: str, R_train_z: np.ndarray) -> dict | None:
    cache = ROOT / f"fingerprint_{ds.lower()}.npz"
    if not cache.exists():
        return None
    d = np.load(cache, allow_pickle=True)
    names = [str(x) for x in d["probe_names"]]

    def _spectrum(R: np.ndarray) -> dict:
        C = np.corrcoef(R, rowvar=False)
        lam = np.linalg.eigvalsh(C)
        lam = np.clip(lam, 0.0, None)
        p = lam / lam.sum()
        pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
        return {"participation_ratio": pr, "effective_rank": erank,
                "eigenvalues": lam[::-1].tolist()}

    R_all = d["residuals"].copy()
    for col in range(R_all.shape[1]):
        vals = R_all[:, col]
        finite = vals[np.isfinite(vals)]
        R_all[np.isnan(vals), col] = float(np.median(finite)) if len(finite) else 0.0
    rho = stats.spearmanr(R_train_z)[0]
    return {"probe_names": names,
            "all_nodes": _spectrum(R_all),
            "train_rows": _spectrum(R_train_z),
            "cross_probe_spearman_train": np.asarray(rho).round(3).tolist()}


# ---------------------------------------------------------------------------
# Evaluation harness (replicate.py pool pattern)
# ---------------------------------------------------------------------------

def _weights_for(sel_local: np.ndarray, n_train: int) -> np.ndarray:
    """Selection mask as a schedule: mean CE over the selected subset (see module doc)."""
    W = np.zeros((T, n_train))
    W[:, sel_local] = n_train / len(sel_local)
    return W


def _init(ds: str) -> None:
    global _DATA
    torch.set_num_threads(1)
    _DATA = _load_data(ds)


def _eval(W: np.ndarray) -> tuple[list[float], list[float]]:
    out = [train_once(_DATA, W, seed=s) for s in HOLD_SEEDS]
    return ([o["test_acc"] for o in out], [o["val_acc"] for o in out])


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def _build_subsets(n_train: int, budgets: tuple[int, ...], draws: int,
                   R_z: np.ndarray | None, E_train: np.ndarray,
                   E: np.ndarray, train_global: np.ndarray, gis: np.ndarray,
                   pc_radii: list[float], y_train: np.ndarray,
                   cln: np.ndarray) -> list[dict]:
    subsets = []
    for k_pct in budgets:
        k = max(1, round(k_pct / 100 * n_train))
        for draw in range(draws):
            rng = np.random.default_rng(1000 + draw)
            subsets.append({"arm": "random", "budget": k_pct, "draw": draw,
                            "sel": rng.choice(n_train, size=k, replace=False)})
        if R_z is not None:
            subsets.append({"arm": "fps_probe", "budget": k_pct, "draw": 0,
                            "sel": _fps(R_z, k)})
            subsets.append({"arm": "ccs_pc1", "budget": k_pct, "draw": 0,
                            "sel": _ccs_pc1(R_z, k)})
        subsets.append({"arm": "fps_sgc", "budget": k_pct, "draw": 0,
                        "sel": _fps(E_train, k)})
        subsets.append({"arm": "easy", "budget": k_pct, "draw": 0,
                        "sel": np.argsort(gis)[:k]})
        subsets.append({"arm": "hard", "budget": k_pct, "draw": 0,
                        "sel": np.argsort(gis)[-k:]})
        # Control arms (2026-07-24): close the class-collapse confound and test
        # whether the easy/hard flip survives CLNode's actual difficulty measure.
        # random_strat isolates stratification itself from within-class easiness.
        for draw in range(draws):
            rng = np.random.default_rng(3000 + draw)
            subsets.append({"arm": "random_strat", "budget": k_pct, "draw": draw,
                            "sel": _stratified(rng.random(n_train), y_train, k)})
        subsets.append({"arm": "easy_strat", "budget": k_pct, "draw": 0,
                        "sel": _stratified(gis, y_train, k, hardest=False)})
        subsets.append({"arm": "hard_strat", "budget": k_pct, "draw": 0,
                        "sel": _stratified(gis, y_train, k, hardest=True)})
        subsets.append({"arm": "clnode_easy", "budget": k_pct, "draw": 0,
                        "sel": np.argsort(cln)[:k]})
        subsets.append({"arm": "clnode_hard", "budget": k_pct, "draw": 0,
                        "sel": np.argsort(cln)[-k:]})
        for draw, r in enumerate(pc_radii):
            subsets.append({"arm": "probcover", "budget": k_pct, "draw": draw,
                            "sel": _probcover(E, train_global, k, r), "radius": r})
    return subsets


def process_dataset(ds: str, budgets: tuple[int, ...], draws: int, workers: int,
                    embeddings_npz: str | None, probcover_sweep: bool) -> None:
    data = _load_data(ds)
    train_global = data.train_mask.nonzero(as_tuple=True)[0].numpy()
    n_train = len(train_global)
    n_classes = int(data.y.max()) + 1
    y_train = data.y.numpy()[train_global]
    print(f"\n{'=' * 60}\n{ds}: {data.num_nodes} nodes, {n_train} train, "
          f"{n_classes} classes\n{'=' * 60}")

    R_z = _load_R_train(data, ds)
    if R_z is None:
        print("  no fingerprint cache — skipping fps_probe and ccs_pc1 arms")

    if embeddings_npz:
        E = np.load(embeddings_npz)["embeddings"].astype(np.float32)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        E = E / norms
        space = f"embeddings:{Path(embeddings_npz).name}"
    else:
        E = _sgc(data)
        space = "sgc2"
    E_train = E[train_global]

    print("  computing GIS difficulty …")
    gis = _gis_early(data)
    print("  computing CLNode difficulty …")
    cln = _clnode_difficulty(data)

    r_star, purity = _choose_radius(E, n_classes)
    D_sample = np.sqrt(_sq_dists(E[np.random.default_rng(0).choice(len(E), 200)], E))
    pc_radii = [r_star]
    if probcover_sweep:
        pc_radii += [float(np.quantile(D_sample[D_sample > 0], q)) for q in (0.25, 0.5)]
    print(f"  probcover r*={r_star:.4f} (purity {purity:.3f}); sweep radii "
          f"{[round(r, 4) for r in pc_radii]}")

    e0 = _e0_diagnostics(ds, R_z) if R_z is not None else None
    if e0:
        print(f"  E0: participation ratio {e0['train_rows']['participation_ratio']:.2f}, "
              f"effective rank {e0['train_rows']['effective_rank']:.2f} (train rows)")

    subsets = _build_subsets(n_train, budgets, draws, R_z, E_train, E,
                             train_global, gis, pc_radii, y_train, cln)

    t0 = time.time()
    pool = mp.Pool(workers, initializer=_init, initargs=(ds,))
    Ws = [_weights_for(s["sel"], n_train) for s in subsets] + [np.ones((T, n_train))]
    res = pool.map(_eval, Ws, chunksize=1)
    pool.close()
    pool.join()
    flat_test = np.array(res[-1][0])
    print(f"  evaluated {len(subsets)} subsets x {len(HOLD_SEEDS)} seeds + flat "
          f"({time.time() - t0:.0f}s)   flat_100: {flat_test.mean():.4f}")

    # ------------------------------------------------------------------ aggregate
    for s, (test, val) in zip(subsets, res):
        s["test"] = np.array(test)
        s["val"] = np.array(val)

    label_aware = {"easy", "hard", "easy_strat", "hard_strat",
                   "clnode_easy", "clnode_hard"}
    arms_out: dict[str, dict] = {}
    print(f"\n  {'arm':16s} {'k%':>4} {'test':>7} {'vs rand':>9} {'sem':>7} {'wins':>6}")
    for k_pct in budgets:
        rand = [s for s in subsets if s["arm"] == "random" and s["budget"] == k_pct]
        rand_mat = np.stack([s["test"] for s in rand])          # (draws, seeds)
        rand_per_seed = rand_mat.mean(axis=0)
        draw_means = rand_mat.mean(axis=1)
        arms_out.setdefault("random", {})[str(k_pct)] = {
            "test_mean": float(rand_mat.mean()),
            "sem": float(draw_means.std(ddof=1) / np.sqrt(len(draw_means))),
            "per_draw_mean": draw_means.tolist(),
            "per_seed_mean": rand_per_seed.tolist(),
            "n_selected": len(rand[0]["sel"]),
            "n_classes_covered": [int(len(np.unique(y_train[s['sel']]))) for s in rand],
            "selected_global_ids": [train_global[s["sel"]].tolist() for s in rand],
            "label_aware": False,
        }
        print(f"  {'random':16s} {k_pct:>4} {rand_mat.mean():7.4f} {'--':>9}")

        rs = [s for s in subsets if s["arm"] == "random_strat" and s["budget"] == k_pct]
        if rs:
            rs_mat = np.stack([s["test"] for s in rs])
            d = rs_mat.mean(axis=0) - rand_per_seed
            arms_out.setdefault("random_strat", {})[str(k_pct)] = {
                "test_mean": float(rs_mat.mean()),
                "gain_vs_random": float(d.mean()),
                "sem": float(rs_mat.mean(axis=1).std(ddof=1) / np.sqrt(len(rs))),
                "wins_vs_random": int((d > 0).sum()),
                "per_draw_mean": rs_mat.mean(axis=1).tolist(),
                "n_selected": len(rs[0]["sel"]),
                "n_classes_covered": [int(len(np.unique(y_train[s['sel']]))) for s in rs],
                "selected_global_ids": [train_global[s["sel"]].tolist() for s in rs],
                "label_aware": True,
            }
            print(f"  {'random_strat':16s} {k_pct:>4} {rs_mat.mean():7.4f} "
                  f"{d.mean():+9.4f} {'':>7} {int((d > 0).sum()):4d}/10")

        for s in subsets:
            if s["arm"] in ("random", "random_strat") or s["budget"] != k_pct:
                continue
            if s["arm"] == "probcover" and s["draw"] > 0:
                key = f"probcover_r{s['draw']}"
            else:
                key = s["arm"]
            d = s["test"] - rand_per_seed
            sem = d.std(ddof=1) / np.sqrt(len(d))
            wins = int((d > 0).sum())
            arms_out.setdefault(key, {})[str(k_pct)] = {
                "test_mean": float(s["test"].mean()),
                "gain_vs_random": float(d.mean()),
                "sem": float(sem),
                "wins_vs_random": wins,
                "per_seed": s["test"].tolist(),
                "n_selected": len(s["sel"]),
                "n_classes_covered": int(len(np.unique(y_train[s["sel"]]))),
                "selected_global_ids": train_global[s["sel"]].tolist(),
                "label_aware": s["arm"] in label_aware,
                **({"radius": s["radius"]} if "radius" in s else {}),
            }
            print(f"  {key:16s} {k_pct:>4} {s['test'].mean():7.4f} "
                  f"{d.mean():+9.4f} {sem:7.4f} {wins:4d}/10")

    # ------------------------------------------------------------------ save
    out = {
        "dataset": ds,
        "config": {
            "budgets": list(budgets), "draws": draws, "hold_seeds": list(HOLD_SEEDS),
            "epochs": T, "renorm": "mean-over-selected",
            "arm_c_space": space,
            "probcover": {"r_star": r_star, "purity": purity,
                          "sweep_radii": pc_radii, "purity_floor": PROBCOVER_PURITY},
            "gis": {"epochs": 15, "seeds": GIS_SEEDS, "variant": "gis_early"},
            "deviations": [
                "CCS degenerated to quantile strata + median representative "
                "(n_select as small as 3-7; no tail pruning) — differs from Zheng 2023",
                "ProbCover candidates restricted to the labeled train pool; "
                "coverage universe is all N nodes",
            ],
        },
        "e0_diagnostics": e0,
        "flat_100": {"test_mean": float(flat_test.mean()),
                     "per_seed": flat_test.tolist()},
        "arms": arms_out,
    }
    (ROOT / f"selection_{ds.lower()}.json").write_text(json.dumps(out, indent=1))

    max_k = max(len(s["sel"]) for s in subsets)
    pad = lambda a: np.pad(a, (0, max_k - len(a)), constant_values=-1)
    np.savez_compressed(
        ROOT / f"selection_{ds.lower()}.npz",
        arm_names=np.array([s["arm"] for s in subsets]),
        budgets=np.array([s["budget"] for s in subsets]),
        draw_ids=np.array([s["draw"] for s in subsets]),
        test_acc=np.stack([s["test"] for s in subsets]),
        val_acc=np.stack([s["val"] for s in subsets]),
        sel_local=np.stack([pad(s["sel"]) for s in subsets]).astype(np.int64),
        sel_global=np.stack([pad(train_global[s["sel"]]) for s in subsets]).astype(np.int64),
        flat_test=flat_test,
        hold_seeds=np.array(HOLD_SEEDS),
    )
    print(f"\n  wrote selection_{ds.lower()}.json + .npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer"])
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--draws", type=int, default=N_RANDOM_DRAWS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--embeddings-npz", default=None,
                    help="optional npz with key 'embeddings' (N,d) to replace SGC space")
    ap.add_argument("--no-probcover-sweep", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(1)
    for ds in a.datasets:
        process_dataset(ds, tuple(a.budgets), a.draws, a.workers,
                        a.embeddings_npz, not a.no_probcover_sweep)


if __name__ == "__main__":
    main()
