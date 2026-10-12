"""Grid search for the hybrid weight and SVD size, on the validation split.

I search SVD factors {20, 50, 100} x alpha {0.0, 0.1, ..., 1.0} and pick the
combo with the best NDCG@5 on validation. The content encoder only trains
once since it doesn't depend on the SVD factors. The blending here uses the
exact same per-user min-max normalisation as HybridModel, so the winning
alpha carries straight over to the test-time hybrid in src/evaluator.py.

Writes logs/grid_search.json with the whole grid plus the best cell.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_all, sha256_of_file
from evaluator import (
    K,
    RELEVANCE_THRESHOLD,
    _git_commit,
    _item_texts,
    _topk_ordered,
    make_splits,
    ranking_metrics_for_user,
)
from hybrid_engine import ContentEncoder, HybridModel, SVDModel

SVD_FACTORS_GRID = [20, 50, 100]
ALPHA_GRID = [round(a, 1) for a in np.arange(0.0, 1.01, 0.1)]
SELECTION_METRIC = "ndcg@5"


def _evaluate_factor_on_validation(svd, content, alpha_grid, val_by_user,
                                   train_items, n_items):
    """Score one SVD factorisation on validation, for every alpha.

    Users in the outer loop, alphas in the inner loop -- that way only two
    score vectors per user are ever in memory at once. (My first version
    stored all users' vectors up front and got OOM-killed at ~6 GB, so this
    ordering is load-bearing.) Blending reuses HybridModel's normalisation
    exactly.
    """
    # Per-alpha accumulators.
    acc = {a: {"ndcg": 0.0, "prec": 0.0, "rec": 0.0, "rec_n": 0} for a in alpha_grid}
    n_users = 0
    for u, g in val_by_user.items():
        rel_items = set(g.loc[g["rating"] >= RELEVANCE_THRESHOLD, "i"].astype(int).tolist())
        cf_n = HybridModel._minmax(svd.score_all_items(u).astype(float))
        co_n = HybridModel._minmax(content.score_all_items(u).astype(float))
        excluded = train_items.get(u, set())
        excl_idx = (np.fromiter(excluded, dtype=int, count=len(excluded))
                    if excluded else None)
        for alpha in alpha_grid:
            blended = alpha * cf_n + (1.0 - alpha) * co_n
            if excl_idx is not None:
                blended[excl_idx] = -np.inf
            top = _topk_ordered(blended, K)
            top = top[blended[top] > -np.inf]
            ndcg, prec, rec = ranking_metrics_for_user(top, rel_items)
            a = acc[alpha]
            a["ndcg"] += ndcg
            a["prec"] += prec
            if rec is not None:
                a["rec"] += rec
                a["rec_n"] += 1
        n_users += 1
    out = []
    for alpha in alpha_grid:
        a = acc[alpha]
        out.append({
            "alpha": alpha,
            "ndcg@5": a["ndcg"] / n_users if n_users else 0.0,
            "precision@5": a["prec"] / n_users if n_users else 0.0,
            "recall@5": a["rec"] / a["rec_n"] if a["rec_n"] else 0.0,
            "n_val_users": n_users,
            "n_val_users_with_relevant": a["rec_n"],
        })
    return out


def run_grid_search(svd_factors_grid: list[int] = SVD_FACTORS_GRID,
                    alpha_grid: list[float] = ALPHA_GRID,
                    seed: int = 42,
                    log_dir: Path | str = "logs",
                    data_dir: Path | str = "data") -> dict:
    import torch
    torch.set_num_threads(1)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    t0 = time.time()

    ratings, metadata = load_all(data_dir / "ratings.csv", data_dir / "metadata.csv")
    train, val, test, _ = make_splits(ratings)
    n_users = int(ratings["user_id"].nunique())
    n_items = int(ratings["item_id"].nunique())
    item_index = np.sort(ratings["item_id"].unique())
    item_texts = _item_texts(metadata, item_index)
    n_ratings_rows = int(len(ratings))
    del ratings, metadata, item_index
    import gc
    gc.collect()

    train_items = {int(u): set(g["i"].astype(int).tolist()) for u, g in train.groupby("u")}
    val_by_user = {int(u): g for u, g in val.groupby("u")}

    print("Training content encoder (once, shared across the grid)...", flush=True)
    content = ContentEncoder(seed=seed).fit(train, n_users, n_items, item_texts)

    config = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "svd_factors_grid": list(svd_factors_grid),
        "alpha_grid": list(alpha_grid),
        "selection_metric": SELECTION_METRIC,
        "selection_split": "validation",
        "svd_lr": 0.005, "svd_reg": 0.02, "svd_epochs": 20,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "ranking_k": K,
        "data": {
            "ratings_rows": n_ratings_rows,
            "n_users": n_users, "n_items": n_items,
            "n_train": int(len(train)), "n_validation": int(len(val)),
            "ratings_sha256": sha256_of_file(data_dir / "ratings.csv"),
            "metadata_sha256": sha256_of_file(data_dir / "metadata.csv"),
        },
    }

    grid_results: list[dict] = []
    best: dict | None = None
    for factors in svd_factors_grid:
        f0 = time.time()
        svd = SVDModel(n_factors=factors, lr=0.005, reg=0.02,
                       n_epochs=20, seed=seed).fit(train, n_users, n_items)
        cells = _evaluate_factor_on_validation(svd, content, list(alpha_grid),
                                               val_by_user, train_items, n_items)
        for cell in cells:
            cell = {"svd_factors": factors, **cell}
            grid_results.append(cell)
            if best is None or cell[SELECTION_METRIC] > best[SELECTION_METRIC]:
                best = dict(cell)
        print(f"factors={factors}: done in {time.time() - f0:.1f}s "
              f"(best so far: {best})", flush=True)

    payload = {"config": config, "grid_results": grid_results, "best": best,
               "elapsed_seconds": time.time() - t0}
    with open(log_dir / "grid_search.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Grid search complete in {time.time() - t0:.1f}s. "
          f"Best: {best}. Log: {log_dir}/grid_search.json", flush=True)
    return payload


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Grid search SVD factors x alpha on validation.")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run_grid_search(log_dir=args.log_dir, data_dir=args.data_dir, seed=args.seed)
