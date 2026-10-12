"""How the models get evaluated (the Milestone 3 redo).

The split: for each user I sort their ratings by timestamp, newest first
(ties broken by item id so it's deterministic, no randomness involved):
  * newest rating        -> test
  * second newest        -> validation
  * everything else      -> train
Users with fewer than 3 ratings can't fill all three buckets, so all of
theirs goes to train. In this dataset that case never happens (it's 5-core),
but the rule is there and documented.

The important part: every model trains ONLY on the train split. The old code
trained on everything and then "tested" on part of the same data, which is
exactly the leakage my supervisor flagged. Test evaluation runs over ALL test
users (not 30 like before), and for each user the candidates exclude their
train and validation items. When tuning on validation, candidates exclude
just the train items. The test item is never used as recommendation context.

Metrics -- everything here is computed by this file, nothing is typed in:
* RMSE / MAE: only for models that actually predict ratings (Popularity,
  ItemKNN, SVD). RMSE on the hybrid's blended cosine score makes no sense,
  so the hybrid and pure-content models only get ranking metrics.
* NDCG@5, Precision@5, Recall@5: for every model, with a fixed relevance
  cutoff of rating >= 4.0. Precision and NDCG average over all evaluated
  users; Recall@5 averages over users who have at least one relevant item
  (for the rest recall is undefined, so they're left out of that average).

Multi-seed: run_experiment() runs the exact same pipeline -- split, train
everything on train, evaluate on test -- once per seed in [42, 123, 2026, 7,
99], then reports mean +/- 95% CI (1.96 * std / sqrt(n)) per model per
metric. Same code path for every model, every seed.

Logs: each run writes logs/eval_seeds.json (per-seed numbers), 
logs/eval_summary.json (means + CIs) and logs/summary.md (readable version).
The config block in there records the git commit, dataset row counts and
SHA-256 hashes, the split rule, seeds, hyper-parameters and a timestamp --
so any number in the report can be traced back to a committed log file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import encode_ids, load_all, sha256_of_file
from hybrid_engine import (
    RELEVANCE_THRESHOLD,
    ContentEncoder,
    HybridModel,
    ItemKNNModel,
    PopularityModel,
    SVDModel,
)

K = 5
SEEDS = [42, 123, 2026, 7, 99]
RATING_MODEL_NAMES = {"popularity", "itemknn", "svd"}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(ratings: pd.DataFrame, min_ratings: int = 3):
    """Per-user train/validation/test split. Deterministic.

    Returns (train, validation, test, test_users) as encoded DataFrames with
    integer ``u``/``i`` columns, plus the sorted user/item id arrays.
    """
    enc, user_to_idx, item_to_idx = encode_ids(ratings)
    # Deterministic ordering: user asc, timestamp desc, item_id asc.
    enc = enc.sort_values(["u", "timestamp", "item_id"],
                          ascending=[True, False, True]).reset_index(drop=True)
    counts = enc.groupby("u").size()
    train_rows, val_rows, test_rows = [], [], []
    for u, g in enc.groupby("u", sort=True):
        idx = g.index.to_numpy()
        if len(idx) < min_ratings:
            train_rows.extend(idx)          # documented rule: all to train
        else:
            test_rows.append(idx[0])        # most recent
            val_rows.append(idx[1])         # second most recent
            train_rows.extend(idx[2:])      # rest
    train = enc.loc[sorted(train_rows)].reset_index(drop=True)
    val = enc.loc[sorted(val_rows)].reset_index(drop=True)
    test = enc.loc[sorted(test_rows)].reset_index(drop=True)
    test_users = np.sort(test["u"].unique())
    return train, val, test, test_users


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def _topk_ordered(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores, ordered by (-score, item index).

    Ties go to the lower item index so results are reproducible.
    argpartition + sorting just the k winners is much faster than a full
    sort when the catalog has 40k+ items.
    """
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=int)
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
    else:
        part = np.arange(n)
    order = np.lexsort((part, -scores[part]))
    return part[order]


def _dcg(relevances: np.ndarray) -> float:
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum((2.0 ** relevances - 1.0) / discounts))


def ranking_metrics_for_user(ranked_items: np.ndarray,
                             rel_item_set: set[int]) -> tuple[float, float, float | None]:
    """Return (ndcg@K, precision@K, recall@K or None if no relevant items)."""
    topk = ranked_items[:K]
    hits = np.array([1.0 if int(i) in rel_item_set else 0.0 for i in topk])
    n_rel = len(rel_item_set)
    precision = float(hits.sum() / K)
    ideal = np.ones(min(n_rel, K))
    ndcg = _dcg(hits) / _dcg(ideal) if n_rel > 0 else 0.0
    recall = float(hits.sum() / n_rel) if n_rel > 0 else None
    return ndcg, precision, recall


# --------------------------------------------------------------------------- #
# Model training (identical code path for every seed and every model)
# --------------------------------------------------------------------------- #
def train_all_models(train: pd.DataFrame, n_users: int, n_items: int,
                     item_texts: list[str], svd_factors: int, alpha: float,
                     seed: int) -> dict:
    """Train Popularity, ItemKNN, SVD, Content and the Hybrid variants.

    All on the train split, all with the same seed. The hybrid_alpha0/alpha1
    entries are the ablation: alpha=0 is pure content, alpha=1 is pure CF.
    """
    models: dict = {}

    pop = PopularityModel().fit(train, n_users, n_items)
    models["popularity"] = pop

    knn = ItemKNNModel(k=50, shrinkage=100.0, seed=seed).fit(train, n_users, n_items)
    models["itemknn"] = knn

    svd = SVDModel(n_factors=svd_factors, lr=0.005, reg=0.02,
                   n_epochs=20, seed=seed).fit(train, n_users, n_items)
    models["svd"] = svd

    content = ContentEncoder(seed=seed).fit(train, n_users, n_items, item_texts)
    models["content"] = content

    # Hybrid at the tuned alpha plus the ablation endpoints.
    models["hybrid"] = HybridModel(svd, content, alpha=alpha)
    models["hybrid_alpha0"] = HybridModel(svd, content, alpha=0.0)  # pure content
    models["hybrid_alpha1"] = HybridModel(svd, content, alpha=1.0)  # pure CF (SVD)
    return models


# --------------------------------------------------------------------------- #
# Evaluation of one model set on one eval partition
# --------------------------------------------------------------------------- #
def evaluate_models(models: dict, train: pd.DataFrame, eval_df: pd.DataFrame,
                    n_items: int, extra_excluded: dict[int, set[int]] | None = None
                    ) -> dict[str, dict[str, float]]:
    """RMSE/MAE for the rating models, NDCG@5/P@5/R@5 for everything.

    Each user's candidates exclude their train items; extra_excluded adds
    more per-user exclusions -- the validation items when we're evaluating
    on the test split.
    """
    train_items = {int(u): set(g["i"].astype(int).tolist())
                   for u, g in train.groupby("u")}
    eval_by_user = {int(u): g for u, g in eval_df.groupby("u")}

    # Precompute the eval triples once for the RMSE/MAE models.
    eval_pairs = eval_df[["u", "i"]].to_numpy()
    eval_true = eval_df["rating"].to_numpy(dtype=float)

    results: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        metrics: dict[str, float] = {}
        if name in RATING_MODEL_NAMES:
            preds = model.predict_ratings(eval_pairs)
            err = preds - eval_true
            metrics["rmse"] = float(np.sqrt(np.mean(err ** 2)))
            metrics["mae"] = float(np.mean(np.abs(err)))

        ndcgs, precs, recs = [], [], []
        for u, g in eval_by_user.items():
            rel_items = set(g.loc[g["rating"] >= RELEVANCE_THRESHOLD, "i"].astype(int).tolist())
            excluded = set(train_items.get(u, set()))
            if extra_excluded:
                excluded |= extra_excluded.get(u, set())
            scores = model.score_all_items(u).astype(float)
            if excluded:
                scores[np.fromiter(excluded, dtype=int, count=len(excluded))] = -np.inf
            # Deterministic top-K: score desc, item index asc.
            top = _topk_ordered(scores, K)
            top = top[scores[top] > -np.inf]
            ndcg, prec, rec = ranking_metrics_for_user(top, rel_items)
            ndcgs.append(ndcg)
            precs.append(prec)
            if rec is not None:
                recs.append(rec)
        metrics["ndcg@5"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        metrics["precision@5"] = float(np.mean(precs)) if precs else 0.0
        metrics["recall@5"] = float(np.mean(recs)) if recs else 0.0
        metrics["n_eval_users"] = int(len(eval_by_user))
        metrics["n_eval_users_with_relevant"] = int(len(recs))
        results[name] = metrics
    return results


# --------------------------------------------------------------------------- #
# Multi-seed experiment driver
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown (not a git repo)"
    except Exception:
        return "unknown (git unavailable)"


def _item_texts(metadata: pd.DataFrame, item_index: pd.Index) -> list[str]:
    # Per-item text truncated to 1500 chars to bound vectoriser memory.
    text_by_id = dict(zip(metadata["item_id"].astype(str),
                          (metadata["title"] + " " + metadata["description"]).astype(str)))
    return [text_by_id.get(str(i), "")[:1500] for i in item_index]


def run_experiment(seeds: list[int] = SEEDS, svd_factors: int = 50,
                   alpha: float = 0.5, log_dir: Path | str = "logs",
                   data_dir: Path | str = "data") -> dict:
    """Full pipeline per seed; writes raw JSON logs; returns the summary dict."""
    import torch
    torch.set_num_threads(1)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    t0 = time.time()
    ratings, metadata = load_all(data_dir / "ratings.csv", data_dir / "metadata.csv")
    train, val, test, test_users = make_splits(ratings)
    n_users = int(ratings["user_id"].nunique())
    n_items = int(ratings["item_id"].nunique())
    item_index = np.sort(ratings["item_id"].unique())
    item_texts = _item_texts(metadata, item_index)
    n_ratings_rows, n_metadata_rows = int(len(ratings)), int(len(metadata))
    # Free the raw frames: only the splits and item texts are needed below.
    # (Keeps peak memory down on small machines.)
    del ratings, metadata, item_index
    import gc
    gc.collect()

    # Validation items per user, needed to exclude them from test candidates.
    val_items_by_user = {int(u): set(g["i"].astype(int).tolist())
                         for u, g in val.groupby("u")}

    config = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "svd_factors": svd_factors,
        "hybrid_alpha": alpha,
        "svd_lr": 0.005, "svd_reg": 0.02, "svd_epochs": 20,
        "itemknn_k": 50, "itemknn_shrinkage": 100.0,
        "content_emb_dim": 32, "content_tfidf_features": 512,
        "content_lr": 0.01, "content_epochs": 10,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "ranking_k": K,
        "split_rule": ("per-user: most recent rating -> test, second most recent -> "
                       "validation, rest -> train; users with <3 ratings all train; "
                       "ordered by timestamp desc, item_id asc; test candidates exclude "
                       "train+validation items"),
        "data": {
            "ratings_rows": n_ratings_rows,
            "metadata_rows": n_metadata_rows,
            "n_users": n_users, "n_items": n_items,
            "n_train": int(len(train)), "n_validation": int(len(val)),
            "n_test": int(len(test)), "n_test_users": int(len(test_users)),
            "ratings_sha256": sha256_of_file(data_dir / "ratings.csv"),
            "metadata_sha256": sha256_of_file(data_dir / "metadata.csv"),
        },
    }

    per_seed: dict[str, dict] = {}
    # Resume support: if a previous run wrote a partial checkpoint with the
    # same config (factors, alpha, data), pick up its finished seeds instead
    # of redoing them.
    partial_path = log_dir / "eval_seeds_partial.json"
    if partial_path.is_file():
        try:
            partial = json.load(open(partial_path))
            pc = partial.get("config", {})
            same = (pc.get("svd_factors") == svd_factors
                    and pc.get("hybrid_alpha") == alpha
                    and pc.get("data", {}).get("ratings_sha256")
                    == config["data"]["ratings_sha256"])
            if same:
                for s in seeds:
                    if str(s) in partial.get("per_seed", {}):
                        per_seed[str(s)] = partial["per_seed"][str(s)]
                if per_seed:
                    print(f"Resuming with {len(per_seed)} finished seed(s) "
                          f"from {partial_path}.", flush=True)
            else:
                print("Partial checkpoint is for a different config; ignoring it.",
                      flush=True)
        except Exception as e:
            print(f"Ignoring unreadable partial checkpoint: {e}", flush=True)

    for seed in seeds:
        if str(seed) in per_seed:
            continue
        s0 = time.time()
        models = train_all_models(train, n_users, n_items, item_texts,
                                  svd_factors=svd_factors, alpha=alpha, seed=seed)
        # Attach validation exclusions for the test-time candidate filter.
        metrics = evaluate_models(models, train, test, n_items,
                                  extra_excluded=val_items_by_user)
        per_seed[str(seed)] = metrics
        # Checkpoint after every seed: a late failure must not lose finished seeds.
        with open(log_dir / "eval_seeds_partial.json", "w") as f:
            json.dump({"config": config, "per_seed": per_seed,
                       "seeds_done": [str(s) for s in seeds if str(s) in per_seed]}, f)
        print(f"[seed {seed}] done in {time.time() - s0:.1f}s", flush=True)

    # Aggregate: mean +/- 95% CI. Each model only aggregates the metrics it
    # actually has (rmse/mae exist only for the rating-prediction models).
    model_names = list(per_seed[str(seeds[0])].keys())
    summary: dict[str, dict[str, dict[str, float]]] = {}
    n = len(seeds)
    for name in model_names:
        present = [m for m in per_seed[str(seeds[0])][name].keys()
                   if not m.startswith("n_eval")]
        summary[name] = {}
        for metric in present:
            vals = np.array([per_seed[str(s)][name][metric] for s in seeds], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
            summary[name][metric] = {"mean": mean, "std": std, "ci95": ci95,
                                     "n_seeds": n, "per_seed": [float(v) for v in vals]}

    payload = {"config": config, "per_seed": per_seed, "summary": summary,
               "elapsed_seconds": time.time() - t0}
    with open(log_dir / "eval_seeds.json", "w") as f:
        json.dump({"config": config, "per_seed": per_seed}, f, indent=2)
    with open(log_dir / "eval_summary.json", "w") as f:
        json.dump(payload, f, indent=2)
    _write_summary_md(log_dir / "summary.md", config, summary)

    print(f"Evaluation complete in {time.time() - t0:.1f}s. "
          f"Logs: {log_dir}/eval_seeds.json, eval_summary.json, summary.md", flush=True)
    return payload


def _write_summary_md(path: Path, config: dict, summary: dict) -> None:
    lines = ["# Evaluation summary (computed by src/evaluator.py)", "",
             f"- Timestamp (UTC): {config['timestamp_utc']}",
             f"- Git commit: {config['git_commit']}",
             f"- Seeds: {config['seeds']}",
             f"- SVD factors: {config['svd_factors']}, hybrid alpha: {config['hybrid_alpha']}",
             f"- Data: {config['data']['ratings_rows']} ratings, "
             f"{config['data']['n_users']} users, {config['data']['n_items']} items",
             f"- Split: train={config['data']['n_train']}, "
             f"validation={config['data']['n_validation']}, test={config['data']['n_test']} "
             f"({config['data']['n_test_users']} test users)",
             f"- Relevance threshold: >= {config['relevance_threshold']}, K={config['ranking_k']}",
             "", "Mean +/- 95% CI over seeds:", ""]
    for name, metrics in summary.items():
        lines.append(f"## {name}")
        for metric, agg in metrics.items():
            lines.append(f"- {metric}: {agg['mean']:.4f} +/- {agg['ci95']:.4f} "
                         f"(per-seed: {', '.join(f'{v:.4f}' for v in agg['per_seed'])})")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Multi-seed evaluation on the test split.")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--svd-factors", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run_experiment(seeds=args.seeds, svd_factors=args.svd_factors,
                   alpha=args.alpha, log_dir=args.log_dir, data_dir=args.data_dir)
