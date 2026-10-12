"""Time each pipeline stage and save to logs/benchmark.json.

Pure measurement: training here uses seed 42 defaults and never touches the
logged evaluation results (those live in logs/eval_seeds.json already).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

import sys
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_all  # noqa: E402
from evaluator import make_splits, train_all_models  # noqa: E402


def _time(label: str, fn, timings: dict):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    timings[label] = round(dt, 2)
    print(f"  {label}: {dt:.1f}s", flush=True)
    return out


def main() -> None:
    timings: dict[str, float] = {}
    print("Benchmarking pipeline stages...", flush=True)

    def _load():
        return load_all(ROOT / "data" / "ratings.csv",
                        ROOT / "data" / "metadata.csv")
    ratings, metadata = _time("data_load_s", _load, timings)
    train, val, test, _ = _time(
        "make_splits_s", lambda: make_splits(ratings), timings)
    n_users = int(ratings["user_id"].nunique())
    n_items = int(ratings["item_id"].nunique())
    item_index = np.sort(ratings["item_id"].unique())
    text_by_id = dict(zip(metadata["item_id"].astype(str),
                          (metadata["title"].fillna("").astype(str) + " "
                           + metadata["description"].fillna("").astype(str))))
    item_texts = [text_by_id.get(str(i), "")[:1500] for i in item_index]

    from hybrid_engine import (ContentEncoder, ItemKNNModel, PopularityModel,
                               SVDModel, HybridModel)
    _time("train_popularity_s",
          lambda: PopularityModel().fit(train, n_users, n_items), timings)
    _time("train_itemknn_s",
          lambda: ItemKNNModel().fit(train, n_users, n_items), timings)
    _time("train_svd_s",
          lambda: SVDModel(n_factors=32, seed=42).fit(train, n_users, n_items),
          timings)
    _time("train_content_s",
          lambda: ContentEncoder().fit(train, n_users, n_items, item_texts),
          timings)

    svd = SVDModel(n_factors=32, seed=42).fit(train, n_users, n_items)
    content = ContentEncoder().fit(train, n_users, n_items, item_texts)
    hybrid = HybridModel(svd, content, alpha=0.7)

    def _rec():
        s = hybrid.score_all_items(0)
        return s.argsort()[-5:]
    _time("hybrid_recommend_one_user_s", _rec, timings)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = {"n_users": n_users, "n_items": n_items, "timings_s": timings}
    (LOG_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
    print("Wrote logs/benchmark.json", flush=True)


if __name__ == "__main__":
    main()
