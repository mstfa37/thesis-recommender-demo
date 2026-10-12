"""Train every model on the train split and save it to models/.

The UI (app.py) loads these files instead of retraining on every start.
SVD uses factors=20 and the hybrid alpha=0.7 comes from the tuning run
(logs/grid_search.json); the UI only needs the trained components.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_all  # noqa: E402
from evaluator import _item_texts, make_splits  # noqa: E402
from hybrid_engine import (  # noqa: E402
    ContentEncoder,
    ItemKNNModel,
    PopularityModel,
    SVDModel,
)

SVD_FACTORS = 20
SEED = 42
MODEL_DIR = ROOT / "models"


def train_and_save(data_dir: Path = ROOT / "data",
                   model_dir: Path = MODEL_DIR) -> dict:
    import torch
    torch.set_num_threads(1)

    model_dir.mkdir(parents=True, exist_ok=True)
    ratings, metadata = load_all(data_dir / "ratings.csv", data_dir / "metadata.csv")
    train, _, _, _ = make_splits(ratings)
    n_users = int(ratings["user_id"].nunique())
    n_items = int(ratings["item_id"].nunique())
    item_texts = _item_texts(metadata, np.sort(ratings["item_id"].unique()))
    del ratings, metadata

    timings = {}

    t = time.time()
    pop = PopularityModel().fit(train, n_users, n_items)
    pop.save(model_dir / "popularity.npz")
    timings["popularity"] = time.time() - t

    t = time.time()
    knn = ItemKNNModel(k=50, shrinkage=100.0, seed=SEED).fit(train, n_users, n_items)
    knn.save(model_dir / "itemknn.npz")
    timings["itemknn"] = time.time() - t

    t = time.time()
    svd = SVDModel(n_factors=SVD_FACTORS, lr=0.005, reg=0.02,
                   n_epochs=20, seed=SEED, verbose=True).fit(train, n_users, n_items)
    svd.save(model_dir / "svd.npz")
    timings["svd"] = time.time() - t

    t = time.time()
    content = ContentEncoder(seed=SEED).fit(train, n_users, n_items, item_texts)
    content.save(model_dir / "content.pt")
    timings["content"] = time.time() - t

    for name, dt in timings.items():
        print(f"{name}: trained+saved in {dt:.1f}s", flush=True)
    print(f"Models saved to {model_dir}", flush=True)
    return timings


if __name__ == "__main__":
    train_and_save()
