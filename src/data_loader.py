"""Loading the dataset.

data/ratings.csv holds (user_id, item_id, rating, timestamp) and
data/metadata.csv holds (item_id, title, description). Both are built by
scripts/download_data.py from the real Amazon Electronics snapshot.

If either file is missing we raise FileNotFoundError and say exactly how to
get it. We never generate fake data as a fallback -- the old Milestone 2
loader silently made up 1200 random ratings when the CSV was missing, which
is also what CI ended up running on. Failing loudly is the whole point.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
METADATA_COLUMNS = ["item_id", "title", "description"]

DEFAULT_RATINGS_PATH = Path("data/ratings.csv")
DEFAULT_METADATA_PATH = Path("data/metadata.csv")


def _require_file(path: Path | str, kind: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {path} ({kind}). "
            "This pipeline does not generate synthetic data. To obtain the real "
            "dataset, run: python scripts/download_data.py "
            "(downloads the Amazon Electronics 5-core snapshot and builds "
            "data/ratings.csv and data/metadata.csv)."
        )
    return path


def load_ratings(path: Path | str = DEFAULT_RATINGS_PATH) -> pd.DataFrame:
    """Load and validate data/ratings.csv. Raises FileNotFoundError if missing."""
    path = _require_file(path, "ratings")
    df = pd.read_csv(path)
    missing = [c for c in RATINGS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df[RATINGS_COLUMNS].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="raise")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    if df[["user_id", "item_id", "rating", "timestamp"]].isna().any().any():
        raise ValueError(f"{path} contains null values in required columns.")
    if ((df["rating"] < 1) | (df["rating"] > 5)).any():
        raise ValueError(f"{path} contains ratings outside the [1, 5] range.")
    return df.reset_index(drop=True)


def load_metadata(path: Path | str = DEFAULT_METADATA_PATH) -> pd.DataFrame:
    """Load and validate data/metadata.csv. Raises FileNotFoundError if missing."""
    path = _require_file(path, "item metadata")
    df = pd.read_csv(path)
    missing = [c for c in METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df[METADATA_COLUMNS].copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df.drop_duplicates(subset="item_id").reset_index(drop=True)


def load_all(
    ratings_path: Path | str = DEFAULT_RATINGS_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and metadata together (both must exist)."""
    return load_ratings(ratings_path), load_metadata(metadata_path)


def encode_ids(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map string user/item ids to 0..n-1 ints.

    Done in sorted id order so the mapping doesn't depend on row order or
    any random state -- same input, same codes, every run.
    """
    users = np.sort(ratings["user_id"].unique())
    items = np.sort(ratings["item_id"].unique())
    user_to_idx = {u: k for k, u in enumerate(users)}
    item_to_idx = {i: k for k, i in enumerate(items)}
    enc = ratings.copy()
    enc["u"] = enc["user_id"].map(user_to_idx).astype(np.int64)
    enc["i"] = enc["item_id"].map(item_to_idx).astype(np.int64)
    return enc, user_to_idx, item_to_idx


def sha256_of_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
