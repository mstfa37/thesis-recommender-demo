"""Build a compact data/reviews.csv from the raw Amazon reviews.

Keeps, per catalog item, up to 3 reviews ranked by helpful votes, with
truncated text. Used by the storefront's product pages.
"""

from __future__ import annotations

import ast
import csv
import gzip
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_REVIEWS = ROOT / "data_raw" / "reviews_Electronics_5.json.gz"
METADATA_CSV = ROOT / "data" / "metadata.csv"
OUT = ROOT / "data" / "reviews.csv"
PER_ITEM = 3
TEXT_LEN = 400


def _helpful(d: dict) -> int:
    h = d.get("helpful") or [0, 0]
    try:
        return int(h[0])
    except Exception:
        return 0


def main() -> None:
    import pandas as pd

    catalog = set(pd.read_csv(METADATA_CSV, usecols=["item_id"],
                              dtype=str)["item_id"])
    print(f"catalog: {len(catalog)} items", flush=True)

    best: dict[str, list] = {}
    t0 = time.time()
    with gzip.open(RAW_REVIEWS, "rt", encoding="utf-8",
                   errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
            except Exception:
                continue
            asin = str(d.get("asin") or "")
            if asin not in catalog:
                continue
            entry = (_helpful(d), str(d.get("reviewerName") or "a shopper"),
                     d.get("overall", 0),
                     str(d.get("summary") or "")[:100].replace("\n", " "),
                     str(d.get("reviewText") or "")[:TEXT_LEN].replace("\n", " "))
            bucket = best.setdefault(asin, [])
            bucket.append(entry)
    print(f"scanned reviews in {time.time() - t0:.1f}s", flush=True)

    n = 0
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "reviewer", "rating", "summary", "text"])
        for asin, entries in best.items():
            entries.sort(key=lambda e: -e[0])
            for _, reviewer, rating, summary, text in entries[:PER_ITEM]:
                w.writerow([asin, reviewer, rating, summary, text])
                n += 1
    mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT}: {n} reviews, {mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
