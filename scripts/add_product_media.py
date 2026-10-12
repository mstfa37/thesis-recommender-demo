"""Add image_url and price columns to data/metadata.csv.

Reads the raw meta_Electronics.json.gz (already in data_raw/) and pulls the
imUrl + price fields per ASIN, then merges them into the existing
metadata.csv. Run once; rerunning just rewrites the same file.
"""

from __future__ import annotations

import ast
import gzip
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_META = ROOT / "data_raw" / "meta_Electronics.json.gz"
METADATA_CSV = ROOT / "data" / "metadata.csv"


def main() -> None:
    import pandas as pd

    print("Reading image urls + prices from the raw metadata...", flush=True)
    t0 = time.time()
    media: dict[str, tuple[str, str, str]] = {}
    with gzip.open(RAW_META, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
            except Exception:
                continue
            asin = d.get("asin")
            if not asin:
                continue
            url = str(d.get("imUrl") or "")
            # normalize to the maintained Amazon CDN host (the old
            # ecx.images-amazon.com host is deprecated and flaky in browsers)
            if "/images/I/" in url:
                url = ("https://m.media-amazon.com/images/I/"
                       + url.split("/images/I/")[-1])
            else:
                url = url.replace("http://", "https://")
            price = d.get("price")
            price = "" if price is None else str(price).strip()
            cats = d.get("categories") or []
            cat = cats[0][-1] if cats and cats[0] else ""
            media[str(asin)] = (url, price, str(cat))
    print(f"  {len(media)} products with media info in {time.time() - t0:.1f}s",
          flush=True)

    meta = pd.read_csv(METADATA_CSV, dtype=str).fillna("")
    meta["image_url"] = meta["item_id"].map(lambda x: media.get(x, ("", "", ""))[0])
    meta["price"] = meta["item_id"].map(lambda x: media.get(x, ("", "", ""))[1])
    meta["category"] = meta["item_id"].map(lambda x: media.get(x, ("", "", ""))[2])
    meta.to_csv(METADATA_CSV, index=False)
    n_img = (meta["image_url"] != "").sum()
    n_price = (meta["price"] != "").sum()
    n_cat = (meta["category"] != "").sum()
    print(f"Wrote {METADATA_CSV}: {n_img}/{len(meta)} with images, "
          f"{n_price}/{len(meta)} with prices, {n_cat}/{len(meta)} with category",
          flush=True)


if __name__ == "__main__":
    main()
