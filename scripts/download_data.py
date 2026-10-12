"""Download the Amazon Electronics data and build data/ratings.csv + data/metadata.csv.

Where the data comes from (checked on 2026-10-11): the Amazon review dataset
from McAuley's lab at UCSD, Electronics 5-core snapshot, mirrored by Stanford
SNAP:
    http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
    http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz

One honest caveat: I wanted the 2018 release, but its official host
(jmcauley.ucsd.edu) is retired and the per-category 2018 files are behind a
manual access-request form now; deepyeti.ucsd.edu doesn't respond at all. So
this uses the same dataset family's 5-core Electronics snapshot (1,689,188
reviews, 1996-2014) from the SNAP mirror. Same schema (reviewerID, asin,
overall, unixReviewTime, plus title/description from the metadata file matched
by ASIN). Nothing synthetic anywhere -- if a download fails, the script fails
instead of inventing data.

Sampling: to keep runtimes sane I keep ~9% of users -- a user is kept iff
int(md5(reviewerID).hexdigest(), 16) % 100 < 9 -- with ALL of their reviews.
Hash-based, so it's deterministic without needing to store any state, and
keeping whole user histories is what the per-user train/val/test split needs.

Outputs: data/ratings.csv (user_id, item_id, rating, timestamp),
data/metadata.csv (item_id, title, description),
logs/download_log.json (URLs, counts, file hashes, timestamp).
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REVIEWS_URL = ("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
               "reviews_Electronics_5.json.gz")
META_URL = ("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
            "meta_Electronics.json.gz")
# Keep users whose md5 hash mod 100 is below this -> ~9% of users.
HASH_MOD = 100
HASH_KEEP_BELOW = 9

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data_raw"
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"


def _download(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"Reusing existing {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"Downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ELTE-BSc-thesis/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _keep_user(reviewer_id: str) -> bool:
    return int(hashlib.md5(reviewer_id.encode("utf-8")).hexdigest(), 16) % HASH_MOD < HASH_KEEP_BELOW


def _parse_line(line: bytes):
    try:
        return json.loads(line)
    except Exception:
        try:
            return ast.literal_eval(line.decode("utf-8", errors="replace"))
        except Exception:
            return None


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v)
    return str(value)


def main() -> None:
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    reviews_gz = RAW_DIR / "reviews_Electronics_5.json.gz"
    meta_gz = RAW_DIR / "meta_Electronics.json.gz"
    _download(REVIEWS_URL, reviews_gz)
    _download(META_URL, meta_gz)

    # --- Pass 1: stream reviews, hash-sample users ---------------------------------
    kept: dict[tuple[str, str], tuple[float, int]] = {}
    wanted_items: set[str] = set()
    n_lines = n_skipped = n_out_of_range = 0
    with gzip.open(reviews_gz, "rb") as f:
        for line in f:
            n_lines += 1
            d = _parse_line(line)
            if not d:
                n_skipped += 1
                continue
            try:
                uid = str(d["reviewerID"])
                asin = str(d["asin"])
                rating = float(d["overall"])
                ts = int(d["unixReviewTime"])
            except (KeyError, TypeError, ValueError):
                n_skipped += 1
                continue
            if not (1.0 <= rating <= 5.0):
                n_out_of_range += 1
                continue
            if not _keep_user(uid):
                continue
            key = (uid, asin)
            # Deduplicate (user, item): keep the most recent rating.
            if key not in kept or ts > kept[key][1]:
                kept[key] = (rating, ts)
            wanted_items.add(asin)
    print(f"Reviews scanned: {n_lines}; kept ratings: {len(kept)}; "
          f"skipped lines: {n_skipped}; out-of-range: {n_out_of_range}", flush=True)

    # --- Pass 2: stream metadata for the sampled items ------------------------------
    meta: dict[str, tuple[str, str]] = {}
    n_meta = 0
    with gzip.open(meta_gz, "rb") as f:
        for line in f:
            d = _parse_line(line)
            if not d:
                continue
            try:
                asin = str(d["asin"])
            except KeyError:
                continue
            if asin not in wanted_items or asin in meta:
                continue
            meta[asin] = (_as_text(d.get("title")), _as_text(d.get("description")))
            n_meta += 1
    n_missing_meta = len(wanted_items) - len(meta)
    print(f"Metadata matched: {n_meta} items; items without metadata: {n_missing_meta}",
          flush=True)

    # --- Write CSVs -----------------------------------------------------------------
    ratings_path = DATA_DIR / "ratings.csv"
    with open(ratings_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "rating", "timestamp"])
        for (uid, asin), (rating, ts) in sorted(kept.items()):
            w.writerow([uid, asin, rating, ts])

    metadata_path = DATA_DIR / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "title", "description"])
        for asin in sorted(wanted_items):
            title, desc = meta.get(asin, ("", ""))
            w.writerow([asin, title, desc])

    def _sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reviews_url": REVIEWS_URL,
        "metadata_url": META_URL,
        "sampling_rule": (f"keep user iff int(md5(reviewerID).hexdigest(),16) % {HASH_MOD} "
                          f"< {HASH_KEEP_BELOW} (deterministic, ~{HASH_KEEP_BELOW}% of users)"),
        "reviews_scanned": n_lines,
        "ratings_kept": len(kept),
        "skipped_lines": n_skipped,
        "out_of_range_ratings": n_out_of_range,
        "unique_users": len({u for u, _ in kept}),
        "unique_items": len(wanted_items),
        "items_with_metadata": n_meta,
        "items_without_metadata": n_missing_meta,
        "ratings_csv": str(ratings_path),
        "ratings_sha256": _sha256(ratings_path),
        "metadata_csv": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(LOG_DIR / "download_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done in {time.time() - t0:.1f}s. Log: {LOG_DIR / 'download_log.json'}")


if __name__ == "__main__":
    main()
