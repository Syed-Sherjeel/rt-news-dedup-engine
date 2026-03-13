"""
ingest.py
---------
Bulk loads stage1_output.csv into Redis with a 1-hour TTL.
Each article is embedded and stored as a JSON + vector entry.

Usage:
    python ingest.py -i out/stage1_output.csv
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from embedder import Embedder
from redis_client import  RedisNewsStore
from models import Article   # reuse from pipeline.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ingest", description="Bulk ingest CSV into Redis")
    p.add_argument("-i", "--input", required=True, type=Path, metavar="PATH",
                   help="Path to stage1_output.csv")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Embedding batch size (default: 64)")
    return p.parse_args()


def parse_timestamp(ts: str) -> int:
    """ISO 8601 string → unix epoch int."""
    return int(datetime.fromisoformat(ts).astimezone(timezone.utc).timestamp())


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pl.read_csv(args.input)
    rows = df.to_dicts()
    total = len(rows)
    print(f"[ingest] {total} articles to load")

    embedder = Embedder()
    store    = RedisNewsStore()

    t0      = time.perf_counter()
    loaded  = 0
    skipped = 0

    # process in batches for efficient embedding
    for i in range(0, total//6, args.batch_size):
        batch = rows[i : i + args.batch_size]

        texts = [r["cleaned_text"] or "" for r in batch]
        vectors = embedder.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for row, vec in zip(batch, vectors):
            if not row.get("cleaned_text"):
                skipped += 1
                continue

            article = Article(
                article_id   = str(row["id"]),
                text         = row["cleaned_text"],
                source       = row.get("message_type", "unknown"),
                published_at = parse_timestamp(row["received_at"]),
            )
            store.store(article, vec)
            loaded += 1

        print(f"  [{i + len(batch)}/{total}] stored {loaded} | skipped {skipped}")

    elapsed = time.perf_counter() - t0
    print(f"\n[ingest] done — {loaded} loaded, {skipped} skipped in {elapsed:.1f}s")


if __name__ == "__main__":
    main()