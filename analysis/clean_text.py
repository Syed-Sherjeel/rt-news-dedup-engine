import argparse
from pathlib import Path
import polars as pl
import regex  
import html

# ── constants ────────────────────────────────────────────────────────────────
EMOJI_PATTERN     = regex.compile(r"\p{Emoji_Presentation}|\p{Extended_Pictographic}")
HTML_PATTERN      = regex.compile(r"<[^>]+>")
PUNCTUATION       = regex.compile(r"[^\w\s]")
WHITESPACE        = regex.compile(r"\s+")
URL_PATTERN       = regex.compile(r"http\S+|www\.\S+|t\.me/\S+|discord\.gg/\S+")
MIN_TOKEN_COUNT   = 4

# ── cleaning functions (operate on plain strings) ─────────────────────────── 
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = URL_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = PUNCTUATION.sub("", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def token_count(text: str) -> int:
    return len(text.split()) if text else 0


# ── build polars pipeline ─────────────────────────────────────────────────── 
def run_pipeline(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .with_columns(
            pl.when(pl.col("text").is_not_null())
              .then(pl.col("text"))
              .otherwise(pl.col("body"))
              .alias("raw_text")
        )
        .with_columns(
            pl.col("raw_text")
              .map_elements(clean_text, return_dtype=pl.String)
              .alias("cleaned_text")
        )
        .with_columns(
            pl.col("cleaned_text")
              .map_elements(token_count, return_dtype=pl.Int32)
              .alias("token_count")
        )
        .with_columns(
            pl.when(pl.col("token_count") < MIN_TOKEN_COUNT)
              .then(pl.lit("JUNK"))
              .otherwise(pl.lit("REVIEW"))
              .alias("verdict")
        )
    )


# ── quick inspection helper ───────────────────────────────────────────────── 
def print_report(df: pl.DataFrame) -> pl.DataFrame:
    results = run_pipeline(df)
    
    print("=" * 60)
    print("PIPELINE REPORT")
    print("=" * 60)
    
    total   = len(results)
    junk_df = results.filter(pl.col("verdict") == "JUNK")
    keep_df = results.filter(pl.col("verdict") == "REVIEW")
    
    print(f"Total messages : {total}")
    print(f"JUNK (dropped) : {len(junk_df)}  ({100*len(junk_df)/total:.1f}%)")
    print(f"REVIEW (kept)  : {len(keep_df)}  ({100*len(keep_df)/total:.1f}%)")
    
    print("\n── Sample: JUNK ──────────────────────────────────────")
    for row in junk_df.select(["id", "raw_text", "cleaned_text", "token_count"]).to_dicts():
        print(f"  id={row['id']} | tokens={row['token_count']}")
        print(f"    raw     : {row['raw_text'][:80]}")
        print(f"    cleaned : {row['cleaned_text'][:80]}")
    
    print("\n── Sample: REVIEW ────────────────────────────────────")
    for row in keep_df.select(["id", "raw_text", "cleaned_text", "token_count"]).to_dicts():
        print(f"  id={row['id']} | tokens={row['token_count']}")
        print(f"    raw     : {row['raw_text'][:80]}")
        print(f"    cleaned : {row['cleaned_text'][:80]}")
    
    return results


# ── CLI ───────────────────────────────────────────────────────────────────── 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="stage1",
        description="Clean & sort news messages — Stage 1 pipeline",
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to input CSV (e.g. data/messages_clean.csv)",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path for output CSV (e.g. out/stage1_output.csv)",
    )
    return p.parse_args()


# ── entry point ───────────────────────────────────────────────────────────── 
if __name__ == "__main__":
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"input  → {args.input}")
    print(f"output → {args.output}")

    df = pl.read_csv(args.input, try_parse_dates=True)
    results = print_report(df)

    results.filter(pl.col("verdict") == "REVIEW") \
           .select(["id", "message_type", "title", "cleaned_text", "received_at"]) \
           .sort("received_at", descending=False) \
           .write_csv(args.output)

    print(f"\n✓ saved {len(results.filter(pl.col('verdict') == 'REVIEW'))} rows → {args.output}")