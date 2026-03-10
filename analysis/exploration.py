"""
Save clean dataset for ML modelling.
Core columns: body, received_at
Supporting: message_type, title (secondary signal)
"""

import polars as pl

FILE = "messages_202603081952.csv"

df = (
    pl.scan_csv(
        FILE,
        infer_schema_length=10_000,
        null_values=["", "NULL", "null"],
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    # Filter test messages first
    .filter(
        pl.col("test_message").is_null() | (pl.col("test_message") == False)
    )
    # Only the columns we need
    .select([
        "id",
        "message_type",
        "title",
        "body",
        "received_at",
    ])
    .collect(streaming=True)
    # Drop rows where body is null — body is our core signal
    .filter(pl.col("body").is_not_null())
    # Parse timestamp
    .with_columns(
        pl.col("received_at")
          .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f %z", strict=False)
    )
    # Combine title + body into single text field
    # title is 34% null so coalesce: use title if present, else just body
    .with_columns(
        pl.when(pl.col("title").is_not_null())
          .then(pl.col("title") + " " + pl.col("body"))
          .otherwise(pl.col("body"))
          .alias("text")
    )
    .sort("received_at")
)

print(f"Clean rows: {df.height:,}")
print(f"Columns: {df.columns}")
print(f"\nMessage type breakdown:")
print(df.group_by("message_type").agg(pl.len()).sort("len", descending=True))
print(f"\nDate range: {df['received_at'].min()} → {df['received_at'].max()}")
print(f"\nSample text field (first 3 rows):")
for row in df.head(3).iter_rows(named=True):
    print(f"  [{row['message_type']}] {row['text'][:120]}...")

df.write_csv("messages_clean.csv")
print(f"\n✓ Saved messages_clean.csv — {df.height:,} rows")