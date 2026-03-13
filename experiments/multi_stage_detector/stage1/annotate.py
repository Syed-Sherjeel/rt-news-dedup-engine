"""
golden_annotate.py
------------------
Builds a golden labeled dataset using Claude Haiku 4.5.

Sliding window (no cold start, each row labeled exactly once):

  Window 1 : rows    0 – 999   → entire window is the batch (1000 labeled)
  Window 2 : context 100 – 999 | batch 1000 – 1099  (100 new)
  Window 3 : context 200 – 1099| batch 1100 – 1199  (100 new)
  ...

Every row is labeled exactly once. No cold-start seeding.

Usage:
    python golden_annotate.py -i out/stage1_output.csv -o out/golden.csv
    python golden_annotate.py -i out/stage1_output.csv -o out/golden.csv --resume
"""

import argparse
from ftplib import all_errors
import json
import os
import time
from pathlib import Path

import anthropic
import polars as pl

# ── config ────────────────────────────────────────────────────────────────────
MODEL       = "claude-haiku-4-5-20251001"
WINDOW_SIZE = 400
STRIDE      = 100
MAX_TOKENS  = 20000

SYSTEM_PROMPT = """You are a financial news deduplication classifier building a golden benchmark dataset.

You will receive either:
  (A) A single BATCH of messages to label sequentially among themselves, or
  (B) A CONTEXT of already-seen messages + a BATCH of new messages to label

Label every message in the BATCH as NEW or DUPE.

Definitions:
  NEW  — distinct event/fact not seen in context or earlier in this batch
  DUPE — substantially equivalent to a context message OR an earlier batch message
         (same entity + same core fact+ same statistics i.e, number; wording/source may differ)

Rules:
  - Crypto pump signals (e.g. "AVER 5x 52M fdv"): DUPE only if EXACT same token ticker
  - Different numbers (price, %, fdv) = NEW even if same ticker
  - Retweet / quote-tweet of identical content = DUPE
  - Same event, different angle or extra detail = DUPE
  - Partial overlap (same topic, different fact) = NEW
  - Within a batch, earlier messages become context for later ones
  - for message that are not semantic and factual,statistical duplicates reasoning will be simple
  - THere could be difference in numbers for example if previous number are 4.5 and now 4.0 then these are not duplicates but different

Respond ONLY with a JSON array — one object per BATCH message, same order:
{{"messages":[
  {{"id": <int>, "verdict": "NEW"|"DUPE", "reason": "<why or why not a duplicate>"}},
  ...
]}}
No preamble. No markdown. No text outside the JSON array."""


# ── prompt builders ───────────────────────────────────────────────────────────
def fmt_rows(rows: list[dict]) -> str:
    return "\n".join(f"[{r['id']}] {str(r.get('text', ''))[:200]}" for r in rows)


def build_prompt_first(batch: list[dict]) -> str:
    return (
        f"BATCH TO LABEL ({len(batch)} messages — label sequentially, "
        f"earlier messages in this batch act as context for later ones):\n"
        f"{fmt_rows(batch)}"
    )


def build_prompt_sliding(context: list[dict], batch: list[dict]) -> str:
    return (
        f"CONTEXT ({len(context)} already-labeled messages):\n"
        f"{fmt_rows(context)}\n\n"
        f"BATCH TO LABEL ({len(batch)} new messages):\n"
        f"{fmt_rows(batch)}"
    )



def call_claude(client: anthropic.Anthropic, prompt: str, batch: list[dict]) -> list[dict]:
    # try:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    result = []
    raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    results = json.loads(raw)
    results = results.get("messages")
    print(results)
    for r in results:
        result.append({"id":r["id"],"verdict": r["verdict"], "reason": r["reason"]})
    return result

    # except (json.JSONDecodeError, KeyError, IndexError) as e:
    #     print(raw)
    #     print(f"  [warn] parse error: {e} — defaulting to NEW")
    #     return [{"id": r["id"], "verdict": "NEW", "reason": "parse_error"} for r in batch]

    # except anthropic.APIError as e:
    #     print(f"  [error] API error: {e} — defaulting to NEW")
    #     return [{"id": r["id"], "verdict": "NEW", "reason": "api_error"} for r in batch]


# ── checkpoint ────────────────────────────────────────────────────────────────
def save(labeled: dict, all_rows: list[dict], path: Path):
    merged = []
    for row in all_rows:
        if int(row["id"]) in labeled:
            v = labeled[int(row["id"])]
            merged.append({**row, "llm_verdict": v["verdict"], "reason": v["reason"]})
    if merged:
        pl.DataFrame(merged).write_csv(path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="golden_annotate")
    p.add_argument("-i", "--input",  required=True, type=Path)
    p.add_argument("-o", "--output", required=True, type=Path)
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    client    = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    df        = pl.read_csv(args.input, separator="|")
    all_rows  = df.to_dicts()
    print(df.head())
    total     = len(all_rows)

    print(f"[annotate] {total} rows | window={args.window} | stride={args.stride}")

    # labeled: id → {verdict, reason}
    labeled: dict[int, dict] = {}
    if args.resume and args.output.exists():
        for r in pl.read_csv(args.output).to_dicts():
            labeled[r["id"]] = {"verdict": r["llm_verdict"], "reason": r.get("reason", "")}
        print(f"[annotate] resuming — {len(labeled)} rows already labeled")

    api_calls = 0
    t_start   = time.perf_counter()

    # ── window 1: entire first window is the batch (no cold start) ───────────
    first_batch = all_rows[:args.window]
    unlabeled   = [r for r in first_batch if r["id"] not in labeled]

    if unlabeled:
        print(f"[annotate] window 1 — labeling {len(unlabeled)} rows as self-contained batch")
        prompt   = build_prompt_first(unlabeled)
        verdicts = call_claude(client, prompt, unlabeled)
        api_calls += 1
        for v in verdicts:
            labeled[int(v["id"])] = {"verdict": v["verdict"], "reason": v["reason"]}
        save(labeled, all_rows, args.output)
        dupes = sum(1 for v in labeled.values() if v["verdict"] == "DUPE")
        print(f"  done | labeled={len(labeled)} | dupes={dupes}")

    # ── sliding windows: label only the new stride rows each time ────────────
    pos = args.window
    while pos < total:
        batch_rows  = all_rows[pos : pos + args.stride]
        unlabeled   = [r for r in batch_rows if r["id"] not in labeled]

        if unlabeled:
            # context = the window_size rows ending just before this batch
            ctx_start   = max(0, pos - args.window + args.stride)
            context_rows = all_rows[ctx_start : pos]

            prompt   = build_prompt_sliding(context_rows, unlabeled)
            verdicts = call_claude(client, prompt, unlabeled)
            api_calls += 1
            for v in verdicts:
                labeled[v["id"]] = {"verdict": v["verdict"], "reason": v["reason"]}

        pos += args.stride

        elapsed = time.perf_counter() - t_start
        dupes   = sum(1 for v in labeled.values() if v["verdict"] == "DUPE")
        print(f"  [pos={pos}/{total}] calls={api_calls} | labeled={len(labeled)} | dupes={dupes} | {elapsed:.0f}s")
        save(labeled, all_rows, args.output)

    # ── final save + report ───────────────────────────────────────────────────
    save(labeled, all_rows, args.output)
    elapsed = time.perf_counter() - t_start
    dupes   = sum(1 for v in labeled.values() if v["verdict"] == "DUPE")
    news    = sum(1 for v in labeled.values() if v["verdict"] == "NEW")

    print(f"\n{'='*50}")
    print(f"DONE  — {len(labeled)} rows | NEW={news} | DUPE={dupes}")
    print(f"API calls : {api_calls} | time : {elapsed:.0f}s")
    print(f"Output    → {args.output}")


if __name__ == "__main__":
    main()