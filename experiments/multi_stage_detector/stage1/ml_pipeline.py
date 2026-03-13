import csv
import re
import time
import numpy as np
import spacy
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from embedder import Embedder

_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "textcat"])
_ENTITY_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "FAC"}

# ── number extraction ─────────────────────────────────────────────────────────
# Matches financial figures: 4.5, 4.5%, 4.5pc, 4.5pct, 3.2bps, $1.2m, -0.5
# Deliberately excludes: bare 4-digit years (2024), and standalone integers
# that look like IDs/counts unless they carry a unit suffix.
_NUM_RE = re.compile(
    r"""
    (?<![/\d])                          # not part of a date or ID
    (?P<num>
        [+\-]?                          # optional sign
        (?:\$|£|€|¥)?                  # optional currency prefix
        \d{1,3}(?:,\d{3})*             # integer part
        (?:\.\d+)?                      # optional decimal
        (?:[kKmMbBtT](?:n|rn)?)?       # optional magnitude: k m b t
        (?:\s*(?:%|pc|pct|bps|bp))?    # optional unit suffix
    )
    (?=[%\s,\.\)\]]|pc|pct|bps|bp|$)   # must be followed by unit/boundary
    """,
    re.VERBOSE,
)

# Years we don't want to treat as meaningful figures
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')


def _extract_numbers(text: str) -> set[float]:
    """
    Extract all financial figures from text as a set of floats.
    Strips units/currency, normalises magnitude suffixes.
    Years (1900-2099) are excluded since they carry no numeric signal.
    """
    # blank out years so they don't match
    cleaned = _YEAR_RE.sub("", text[:512])

    nums = set()
    for m in _NUM_RE.finditer(cleaned):
        raw = m.group("num").strip()
        if not raw or raw in ("+", "-"):
            continue

        s = raw.lower()
        s = re.sub(r"[$£€¥]", "", s)
        s = re.sub(r"\s*(%|pc|pct|bps|bp)\s*$", "", s)

        mag = 1.0
        if s and s[-1] in {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}:
            mag = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[s[-1]]
            s = s[:-1]

        s = s.replace(",", "")
        try:
            nums.add(round(float(s) * mag, 6))
        except ValueError:
            continue

    return nums


def _number_verdict(a: str, b: str) -> str | None:
    """
    Compare the number sets of two texts.

    Returns:
        "DUPE"      — sets are identical (same facts)
        "PARAPHRASE"— one is a strict subset of the other (partial coverage)
        "NEW"       — sets differ in at least one value (different fact)
        None        — one or both texts have no numbers; can't decide
    """
    nums_a = _extract_numbers(a)
    nums_b = _extract_numbers(b)

    if not nums_a or not nums_b:
        return None             # no numeric content → fall back to embedding verdict

    if nums_a == nums_b:
        return "DUPE"

    if nums_a.issubset(nums_b) or nums_b.issubset(nums_a):
        return "PARAPHRASE"     # one covers a subset of the other's figures

    return "NEW"                # at least one figure differs → different story


# ── entity helpers ─────────────────────────────────────────────────────────────

def _entities(text: str) -> set[str]:
    doc = _nlp(text[:512])
    return {ent.text.upper() for ent in doc.ents if ent.label_ in _ENTITY_LABELS}


def _entity_clash(a: str, b: str) -> bool:
    ents_a, ents_b = _entities(a), _entities(b)
    return bool(ents_a) and bool(ents_b) and ents_a != ents_b


# ── timing ────────────────────────────────────────────────────────────────────

@dataclass
class Timing:
    embed_ms:    float
    sim_ms:      float
    spacy_ms:    float
    total_ms:    float

    def __str__(self) -> str:
        spacy_str = f"  spacy={self.spacy_ms:.1f}ms" if self.spacy_ms else ""
        return (
            f"embed={self.embed_ms:.1f}ms  "
            f"sim={self.sim_ms:.2f}ms"
            f"{spacy_str}"
            f"  total={self.total_ms:.1f}ms"
        )


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class Message:
    id: int
    message_type: str
    title: str
    cleaned_text: str
    received_at: datetime
    embedding: np.ndarray = field(default=None, repr=False)


@dataclass
class WindowResult:
    current: Message
    window: list[Message]
    similarities: np.ndarray
    thresholds: dict
    timing: Timing
    verdict: str = field(init=False)
    top_match: tuple = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "top_match", self._compute_top_match())
        object.__setattr__(self, "verdict", self._compute_verdict())

    def most_similar(self) -> tuple[Message | None, float]:
        if len(self.similarities) == 0:
            return None, 0.0
        idx = int(np.argmax(self.similarities))
        return self.window[idx], float(self.similarities[idx])

    def above_threshold(self, threshold: float) -> list[tuple[Message, float]]:
        pairs = [
            (self.window[i], float(self.similarities[i]))
            for i in range(len(self.window))
            if self.similarities[i] >= threshold
        ]
        return sorted(pairs, key=lambda x: x[1], reverse=True)

    def _compute_top_match(self) -> tuple[Message | None, float]:
        return self.most_similar()

    def _compute_verdict(self) -> str:
        neighbour, score = self.top_match

        if score < self.thresholds["paraphrase"]:
            return "NEW"

        # ── number check: only fires in dupe zone (>= 0.9) ───────────────────
        # Below dupe threshold the embedding difference is already meaningful
        # enough — no need to second-guess it with numbers.
        if score >= self.thresholds["dupe"] and len(self.current.cleaned_text)<100:
            num_v = _number_verdict(self.current.cleaned_text, neighbour.cleaned_text)
            if num_v is not None:
                return num_v

        # ── fallback: entity clash (original logic) ───────────────────────────
        clash = _entity_clash(self.current.cleaned_text, neighbour.cleaned_text)
        if score >= self.thresholds["dupe"]:
            return "PARAPHRASE" if clash else "DUPE"
        if score >= self.thresholds["paraphrase"]:
            return "NEW" if clash else "PARAPHRASE"

        return "NEW"


# ── pipeline ──────────────────────────────────────────────────────────────────

class HourWindowPipeline:
    def __init__(
        self,
        csv_path: str | Path,
        window_hours: float = 1.0,
        embedder: Embedder | None = None,
        batch_embed: int = 32,
        thresholds: dict | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.window = timedelta(hours=window_hours)
        self.embedder = embedder or Embedder()
        self.batch_embed = batch_embed
        self.thresholds = thresholds or {
            "dupe":       0.9,
            "paraphrase": 0.85,
            "topic":      0.70,
        }
        self._buffer: deque[Message] = deque()

    @staticmethod
    def _parse_row(row: dict) -> Message:
        return Message(
            id=int(row["id"]),
            message_type=row["message_type"],
            title=row["title"],
            cleaned_text=row["cleaned_text"],
            received_at=datetime.fromisoformat(row["received_at"]),
        )

    def _embed_batch(self, msgs: list[Message]) -> float:
        t0 = time.perf_counter()
        texts = [m.cleaned_text for m in msgs]
        vecs = self.embedder.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=self.batch_embed,
            show_progress_bar=False,
        ).astype(np.float32)
        for msg, vec in zip(msgs, vecs):
            msg.embedding = vec
        return (time.perf_counter() - t0) * 1000 / len(msgs)

    def _evict(self, cutoff: datetime) -> None:
        while self._buffer and self._buffer[0].received_at < cutoff:
            self._buffer.popleft()

    def _similarities(self, vec: np.ndarray) -> tuple[np.ndarray, float]:
        if not self._buffer:
            return np.array([], dtype=np.float32), 0.0
        t0 = time.perf_counter()
        matrix = np.stack([m.embedding for m in self._buffer])
        sims = matrix @ vec
        return sims, (time.perf_counter() - t0) * 1000

    def stream(self) -> Iterator[WindowResult]:
        pending: list[Message] = []

        def _flush(force: bool = False) -> Iterator[WindowResult]:
            nonlocal pending
            if not pending:
                return
            if not force and len(pending) < self.batch_embed:
                return

            embed_ms_per = self._embed_batch(pending)

            for msg in pending:
                t_total = time.perf_counter()

                cutoff = msg.received_at - self.window
                self._evict(cutoff)

                sims, sim_ms = self._similarities(msg.embedding)

                neighbour_score = float(sims[int(np.argmax(sims))]) if len(sims) else 0.0
                spacy_ms = 0.0
                if neighbour_score >= self.thresholds["dupe"]:
                    t_spacy = time.perf_counter()
                    _entities(msg.cleaned_text)
                    spacy_ms = (time.perf_counter() - t_spacy) * 1000

                timing = Timing(
                    embed_ms=embed_ms_per,
                    sim_ms=sim_ms,
                    spacy_ms=spacy_ms,
                    total_ms=(time.perf_counter() - t_total) * 1000 + embed_ms_per,
                )

                yield WindowResult(
                    current=msg,
                    window=list(self._buffer),
                    similarities=sims,
                    thresholds=self.thresholds,
                    timing=timing,
                )
                self._buffer.append(msg)

            pending = []

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pending.append(self._parse_row(row))
                yield from _flush()

        yield from _flush(force=True)


# ── example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import csv as csv_mod

    OUTPUT_PATH = "../../../data/bgi_large.csv"
    OUTPUT_COLS = ["verdict", "conf", "id", "similar_to_id", "text"]

    pipeline = HourWindowPipeline(
        csv_path="../../../data/stage1_output.csv",
        window_hours=1.0,
        batch_embed=64,
    )

    seen = dupes = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as out_f:
        writer = csv_mod.DictWriter(out_f, fieldnames=OUTPUT_COLS, delimiter="|")
        writer.writeheader()

        for result in pipeline.stream():
            seen += 1
            msg = result.current
            neighbour, score = result.top_match

            writer.writerow({
                "verdict":       result.verdict,
                "conf":          f"{score:.3f}",
                "id":            msg.id,
                "similar_to_id": neighbour.id if neighbour else None,
                "text":          msg.cleaned_text,
            })
            out_f.flush()

            if result.verdict in ("DUPE", "PARAPHRASE"):
                dupes += 1
                print(f"{result.verdict}|{score:.3f}|id={msg.id:>7} ← id={neighbour.id:>7}|{msg.cleaned_text}|{result.timing}")
            else:
                print(f"{result.verdict:<10}|{score:.3f}|id={msg.id:>7}|similar to None|{msg.cleaned_text}|{result.timing}")

            if seen % 10_000 == 0:
                print(f"  → processed {seen:,} rows, {dupes:,} dupes so far", file=sys.stderr)