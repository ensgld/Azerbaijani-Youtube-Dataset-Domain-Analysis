"""Attach source_tag to labeled samples by matching normalized text to corpus_all.txt.
Outputs:
- data/part1/processed/part1_master_labeled_with_source.csv
- data/analytics/part1_source_match_rate.json
"""

from __future__ import annotations
import json
import pandas as pd

from config import PART1_RAW_DIR, PART1_PROCESSED_DIR, ANALYTICS_DIR
from utils.text_normalize import normalize_text

def parse_corpus_line(line: str) -> tuple[str | None, str | None]:
    line = (line or "").strip()
    if not line:
        return None, None
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]

def main():
    corpus_path = PART1_RAW_DIR / "corpus_all.txt"
    labeled_path = PART1_PROCESSED_DIR / "part1_master_labeled.csv"
    out_path = PART1_PROCESSED_DIR / "part1_master_labeled_with_source.csv"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")
    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing labeled file: {labeled_path}")

    text_to_source: dict[str, str] = {}
    collisions = 0
    total_corpus_lines = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            total_corpus_lines += 1
            source_tag, text = parse_corpus_line(line)
            if not source_tag or not text:
                continue
            text_norm = normalize_text(text)
            if not text_norm:
                continue
            if text_norm in text_to_source and text_to_source[text_norm] != source_tag:
                collisions += 1
                continue
            text_to_source.setdefault(text_norm, source_tag)

    df = pd.read_csv(labeled_path)
    if "text" not in df.columns:
        raise ValueError("Expected 'text' column in part1_master_labeled.csv")

    df["text_norm"] = df["text"].astype(str).map(normalize_text)
    df["source_tag"] = df["text_norm"].map(text_to_source)

    matched = int(df["source_tag"].notna().sum())
    total = int(len(df))
    match_rate = matched / total if total else 0.0
    df["source_tag"] = df["source_tag"].fillna("unknown")

    df.drop(columns=["text_norm"], inplace=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "total_labeled_rows": total,
        "matched_rows": matched,
        "match_rate": round(match_rate, 6),
        "corpus_lines": total_corpus_lines,
        "corpus_unique_texts": len(text_to_source),
        "corpus_collisions": collisions,
    }
    with open(ANALYTICS_DIR / "part1_source_match_rate.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Loaded corpus lines:", total_corpus_lines)
    print("Unique corpus texts:", len(text_to_source))
    print("Corpus collisions:", collisions)
    print("Matched rows:", matched, "/", total, f"({match_rate:.2%})")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
