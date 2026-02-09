"""Merge candidate CSVs and dedupe by video_id, keeping higher commentCount."""

from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

def to_int(v) -> int:
    try:
        if pd.isna(v):
            return 0
    except Exception:
        pass
    try:
        return int(float(v))
    except Exception:
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Old candidates CSV")
    ap.add_argument("--new", required=True, help="New candidates CSV")
    ap.add_argument("--output", required=True, help="Merged output CSV")
    args = ap.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    out_path = Path(args.output)

    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    old_df["commentCount"] = old_df["commentCount"].map(to_int)
    new_df["commentCount"] = new_df["commentCount"].map(to_int)

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.sort_values("commentCount", ascending=False)
    merged = combined.drop_duplicates(subset=["video_id"], keep="first")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")

    old_count = len(old_df)
    new_count = len(new_df)
    merged_count = len(merged)
    new_unique_added = merged_count - len(old_df.drop_duplicates(subset=["video_id"]))

    print("old_count:", old_count)
    print("new_count:", new_count)
    print("merged_count:", merged_count)
    print("num_new_unique_added:", new_unique_added)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
