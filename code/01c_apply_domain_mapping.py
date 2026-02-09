"""Apply source_tag -> domain_5 mapping for Part-1 labeled data.
Outputs:
- data/part1/processed/part1_master_labeled_final.csv
"""

from __future__ import annotations
import pandas as pd

from config import PART1_PROCESSED_DIR

def main():
    labeled_path = PART1_PROCESSED_DIR / "part1_master_labeled_with_source.csv"
    mapping_path = PART1_PROCESSED_DIR / "source_to_domain.csv"
    out_path = PART1_PROCESSED_DIR / "part1_master_labeled_final.csv"

    if not labeled_path.exists():
        raise FileNotFoundError(f"Missing labeled file: {labeled_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")

    df = pd.read_csv(labeled_path)
    mapping_df = pd.read_csv(mapping_path)
    if "source_tag" not in mapping_df.columns or "domain_5" not in mapping_df.columns:
        raise ValueError("source_to_domain.csv must include columns: source_tag, domain_5")
    if "source_tag" not in df.columns:
        raise ValueError("Expected source_tag column in part1_master_labeled_with_source.csv")

    mapping = dict(zip(mapping_df["source_tag"], mapping_df["domain_5"]))
    df["domain_5"] = df["source_tag"].map(mapping)

    source_counts = df["source_tag"].value_counts(dropna=False)
    domain_counts = df["domain_5"].value_counts(dropna=False)
    missing = sorted(set(df["source_tag"].dropna()) - set(mapping.keys()))

    print("Counts by source_tag:")
    print(source_counts.to_string())
    print("Counts by domain_5:")
    print(domain_counts.to_string())
    if missing:
        print("Unmapped source_tag values:", ", ".join(missing))

    df.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
