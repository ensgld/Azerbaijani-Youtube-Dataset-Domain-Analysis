#!/usr/bin/env python3
"""Report domain_5 coverage for top-errors outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_errors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")
    return pd.read_csv(path)


def domain_stats(df: pd.DataFrame) -> dict:
    if "domain_5" not in df.columns:
        return {
            "total_rows": int(len(df)),
            "missing_domain_5": int(len(df)),
            "na_domain_5": 0,
            "domain_counts": {},
        }
    domain_col = df["domain_5"].fillna("")
    missing = domain_col.eq("").sum()
    na_count = domain_col.eq("N/A").sum()
    counts = domain_col.value_counts()
    return {
        "total_rows": int(len(df)),
        "missing_domain_5": int(missing),
        "na_domain_5": int(na_count),
        "domain_counts": {str(k): int(v) for k, v in counts.items()},
    }


def main() -> None:
    report_dir = Path("data/analytics")
    out_csv = report_dir / "errors_domain5_coverage_report.csv"
    out_json = report_dir / "errors_domain5_coverage_report.json"

    sample_path = Path("report_tables/top_errors_sample_20.csv")
    conf_path = Path("report_tables/top_errors_confident_wrong.csv")

    sample_df = load_errors(sample_path)
    conf_df = load_errors(conf_path)

    sample_stats = domain_stats(sample_df)
    conf_stats = domain_stats(conf_df)

    rows = []
    for name, stats in [("sample_20", sample_stats), ("confident_wrong", conf_stats)]:
        domain_counts = stats.get("domain_counts", {})
        for domain, count in domain_counts.items():
            rows.append({
                "set": name,
                "domain_5": domain,
                "count": count,
                "total_rows": stats["total_rows"],
                "missing_domain_5": stats["missing_domain_5"],
                "na_domain_5": stats["na_domain_5"],
            })

    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(
        json.dumps({
            "sample_20": sample_stats,
            "confident_wrong": conf_stats,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Wrote:")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
