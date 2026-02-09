"""Check and harden domain mapping integrity for Part-1 splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config import PART1_PROCESSED_DIR, DOMAINS, ANALYTICS_DIR

WHITESPACE_RE = re.compile(r"\s+")


def canonical_text(text: str, lower: bool = True) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)
    t = unicodedata.normalize("NFKC", t)
    t = t.strip()
    t = WHITESPACE_RE.sub(" ", t)
    if lower:
        t = t.lower()
    return t


def add_sample_id(df: pd.DataFrame, lower: bool = True) -> pd.DataFrame:
    canon = df["text"].map(lambda t: canonical_text(t, lower=lower))
    payload = df["source_id"].astype(str) + "\n" + canon
    df = df.copy()
    df["sample_id"] = payload.map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
    return df


def merge_with_strategy(split_df: pd.DataFrame, master_df: pd.DataFrame, strategy: str, lower: bool) -> Tuple[pd.DataFrame, float]:
    if strategy == "exact":
        merged = split_df.merge(master_df[["source_id", "text", "domain_5", "__match_flag"]], on=["source_id", "text"], how="left")
    elif strategy == "normalized":
        split_df = split_df.copy()
        master_df = master_df.copy()
        split_df["canon_text"] = split_df["text"].map(lambda t: canonical_text(t, lower=lower))
        master_df["canon_text"] = master_df["text"].map(lambda t: canonical_text(t, lower=lower))
        merged = split_df.merge(master_df[["source_id", "canon_text", "domain_5", "__match_flag"]], on=["source_id", "canon_text"], how="left")
    elif strategy == "sample_id":
        split_df = add_sample_id(split_df, lower=lower)
        master_df = add_sample_id(master_df, lower=lower)
        merged = split_df.merge(master_df[["sample_id", "domain_5", "__match_flag"]], on=["sample_id"], how="left")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    matched = merged["__match_flag"].notna().sum()
    total = len(merged)
    rate = matched / total if total else 0.0
    return merged, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_path", type=str, default=str(PART1_PROCESSED_DIR / "part1_master_labeled_final.csv"))
    ap.add_argument("--splits_dir", type=str, default=str(PART1_PROCESSED_DIR / "splits"))
    ap.add_argument("--output_dir", type=str, default=str(PART1_PROCESSED_DIR / "splits_with_domain"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--lower", action="store_true", default=True)
    ap.add_argument("--no_lower", action="store_false", dest="lower")
    args = ap.parse_args()

    master_path = Path(args.master_path)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    master_df = pd.read_csv(master_path)
    required_cols = {"source_id", "text", "domain_5"}
    if not required_cols.issubset(master_df.columns):
        raise ValueError(f"Master file missing required columns: {required_cols - set(master_df.columns)}")
    master_df = master_df.copy()
    master_df["__match_flag"] = 1

    report_rows = []
    mismatch_examples = []
    output_files = {}

    for split in ["train", "val", "test"]:
        split_path = splits_dir / f"{split}.csv"
        if not split_path.exists():
            continue
        split_df = pd.read_csv(split_path)

        if "domain_5" in split_df.columns:
            non_null = split_df["domain_5"].notna().sum()
            total = len(split_df)
            valid = split_df["domain_5"].dropna().isin(DOMAINS).all()
            report_rows.append({
                "split": split,
                "strategy": "already_present",
                "matched": int(non_null),
                "total": int(total),
                "merge_match_rate": float(non_null / total if total else 0.0),
                "domain_5_coverage": float(non_null / total if total else 0.0),
                "valid_domains": bool(valid),
            })
            output_files[split] = str(split_path)
            continue

        # exact merge
        merged_exact, rate_exact = merge_with_strategy(split_df, master_df, "exact", args.lower)
        report_rows.append({
            "split": split,
            "strategy": "exact",
            "matched": int(merged_exact["__match_flag"].notna().sum()),
            "total": int(len(merged_exact)),
            "merge_match_rate": float(rate_exact),
            "domain_5_coverage": float(merged_exact["domain_5"].notna().sum() / len(merged_exact) if len(merged_exact) else 0.0),
            "valid_domains": None,
        })
        best = ("exact", merged_exact, rate_exact)

        # normalized merge
        if rate_exact < 0.995:
            merged_norm, rate_norm = merge_with_strategy(split_df, master_df, "normalized", args.lower)
            report_rows.append({
                "split": split,
                "strategy": "normalized",
                "matched": int(merged_norm["__match_flag"].notna().sum()),
                "total": int(len(merged_norm)),
                "merge_match_rate": float(rate_norm),
                "domain_5_coverage": float(merged_norm["domain_5"].notna().sum() / len(merged_norm) if len(merged_norm) else 0.0),
                "valid_domains": None,
            })
            if rate_norm > best[2]:
                best = ("normalized", merged_norm, rate_norm)

        # sample_id merge
        if best[2] < 0.995:
            merged_sid, rate_sid = merge_with_strategy(split_df, master_df, "sample_id", args.lower)
            report_rows.append({
                "split": split,
                "strategy": "sample_id",
                "matched": int(merged_sid["__match_flag"].notna().sum()),
                "total": int(len(merged_sid)),
                "merge_match_rate": float(rate_sid),
                "domain_5_coverage": float(merged_sid["domain_5"].notna().sum() / len(merged_sid) if len(merged_sid) else 0.0),
                "valid_domains": None,
            })
            if rate_sid > best[2]:
                best = ("sample_id", merged_sid, rate_sid)

        strategy_used, merged_best, rate_best = best
        merged_best["domain_5"] = merged_best["domain_5"].fillna("N/A")
        missing = merged_best[merged_best["domain_5"] == "N/A"].head(20)
        for _, row in missing.iterrows():
            mismatch_examples.append({
                "split": split,
                "source_id": row.get("source_id", ""),
                "text": row.get("text", ""),
            })

        # write output with domain_5
        out_path = out_dir / f"{split}.csv"
        merged_best.to_csv(out_path, index=False, encoding="utf-8")
        output_files[split] = str(out_path)

        if args.overwrite:
            merged_best.to_csv(split_path, index=False, encoding="utf-8")
            output_files[split] = str(split_path)

        report_rows.append({
            "split": split,
            "strategy": f"{strategy_used}_used",
            "matched": int(merged_best["__match_flag"].notna().sum()),
            "total": int(len(merged_best)),
            "merge_match_rate": float(rate_best),
            "domain_5_coverage": float((merged_best["domain_5"] != "N/A").sum() / len(merged_best) if len(merged_best) else 0.0),
            "valid_domains": bool(merged_best["domain_5"].isin(DOMAINS + ["N/A"]).all()),
        })

    report = {
        "master_path": str(master_path),
        "splits_dir": str(splits_dir),
        "output_files": output_files,
        "lower": bool(args.lower),
        "report_rows": report_rows,
        "mismatch_examples": mismatch_examples,
    }

    report_json = ANALYTICS_DIR / "domain_mapping_integrity_report.json"
    report_csv = ANALYTICS_DIR / "domain_mapping_integrity_report.csv"
    pd.DataFrame(report_rows).to_csv(report_csv, index=False, encoding="utf-8")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    mismatch_csv = ANALYTICS_DIR / "domain_mapping_mismatches.csv"
    pd.DataFrame(mismatch_examples).to_csv(mismatch_csv, index=False, encoding="utf-8")

    print("Wrote:", report_csv)
    print("Wrote:", report_json)
    print("Wrote:", mismatch_csv)


if __name__ == "__main__":
    main()
