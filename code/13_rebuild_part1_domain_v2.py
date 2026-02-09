#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\\S+")
PUNCT_RE = re.compile(r"[\\.,;:!?\\(\\)\\[\\]{}\\\"'`~@#$%^&*_+=|<>/\\\\-]")
SPACE_RE = re.compile(r"\\s+")


def normalize_text_for_join(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def text_key(text: str) -> str:
    norm = normalize_text_for_join(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def infer_source_tag(path_str: str) -> str:
    s = path_str.lower()
    for tag in ["domgeneral", "domsocial", "domreviews", "domnews"]:
        if tag in s:
            return tag
    for tag in ["general", "social", "reviews", "review", "news"]:
        if tag in s:
            return f"dom{tag}"
    name = Path(path_str).stem
    return name


def infer_domain_guess(path_str: str) -> str:
    s = path_str.lower()
    if any(k in s for k in ["tech", "digital", "internet", "telefon", "app"]):
        return "Technology & Digital Services"
    if any(k in s for k in ["finance", "business", "bank", "kredit", "invest", "valyuta"]):
        return "Finance & Business"
    if any(k in s for k in ["social", "entertainment", "music", "film", "serial", "vlog"]):
        return "Social Life & Entertainment"
    if any(k in s for k in ["retail", "review", "market", "shopping", "lifestyle", "qiymet"]):
        return "Retail & Lifestyle"
    if any(k in s for k in ["public", "news", "government", "dovlet", "vergi"]):
        return "Public Services"
    return ""


def build_inventory(data_root: Path, out_path: Path) -> None:
    rows = []
    for p in sorted(data_root.rglob("*")):
        if p.suffix.lower() not in [".xlsx", ".csv", ".tsv"]:
            continue
        path_str = str(p)
        rows.append(
            {
                "path": path_str,
                "filename": p.name,
                "inferred_source_tag": infer_source_tag(path_str),
                "inferred_domain_guess": infer_domain_guess(path_str),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_lookup(corpus_path: Path, master_keys: set) -> tuple[dict, Counter]:
    key_to_tag_counts = defaultdict(Counter)
    tag_counts = Counter()
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if " " not in line:
                continue
            tag, text = line.split(" ", 1)
            key = text_key(text)
            if key not in master_keys:
                continue
            key_to_tag_counts[key][tag] += 1
            tag_counts[tag] += 1
    return key_to_tag_counts, tag_counts


def write_lookup_csv(key_to_tag_counts: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text_key", "source_tag", "source_id", "count"])
        for key, tags in key_to_tag_counts.items():
            for tag, count in tags.items():
                writer.writerow([key, tag, f"corpus_all:{tag}", count])


def choose_tags(key_to_tag_counts: dict) -> tuple[dict, dict]:
    chosen = {}
    ambiguous = {}
    for key, tags in key_to_tag_counts.items():
        best_tag = None
        best_count = -1
        tie = False
        for tag, count in tags.items():
            if count > best_count:
                best_tag = tag
                best_count = count
                tie = False
            elif count == best_count:
                tie = True
        chosen[key] = best_tag
        ambiguous[key] = tie and len(tags) > 1
    return chosen, ambiguous


def build_mapping(unique_tags: list, out_path: Path, config_out: Path) -> dict:
    default_map = {
        "domreviews": "Retail & Lifestyle",
        "domsocial": "Social Life & Entertainment",
        "domnews": "Public Services",
        "domgeneral": "Public Services",
        "unknown": "N/A",
        "nan": "N/A",
    }
    mapping = {}
    for tag in unique_tags:
        t = str(tag)
        if t in default_map:
            mapping[t] = default_map[t]
            continue
        low = t.lower()
        if "tech" in low:
            mapping[t] = "Technology & Digital Services"
        elif "finance" in low or "bank" in low or "business" in low:
            mapping[t] = "Finance & Business"
        elif "social" in low or "entertain" in low:
            mapping[t] = "Social Life & Entertainment"
        elif "review" in low or "retail" in low:
            mapping[t] = "Retail & Lifestyle"
        elif "news" in low or "public" in low:
            mapping[t] = "Public Services"
        else:
            mapping[t] = "N/A"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"source_tag": k, "domain_5": v} for k, v in sorted(mapping.items())]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    pd.DataFrame(rows).to_csv(config_out, index=False)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_path", default="data/part1/processed/part1_master_labeled_final.csv")
    parser.add_argument("--corpus_path", default="data/part1/raw/corpus_all.txt")
    parser.add_argument("--splits_dir", default="data/part1/processed/splits")
    parser.add_argument("--out_master", default="data/part1/processed/part1_master_labeled_final_v2.csv")
    parser.add_argument("--out_splits_dir", default="data/part1/processed/splits_with_domain_v2")
    parser.add_argument("--analytics_dir", default="data/analytics")
    parser.add_argument("--inventory_out", default="data/analytics/part1_source_inventory.csv")
    parser.add_argument("--lookup_out", default="data/analytics/part1_textkey_to_source.csv")
    parser.add_argument("--mapping_out", default="data/analytics/source_to_domain5_mapping.csv")
    parser.add_argument("--config_mapping_out", default="config/source_to_domain5_mapping.csv")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    analytics_dir = Path(args.analytics_dir)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    master_path = Path(args.master_path)
    corpus_path = Path(args.corpus_path)

    print(f"Loading master: {master_path}")
    master = pd.read_csv(master_path)

    # Step 1: inventory
    print("Building Part-1 source inventory...")
    build_inventory(Path("data/part1/raw"), Path(args.inventory_out))

    # Step 2: build lookup
    print("Building text_key for master...")
    master["text_key"] = master["text"].map(text_key)
    master_keys = set(master["text_key"].unique())
    print(f"Master unique text_keys: {len(master_keys)}")

    print("Scanning corpus_all for matching keys...")
    key_to_tag_counts, tag_counts = build_lookup(corpus_path, master_keys)
    write_lookup_csv(key_to_tag_counts, Path(args.lookup_out))

    chosen_tag, ambiguous_map = choose_tags(key_to_tag_counts)
    master["lookup_tag"] = master["text_key"].map(chosen_tag)
    master["lookup_ambiguous"] = master["text_key"].map(ambiguous_map).fillna(False)

    source_tag_series = master.get("source_tag", pd.Series([None] * len(master)))
    source_tag_str = source_tag_series.fillna("unknown").astype(str)
    unknown_before = int((source_tag_str == "unknown").sum())

    update_mask = source_tag_str.eq("unknown") & master["lookup_tag"].notna()
    master.loc[update_mask, "source_tag"] = master.loc[update_mask, "lookup_tag"]

    source_tag_after = master["source_tag"].fillna("unknown").astype(str)
    unknown_after = int((source_tag_after == "unknown").sum())

    matched_exact = int((master["lookup_tag"].notna() & ~master["lookup_ambiguous"]).sum())
    ambiguous_matched = int(master["lookup_ambiguous"].sum())
    still_unknown = unknown_after

    report = {
        "total_rows": int(len(master)),
        "unknown_before": unknown_before,
        "unknown_after": unknown_after,
        "matched_exact": matched_exact,
        "ambiguous_matched": ambiguous_matched,
        "still_unknown": still_unknown,
        "corpus_unique_tags": sorted(tag_counts.keys()),
        "corpus_tag_counts": dict(tag_counts),
    }
    report_path = analytics_dir / "source_attach_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Ambiguous samples
    ambiguous_rows = master[master["lookup_ambiguous"]].head(args.max_samples)
    if not ambiguous_rows.empty:
        rows = []
        for _, row in ambiguous_rows.iterrows():
            key = row["text_key"]
            candidates = sorted(key_to_tag_counts.get(key, {}).items(), key=lambda x: -x[1])
            rows.append(
                {
                    "text": row["text"],
                    "candidates": ";".join([f"{t}:{c}" for t, c in candidates]),
                    "chosen_candidate": row.get("lookup_tag", ""),
                }
            )
        pd.DataFrame(rows).to_csv(analytics_dir / "source_attach_ambiguous_samples.csv", index=False)
    else:
        (analytics_dir / "source_attach_ambiguous_samples.csv").write_text("text,candidates,chosen_candidate\n")

    # Unmatched samples
    unmatched_rows = master[master["lookup_tag"].isna()].head(args.max_samples)
    if not unmatched_rows.empty:
        pd.DataFrame(
            {
                "text": unmatched_rows["text"].tolist(),
                "source_id": unmatched_rows.get("source_id", pd.Series([""] * len(unmatched_rows))).tolist(),
            }
        ).to_csv(analytics_dir / "source_attach_unmatched_samples.csv", index=False)
    else:
        (analytics_dir / "source_attach_unmatched_samples.csv").write_text("text,source_id\n")

    # Step 3: build mapping and apply
    unique_tags = sorted(source_tag_after.unique().tolist())
    mapping = build_mapping(unique_tags, Path(args.mapping_out), Path(args.config_mapping_out))

    master["domain_5"] = master["source_tag"].map(mapping).fillna("N/A")

    # Coverage report
    coverage = master["domain_5"].value_counts(dropna=False).reset_index()
    coverage.columns = ["domain_5", "count"]
    coverage["pct"] = coverage["count"] / max(len(master), 1)
    coverage.to_csv(analytics_dir / "domain5_mapping_coverage.csv", index=False)

    # Step 4: write master v2
    out_master = Path(args.out_master)
    out_master.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["text", "label_id", "source_tag", "source_id", "domain_5"] if c in master.columns]
    master[cols].to_csv(out_master, index=False)

    # Step 4: splits_with_domain_v2
    splits_dir = Path(args.splits_dir)
    out_splits_dir = Path(args.out_splits_dir)
    out_splits_dir.mkdir(parents=True, exist_ok=True)

    join_key = master[["source_id", "text", "domain_5"]].copy()
    join_key["text_key"] = join_key["text"].map(text_key)
    join_key = join_key.drop(columns=["text"])

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.csv"
        if not split_path.exists():
            print(f"Split missing: {split_path}")
            continue
        split_df = pd.read_csv(split_path)
        split_df["text_key"] = split_df["text"].map(text_key)
        merged = split_df.merge(
            join_key,
            on=["source_id", "text_key"],
            how="left",
        )
        merged = merged.drop(columns=["text_key"])
        merged["domain_5"] = merged["domain_5"].fillna("N/A")
        merged.to_csv(out_splits_dir / f"{split_name}.csv", index=False)

    print("Done. Outputs written under data/analytics and data/part1/processed/*_v2.")


if __name__ == "__main__":
    main()
