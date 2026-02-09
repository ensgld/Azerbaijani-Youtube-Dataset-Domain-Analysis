"""Audit filtered comments for Cyrillic/Turkish/non-Latin leakage."""

from __future__ import annotations
import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from config import ANALYTICS_DIR, YOUTUBE_COMMENTS_FILTERED_DIR, DELIVERABLES_DIR, DOMAINS
from utils.az_filter import analyze_text

def has_non_latin_letter(text: str) -> bool:
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name or "CYRILLIC" in name:
            continue
        return True
    return False

def load_comments_from_jsonl(path: Path) -> List[str]:
    comments: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("comment", "")
                if text:
                    comments.append(str(text))
            except Exception:
                continue
    return comments

def load_comments_from_excel(path: Path) -> List[str]:
    try:
        df = pd.read_excel(path, header=None)
    except Exception:
        return []
    if df.shape[1] < 2 or df.shape[0] < 2:
        return []
    comments = df.iloc[1:, 1].dropna().astype(str).tolist()
    return [c for c in comments if c.strip()]

def choose_source(source: str | None) -> str:
    if source in {"filtered", "excel"}:
        return source
    filtered_files = list(YOUTUBE_COMMENTS_FILTERED_DIR.rglob("*.jsonl"))
    if filtered_files:
        return "filtered"
    return "excel"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None, choices=["filtered", "excel"])
    ap.add_argument("--suffix", type=str, default=None)
    args = ap.parse_args()

    source = choose_source(args.source)
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    domain_stats: Dict[str, Dict[str, object]] = {}
    per_video_rows: List[Dict[str, object]] = []

    for domain in DOMAINS:
        domain_stats[domain] = {
            "domain": domain,
            "total_comments": 0,
            "cyrillic_comments": 0,
            "turkish_flagged_comments": 0,
            "non_latin_comments": 0,
            "sample_cyrillic": [],
            "sample_turkish": [],
            "sample_non_latin": [],
        }

    if source == "filtered":
        files = list(YOUTUBE_COMMENTS_FILTERED_DIR.rglob("*.jsonl"))
        for path in files:
            domain = path.parent.name
            if domain not in domain_stats:
                continue
            comments = load_comments_from_jsonl(path)
            video_id = path.stem
            v_stats = {
                "domain": domain,
                "video_id": video_id,
                "total_comments": 0,
                "cyrillic_comments": 0,
                "turkish_flagged_comments": 0,
                "non_latin_comments": 0,
            }
            for text in comments:
                v_stats["total_comments"] += 1
                d = domain_stats[domain]
                d["total_comments"] += 1
                stats = analyze_text(text)
                cyr_reject = (
                    stats["cyrillic_count"] >= 3
                    or stats["cyrillic_ratio"] >= 0.02
                )
                tr_flag = stats["tr_hits"] > 0
                non_latin = has_non_latin_letter(text)

                if cyr_reject:
                    d["cyrillic_comments"] += 1
                    v_stats["cyrillic_comments"] += 1
                    if len(d["sample_cyrillic"]) < 3:
                        d["sample_cyrillic"].append(text[:300])
                if tr_flag:
                    d["turkish_flagged_comments"] += 1
                    v_stats["turkish_flagged_comments"] += 1
                    if len(d["sample_turkish"]) < 3:
                        d["sample_turkish"].append(text[:300])
                if non_latin:
                    d["non_latin_comments"] += 1
                    v_stats["non_latin_comments"] += 1
                    if len(d["sample_non_latin"]) < 3:
                        d["sample_non_latin"].append(text[:300])

            if (
                v_stats["cyrillic_comments"]
                or v_stats["turkish_flagged_comments"]
                or v_stats["non_latin_comments"]
            ):
                per_video_rows.append(v_stats)

    else:
        files = list(DELIVERABLES_DIR.rglob("*.xlsx"))
        for path in files:
            domain = path.parent.name
            if domain not in domain_stats:
                continue
            comments = load_comments_from_excel(path)
            video_id = path.stem
            v_stats = {
                "domain": domain,
                "video_id": video_id,
                "total_comments": 0,
                "cyrillic_comments": 0,
                "turkish_flagged_comments": 0,
                "non_latin_comments": 0,
            }
            for text in comments:
                v_stats["total_comments"] += 1
                d = domain_stats[domain]
                d["total_comments"] += 1
                stats = analyze_text(text)
                cyr_reject = (
                    stats["cyrillic_count"] >= 3
                    or stats["cyrillic_ratio"] >= 0.02
                )
                tr_flag = stats["tr_hits"] > 0
                non_latin = has_non_latin_letter(text)

                if cyr_reject:
                    d["cyrillic_comments"] += 1
                    v_stats["cyrillic_comments"] += 1
                    if len(d["sample_cyrillic"]) < 3:
                        d["sample_cyrillic"].append(text[:300])
                if tr_flag:
                    d["turkish_flagged_comments"] += 1
                    v_stats["turkish_flagged_comments"] += 1
                    if len(d["sample_turkish"]) < 3:
                        d["sample_turkish"].append(text[:300])
                if non_latin:
                    d["non_latin_comments"] += 1
                    v_stats["non_latin_comments"] += 1
                    if len(d["sample_non_latin"]) < 3:
                        d["sample_non_latin"].append(text[:300])

            if (
                v_stats["cyrillic_comments"]
                or v_stats["turkish_flagged_comments"]
                or v_stats["non_latin_comments"]
            ):
                per_video_rows.append(v_stats)

    df = pd.DataFrame(domain_stats.values())
    out_csv = ANALYTICS_DIR / "lang_audit_by_domain.csv"
    out_json = ANALYTICS_DIR / "lang_audit_by_domain.json"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(domain_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.suffix:
        df.to_csv(ANALYTICS_DIR / f"lang_audit_by_domain_{args.suffix}.csv", index=False, encoding="utf-8")
        (ANALYTICS_DIR / f"lang_audit_by_domain_{args.suffix}.json").write_text(
            json.dumps(domain_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if per_video_rows:
        leak_csv = ANALYTICS_DIR / "lang_audit_leakage_videos.csv"
        pd.DataFrame(per_video_rows).to_csv(leak_csv, index=False, encoding="utf-8")
        cyr_ids = [r["video_id"] for r in per_video_rows if r.get("cyrillic_comments", 0) > 0]
        (ANALYTICS_DIR / "lang_audit_cyrillic_video_ids.txt").write_text(
            "\n".join(sorted(set(cyr_ids))),
            encoding="utf-8",
        )
    else:
        (ANALYTICS_DIR / "lang_audit_leakage_videos.csv").write_text(
            "domain,video_id,total_comments,cyrillic_comments,turkish_flagged_comments,non_latin_comments\n",
            encoding="utf-8",
        )
        (ANALYTICS_DIR / "lang_audit_cyrillic_video_ids.txt").write_text("", encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    if args.suffix:
        print("Wrote:", ANALYTICS_DIR / f"lang_audit_by_domain_{args.suffix}.csv")
        print("Wrote:", ANALYTICS_DIR / f"lang_audit_by_domain_{args.suffix}.json")
    print("Wrote:", ANALYTICS_DIR / "lang_audit_leakage_videos.csv")
    print("Wrote:", ANALYTICS_DIR / "lang_audit_cyrillic_video_ids.txt")
    print("Source used:", source)

if __name__ == "__main__":
    main()
