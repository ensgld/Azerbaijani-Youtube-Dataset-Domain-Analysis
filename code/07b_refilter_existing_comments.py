"""Re-filter existing comments with the updated AZ-only filter (no refetch)."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from config import (
    DOMAINS,
    YOUTUBE_COMMENTS_FILTERED_DIR,
    YOUTUBE_COMMENTS_RAW_DIR,
    DELIVERABLES_DIR,
    ANALYTICS_DIR,
    AZ_FILTER_DEFAULT_THRESHOLD,
)
from utils.az_filter import is_azerbaijani
from utils.excel_export import write_video_excel

def parse_domains(domains_arg: str | None) -> List[str]:
    if not domains_arg:
        return DOMAINS
    parts = [d.strip() for d in domains_arg.split(",") if d.strip()]
    invalid = [d for d in parts if d not in DOMAINS]
    if invalid:
        raise ValueError(f"Invalid domain(s): {invalid}. Must be one of: {DOMAINS}")
    return parts

def load_jsonl(path: Path) -> List[str]:
    comments: List[str] = []
    if not path.exists():
        return comments
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

def write_jsonl(path: Path, comments: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps({"comment": c}, ensure_ascii=False) + "\n")

def ensure_progress_files():
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    by_video_path = ANALYTICS_DIR / "comments_progress_by_video.csv"
    if not by_video_path.exists():
        df = pd.DataFrame(columns=[
            "timestamp",
            "domain",
            "video_id",
            "fetched_raw_count",
            "az_accepted_count",
            "saved_to_excel_count",
            "excel_path",
            "status",
            "error_message",
        ])
        df.to_csv(by_video_path, index=False, encoding="utf-8")

    by_domain_path = ANALYTICS_DIR / "comments_progress_by_domain.json"
    if not by_domain_path.exists():
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "domains": {
                d: {
                    "videos_processed": 0,
                    "videos_with_data": 0,
                    "raw_fetched_total": 0,
                    "az_accepted_total": 0,
                    "saved_to_excel_total": 0,
                    "target_10000_reached": False,
                } for d in DOMAINS
            }
        }
        by_domain_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def update_progress_by_video(row: dict) -> None:
    by_video_path = ANALYTICS_DIR / "comments_progress_by_video.csv"
    if by_video_path.exists():
        df = pd.read_csv(by_video_path)
    else:
        df = pd.DataFrame()

    if not df.empty and "domain" in df.columns and "video_id" in df.columns:
        df = df[
            ~(
                (df["domain"].astype(str) == str(row["domain"]))
                & (df["video_id"].astype(str) == str(row["video_id"]))
            )
        ]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(by_video_path, index=False, encoding="utf-8")

    # Recompute domain totals
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["domain", "video_id"], keep="last")
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "domains": {}}
    for domain in DOMAINS:
        sub = df[df["domain"] == domain]
        summary["domains"][domain] = {
            "videos_processed": int(sub["video_id"].nunique()) if not sub.empty else 0,
            "videos_with_data": int((sub["saved_to_excel_count"] > 0).sum()) if not sub.empty else 0,
            "raw_fetched_total": int(sub["fetched_raw_count"].sum()) if not sub.empty else 0,
            "az_accepted_total": int(sub["az_accepted_count"].sum()) if not sub.empty else 0,
            "saved_to_excel_total": int(sub["saved_to_excel_count"].sum()) if not sub.empty else 0,
            "target_10000_reached": int(sub["saved_to_excel_count"].sum()) >= 10000 if not sub.empty else False,
        }
    (ANALYTICS_DIR / "comments_progress_by_domain.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=AZ_FILTER_DEFAULT_THRESHOLD)
    ap.add_argument("--prefer_raw", action="store_true")
    args = ap.parse_args()

    run_domains = parse_domains(args.domains)
    ensure_progress_files()

    started_at = datetime.now(timezone.utc).isoformat()
    total_raw_comments_read = 0
    total_written_filtered = 0
    total_rejected_cyrillic = 0
    total_rejected_nonlatin = 0
    total_rejected_turkish = 0
    files_processed = 0
    files_using_raw = 0
    files_using_filtered = 0

    total_refiltered = 0
    for domain in run_domains:
        filtered_dir = YOUTUBE_COMMENTS_FILTERED_DIR / domain
        if not filtered_dir.exists():
            print(f"{domain}: no filtered dir found, skipping.")
            continue
        files = sorted(filtered_dir.glob("*.jsonl"))
        if not files:
            print(f"{domain}: no filtered files found, skipping.")
            continue

        for fpath in files:
            video_id = fpath.stem
            raw_path = YOUTUBE_COMMENTS_RAW_DIR / f"{video_id}.jsonl"
            if args.prefer_raw and raw_path.exists():
                source_comments = load_jsonl(raw_path)
                source_label = "raw"
                files_using_raw += 1
            else:
                source_comments = load_jsonl(fpath)
                source_label = "filtered"
                files_using_filtered += 1
            files_processed += 1
            total_raw_comments_read += len(source_comments)

            filtered_comments = []
            for c in source_comments:
                passed, debug = is_azerbaijani(c, threshold=args.threshold, return_debug=True)
                if passed:
                    filtered_comments.append(c)
                else:
                    if debug.get("cyrillic_reject"):
                        total_rejected_cyrillic += 1
                    if debug.get("tr_reject"):
                        total_rejected_turkish += 1
                    if debug.get("non_latin_reject"):
                        total_rejected_nonlatin += 1

            # Overwrite filtered JSONL and Excel
            write_jsonl(fpath, filtered_comments)
            url = f"https://www.youtube.com/watch?v={video_id}"
            out_path = DELIVERABLES_DIR / domain / f"{video_id}.xlsx"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_video_excel(str(out_path), url, domain, comments=filtered_comments)

            update_progress_by_video({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain": domain,
                "video_id": video_id,
                "fetched_raw_count": len(source_comments),
                "az_accepted_count": len(filtered_comments),
                "saved_to_excel_count": len(filtered_comments),
                "excel_path": str(out_path),
                "status": "refiltered",
                "error_message": f"source={source_label}",
            })
            total_written_filtered += len(filtered_comments)
            total_refiltered += 1

        print(f"{domain}: refiltered {len(files)} files.")

    finished_at = datetime.now(timezone.utc).isoformat()
    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "threshold": args.threshold,
        "prefer_raw": bool(args.prefer_raw),
        "domains": run_domains,
        "files_processed": files_processed,
        "files_using_raw": files_using_raw,
        "files_using_filtered": files_using_filtered,
        "total_raw_comments_read": total_raw_comments_read,
        "total_written_filtered": total_written_filtered,
        "total_rejected_cyrillic": total_rejected_cyrillic,
        "total_rejected_nonlatin": total_rejected_nonlatin,
        "total_rejected_turkish": total_rejected_turkish,
        "rejects_not_mutually_exclusive": True,
    }
    report_path = ANALYTICS_DIR / "refilter_run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Refilter complete. Files updated:", total_refiltered)
    print("Wrote:", report_path)

if __name__ == "__main__":
    main()
