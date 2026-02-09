"""Verify raw cache coverage and regeneration timestamps."""

from __future__ import annotations
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import ANALYTICS_DIR, YOUTUBE_COMMENTS_RAW_DIR, YOUTUBE_COMMENTS_FILTERED_DIR, DELIVERABLES_DIR

def file_stat(path: Path, ref_ts: float | None) -> dict:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "mtime_after_refilter": (st.st_mtime >= ref_ts) if ref_ts is not None else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--refilter_report", type=str, default=str(ANALYTICS_DIR / "refilter_run_report.json"))
    ap.add_argument("--output", type=str, default=str(ANALYTICS_DIR / "raw_cache_coverage_report.json"))
    args = ap.parse_args()

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(YOUTUBE_COMMENTS_RAW_DIR.glob("*.jsonl"))
    filtered_files = list(YOUTUBE_COMMENTS_FILTERED_DIR.rglob("*.jsonl"))
    excel_files = list(DELIVERABLES_DIR.rglob("*.xlsx"))

    ref_ts = None
    refilter_report = {}
    ref_path = Path(args.refilter_report)
    if ref_path.exists():
        refilter_report = json.loads(ref_path.read_text(encoding="utf-8"))
        started_at = refilter_report.get("started_at")
        if started_at:
            ref_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()

    # pick samples across domains from progress
    progress_path = ANALYTICS_DIR / "comments_progress_by_video.csv"
    samples: List[Dict[str, object]] = []
    if progress_path.exists():
        df = pd.read_csv(progress_path)
        random.seed(42)
        for domain, group in df.groupby("domain"):
            if len(samples) >= args.samples:
                break
            row = group.sample(1, random_state=42).iloc[0]
            vid = str(row["video_id"])
            raw_path = YOUTUBE_COMMENTS_RAW_DIR / f"{vid}.jsonl"
            filtered_path = YOUTUBE_COMMENTS_FILTERED_DIR / domain / f"{vid}.jsonl"
            excel_path = DELIVERABLES_DIR / domain / f"{vid}.xlsx"
            samples.append({
                "domain": domain,
                "video_id": vid,
                "raw": file_stat(raw_path, ref_ts),
                "filtered": file_stat(filtered_path, ref_ts),
                "excel": file_stat(excel_path, ref_ts),
            })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "refilter_started_at": refilter_report.get("started_at"),
        "raw_files_count": len(raw_files),
        "filtered_files_count": len(filtered_files),
        "excel_files_count": len(excel_files),
        "samples": samples,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("raw_files_count", len(raw_files))
    print("filtered_files_count", len(filtered_files))
    print("excel_files_count", len(excel_files))
    if report["refilter_started_at"]:
        print("refilter_started_at", report["refilter_started_at"])
    for s in samples:
        print("\n", s["domain"], s["video_id"])
        print("  raw", s["raw"])
        print("  filtered", s["filtered"])
        print("  excel", s["excel"])
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
