"""Apply Azerbaijani-only filter and export strict Excel files per video.
Outputs:
- deliverables/<domain>/<video_id>.xlsx
- data/youtube/comments_filtered/<domain>/<video_id>.jsonl
- data/analytics/comments_progress_by_video.csv
- data/analytics/comments_progress_by_domain.json
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from config import (
    YOUTUBE_VIDEOS_LABELED_DIR,
    DELIVERABLES_DIR,
    YOUTUBE_COMMENTS_RAW_DIR,
    YOUTUBE_COMMENTS_FILTERED_DIR,
    ANALYTICS_DIR,
    DOMAINS,
    AZ_FILTER_DEFAULT_THRESHOLD,
)
from utils.excel_export import write_video_excel
from utils.az_filter import is_azerbaijani

def extract_comment_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ["comment", "text", "textDisplay", "textOriginal", "text_display", "text_original"]:
            if obj.get(key):
                return str(obj.get(key))
        snippet = obj.get("snippet")
        if isinstance(snippet, dict):
            for key in ["textDisplay", "textOriginal"]:
                if snippet.get(key):
                    return str(snippet.get(key))
            tlc = snippet.get("topLevelComment")
            if isinstance(tlc, dict):
                sn = tlc.get("snippet", {})
                if isinstance(sn, dict):
                    for key in ["textDisplay", "textOriginal"]:
                        if sn.get(key):
                            return str(sn.get(key))
    return ""

def read_raw_comments(path: Path) -> tuple[list[str], int, str | None]:
    if not path.exists():
        return [], 0, "raw_comments_missing"
    comments: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = ""
            try:
                obj = json.loads(line)
                text = extract_comment_text(obj)
            except json.JSONDecodeError:
                text = line
            if text:
                comments.append(text)
    return comments, len(comments), None

def write_filtered_jsonl(path: Path, comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps({"comment": c}, ensure_ascii=False) + "\n")

def update_progress_by_video(row: dict) -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    by_video_path = ANALYTICS_DIR / "comments_progress_by_video.csv"
    if by_video_path.exists():
        df = pd.read_csv(by_video_path)
    else:
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

    df = df[
        ~(
            (df["domain"].astype(str) == str(row["domain"]))
            & (df["video_id"].astype(str) == str(row["video_id"]))
        )
    ]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(by_video_path, index=False, encoding="utf-8")

    # Update domain totals
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["domain", "video_id"], keep="last")
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "domains": {}}
    for domain in DOMAINS:
        sub = df[df["domain"] == domain]
        summary["domains"][domain] = {
            "videos_processed": int(sub["video_id"].nunique()),
            "videos_with_data": int((sub["saved_to_excel_count"] > 0).sum()),
            "raw_fetched_total": int(sub["fetched_raw_count"].sum()),
            "az_accepted_total": int(sub["az_accepted_count"].sum()),
            "saved_to_excel_total": int(sub["saved_to_excel_count"].sum()),
            "target_10000_reached": int(sub["saved_to_excel_count"].sum()) >= 10000,
        }
    with (ANALYTICS_DIR / "comments_progress_by_domain.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=AZ_FILTER_DEFAULT_THRESHOLD)
    args = ap.parse_args()

    videos_path = YOUTUBE_VIDEOS_LABELED_DIR / "videos_with_domain.csv"
    df = pd.read_csv(videos_path)

    for _, r in df.iterrows():
        domain = r["domain_assigned"]
        vid = r.get("video_id", "UNKNOWN_VIDEO")
        url = f"https://www.youtube.com/watch?v={vid}"
        raw_path = YOUTUBE_COMMENTS_RAW_DIR / domain / f"{vid}.jsonl"
        comments, raw_count, err = read_raw_comments(raw_path)

        status = "ok"
        error_message = ""
        if err:
            status = "failed"
            error_message = err
            filtered = []
        else:
            filtered = [c for c in comments if is_azerbaijani(c, threshold=args.threshold)]
            if raw_count == 0:
                status = "comments_disabled"

        saved_count = len(filtered)
        excel_path = ""
        if not err:
            out_path = (DELIVERABLES_DIR / domain / f"{vid}.xlsx")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_video_excel(str(out_path), url, domain, comments=filtered)
            excel_path = str(out_path)
            filtered_path = YOUTUBE_COMMENTS_FILTERED_DIR / domain / f"{vid}.jsonl"
            write_filtered_jsonl(filtered_path, filtered)

        update_progress_by_video({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": domain,
            "video_id": vid,
            "fetched_raw_count": raw_count,
            "az_accepted_count": len(filtered),
            "saved_to_excel_count": saved_count,
            "excel_path": excel_path,
            "status": status,
            "error_message": error_message,
        })

    print("Processed videos. Progress files updated.")

if __name__ == "__main__":
    main()
