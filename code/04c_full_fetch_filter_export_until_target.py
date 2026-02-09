"""Full comment crawl + AZ filter + strict Excel export until per-domain target reached."""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from config import (
    DOMAINS,
    YOUTUBE_DIR,
    YOUTUBE_VIDEOS_LABELED_DIR,
    YOUTUBE_COMMENTS_RAW_DIR,
    YOUTUBE_COMMENTS_FILTERED_DIR,
    DELIVERABLES_DIR,
    ANALYTICS_DIR,
    AZ_FILTER_DEFAULT_THRESHOLD,
)
from utils.az_filter import is_azerbaijani
from utils.youtube_api import get_youtube_client, fetch_comments_paged
from utils.excel_export import write_video_excel

PERMANENT_FAILURES = {"commentsdisabled", "videonotfound", "invalidvideoid"}

def parse_domains(domains_arg: str | None) -> List[str]:
    if not domains_arg:
        return DOMAINS
    parts = [d.strip() for d in domains_arg.split(",") if d.strip()]
    invalid = [d for d in parts if d not in DOMAINS]
    if invalid:
        raise ValueError(f"Invalid domain(s): {invalid}. Must be one of: {DOMAINS}")
    return parts

def load_skip_ids(skip_failures_csv: str | None) -> set[str]:
    skip_ids: set[str] = set()
    if not skip_failures_csv:
        return skip_ids
    path = Path(skip_failures_csv)
    if not path.exists():
        return skip_ids
    df = pd.read_csv(path)
    if "video_id" not in df.columns:
        return skip_ids
    if "reason" in df.columns:
        df["reason"] = df["reason"].astype(str).str.lower()
        df = df[df["reason"].isin(PERMANENT_FAILURES)]
    skip_ids.update(df["video_id"].astype(str).tolist())
    return skip_ids

def load_force_ids(path_str: str | None) -> set[str]:
    if not path_str:
        return set()
    path = Path(path_str)
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            vid = line.strip()
            if vid:
                ids.add(vid)
    return ids

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

def load_progress_by_video() -> pd.DataFrame:
    by_video_path = ANALYTICS_DIR / "comments_progress_by_video.csv"
    if not by_video_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(by_video_path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    if "domain" in df.columns and "video_id" in df.columns:
        df = df.drop_duplicates(subset=["domain", "video_id"], keep="last")
    return df

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

    # Update domain totals
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

def get_domain_saved_total(domain: str) -> int:
    by_domain_path = ANALYTICS_DIR / "comments_progress_by_domain.json"
    if not by_domain_path.exists():
        return 0
    try:
        payload = json.loads(by_domain_path.read_text(encoding="utf-8"))
        return int(payload.get("domains", {}).get(domain, {}).get("saved_to_excel_total", 0))
    except Exception:
        return 0

def write_jsonl(path: Path, comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in comments:
            f.write(json.dumps({"comment": c}, ensure_ascii=False) + "\n")

def load_jsonl(path: Path) -> list[str]:
    comments: list[str] = []
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=str, default=None)
    ap.add_argument("--target_per_domain", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=AZ_FILTER_DEFAULT_THRESHOLD)
    ap.add_argument("--max_comments_per_video", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_failures_csv", type=str, default=None)
    ap.add_argument("--force_video_ids_file", type=str, default=None)
    args = ap.parse_args()

    run_domains = parse_domains(args.domains)
    ensure_progress_files()

    load_dotenv()
    yt = get_youtube_client()
    YOUTUBE_COMMENTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    selected_path = YOUTUBE_DIR / "selection" / "selected_videos_for_full_fetch.csv"
    if not selected_path.exists():
        alt_path = ANALYTICS_DIR / "selected_videos_for_full_fetch.csv"
        if alt_path.exists():
            selected_path = alt_path
        else:
            raise FileNotFoundError(f"Missing selection file: {selected_path}")
    sel = pd.read_csv(selected_path)

    skip_ids = load_skip_ids(args.skip_failures_csv)
    force_ids = load_force_ids(args.force_video_ids_file)
    progress_df = load_progress_by_video()

    stop_report_path = ANALYTICS_DIR / "quota_stop_report.json"
    full_failures_path = ANALYTICS_DIR / "youtube_full_fetch_failures.csv"
    selection_failures_path = ANALYTICS_DIR / "youtube_selection_failures.csv"
    full_failures = []
    selection_failures = []

    stop_all = False
    for domain in run_domains:
        domain_force_ids = set()
        if force_ids:
            domain_ids = set(sel[sel["domain_assigned"] == domain]["video_id"].astype(str))
            domain_force_ids = force_ids.intersection(domain_ids)

        domain_total = get_domain_saved_total(domain)
        if domain_total >= args.target_per_domain and not domain_force_ids:
            print(f"{domain}: target already reached ({domain_total}). Skipping.")
            continue

        domain_sel = sel[sel["domain_assigned"] == domain].copy()
        if domain_sel.empty:
            print(f"{domain}: no selected videos found.")
            continue

        domain_sel = domain_sel.sort_values("expected_az_comments", ascending=False)
        attempted = 0
        success = 0
        failed = 0

        for _, row in domain_sel.iterrows():
            if domain_total >= args.target_per_domain and not domain_force_ids:
                break

            vid = str(row.get("video_id", "")).strip()
            if not vid:
                continue
            if vid in skip_ids:
                continue
            if domain_force_ids and vid not in domain_force_ids:
                continue

            prev_saved = 0
            has_prev = False

            if args.resume and not progress_df.empty:
                existing = progress_df[
                    (progress_df["domain"].astype(str) == domain)
                    & (progress_df["video_id"].astype(str) == vid)
                    & (progress_df["status"].astype(str).isin(["ok", "no_az_comments", "comments_disabled"]))
                ]
                if not existing.empty:
                    has_prev = True
                    try:
                        prev_saved = int(existing.iloc[-1].get("saved_to_excel_count", 0))
                    except Exception:
                        prev_saved = 0
                if not existing.empty and (not domain_force_ids or vid not in domain_force_ids):
                    continue

            attempted += 1
            raw_path = YOUTUBE_COMMENTS_RAW_DIR / f"{vid}.jsonl"
            used_raw = False
            if raw_path.exists():
                comments = load_jsonl(raw_path)
                error_info = None
                used_raw = True
            else:
                comments, error_info = fetch_comments_paged(
                    yt,
                    vid,
                    max_comments=args.max_comments_per_video,
                )

            if error_info:
                failed += 1
                reason = (error_info.get("reason") or "httpError").lower()
                message = error_info.get("message") or ""
                status = "comments_disabled" if reason == "commentsdisabled" else "failed"
                full_failures.append({
                    "domain": domain,
                    "video_id": vid,
                    "reason": reason,
                    "error_message": message,
                })
                selection_failures.append({
                    "domain": domain,
                    "video_id": vid,
                    "reason": reason,
                    "error_message": message,
                    "status": "full_fetch",
                })
                update_progress_by_video({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "domain": domain,
                    "video_id": vid,
                    "fetched_raw_count": 0,
                    "az_accepted_count": 0,
                    "saved_to_excel_count": prev_saved if domain_force_ids and has_prev else 0,
                    "excel_path": "",
                    "status": "failed_keep_previous" if domain_force_ids and has_prev else status,
                    "error_message": message,
                })

                if reason in {"quotaexceeded", "dailylimitexceeded", "ratelimitexceeded"}:
                    stop_report_path.write_text(json.dumps({
                        "run_status": "stopped_quota",
                        "domain": domain,
                        "video_id": vid,
                        "reason": reason,
                        "error_message": message,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"{domain}: stopped due to quota at video {vid}")
                    stop_all = True
                    break
                continue

            if not used_raw:
                write_jsonl(raw_path, comments)
            success += 1
            raw_count = len(comments)
            filtered = [c for c in comments if is_azerbaijani(c, threshold=args.threshold)]
            az_count = len(filtered)
            excel_path = ""
            status = "ok"

            if az_count == 0:
                status = "no_az_comments"
                domain_total = max(0, domain_total - prev_saved)
            else:
                url = f"https://www.youtube.com/watch?v={vid}"
                out_path = DELIVERABLES_DIR / domain / f"{vid}.xlsx"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_video_excel(str(out_path), url, domain, comments=filtered)
                excel_path = str(out_path)
                filtered_path = YOUTUBE_COMMENTS_FILTERED_DIR / domain / f"{vid}.jsonl"
                write_jsonl(filtered_path, filtered)
                domain_total = max(0, domain_total - prev_saved) + az_count

            update_progress_by_video({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain": domain,
                "video_id": vid,
                "fetched_raw_count": raw_count,
                "az_accepted_count": az_count,
                "saved_to_excel_count": az_count,
                "excel_path": excel_path,
                "status": status,
                "error_message": "",
            })

        print(
            f"{domain}: attempted={attempted}, success={success}, failed={failed}, "
            f"saved_total={domain_total}"
        )
        if stop_all:
            break

    if full_failures:
        df_fail = pd.DataFrame(full_failures)
        if full_failures_path.exists():
            existing = pd.read_csv(full_failures_path)
            df_fail = pd.concat([existing, df_fail], ignore_index=True)
        df_fail.to_csv(full_failures_path, index=False, encoding="utf-8")
    if selection_failures:
        df_sel = pd.DataFrame(selection_failures)
        if selection_failures_path.exists():
            existing = pd.read_csv(selection_failures_path)
            df_sel = pd.concat([existing, df_sel], ignore_index=True)
        df_sel = df_sel.drop_duplicates(subset=["domain", "video_id", "reason", "status"], keep="last")
        df_sel.to_csv(selection_failures_path, index=False, encoding="utf-8")

    # Update current progress dashboard
    status_script = Path(__file__).resolve().parent / "06_progress_status.py"
    if status_script.exists():
        subprocess.run([sys.executable, str(status_script)], check=False)

if __name__ == "__main__":
    main()
