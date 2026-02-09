"""Select videos for comment crawling using metadata + quick AZ yield scan.
Outputs:
- data/youtube/selection/video_quality_by_domain.csv
- data/youtube/selection/selected_videos_for_full_fetch.csv
- data/analytics/youtube_selection_summary.json
- data/analytics/youtube_selection_failures.csv
- data/analytics/youtube_top20_selected_per_domain.csv
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv

from config import DOMAINS, ANALYTICS_DIR, YOUTUBE_DIR, YOUTUBE_VIDEOS_LABELED_DIR, AZ_FILTER_DEFAULT_THRESHOLD
from utils.az_filter import is_azerbaijani
from utils.youtube_api import get_youtube_client, fetch_comments_sample

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

def build_thresholds(min_comment_count: int) -> List[int]:
    raw = [min_comment_count, 100, 50, 0]
    seen = set()
    thresholds = []
    for t in raw:
        if t not in seen:
            thresholds.append(t)
            seen.add(t)
    return thresholds

def parse_domains(domains_arg: str | None) -> List[str]:
    if not domains_arg:
        return DOMAINS
    parts = [d.strip() for d in domains_arg.split(",") if d.strip()]
    invalid = [d for d in parts if d not in DOMAINS]
    if invalid:
        raise ValueError(f"Invalid domain(s): {invalid}. Must be one of: {DOMAINS}")
    return parts

def load_skip_video_ids(
    skip_failures_csv: str | None,
    skip_video_ids_file: str | None,
    domains: List[str],
) -> set[str]:
    skip_ids: set[str] = set()
    if skip_failures_csv:
        try:
            df = pd.read_csv(skip_failures_csv)
            if "video_id" in df.columns:
                if "reason" in df.columns:
                    df["reason"] = df["reason"].astype(str)
                    df = df[df["reason"].str.lower() == "commentsdisabled"]
                if "domain" in df.columns:
                    df = df[df["domain"].isin(domains)]
                skip_ids.update(df["video_id"].astype(str).tolist())
        except FileNotFoundError:
            pass
    if skip_video_ids_file:
        try:
            with open(skip_video_ids_file, "r", encoding="utf-8") as f:
                for line in f:
                    vid = line.strip()
                    if vid:
                        skip_ids.add(vid)
        except FileNotFoundError:
            pass
    return skip_ids

def is_quota_error(error_info: dict | None) -> bool:
    if not error_info:
        return False
    reason = (error_info.get("reason") or "").lower()
    message = (error_info.get("message") or "").lower()
    status = error_info.get("status")
    if reason == "quotaexceeded":
        return True
    if status == 403 and "quota" in message:
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(YOUTUBE_VIDEOS_LABELED_DIR / "videos_with_domain.csv"))
    ap.add_argument("--domains", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_failures_csv", type=str, default=None)
    ap.add_argument("--skip_video_ids_file", type=str, default=None)
    ap.add_argument("--min_comment_count", type=int, default=100)
    ap.add_argument("--min_success_per_domain", type=int, default=150)
    ap.add_argument("--min_az_rate", type=float, default=0.20)
    ap.add_argument("--min_expected_az", type=int, default=200)
    ap.add_argument("--sample_n", type=int, default=100)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_scan_per_domain", type=int, default=400)
    ap.add_argument("--expand_step", type=int, default=20)
    ap.add_argument("--max_candidates_to_consider", type=int, default=2000)
    ap.add_argument("--target_per_domain", type=int, default=10000)
    ap.add_argument("--az_threshold", type=float, default=AZ_FILTER_DEFAULT_THRESHOLD)
    args = ap.parse_args()

    run_domains = parse_domains(args.domains)
    skip_ids = load_skip_video_ids(args.skip_failures_csv, args.skip_video_ids_file, run_domains)

    load_dotenv()
    yt = get_youtube_client()

    in_path = Path(args.input)
    df = pd.read_csv(in_path)
    if "domain_assigned" not in df.columns:
        raise ValueError("domain_assigned column missing in input file.")

    df["commentCount"] = df["commentCount"].map(to_int)
    if "az_meta_flag" in df.columns:
        df["az_meta_flag"] = (
            df["az_meta_flag"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        df["az_meta_flag"] = False
    if "az_meta_score" not in df.columns:
        df["az_meta_score"] = 0.0

    selection_dir = YOUTUBE_DIR / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    quality_path = selection_dir / "video_quality_by_domain.csv"
    selected_path = selection_dir / "selected_videos_for_full_fetch.csv"
    failures_path = ANALYTICS_DIR / "youtube_selection_failures.csv"

    existing_quality = pd.read_csv(quality_path) if args.resume and quality_path.exists() else pd.DataFrame()
    existing_selected = pd.read_csv(selected_path) if args.resume and selected_path.exists() else pd.DataFrame()
    existing_failures = pd.read_csv(failures_path) if args.resume and failures_path.exists() else pd.DataFrame()

    ok_pairs = set()
    if not existing_quality.empty and "status" in existing_quality.columns:
        ok_df = existing_quality[existing_quality["status"] == "ok"]
        ok_pairs = set(zip(ok_df["domain_assigned"], ok_df["video_id"]))

    quality_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []

    summary: Dict[str, Dict[str, object]] = {}
    run_status = "complete"
    stop_info: Dict[str, object] = {}
    stop_all = False
    if args.min_success_per_domain > args.max_scan_per_domain:
        print(
            "Warning: min_success_per_domain > max_scan_per_domain; using min_success as effective success target."
        )
    success_target = max(args.min_success_per_domain, args.max_scan_per_domain)

    for domain in run_domains:
        domain_df = df[df["domain_assigned"] == domain].copy()
        domain_df = domain_df.sort_values(
            ["az_meta_flag", "commentCount"],
            ascending=[False, False],
        )
        num_candidates = int(len(domain_df))

        thresholds = build_thresholds(args.min_comment_count)
        scanned_ids = set()
        scanned_attempted = 0
        scanned_success = 0
        scanned_failed = 0
        candidates_considered = 0
        sum_expected = 0.0
        selected_domain_rows = []
        scan_cap = min(args.top_k, success_target)
        threshold_used = None

        for threshold in thresholds:
            eligible = domain_df[domain_df["commentCount"] >= threshold]
            if threshold_used is None and len(eligible) > 0:
                threshold_used = threshold
            if len(eligible) < args.top_k and threshold != thresholds[-1]:
                print(f"{domain}: only {len(eligible)} videos with commentCount>={threshold}, lowering threshold...")

            for _, row in eligible.iterrows():
                if stop_all:
                    break
                vid = row.get("video_id")
                if not vid:
                    continue
                if vid in skip_ids:
                    continue
                if vid in scanned_ids:
                    continue
                if (domain, vid) in ok_pairs:
                    continue
                if scanned_success >= success_target:
                    break
                if candidates_considered >= args.max_candidates_to_consider:
                    break
                if scanned_success >= scan_cap and sum_expected >= args.target_per_domain and scanned_success >= args.min_success_per_domain:
                    break
                if scanned_success >= scan_cap and sum_expected < args.target_per_domain:
                    scan_cap = min(scan_cap + args.expand_step, success_target)

                scanned_ids.add(vid)
                candidates_considered += 1
                scanned_attempted += 1

                comments, error_info = fetch_comments_sample(yt, vid, max_comments=args.sample_n)
                if error_info:
                    reason = error_info.get("reason") or "httpError"
                    message = error_info.get("message") or ""
                    status = error_info.get("status")
                    scanned_failed += 1
                    failure_rows.append({
                        "domain": domain,
                        "video_id": vid,
                        "reason": reason,
                        "error_message": message,
                        "status": status,
                    })
                    record = {
                        "domain_assigned": domain,
                        "video_id": vid,
                        "commentCount": int(row.get("commentCount", 0)),
                        "sampled_total": 0,
                        "az_accepted": 0,
                        "az_rate": 0.0,
                        "expected_az_comments": 0.0,
                        "az_meta_score": float(row.get("az_meta_score", 0.0)),
                        "az_meta_flag": bool(row.get("az_meta_flag", False)),
                        "title": row.get("title", ""),
                        "channelTitle": row.get("channelTitle", ""),
                        "publishedAt": row.get("publishedAt", ""),
                        "status": "error",
                        "error_reason": reason,
                        "error_message": message,
                    }
                    quality_rows.append(record)
                    if is_quota_error(error_info):
                        run_status = "stopped_quota"
                        stop_info = {
                            "domain": domain,
                            "video_id": vid,
                            "reason": reason,
                            "error_message": message,
                        }
                        stop_all = True
                    continue

                scanned_success += 1
                sampled_total = len(comments)
                az_accepted = 0
                if comments:
                    az_accepted = sum(1 for c in comments if is_azerbaijani(c, threshold=args.az_threshold))
                az_rate = (az_accepted / sampled_total) if sampled_total > 0 else 0.0
                expected_az = float(row.get("commentCount", 0)) * az_rate

                record = {
                    "domain_assigned": domain,
                    "video_id": vid,
                    "commentCount": int(row.get("commentCount", 0)),
                    "sampled_total": sampled_total,
                    "az_accepted": az_accepted,
                    "az_rate": round(az_rate, 6),
                    "expected_az_comments": round(expected_az, 2),
                    "az_meta_score": float(row.get("az_meta_score", 0.0)),
                    "az_meta_flag": bool(row.get("az_meta_flag", False)),
                    "title": row.get("title", ""),
                    "channelTitle": row.get("channelTitle", ""),
                    "publishedAt": row.get("publishedAt", ""),
                    "status": "ok",
                    "error_reason": "",
                    "error_message": "",
                }
                quality_rows.append(record)

                keep = False
                if az_rate >= args.min_az_rate and row.get("commentCount", 0) >= args.min_comment_count:
                    keep = True
                if expected_az >= args.min_expected_az:
                    keep = True
                if keep:
                    selected_domain_rows.append(record)
                    selected_rows.append({
                        "domain_assigned": domain,
                        "video_id": vid,
                        "expected_az_comments": round(expected_az, 2),
                        "az_rate": round(az_rate, 6),
                        "commentCount": int(row.get("commentCount", 0)),
                        "az_meta_score": float(row.get("az_meta_score", 0.0)),
                        "title": row.get("title", ""),
                    })
                    sum_expected += expected_az

            if stop_all:
                break
            if scanned_success >= success_target or candidates_considered >= args.max_candidates_to_consider:
                break

        az_rates = [r["az_rate"] for r in selected_domain_rows if r.get("sampled_total", 0) > 0]
        avg_az_rate = sum(az_rates) / len(az_rates) if az_rates else 0.0
        expected_vals = [r["expected_az_comments"] for r in selected_domain_rows]
        avg_expected = sum(expected_vals) / len(expected_vals) if expected_vals else 0.0
        sum_expected = sum(expected_vals) if expected_vals else 0.0
        est_needed = int(math.ceil(args.target_per_domain / avg_expected)) if avg_expected > 0 else None

        summary[domain] = {
            "num_candidates": num_candidates,
            "num_prefiltered": int(len(domain_df[domain_df["commentCount"] >= (threshold_used or 0)])),
            "scanned_attempted": int(scanned_attempted),
            "scanned_success": int(scanned_success),
            "scanned_failed": int(scanned_failed),
            "candidates_considered": int(candidates_considered),
            "selected_count": int(len(selected_domain_rows)),
            "avg_az_rate": round(avg_az_rate, 6),
            "avg_expected_az_comments": round(avg_expected, 2),
            "sum_expected_az_comments": round(sum_expected, 2),
            "min_comment_count_used": int(threshold_used if threshold_used is not None else 0),
            "estimated_videos_needed": est_needed,
            "target_feasible": sum_expected >= args.target_per_domain,
        }

        print(
            f"{domain}: attempted={scanned_attempted}, success={scanned_success}, "
            f"failed={scanned_failed}, selected={len(selected_domain_rows)}, sum_expected={round(sum_expected,2)}"
        )

        if stop_all:
            break

    quality_df_new = pd.DataFrame(quality_rows)
    selected_df_new = pd.DataFrame(selected_rows)
    failures_df_new = pd.DataFrame(failure_rows)

    if args.resume and not existing_quality.empty:
        quality_df = pd.concat([existing_quality, quality_df_new], ignore_index=True)
        quality_df = quality_df.drop_duplicates(subset=["domain_assigned", "video_id"], keep="last")
    else:
        quality_df = quality_df_new
    quality_df.to_csv(quality_path, index=False, encoding="utf-8")

    if args.resume and not existing_selected.empty:
        selected_df = pd.concat([existing_selected, selected_df_new], ignore_index=True)
        selected_df = selected_df.drop_duplicates(subset=["domain_assigned", "video_id"], keep="last")
    else:
        selected_df = selected_df_new
    if not selected_df.empty:
        selected_df = selected_df.sort_values(["domain_assigned", "expected_az_comments"], ascending=[True, False])
    selected_df.to_csv(selected_path, index=False, encoding="utf-8")

    if args.resume and not existing_failures.empty:
        failures_df = pd.concat([existing_failures, failures_df_new], ignore_index=True)
        failures_df = failures_df.drop_duplicates(subset=["domain", "video_id", "reason"], keep="last")
    else:
        failures_df = failures_df_new
    failures_df.to_csv(failures_path, index=False, encoding="utf-8")

    summary_payload = {
        "run_status": run_status,
        "run_domains": run_domains,
        "stop_info": stop_info if stop_info else None,
        "per_domain": summary,
    }
    summary_path = ANALYTICS_DIR / "youtube_selection_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    top20_rows = []
    if not selected_df.empty:
        for domain, group in selected_df.groupby("domain_assigned"):
            top = group.sort_values("expected_az_comments", ascending=False).head(20)
            for _, row in top.iterrows():
                top20_rows.append({
                    "domain": domain,
                    "video_id": row.get("video_id", ""),
                    "expected_az_comments": row.get("expected_az_comments", 0),
                    "az_rate": row.get("az_rate", 0),
                    "commentCount": row.get("commentCount", 0),
                    "az_meta_score": row.get("az_meta_score", 0),
                    "title": row.get("title", ""),
                })
    top20_path = ANALYTICS_DIR / "youtube_top20_selected_per_domain.csv"
    pd.DataFrame(top20_rows).to_csv(top20_path, index=False, encoding="utf-8")

    print("Wrote:", quality_path)
    print("Wrote:", selected_path)
    print("Wrote:", failures_path)
    print("Wrote:", summary_path)
    print("Wrote:", top20_path)

    if run_status != "complete":
        print("Run stopped:", run_status, stop_info)

if __name__ == "__main__":
    main()
