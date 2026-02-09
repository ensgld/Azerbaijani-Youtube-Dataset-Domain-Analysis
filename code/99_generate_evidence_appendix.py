#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "report_tables" / "EVIDENCE_APPENDIX.md"


def relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fmt_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except FileNotFoundError:
        return "unknown"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def fmt_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return "unknown"
    return f"{size} bytes"


def csv_header_hint(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
        if not first:
            return ""
        cols = [c.strip() for c in first.split(",") if c.strip()]
        if not cols:
            return ""
        preview = ",".join(cols[:8])
        if len(cols) > 8:
            preview += ",..."
        return f"schema: {preview}"
    except Exception:
        return ""


def json_key_hint(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size > 1024 * 1024:
            return ""
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = list(data.keys())
            preview = ",".join(keys[:8])
            if len(keys) > 8:
                preview += ",..."
            return f"schema: keys({preview})"
        if isinstance(data, list):
            return f"schema: list(len={len(data)})"
        return ""
    except Exception:
        return ""


def schema_hint(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return csv_header_hint(path)
    if path.suffix.lower() == ".json":
        return json_key_hint(path)
    return ""


def collect_patterns(patterns, include_dirs=False):
    matched = []
    for pattern in patterns:
        for p in ROOT.glob(pattern):
            if p.is_dir() and not include_dirs:
                continue
            if p.exists():
                matched.append(p)
    return sorted(set(matched))


def add_entries(lines, title, items):
    lines.append(f"## {title}")
    for item in items:
        label = item["label"]
        patterns = item.get("patterns")
        paths = item.get("paths")
        include_dirs = item.get("include_dirs", False)
        limit = item.get("limit")
        desc = item.get("desc", "")

        matched = []
        if paths:
            for p in paths:
                if p.exists():
                    matched.append(p)
        elif patterns:
            matched = collect_patterns(patterns, include_dirs=include_dirs)

        if not matched:
            if patterns:
                pat_str = "; ".join(patterns)
            else:
                pat_str = "; ".join([str(p) for p in paths])
            lines.append(f"- MISSING: {pat_str} — {desc}")
            continue

        if limit is not None and len(matched) > limit:
            shown = matched[:limit]
            extra = len(matched) - limit
        else:
            shown = matched
            extra = 0

        for p in shown:
            hint = schema_hint(p)
            meta = f"({fmt_size(p)}, mtime {fmt_mtime(p)})"
            hint_str = f" {hint}" if hint else ""
            lines.append(f"- `{relpath(p)}` — {desc}{hint_str} {meta}")

        if extra > 0:
            lines.append(
                f"- … ({extra} more files matched; showing first {limit} for `{label}`)"
            )
    lines.append("")


def main():
    lines = []
    lines.append("# Evidence Appendix")
    lines.append("")

    sections = [
        {
            "title": "A) Domain folders & per-video Excel files",
            "items": [
                {
                    "label": "deliverables_dir",
                    "paths": [ROOT / "deliverables"],
                    "include_dirs": True,
                    "desc": "Domain-segmented Excel deliverables folder (per-video exports).",
                },
                {
                    "label": "deliverables_xlsx",
                    "patterns": ["deliverables/*/*.xlsx"],
                    "limit": 5,
                    "desc": "Per-video Excel export (A1 URL; rows = domain, comment).",
                },
                {
                    "label": "comments_filtered_jsonl",
                    "patterns": ["data/youtube/comments_filtered/*/*.jsonl"],
                    "limit": 5,
                    "desc": "Filtered Azerbaijani-only comments (JSONL per video).",
                },
                {
                    "label": "comments_raw_jsonl",
                    "patterns": ["data/youtube/comments_raw/*.jsonl", "data/youtube/comments_raw/*/*.jsonl"],
                    "limit": 5,
                    "desc": "Raw comment cache per video (for refilter/reproducibility).",
                },
                {
                    "label": "progress_current",
                    "patterns": [
                        "data/analytics/comments_progress_current.csv",
                        "data/analytics/comments_progress_current.json",
                        "data/analytics/comments_progress_by_video.csv",
                        "data/analytics/comments_progress_by_domain.json",
                    ],
                    "desc": "Ground-truth progress tracking (saved comment counts by video/domain).",
                },
                {
                    "label": "selection_outputs",
                    "patterns": [
                        "data/youtube/video_candidates/videos_raw.csv",
                        "data/youtube/video_candidates/videos_raw_expanded*.csv",
                        "data/youtube/video_candidates/videos_raw_expanded_merged.csv",
                        "data/youtube/videos_labeled/videos_with_domain.csv",
                        "data/youtube/selection/video_quality_by_domain.csv",
                        "data/youtube/selection/selected_videos_for_full_fetch.csv",
                        "data/analytics/youtube_selection_summary.json",
                        "data/analytics/youtube_top20_selected_per_domain.csv",
                    ],
                    "desc": "YouTube discovery/selection artifacts (metadata-based domain assignment + yield scan).",
                },
            ],
        },
        {
            "title": "B) Code pipeline evidence",
            "items": [
                {
                    "label": "collection_scripts",
                    "patterns": [
                        "code/02_youtube_discover_videos.py",
                        "code/02b_expand_video_candidates.py",
                        "code/03_domain_assign_metadata.py",
                        "code/03b_select_videos_for_comments.py",
                        "code/04c_full_fetch_filter_export_until_target.py",
                        "code/utils/az_filter.py",
                        "code/utils/youtube_api.py",
                        "code/06_progress_status.py",
                        "code/07_train_gru_4runs.py",
                        "code/08_evaluate_domainwise_A.py",
                        "code/09_evaluate_domainshift_B.py",
                        "code/10_predict_youtube_domain_distributions.py",
                        "code/11_generate_mermaid_diagram.py",
                        "code/13d_extract_top_errors.py",
                        "code/13f_temperature_scaling.py",
                        "code/99_generate_evidence_appendix.py",
                    ],
                    "desc": "Core pipeline scripts for collection, filtering, training, evaluation, and reporting.",
                }
            ],
        },
        {
            "title": "C) Modeling & evaluation evidence",
            "items": [
                {
                    "label": "embedding_models",
                    "patterns": [
                        "models/embeddings/*.model",
                        "data/youtube/corpora/combined_corpus.txt",
                        "data/youtube/corpora/combined_corpus_stats.json",
                        "data/analytics/oov_rates.csv",
                        "data/analytics/oov_rates.json",
                    ],
                    "desc": "Embedding artifacts and OOV analysis (Part-1 + YouTube corpus).",
                },
                {
                    "label": "runs_models",
                    "patterns": [
                        "runs/*/model.keras",
                        "runs/*/metrics.json",
                        "runs/*/history.json",
                        "runs/*/classification_report.txt",
                        "runs/*/confusion_matrix.csv",
                        "runs/*/test_pred_distribution.csv",
                    ],
                    "desc": "GRU training artifacts for W2V/FT and variants (frozen/tuned/cw/ls).",
                },
                {
                    "label": "best_model",
                    "patterns": [
                        "data/analytics/best_model_path.txt",
                        "data/analytics/final_metrics_summary.json",
                    ],
                    "desc": "Best model pointer and final consolidated metrics (if present).",
                },
                {
                    "label": "expA_expB",
                    "patterns": [
                        "data/analytics/expA_domainwise_macro_f1*.csv",
                        "data/analytics/expA_domainwise_macro_f1*.json",
                        "data/analytics/expA_domainwise_mapping_stats*.json",
                        "data/analytics/expB_lodo_macro_f1*.csv",
                        "data/analytics/expB_lodo_macro_f1*.json",
                    ],
                    "desc": "Experiment A (domain-wise Macro-F1) and Experiment B (LODO) outputs.",
                },
                {
                    "label": "youtube_predictions",
                    "patterns": [
                        "data/analytics/youtube_sentiment_distribution*.csv",
                        "data/analytics/youtube_sentiment_distribution*.json",
                        "data/analytics/youtube_sentiment_top_examples*.csv",
                    ],
                    "desc": "YouTube descriptive predictions (unlabeled; distribution + examples).",
                },
                {
                    "label": "report_tables_best",
                    "patterns": [
                        "report_tables/expA_domainwise_best.csv",
                        "report_tables/expB_lodo_best.csv",
                        "report_tables/youtube_distribution_best.csv",
                        "report_tables/oov_rates.csv",
                    ],
                    "desc": "Final report tables (best model versions).",
                },
                {
                    "label": "mermaid",
                    "patterns": ["mermaid_gru_diagram.md"],
                    "desc": "Mermaid GRU pipeline diagram for report.",
                },
            ],
        },
        {
            "title": "D) Diagnostics & error analysis",
            "items": [
                {
                    "label": "language_audit",
                    "patterns": [
                        "data/analytics/lang_audit_by_domain_after_nonlatin.csv",
                        "data/analytics/lang_audit_by_domain_after_nonlatin.json",
                        "data/analytics/lang_audit_by_domain_after_tr*.csv",
                        "data/analytics/lang_audit_by_domain_after_tr*.json",
                    ],
                    "desc": "Language leakage audit (Cyrillic/Turkish/Non-Latin counts).",
                },
                {
                    "label": "mapping_limitations",
                    "patterns": [
                        "data/analytics/part1_domain5_distribution_v2.csv",
                        "data/analytics/part1_domain5_distribution_v2.json",
                        "data/analytics/test_domain5_distribution_v2.csv",
                        "data/analytics/test_domain5_distribution_v2.json",
                        "data/analytics/part1_source_tag_counts_v2.csv",
                        "data/analytics/part1_source_id_counts_v2.csv",
                        "data/analytics/part1_na_origin_summary_v2.csv",
                    ],
                    "desc": "Domain mapping coverage evidence (Tech/Finance = 0 due to source limits).",
                },
                {
                    "label": "top_errors",
                    "patterns": [
                        "report_tables/top_errors_sample_20*.csv",
                        "report_tables/top_errors_confident_wrong*.csv",
                        "report_tables/errors_domain5_coverage_summary.md",
                        "report_tables/top_errors_domain5_missing.csv",
                    ],
                    "desc": "Top error examples and domain_5 coverage of error analysis.",
                },
                {
                    "label": "diagnostics",
                    "patterns": [
                        "data/analytics/diagnostics/*.csv",
                        "data/analytics/diagnostics/*.json",
                        "data/analytics/diagnostics/*.txt",
                        "data/analytics/compare_macro_f1_before_after.csv",
                        "data/analytics/youtube_dist_before_after_best.csv",
                    ],
                    "desc": "Diagnostics for Macro-F1 issues, prediction collapse, and label sanity.",
                },
                {
                    "label": "calibration",
                    "patterns": ["data/analytics/calibration/*.json", "data/analytics/calibration/*.csv"],
                    "desc": "Temperature scaling outputs and calibration summary.",
                },
            ],
        },
    ]

    for section in sections:
        add_entries(lines, section["title"], section["items"])

    # Drive folder checklist
    lines.append("## Drive folder checklist")
    required_dirs = [
        "code",
        "data",
        "deliverables",
        "report_tables",
        "runs",
        "models",
        "docs",
        "config",
    ]
    for d in required_dirs:
        p = ROOT / d
        status = "OK" if p.exists() else "MISSING"
        lines.append(f"- `{d}/` — {status}")
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Print first ~80 lines for quick review
    preview_lines = lines[:80]
    print("\n".join(preview_lines))


if __name__ == "__main__":
    main()
