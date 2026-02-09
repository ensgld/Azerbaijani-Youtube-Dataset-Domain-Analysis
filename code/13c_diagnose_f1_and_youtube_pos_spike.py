#!/usr/bin/env python3
"""End-to-end diagnostics for low Macro-F1 and YouTube positive spike."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from config import (
    PART1_PROCESSED_DIR,
    RUNS_DIR,
    TOKENIZERS_DIR,
    MAX_LEN,
    YOUTUBE_COMMENTS_FILTERED_DIR,
    DELIVERABLES_DIR,
    ANALYTICS_DIR,
    DOMAINS,
)
from utils.io import load_jsonl

LABEL_MAP = {0: "neg", 1: "neu", 2: "pos"}


def resolve_label_col(df: pd.DataFrame) -> str:
    for col in ["label_id", "label", "y", "sentiment_label"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found. Expected one of label_id/label/y/sentiment_label")


def resolve_text_col(df: pd.DataFrame) -> str:
    for col in ["text", "cleaned_text", "comment"]:
        if col in df.columns:
            return col
    raise ValueError("No text column found. Expected one of text/cleaned_text/comment")


def load_tokenizer(path: Path):
    return tokenizer_from_json(path.read_text(encoding="utf-8"))


def save_label_dist(df: pd.DataFrame, label_col: str, out_path: Path) -> None:
    counts = df[label_col].value_counts(dropna=False)
    dist = counts.reset_index()
    dist.columns = ["label", "count"]
    dist["pct"] = dist["count"] / max(len(df), 1)
    dist.to_csv(out_path, index=False)


def sample_label_texts(df: pd.DataFrame, label_col: str, text_col: str, out_path: Path) -> None:
    random.seed(42)
    samples = []
    for label, group in df.groupby(label_col):
        n = min(20, len(group))
        sampled = group.sample(n=n, random_state=42)
        for _, row in sampled.iterrows():
            samples.append({"label_id": int(label), "text": str(row[text_col])})
    pd.DataFrame(samples).to_csv(out_path, index=False)


def texts_to_sequences(tok, texts: List[str], max_len: int) -> Tuple[np.ndarray, List[List[int]]]:
    seqs = tok.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded, seqs


def oov_stats(seqs: List[List[int]], oov_idx: int) -> Dict[str, float]:
    total_tokens = 0
    total_oov = 0
    empty = 0
    for s in seqs:
        if not s:
            empty += 1
            continue
        total_tokens += len(s)
        if oov_idx is not None:
            total_oov += sum(1 for t in s if t == oov_idx)
    oov_rate = (total_oov / total_tokens) if total_tokens > 0 else 0.0
    empty_ratio = empty / max(len(seqs), 1)
    avg_len = (total_tokens / max(len(seqs) - empty, 1)) if len(seqs) > 0 else 0.0
    return {
        "total_comments": int(len(seqs)),
        "total_tokens": int(total_tokens),
        "oov_tokens": int(total_oov),
        "oov_rate": float(oov_rate),
        "empty_ratio": float(empty_ratio),
        "avg_len": float(avg_len),
    }


def collect_youtube_comments() -> Dict[str, List[str]]:
    comments_by_domain: Dict[str, List[str]] = defaultdict(list)

    if YOUTUBE_COMMENTS_FILTERED_DIR.exists():
        for domain_dir in sorted(YOUTUBE_COMMENTS_FILTERED_DIR.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for jsonl_path in domain_dir.glob("*.jsonl"):
                for record in load_jsonl(jsonl_path):
                    text = record.get("comment", "")
                    if text:
                        comments_by_domain[domain].append(str(text))

    # Optional: deliverables jsonl if present
    if DELIVERABLES_DIR.exists():
        for jsonl_path in DELIVERABLES_DIR.rglob("*.jsonl"):
            domain = jsonl_path.parent.name
            for record in load_jsonl(jsonl_path):
                text = record.get("comment", "")
                if text:
                    comments_by_domain[domain].append(str(text))

    return comments_by_domain


def compute_prob_stats(probs: np.ndarray) -> Dict[str, float]:
    max_prob = probs.max(axis=1)
    mean_prob_per_class = probs.mean(axis=0).tolist()
    return {
        "mean_prob_per_class": mean_prob_per_class,
        "mean_max_prob": float(max_prob.mean()) if len(max_prob) else 0.0,
        "pct_maxprob_gt_0.9": float((max_prob > 0.9).mean()) if len(max_prob) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--tokenizer_path", type=str, default=str(TOKENIZERS_DIR / "tokenizer.json"))
    args = parser.parse_args()

    diag_dir = ANALYTICS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # A) Label sanity
    splits_dir = PART1_PROCESSED_DIR / "splits_with_domain"
    splits = {
        "train": splits_dir / "train.csv",
        "val": splits_dir / "val.csv",
        "test": splits_dir / "test.csv",
    }

    for split_name, path in splits.items():
        df = pd.read_csv(path)
        label_col = resolve_label_col(df)
        out_path = diag_dir / f"diag_labels_{split_name}.csv"
        save_label_dist(df, label_col, out_path)

    test_df = pd.read_csv(splits["test"])
    label_col = resolve_label_col(test_df)
    text_col = resolve_text_col(test_df)
    sample_label_texts(test_df, label_col, text_col, diag_dir / "diag_label_samples_test.csv")

    # Tokenizer
    tok = load_tokenizer(Path(args.tokenizer_path))

    # B) Model prediction sanity on test
    collapse_runs = []
    X_test, _ = texts_to_sequences(tok, test_df[text_col].astype(str).tolist(), args.max_len)
    y_true = test_df[label_col].astype(int).values

    for run_dir in sorted(RUNS_DIR.iterdir()):
        model_path = run_dir / "model.keras"
        if not model_path.exists():
            continue
        run_name = run_dir.name
        model = tf.keras.models.load_model(model_path)
        y_prob = model.predict(X_test, batch_size=256, verbose=0)
        assert y_prob.ndim == 2 and y_prob.shape[1] == 3, "Expected shape (n, 3) for class probabilities"
        y_pred = y_prob.argmax(axis=1)

        # Pred distribution
        counts = np.bincount(y_pred, minlength=3)
        dist_rows = []
        for cls_id in range(3):
            dist_rows.append({
                "class_id": cls_id,
                "class_name": LABEL_MAP.get(cls_id, str(cls_id)),
                "count": int(counts[cls_id]),
                "pct": float(counts[cls_id] / max(len(y_pred), 1)),
            })
        pd.DataFrame(dist_rows).to_csv(diag_dir / f"diag_test_pred_dist_{run_name}.csv", index=False)

        max_pct = max([r["pct"] for r in dist_rows]) if dist_rows else 0.0
        if max_pct > 0.9:
            collapse_runs.append(run_name)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        pd.DataFrame(cm, columns=["pred_0", "pred_1", "pred_2"], index=["true_0", "true_1", "true_2"]).to_csv(
            diag_dir / f"diag_test_confusion_{run_name}.csv"
        )

        # Classification report
        report = classification_report(y_true, y_pred, digits=4)
        (diag_dir / f"diag_test_report_{run_name}.txt").write_text(report, encoding="utf-8")

        # Prob stats
        prob_stats = compute_prob_stats(y_prob)
        prob_stats.update({
            "run": run_name,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        })
        (diag_dir / f"diag_test_prob_stats_{run_name}.json").write_text(
            json.dumps(prob_stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # C) YouTube prediction sanity + OOV
    comments_by_domain = collect_youtube_comments()
    oov_idx = tok.word_index.get(tok.oov_token) if tok.oov_token else None

    dist_rows = []
    oov_stats_by_domain = {}
    prob_stats_by_domain = {}
    top_pos = []
    top_neg = []

    yt_model_path = RUNS_DIR / "ft_tuned" / "model.keras"
    if not yt_model_path.exists():
        for run_dir in sorted(RUNS_DIR.iterdir()):
            candidate = run_dir / "model.keras"
            if candidate.exists():
                yt_model_path = candidate
                break
    if not yt_model_path.exists():
        raise SystemExit("No model.keras found under runs/ for YouTube diagnostics")

    yt_model = tf.keras.models.load_model(yt_model_path)

    for domain in sorted(comments_by_domain.keys()):
        comments = comments_by_domain[domain]
        if not comments:
            continue
        X, seqs = texts_to_sequences(tok, comments, args.max_len)
        oov_stats_by_domain[domain] = oov_stats(seqs, oov_idx)

        probs = yt_model.predict(X, batch_size=256, verbose=0)
        assert probs.ndim == 2 and probs.shape[1] == 3, "Expected shape (n, 3) for class probabilities"
        preds = probs.argmax(axis=1)
        counts = np.bincount(preds, minlength=3)
        total = int(counts.sum())
        dist_rows.append({
            "domain": domain,
            "total": total,
            "neg_count": int(counts[0]),
            "neu_count": int(counts[1]),
            "pos_count": int(counts[2]),
            "neg_pct": round(int(counts[0]) / max(total, 1), 4),
            "neu_pct": round(int(counts[1]) / max(total, 1), 4),
            "pos_pct": round(int(counts[2]) / max(total, 1), 4),
        })

        prob_stats_by_domain[domain] = compute_prob_stats(probs)

        for i, text in enumerate(comments):
            top_pos.append({
                "domain": domain,
                "prob": float(probs[i, 2]),
                "text": text,
                "neg_prob": float(probs[i, 0]),
                "neu_prob": float(probs[i, 1]),
                "pos_prob": float(probs[i, 2]),
            })
            top_neg.append({
                "domain": domain,
                "prob": float(probs[i, 0]),
                "text": text,
                "neg_prob": float(probs[i, 0]),
                "neu_prob": float(probs[i, 1]),
                "pos_prob": float(probs[i, 2]),
            })

    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(diag_dir / "diag_youtube_pred_dist_by_domain.csv", index=False)

    (diag_dir / "diag_youtube_oov_and_length_stats.json").write_text(
        json.dumps(oov_stats_by_domain, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (diag_dir / "diag_youtube_prob_stats.json").write_text(
        json.dumps(prob_stats_by_domain, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    top_pos_sorted = sorted(top_pos, key=lambda x: -x["prob"])[:50]
    top_neg_sorted = sorted(top_neg, key=lambda x: -x["prob"])[:50]
    pd.DataFrame(top_pos_sorted).to_csv(diag_dir / "diag_youtube_top_examples_pos.csv", index=False)
    pd.DataFrame(top_neg_sorted).to_csv(diag_dir / "diag_youtube_top_examples_neg.csv", index=False)

    # D) Before/after diff for YouTube distribution if prior exists
    youtube_pos_spike = False
    if not dist_df.empty and "pos_pct" in dist_df.columns:
        if (dist_df["pos_pct"] > 0.9).all():
            youtube_pos_spike = True
    before_path = ANALYTICS_DIR / "youtube_sentiment_distribution.csv"
    if before_path.exists() and not dist_df.empty:
        before_df = pd.read_csv(before_path)
        merged = before_df.merge(dist_df, on="domain", suffixes=("_before", "_after"))
        diff_rows = []
        for _, row in merged.iterrows():
            for col in ["neg_pct", "neu_pct", "pos_pct", "neg_count", "neu_count", "pos_count", "total"]:
                before_val = row.get(f"{col}_before", None)
                after_val = row.get(f"{col}_after", None)
                if pd.isna(before_val) or pd.isna(after_val):
                    continue
                if float(before_val) != float(after_val):
                    diff_rows.append({
                        "domain": row["domain"],
                        "metric": col,
                        "before": before_val,
                        "after": after_val,
                        "delta": float(after_val) - float(before_val),
                    })
        if diff_rows:
            pd.DataFrame(diff_rows).to_csv(diag_dir / "diag_youtube_dist_before_after.csv", index=False)

    # Summary heuristic
    problem_type = []
    evidence = []

    if collapse_runs:
        problem_type.append("model_collapse_argmax_bias")
        evidence.append(str(diag_dir / f"diag_test_pred_dist_{collapse_runs[0]}.csv"))

    if youtube_pos_spike:
        problem_type.append("youtube_pos_spike")
        evidence.append(str(diag_dir / "diag_youtube_pred_dist_by_domain.csv"))

    for domain, stats in oov_stats_by_domain.items():
        if stats.get("oov_rate", 0) > 0.5 or stats.get("empty_ratio", 0) > 0.2:
            problem_type.append("oov_empty_bias")
            evidence.append(str(diag_dir / "diag_youtube_oov_and_length_stats.json"))
            break

    if not problem_type:
        problem_type.append("no_obvious_collapse_or_oov_bias")

    print("Diagnostics complete.")
    print("Problem type:", ", ".join(sorted(set(problem_type))))
    print("Evidence:")
    for e in sorted(set(evidence)):
        print("-", e)


if __name__ == "__main__":
    main()
