#!/usr/bin/env python3
"""Build calibration summary comparing before/after calibration and label smoothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from config import TOKENIZERS_DIR, MAX_LEN


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    probs = np.clip(probs, eps, 1.0)
    logp = np.log(probs) / temperature
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def resolve_label_col(df: pd.DataFrame) -> str:
    for col in ["label_id", "label", "y", "sentiment_label"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found in test split")


def resolve_text_col(df: pd.DataFrame) -> str:
    for col in ["text", "cleaned_text", "comment"]:
        if col in df.columns:
            return col
    raise ValueError("No text column found in test split")


def load_top_error_direction(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not set(["true_label_id", "pred_label_id"]).issubset(df.columns):
        return {}
    counts = (
        df.groupby(["true_label_id", "pred_label_id"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    if counts.empty:
        return {}
    row = counts.iloc[0]
    return {
        "true_label_id": int(row["true_label_id"]),
        "pred_label_id": int(row["pred_label_id"]),
        "count": int(row["count"]),
    }


def get_pred_dist_from_probs(y_pred: np.ndarray) -> list:
    counts = np.bincount(y_pred, minlength=3)
    total = int(counts.sum())
    rows = []
    for cls_id in range(3):
        rows.append({
            "class_id": cls_id,
            "count": int(counts[cls_id]),
            "pct": float(counts[cls_id] / max(total, 1)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_model_path", type=str, default="data/analytics/best_model_path.txt")
    ap.add_argument("--temperature_json", type=str, default="data/analytics/calibration/temperature.json")
    ap.add_argument("--test_path", type=str, default="data/part1/processed/splits_with_domain/test.csv")
    ap.add_argument("--tokenizer_path", type=str, default=str(TOKENIZERS_DIR / "tokenizer.json"))
    ap.add_argument("--out_dir", type=str, default="data/analytics/calibration")
    ap.add_argument("--errors_before", type=str, default="report_tables/top_errors_confident_wrong.csv")
    ap.add_argument("--errors_after", type=str, default="report_tables/top_errors_confident_wrong_calibrated.csv")
    ap.add_argument("--youtube_before", type=str, default="data/analytics/youtube_sentiment_distribution_best.csv")
    ap.add_argument("--youtube_after", type=str, default="data/analytics/youtube_sentiment_distribution_best_calibrated.csv")
    ap.add_argument("--ls_metrics", type=str, default="runs/w2v_tuned_ls/metrics.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path_txt = Path(args.best_model_path)
    if not best_path_txt.exists():
        raise SystemExit("best_model_path.txt not found")
    model_path = Path(best_path_txt.read_text().strip())
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    temperature_path = Path(args.temperature_json)
    if not temperature_path.exists():
        raise SystemExit(f"temperature_json not found: {temperature_path}")
    temperature_payload = json.loads(temperature_path.read_text(encoding="utf-8"))
    temperature = float(temperature_payload.get("temperature"))

    # Metrics before
    metrics_path = model_path.parent / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"metrics.json not found for best model: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    test_macro_f1_before = metrics.get("test_macro_f1", metrics.get("macro_f1"))

    # Metrics label smoothing
    ls_metrics_path = Path(args.ls_metrics)
    test_macro_f1_ls = None
    if ls_metrics_path.exists():
        ls_metrics = json.loads(ls_metrics_path.read_text(encoding="utf-8"))
        test_macro_f1_ls = ls_metrics.get("test_macro_f1", ls_metrics.get("macro_f1"))

    # Test distribution before (from run file)
    pred_dist_before_path = model_path.parent / "test_pred_distribution.csv"
    pred_dist_before = []
    if pred_dist_before_path.exists():
        pred_dist_before = pd.read_csv(pred_dist_before_path).to_dict(orient="records")

    # Test distribution after calibration (compute)
    test_path = Path(args.test_path)
    df = pd.read_csv(test_path)
    label_col = resolve_label_col(df)
    text_col = resolve_text_col(df)
    tok = tokenizer_from_json(Path(args.tokenizer_path).read_text(encoding="utf-8"))

    X = pad_sequences(tok.texts_to_sequences(df[text_col].astype(str).tolist()), maxlen=MAX_LEN, padding="post", truncating="post")
    y_true = df[label_col].astype(int).values

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=256, verbose=0)
    assert probs.ndim == 2 and probs.shape[1] == 3, "Expected shape (n, 3) for class probabilities"
    probs_cal = apply_temperature(probs, temperature)
    y_pred_cal = probs_cal.argmax(axis=1)
    pred_dist_after = get_pred_dist_from_probs(y_pred_cal)

    # Approx macro-f1 after calibration from ExpA calibrated file if present
    expA_cal_path = Path("data/analytics/expA_domainwise_macro_f1_best_calibrated.csv")
    test_macro_f1_after = None
    if expA_cal_path.exists():
        expA_df = pd.read_csv(expA_cal_path)
        overall = expA_df[expA_df["scope"] == "overall_all_rows"]
        if not overall.empty:
            test_macro_f1_after = float(overall.iloc[0]["macro_f1"])

    # Error directions
    top_error_before = load_top_error_direction(Path(args.errors_before))
    top_error_after = load_top_error_direction(Path(args.errors_after))

    # YouTube distribution before/after
    yt_before = []
    yt_after = []
    if Path(args.youtube_before).exists():
        yt_before = pd.read_csv(args.youtube_before).to_dict(orient="records")
    if Path(args.youtube_after).exists():
        yt_after = pd.read_csv(args.youtube_after).to_dict(orient="records")

    def avg_pos_pct(rows):
        if not rows:
            return None
        vals = [r.get("pos_pct") for r in rows if r.get("pos_pct") is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "best_run_name": model_path.parent.name,
        "best_model_path": str(model_path),
        "test_macro_f1_before": test_macro_f1_before,
        "test_macro_f1_after_calibrated": test_macro_f1_after,
        "test_macro_f1_label_smoothing": test_macro_f1_ls,
        "top_error_direction_before": top_error_before,
        "top_error_direction_after": top_error_after,
        "test_pred_distribution_before": pred_dist_before,
        "test_pred_distribution_after_calibrated": pred_dist_after,
        "youtube_distribution_before": yt_before,
        "youtube_distribution_after_calibrated": yt_after,
    }

    out_json = out_dir / "calibration_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = out_dir / "calibration_summary.csv"
    csv_row = {
        "best_run_name": model_path.parent.name,
        "test_macro_f1_before": test_macro_f1_before,
        "test_macro_f1_after_calibrated": test_macro_f1_after,
        "test_macro_f1_label_smoothing": test_macro_f1_ls,
        "top_error_before": f"{top_error_before.get('true_label_id')}->{top_error_before.get('pred_label_id')} ({top_error_before.get('count')})" if top_error_before else None,
        "top_error_after": f"{top_error_after.get('true_label_id')}->{top_error_after.get('pred_label_id')} ({top_error_after.get('count')})" if top_error_after else None,
        "test_pos_pct_before": next((r["pct"] for r in pred_dist_before if r["class_id"] == 2), None),
        "test_pos_pct_after": next((r["pct"] for r in pred_dist_after if r["class_id"] == 2), None),
        "youtube_pos_pct_avg_before": avg_pos_pct(yt_before),
        "youtube_pos_pct_avg_after": avg_pos_pct(yt_after),
    }
    pd.DataFrame([csv_row]).to_csv(out_csv, index=False, encoding="utf-8")

    print("Wrote:")
    print(out_json)
    print(out_csv)


if __name__ == "__main__":
    main()
