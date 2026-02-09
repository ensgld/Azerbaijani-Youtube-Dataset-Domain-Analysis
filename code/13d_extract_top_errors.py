#!/usr/bin/env python3
"""Extract top misclassification examples from a specified test split."""

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

LABEL_MAP = {0: "neg", 1: "neu", 2: "pos"}


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


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    probs = np.clip(probs, eps, 1.0)
    logp = np.log(probs) / temperature
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract top misclassification examples from a specified test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python code/13d_extract_top_errors.py \\\n"
            "    --test_path data/part1/processed/splits_with_domain/test.csv\n"
        ),
    )
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--tokenizer_path", type=str, default=str(TOKENIZERS_DIR / "tokenizer.json"))
    ap.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test split CSV (must be splits_with_domain/test.csv recommended)",
    )
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--out_dir", type=str, default="report_tables")
    ap.add_argument("--temperature_json", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_path = Path(args.test_path)
    if not test_path.exists():
        raise SystemExit(f"Test split not found: {test_path}")

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path}")
    else:
        best_path_txt = Path("data/analytics/best_model_path.txt")
        if not best_path_txt.exists():
            raise SystemExit("best_model_path.txt not found; pass --model_path")
        raw = best_path_txt.read_text().strip()
        if not raw:
            raise SystemExit("best_model_path.txt is empty; pass --model_path")
        model_path = Path(raw)
        if not model_path.exists():
            raise SystemExit(f"Model not found: {model_path} (from best_model_path.txt)")

    temperature = None
    if args.temperature_json:
        temp_path = Path(args.temperature_json)
        if not temp_path.exists():
            raise SystemExit(f"temperature_json not found: {temp_path}")
        temp_payload = json.loads(temp_path.read_text(encoding="utf-8"))
        temperature = float(temp_payload.get("temperature"))

    tok = tokenizer_from_json(Path(args.tokenizer_path).read_text(encoding="utf-8"))
    df = pd.read_csv(test_path)
    label_col = resolve_label_col(df)
    text_col = resolve_text_col(df)
    has_domain_5 = "domain_5" in df.columns
    if not has_domain_5:
        print("Warning: domain_5 not found; errors will be reported without domain breakdown.")

    print(f"test_path={test_path}")
    print(f"loaded_rows={len(df)}")
    print(f"label_col={label_col}")
    print(f"has_domain_5={has_domain_5}")
    print(f"model_path={model_path}")
    if temperature is not None:
        print(f"temperature={temperature}")

    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).values
    domain_values = df["domain_5"].tolist() if has_domain_5 else None

    seqs = tok.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=args.max_len, padding="post", truncating="post")

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=256, verbose=0)
    assert probs.ndim == 2 and probs.shape[1] == 3, "Expected shape (n, 3) for class probabilities"
    if temperature is not None:
        probs = apply_temperature(probs, temperature)
    y_pred = probs.argmax(axis=1)

    wrong_mask = y_pred != y_true
    if wrong_mask.sum() == 0:
        print("No misclassifications found.")
        return

    wrong_idx = np.where(wrong_mask)[0]
    wrong_rows = []
    for idx in wrong_idx:
        domain_5 = domain_values[idx] if domain_values is not None else ""
        wrong_rows.append({
            "text": texts[idx],
            "domain_5": domain_5,
            "true_label_id": int(y_true[idx]),
            "pred_label_id": int(y_pred[idx]),
            "true_label": LABEL_MAP.get(int(y_true[idx]), str(int(y_true[idx]))),
            "pred_label": LABEL_MAP.get(int(y_pred[idx]), str(int(y_pred[idx]))),
            "prob_true": float(probs[idx, int(y_true[idx])]),
            "prob_pred": float(probs[idx, int(y_pred[idx])]),
            "prob_max": float(probs[idx].max()),
        })

    wrong_df = pd.DataFrame(wrong_rows)

    suffix = "_calibrated" if temperature is not None else ""

    # Sample 20 random wrong examples (deterministic)
    sample_n = min(20, len(wrong_df))
    sample_df = wrong_df.sample(n=sample_n, random_state=42)
    sample_df.to_csv(out_dir / f"top_errors_sample_20{suffix}.csv", index=False, encoding="utf-8")

    # Top 20 confident wrong examples by prob_max
    top_conf = wrong_df.sort_values("prob_max", ascending=False).head(20)
    top_conf.to_csv(out_dir / f"top_errors_confident_wrong{suffix}.csv", index=False, encoding="utf-8")

    print("Wrote:")
    print(out_dir / f"top_errors_sample_20{suffix}.csv")
    print(out_dir / f"top_errors_confident_wrong{suffix}.csv")


if __name__ == "__main__":
    main()
