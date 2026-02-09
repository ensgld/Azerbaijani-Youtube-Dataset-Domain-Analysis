#!/usr/bin/env python3
"""Fit a single temperature parameter using validation set (post-hoc calibration)."""

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


def resolve_label_col(df: pd.DataFrame) -> str:
    for col in ["label_id", "label", "y", "sentiment_label"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found in validation split")


def resolve_text_col(df: pd.DataFrame) -> str:
    for col in ["text", "cleaned_text", "comment"]:
        if col in df.columns:
            return col
    raise ValueError("No text column found in validation split")


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    probs = np.clip(probs, eps, 1.0)
    logp = np.log(probs) / temperature
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
    probs = np.clip(probs, eps, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--val_path", type=str, default="data/part1/processed/splits_with_domain/val.csv")
    ap.add_argument("--tokenizer_path", type=str, default=str(TOKENIZERS_DIR / "tokenizer.json"))
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--out_dir", type=str, default="data/analytics/calibration")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        best_path_txt = Path("data/analytics/best_model_path.txt")
        if not best_path_txt.exists():
            raise SystemExit("best_model_path.txt not found; pass --model_path")
        raw = best_path_txt.read_text().strip()
        if not raw:
            raise SystemExit("best_model_path.txt is empty; pass --model_path")
        model_path = Path(raw)

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    val_path = Path(args.val_path)
    if not val_path.exists():
        raise SystemExit(f"Val split not found: {val_path}")

    tok = tokenizer_from_json(Path(args.tokenizer_path).read_text(encoding="utf-8"))
    df = pd.read_csv(val_path)
    label_col = resolve_label_col(df)
    text_col = resolve_text_col(df)

    texts = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).values
    X = pad_sequences(tok.texts_to_sequences(texts), maxlen=args.max_len, padding="post", truncating="post")

    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X, batch_size=256, verbose=0)
    assert probs.ndim == 2 and probs.shape[1] == 3, "Expected shape (n, 3) for class probabilities"

    nll_before = nll(probs, y_true)
    acc_before = float((probs.argmax(axis=1) == y_true).mean())

    # Grid search for temperature
    candidates = np.linspace(0.5, 5.0, 91)
    best_t = 1.0
    best_nll = nll_before

    for t in candidates:
        cal = apply_temperature(probs, float(t))
        val_nll = nll(cal, y_true)
        if val_nll < best_nll:
            best_nll = val_nll
            best_t = float(t)

    # Refine around best
    fine_candidates = np.linspace(max(0.1, best_t - 0.2), best_t + 0.2, 41)
    for t in fine_candidates:
        cal = apply_temperature(probs, float(t))
        val_nll = nll(cal, y_true)
        if val_nll < best_nll:
            best_nll = val_nll
            best_t = float(t)

    probs_after = apply_temperature(probs, best_t)
    nll_after = nll(probs_after, y_true)
    acc_after = float((probs_after.argmax(axis=1) == y_true).mean())

    payload = {
        "temperature": best_t,
        "val_nll_before": nll_before,
        "val_nll_after": nll_after,
        "val_acc_before": acc_before,
        "val_acc_after": acc_after,
        "val_size": int(len(y_true)),
        "model_path": str(model_path),
        "val_path": str(val_path),
    }

    out_path = out_dir / "temperature.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:", out_path)
    print(f"Best temperature: {best_t:.3f}")
    print(f"NLL before/after: {nll_before:.4f} -> {nll_after:.4f}")


if __name__ == "__main__":
    main()
