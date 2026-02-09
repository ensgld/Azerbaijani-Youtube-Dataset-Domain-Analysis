"""Experiment A: train on all domains, test per domain (Part-1 only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import (
    DOMAINS,
    TOKENIZERS_DIR,
    ANALYTICS_DIR,
    MAX_LEN,
)


def load_tokenizer(path: Path):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json(path.read_text(encoding="utf-8"))


def pick_label_column(df: pd.DataFrame) -> str:
    for col in ["label", "y", "sentiment_label", "label_id"]:
        if col in df.columns:
            return col
    raise ValueError("Label column not found. Expected one of: label, y, sentiment_label, label_id")


def pick_text_column(df: pd.DataFrame) -> str:
    for col in ["text", "cleaned_text", "comment"]:
        if col in df.columns:
            return col
    raise ValueError("Text column not found. Expected one of: text, cleaned_text, comment")


def apply_temperature(probs: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    probs = np.clip(probs, eps, 1.0)
    logp = np.log(probs) / temperature
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=str(TOKENIZERS_DIR / "tokenizer.json"))
    ap.add_argument("--test_path", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--temperature_json", type=str, default=None)
    ap.add_argument("--output_tag", type=str, default="")
    args = ap.parse_args()

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    test_path = Path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    test_df = pd.read_csv(test_path)
    if "domain_5" not in test_df.columns:
        raise ValueError("domain_5 column missing in test split")

    text_col = pick_text_column(test_df)
    label_col = pick_label_column(test_df)

    test_df["domain_5"] = test_df["domain_5"].fillna("N/A")

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
    except Exception as exc:
        raise RuntimeError("TensorFlow is required to run this script. Ensure your venv has tensorflow installed.") from exc

    tok = load_tokenizer(Path(args.tokenizer_path))
    X_test = pad_sequences(
        tok.texts_to_sequences(test_df[text_col].astype(str).tolist()),
        maxlen=args.max_len,
        padding="post",
        truncating="post",
    )
    y_test = test_df[label_col].astype(int).values

    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(X_test, batch_size=256, verbose=0)
    assert y_prob.ndim == 2 and y_prob.shape[1] == 3, "Expected shape (n, 3) for class probabilities"

    if args.temperature_json:
        temp_path = Path(args.temperature_json)
        if not temp_path.exists():
            raise FileNotFoundError(f"temperature_json not found: {temp_path}")
        temp_payload = json.loads(temp_path.read_text(encoding="utf-8"))
        temperature = float(temp_payload.get("temperature"))
        y_prob = apply_temperature(y_prob, temperature)

    y_pred = y_prob.argmax(axis=1)

    rows = []
    overall_macro = f1_score(y_test, y_pred, average="macro")
    rows.append({
        "scope": "overall_all_rows",
        "domain": "ALL",
        "macro_f1": float(overall_macro),
        "n_samples": int(len(test_df)),
    })

    mapped_mask = test_df["domain_5"].isin(DOMAINS)
    mapped_count = int(mapped_mask.sum())
    unmapped_count = int(len(test_df) - mapped_count)
    per_domain_counts = {}

    for domain in DOMAINS:
        sub = test_df[test_df["domain_5"] == domain]
        per_domain_counts[domain] = int(len(sub))
        if sub.empty:
            rows.append({
                "scope": "domain_only_official_5",
                "domain": domain,
                "macro_f1": None,
                "n_samples": 0,
            })
            continue
        idx = sub.index
        macro = f1_score(y_test[idx], y_pred[idx], average="macro")
        rows.append({
            "scope": "domain_only_official_5",
            "domain": domain,
            "macro_f1": float(macro),
            "n_samples": int(len(sub)),
        })

    tag = args.output_tag or ""
    out_csv = ANALYTICS_DIR / f"expA_domainwise_macro_f1{tag}.csv"
    out_json = ANALYTICS_DIR / f"expA_domainwise_macro_f1{tag}.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    mapping_stats = {
        "test_total_samples": int(len(test_df)),
        "test_mapped_samples": mapped_count,
        "test_unmapped_samples": unmapped_count,
        "mapped_pct": float(mapped_count / len(test_df)) if len(test_df) else 0.0,
        "unmapped_pct": float(unmapped_count / len(test_df)) if len(test_df) else 0.0,
        "per_domain_counts": per_domain_counts,
    }
    mapping_path = ANALYTICS_DIR / f"expA_domainwise_mapping_stats{tag}.json"
    mapping_path.write_text(json.dumps(mapping_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Overall Macro-F1: {overall_macro:.4f} (n={len(test_df)})")
    print("Domain-wise Macro-F1 (official 5 only):")
    for domain in DOMAINS:
        n = per_domain_counts[domain]
        if n == 0:
            print(f"- {domain}: macro_f1=None n=0")
        else:
            df_domain = [r for r in rows if r["scope"] == "domain_only_official_5" and r["domain"] == domain][0]
            print(f"- {domain}: macro_f1={df_domain['macro_f1']:.4f} n={n}")
    print(f"Mapping coverage: mapped={mapped_count} unmapped={unmapped_count} (mapped_pct={mapping_stats['mapped_pct']:.3f})")

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("Wrote:", mapping_path)


if __name__ == "__main__":
    main()
