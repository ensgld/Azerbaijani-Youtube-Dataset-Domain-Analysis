"""Predict sentiment distribution on unlabeled YouTube comments (descriptive only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import YOUTUBE_COMMENTS_FILTERED_DIR, ANALYTICS_DIR, TOKENIZERS_DIR, MAX_LEN
from utils.io import load_jsonl

LABEL_MAP = {0: "neg", 1: "neu", 2: "pos"}


def load_tokenizer(path: Path):
    return tokenizer_from_json(path.read_text(encoding="utf-8"))


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
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--max_per_domain", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--temperature_json", type=str, default=None)
    ap.add_argument("--output_tag", type=str, default="")
    args = ap.parse_args()

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(Path(args.tokenizer_path))
    model = tf.keras.models.load_model(args.model_path)

    temperature = None
    if args.temperature_json:
        temp_path = Path(args.temperature_json)
        if not temp_path.exists():
            raise FileNotFoundError(f"temperature_json not found: {temp_path}")
        temp_payload = json.loads(temp_path.read_text(encoding="utf-8"))
        temperature = float(temp_payload.get("temperature"))

    dist_rows = []
    example_rows = []

    for domain_dir in sorted(YOUTUBE_COMMENTS_FILTERED_DIR.iterdir()):
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name
        comments: List[str] = []
        for jsonl_path in domain_dir.glob("*.jsonl"):
            for record in load_jsonl(jsonl_path):
                text = record.get("comment", "")
                if text:
                    comments.append(str(text))
                    if args.max_per_domain and len(comments) >= args.max_per_domain:
                        break
            if args.max_per_domain and len(comments) >= args.max_per_domain:
                break

        if not comments:
            dist_rows.append({
                "domain": domain,
                "total": 0,
                "neg_count": 0,
                "neu_count": 0,
                "pos_count": 0,
                "neg_pct": 0.0,
                "neu_pct": 0.0,
                "pos_pct": 0.0,
            })
            continue

        X = pad_sequences(
            tok.texts_to_sequences(comments),
            maxlen=args.max_len,
            padding="post",
            truncating="post",
        )
        probs = model.predict(X, batch_size=256, verbose=0)
        assert probs.ndim == 2 and probs.shape[1] == 3, "Expected shape (n, 3) for class probabilities"
        if temperature is not None:
            probs = apply_temperature(probs, temperature)
        preds = probs.argmax(axis=1)

        counts = np.bincount(preds, minlength=3)
        total = int(counts.sum())
        neg_count, neu_count, pos_count = [int(c) for c in counts]
        dist_rows.append({
            "domain": domain,
            "total": total,
            "neg_count": neg_count,
            "neu_count": neu_count,
            "pos_count": pos_count,
            "neg_pct": round(neg_count / total, 4),
            "neu_pct": round(neu_count / total, 4),
            "pos_pct": round(pos_count / total, 4),
        })

        # Top confident examples per sentiment
        for label_id, label in LABEL_MAP.items():
            idx = np.where(preds == label_id)[0]
            if idx.size == 0:
                continue
            conf = probs[idx, label_id]
            top_idx = idx[np.argsort(-conf)[: args.top_k]]
            for i in top_idx:
                example_rows.append({
                    "domain": domain,
                    "sentiment": label,
                    "confidence": float(probs[i, label_id]),
                    "comment": comments[i],
                })

    tag = args.output_tag or ""
    dist_df = pd.DataFrame(dist_rows)
    dist_csv = ANALYTICS_DIR / f"youtube_sentiment_distribution{tag}.csv"
    dist_json = ANALYTICS_DIR / f"youtube_sentiment_distribution{tag}.json"
    dist_df.to_csv(dist_csv, index=False, encoding="utf-8")
    dist_json.write_text(json.dumps(dist_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    examples_df = pd.DataFrame(example_rows)
    examples_csv = ANALYTICS_DIR / f"youtube_sentiment_top_examples{tag}.csv"
    examples_df.to_csv(examples_csv, index=False, encoding="utf-8")

    print("Wrote:", dist_csv)
    print("Wrote:", dist_json)
    print("Wrote:", examples_csv)
    print("Note: These are model predictions on unlabeled YouTube data, not ground truth.")


if __name__ == "__main__":
    main()
