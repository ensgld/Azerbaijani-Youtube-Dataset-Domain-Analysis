"""Part-1 master labeled dataset preparation (3-class, tri only).
Outputs:
- data/part1/processed/part1_master_labeled.csv
- data/part1/processed/splits/train|val|test.csv
- data/part1/processed/part1_master_summary.json
"""

from __future__ import annotations
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from config import PART1_RAW_DIR, PART1_PROCESSED_DIR, SEED, LABEL_MAP
from utils.text_normalize import normalize_text

LEAK_RE = re.compile(r"_(NEG|POS|NEU)\b", re.IGNORECASE)
LABEL_VALUE_TO_ID = {
    LABEL_MAP.neg_value: LABEL_MAP.neg_id,
    LABEL_MAP.neu_value: LABEL_MAP.neu_id,
    LABEL_MAP.pos_value: LABEL_MAP.pos_id,
}
TRI_FILES = [
    "labeled-sentiment.xlsx",
    "train-00000-of-00001.xlsx",
]

def clean_text(t: str) -> str:
    t = normalize_text(t)
    t = LEAK_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_label_value(v) -> float:
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v = v.strip().replace(",", ".")
    try:
        val = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid label value: {v}") from exc
    return round(val, 3)

def map_label(v: float) -> int:
    if v in LABEL_VALUE_TO_ID:
        return LABEL_VALUE_TO_ID[v]
    raise ValueError(f"Unexpected label value: {v}")

def main():
    rows = []
    for fname in TRI_FILES:
        path = PART1_RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_excel(path)

        if "cleaned_text" not in df.columns or "sentiment_value" not in df.columns:
            raise ValueError(f"Expected columns missing in {fname}: cleaned_text, sentiment_value")

        y = df["sentiment_value"].apply(parse_label_value)

        for text, val in zip(df["cleaned_text"].astype(str), y):
            if val is None:
                continue
            label_id = map_label(val)
            rows.append({
                "text": clean_text(text),
                "label_value": float(val),
                "label_id": label_id,
                "source_id": fname,
            })

    out = pd.DataFrame(rows)
    out = out[out["text"].astype(str).str.strip().ne("")]
    before = len(out)
    out = out.drop_duplicates(subset=["text", "label_id"])
    dedup = before - len(out)

    PART1_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PART1_PROCESSED_DIR / "part1_master_labeled.csv"
    out.to_csv(master_path, index=False, encoding="utf-8")

    train_df, test_df = train_test_split(
        out,
        test_size=0.1,
        random_state=SEED,
        stratify=out["label_id"],
    )
    val_size = 0.1 / 0.9
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=SEED,
        stratify=train_df["label_id"],
    )

    split_dir = PART1_PROCESSED_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(split_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(split_dir / "test.csv", index=False, encoding="utf-8")

    summary = {
        "master_rows": int(len(out)),
        "dedup_removed": int(dedup),
        "label_counts": out["label_id"].value_counts().to_dict(),
        "sources": out["source_id"].value_counts().to_dict(),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_label_counts": {
            "train": train_df["label_id"].value_counts().to_dict(),
            "val": val_df["label_id"].value_counts().to_dict(),
            "test": test_df["label_id"].value_counts().to_dict(),
        },
    }
    with open(PART1_PROCESSED_DIR / "part1_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Wrote:", master_path)
    print("Split sizes:", summary["split_sizes"])
    print("Dedup removed:", dedup)
    print("Label counts:", summary["label_counts"])

if __name__ == "__main__":
    main()
