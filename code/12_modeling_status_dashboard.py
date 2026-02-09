"""Audit modeling artifacts and produce a status dashboard (CSV/JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import (
    EMBEDDINGS_DIR,
    YOUTUBE_CORPORA_DIR,
    RUNS_DIR,
    ANALYTICS_DIR,
    DATA_DIR,
)


def file_info(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "exists": False,
            "size": None,
            "mtime": None,
        }
    st = path.stat()
    return {
        "exists": True,
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def run_artifacts(run_name: str) -> List[Dict[str, object]]:
    base = RUNS_DIR / run_name
    required = [
        "model.keras",
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "history.json",
    ]
    rows = []
    for fname in required:
        path = base / fname
        info = file_info(path)
        rows.append({
            "artifact": f"runs/{run_name}/{fname}",
            **info,
        })
    return rows


def main():
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = []

    # Embeddings & corpus
    artifacts.append({"artifact": "models/embeddings/w2v_combined.model", **file_info(EMBEDDINGS_DIR / "w2v_combined.model")})
    artifacts.append({"artifact": "models/embeddings/ft_combined.model", **file_info(EMBEDDINGS_DIR / "ft_combined.model")})
    artifacts.append({"artifact": "data/youtube/corpora/combined_corpus.txt", **file_info(YOUTUBE_CORPORA_DIR / "combined_corpus.txt")})
    artifacts.append({"artifact": "data/youtube/corpora/combined_corpus_stats.json", **file_info(YOUTUBE_CORPORA_DIR / "combined_corpus_stats.json")})

    # Runs
    for run_name in ["w2v_frozen", "w2v_tuned", "ft_frozen", "ft_tuned"]:
        artifacts.extend(run_artifacts(run_name))

    # Analytics outputs
    analytics_files = [
        "oov_rates.csv",
        "oov_rates.json",
        "expA_domainwise_macro_f1.csv",
        "expA_domainwise_macro_f1.json",
        "expB_lodo_macro_f1.csv",
        "expB_lodo_macro_f1.json",
        "youtube_sentiment_distribution.csv",
        "youtube_sentiment_distribution.json",
        "youtube_sentiment_top_examples.csv",
        "mermaid_gru_diagram.md",
    ]
    for fname in analytics_files:
        artifacts.append({"artifact": f"data/analytics/{fname}", **file_info(ANALYTICS_DIR / fname)})

    df = pd.DataFrame(artifacts)
    out_csv = ANALYTICS_DIR / "modeling_artifacts_status.csv"
    out_json = ANALYTICS_DIR / "modeling_artifacts_status.json"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print(df.head(10))


if __name__ == "__main__":
    main()
