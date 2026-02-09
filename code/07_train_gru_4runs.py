"""Train 4 required runs: W2V/FT x frozen/tuned using Part-1 labeled data only.
Also supports class-weighted runs and a label-smoothing run for w2v_tuned.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec, FastText
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from config import (
    PART1_PROCESSED_DIR,
    EMBEDDINGS_DIR,
    TOKENIZERS_DIR,
    RUNS_DIR,
    ANALYTICS_DIR,
    MAX_LEN,
    MAX_NUM_WORDS,
    GRU_UNITS,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    NUM_CLASSES,
    SEED,
)

LABEL_MAP = {0: "neg", 1: "neu", 2: "pos"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_splits(train_path: Path, val_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def resolve_label_col(df: pd.DataFrame) -> str:
    for col in ["label_id", "label", "y", "sentiment_label"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found in split")


def build_tokenizer(texts, max_words: int) -> Tokenizer:
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok


def texts_to_padded(tok: Tokenizer, texts, max_len: int) -> np.ndarray:
    seq = tok.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")


def build_embedding_matrix(word_index: Dict[str, int], keyed_vectors, num_words: int) -> Tuple[np.ndarray, float]:
    embedding_dim = keyed_vectors.vector_size
    matrix = np.random.normal(scale=0.02, size=(num_words, embedding_dim)).astype(np.float32)
    matrix[0] = 0.0

    oov = 0
    total = 0
    for word, idx in word_index.items():
        if idx >= num_words:
            continue
        total += 1
        try:
            matrix[idx] = keyed_vectors.get_vector(word)
        except KeyError:
            oov += 1

    oov_rate = oov / max(total, 1)
    return matrix, oov_rate


def build_gru_model(vocab_size: int, embedding_dim: int, max_len: int,
                    embedding_matrix: np.ndarray, embedding_trainable: bool,
                    num_classes: int = NUM_CLASSES, gru_units: int = GRU_UNITS,
                    dropout_rate: float = DROPOUT_RATE) -> Model:
    inp = Input(shape=(max_len,), name="input_ids")
    emb = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=embedding_trainable,
        name="embedding",
    )(inp)
    x = GRU(gru_units, name="gru")(emb)
    x = Dropout(dropout_rate, name="dropout")(x)
    out = Dense(num_classes, activation="softmax", name="out")(x)
    model = Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_pred_distribution(y_pred: np.ndarray, out_path: Path) -> float:
    counts = np.bincount(y_pred, minlength=3)
    total = int(counts.sum())
    rows = []
    for cls_id in range(3):
        rows.append({
            "class_id": cls_id,
            "class_name": LABEL_MAP.get(cls_id, str(cls_id)),
            "count": int(counts[cls_id]),
            "pct": float(counts[cls_id] / max(total, 1)),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    return float(counts[2] / max(total, 1))


def evaluate_and_save(model: Model, X_test: np.ndarray, y_test: np.ndarray, out_dir: Path) -> Dict[str, float]:
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    macro = f1_score(y_test, y_pred, average="macro")

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm).to_csv(out_dir / "confusion_matrix.csv", index=False, encoding="utf-8")

    pred_pos_pct = save_pred_distribution(y_pred, out_dir / "test_pred_distribution.csv")

    return {
        "macro_f1": float(macro),
        "test_macro_f1": float(macro),
        "accuracy": float((y_pred == y_test).mean()),
        "pred_pos_pct": float(pred_pos_pct),
    }


def save_tokenizer(tok: Tokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tok.to_json(), encoding="utf-8")


def load_best_model_path() -> Path | None:
    best = None
    best_score = -1.0
    for metrics_path in RUNS_DIR.glob("*/metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        score = metrics.get("test_macro_f1")
        if score is None:
            score = metrics.get("macro_f1")
        if score is None:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best = metrics_path.parent / "model.keras"
    return best


def build_compare_before_after(diag_dir: Path) -> None:
    rows = []
    base_runs = ["w2v_frozen", "w2v_tuned", "ft_frozen", "ft_tuned"]
    for base in base_runs:
        before_metrics = RUNS_DIR / base / "metrics.json"
        after_metrics = RUNS_DIR / f"{base}_cw" / "metrics.json"
        before_pred = RUNS_DIR / base / "test_pred_distribution.csv"
        after_pred = RUNS_DIR / f"{base}_cw" / "test_pred_distribution.csv"

        macro_before = None
        macro_after = None
        pos_before = None
        pos_after = None

        if before_metrics.exists():
            m = json.loads(before_metrics.read_text(encoding="utf-8"))
            macro_before = m.get("test_macro_f1", m.get("macro_f1"))
        if after_metrics.exists():
            m = json.loads(after_metrics.read_text(encoding="utf-8"))
            macro_after = m.get("test_macro_f1", m.get("macro_f1"))

        if before_pred.exists():
            df = pd.read_csv(before_pred)
            pos = df.loc[df["class_id"] == 2, "pct"]
            pos_before = float(pos.values[0]) if len(pos) else None
        if after_pred.exists():
            df = pd.read_csv(after_pred)
            pos = df.loc[df["class_id"] == 2, "pct"]
            pos_after = float(pos.values[0]) if len(pos) else None

        rows.append({
            "run_name": base,
            "macro_f1_before": macro_before,
            "macro_f1_after_cw": macro_after,
            "pred_pos_pct_before": pos_before,
            "pred_pos_pct_after": pos_after,
        })

    diag_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(diag_dir / "compare_macro_f1_before_after.csv", index=False, encoding="utf-8")


def train_label_smoothing_run(
    w2v_matrix: np.ndarray,
    num_words: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> None:
    run_dir = RUNS_DIR / "w2v_tuned_ls"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_gru_model(
        vocab_size=num_words,
        embedding_dim=w2v_matrix.shape[1],
        max_len=args.max_len,
        embedding_matrix=w2v_matrix,
        embedding_trainable=True,
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing_value)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    y_train_oh = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_oh = to_categorical(y_val, num_classes=NUM_CLASSES)

    history = model.fit(
        X_train,
        y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    model.save(run_dir / "model.keras")

    metrics = evaluate_and_save(model, X_test, y_test, run_dir)
    metrics.update({
        "run": "w2v_tuned_ls",
        "base_run": "w2v_tuned",
        "embedding_trainable": True,
        "class_weighted": False,
        "label_smoothing": float(args.label_smoothing_value),
        "num_words": int(num_words),
        "max_len": int(args.max_len),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
    })
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history.history, ensure_ascii=False, indent=2), encoding="utf-8")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default=str(PART1_PROCESSED_DIR / "splits_with_domain" / "train.csv"))
    ap.add_argument("--val_path", type=str, default=str(PART1_PROCESSED_DIR / "splits_with_domain" / "val.csv"))
    ap.add_argument("--test_path", type=str, default=str(PART1_PROCESSED_DIR / "splits_with_domain" / "test.csv"))
    ap.add_argument("--w2v_model", type=str, default=str(EMBEDDINGS_DIR / "w2v_combined.model"))
    ap.add_argument("--ft_model", type=str, default=str(EMBEDDINGS_DIR / "ft_combined.model"))
    ap.add_argument("--max_words", type=int, default=MAX_NUM_WORDS)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--label_smoothing_only", action="store_true")
    ap.add_argument("--label_smoothing_value", type=float, default=0.05)
    args = ap.parse_args()

    set_seed(args.seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    diag_dir = ANALYTICS_DIR / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(Path(args.train_path), Path(args.val_path), Path(args.test_path))
    label_col = resolve_label_col(train_df)

    # Train label distribution
    label_counts = train_df[label_col].value_counts(dropna=False)
    dist_df = label_counts.reset_index()
    dist_df.columns = ["label", "count"]
    dist_df["pct"] = dist_df["count"] / max(len(train_df), 1)
    dist_path = diag_dir / "train_label_distribution.csv"
    dist_df.to_csv(dist_path, index=False, encoding="utf-8")
    print("Train label distribution written:", dist_path)
    print(dist_df.to_string(index=False))

    tok = build_tokenizer(train_df["text"].astype(str).tolist(), args.max_words)
    tokenizer_path = TOKENIZERS_DIR / "tokenizer.json"
    save_tokenizer(tok, tokenizer_path)

    X_train = texts_to_padded(tok, train_df["text"].astype(str).tolist(), args.max_len)
    X_val = texts_to_padded(tok, val_df["text"].astype(str).tolist(), args.max_len)
    X_test = texts_to_padded(tok, test_df["text"].astype(str).tolist(), args.max_len)

    y_train = train_df[label_col].astype(int).values
    y_val = val_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    num_words = min(args.max_words, len(tok.word_index) + 1)

    w2v = Word2Vec.load(args.w2v_model)
    ft = FastText.load(args.ft_model)

    w2v_matrix, oov_w2v = build_embedding_matrix(tok.word_index, w2v.wv, num_words)
    ft_matrix, oov_ft = build_embedding_matrix(tok.word_index, ft.wv, num_words)

    oov_rows = [
        {"embedding": "word2vec", "oov_rate": oov_w2v},
        {"embedding": "fasttext", "oov_rate": oov_ft},
    ]
    oov_path_csv = ANALYTICS_DIR / "oov_rates.csv"
    oov_path_json = ANALYTICS_DIR / "oov_rates.json"
    pd.DataFrame(oov_rows).to_csv(oov_path_csv, index=False, encoding="utf-8")
    oov_path_json.write_text(json.dumps(oov_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.label_smoothing_only:
        train_label_smoothing_run(
            w2v_matrix,
            num_words,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            args,
        )
        print("Label smoothing run complete: runs/w2v_tuned_ls")
        return

    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Class weights:", class_weight)

    runs = [
        ("w2v_frozen", w2v_matrix, False, oov_w2v),
        ("w2v_tuned", w2v_matrix, True, oov_w2v),
        ("ft_frozen", ft_matrix, False, oov_ft),
        ("ft_tuned", ft_matrix, True, oov_ft),
    ]

    for run_name, emb_matrix, trainable, oov_rate in runs:
        for use_cw in [False, True]:
            suffix = "_cw" if use_cw else ""
            final_name = f"{run_name}{suffix}"
            run_dir = RUNS_DIR / final_name
            run_dir.mkdir(parents=True, exist_ok=True)

            model = build_gru_model(
                vocab_size=num_words,
                embedding_dim=emb_matrix.shape[1],
                max_len=args.max_len,
                embedding_matrix=emb_matrix,
                embedding_trainable=trainable,
            )
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                class_weight=class_weight if use_cw else None,
            )
            model.save(run_dir / "model.keras")

            metrics = evaluate_and_save(model, X_test, y_test, run_dir)
            metrics.update({
                "run": final_name,
                "base_run": run_name,
                "embedding_trainable": bool(trainable),
                "class_weighted": bool(use_cw),
                "class_weight": class_weight if use_cw else None,
                "oov_rate": float(oov_rate),
                "num_words": int(num_words),
                "max_len": int(args.max_len),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
            })
            (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            (run_dir / "history.json").write_text(json.dumps(history.history, ensure_ascii=False, indent=2), encoding="utf-8")

    # Select best model
    best_model = load_best_model_path()
    if best_model is None:
        raise SystemExit("No metrics.json found to select best model")
    best_path_txt = ANALYTICS_DIR / "best_model_path.txt"
    best_path_txt.write_text(str(best_model), encoding="utf-8")
    print("Best model:", best_model)

    # Run ExpA and YouTube distribution with best model
    test_path = PART1_PROCESSED_DIR / "splits_with_domain" / "test.csv"
    v2_test_path = PART1_PROCESSED_DIR / "splits_with_domain_v2" / "test.csv"
    if v2_test_path.exists():
        test_path = v2_test_path

    if test_path.exists():
        subprocess.run(
            [
                sys.executable,
                "code/08_evaluate_domainwise_A.py",
                "--model_path",
                str(best_model),
                "--test_path",
                str(test_path),
            ],
            check=True,
        )
    else:
        print("Warning: domain-aware test split not found, skipping ExpA.")

    before_dist = ANALYTICS_DIR / "youtube_sentiment_distribution.csv"
    before_df = None
    if before_dist.exists():
        before_df = pd.read_csv(before_dist)

    subprocess.run(
        [
            sys.executable,
            "code/10_predict_youtube_domain_distributions.py",
            "--model_path",
            str(best_model),
        ],
        check=True,
    )

    if before_df is not None:
        after_df = pd.read_csv(ANALYTICS_DIR / "youtube_sentiment_distribution.csv")
        merged = before_df.merge(after_df, on="domain", suffixes=("_before", "_after"))
        diff_rows = []
        for _, row in merged.iterrows():
            for col in ["neg_pct", "neu_pct", "pos_pct", "neg_count", "neu_count", "pos_count", "total"]:
                b = row.get(f"{col}_before")
                a = row.get(f"{col}_after")
                if pd.isna(b) or pd.isna(a):
                    continue
                diff_rows.append({
                    "domain": row["domain"],
                    "metric": col,
                    "before": b,
                    "after": a,
                    "delta": float(a) - float(b),
                })
        pd.DataFrame(diff_rows).to_csv(diag_dir / "youtube_dist_before_after_best.csv", index=False, encoding="utf-8")
    else:
        (diag_dir / "youtube_dist_before_after_best.csv").write_text(
            "domain,metric,before,after,delta\n",
            encoding="utf-8",
        )

    # Compare before/after for class weights
    build_compare_before_after(diag_dir)

    print("Tokenizer:", tokenizer_path)
    print("OOV rates:", oov_path_csv, oov_path_json)
    print("Runs saved in:", RUNS_DIR)


if __name__ == "__main__":
    main()
