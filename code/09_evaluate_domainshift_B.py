"""Experiment B: Leave-one-domain-out (LODO) evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout
from tensorflow.keras.models import Model

from config import (
    PART1_PROCESSED_DIR,
    EMBEDDINGS_DIR,
    ANALYTICS_DIR,
    DOMAINS,
    MAX_LEN,
    MAX_NUM_WORDS,
    GRU_UNITS,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    NUM_CLASSES,
    SEED,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_tokenizer(texts, max_words: int) -> Tokenizer:
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok


def texts_to_padded(tok: Tokenizer, texts, max_len: int) -> np.ndarray:
    seq = tok.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")


def build_embedding_matrix(word_index, keyed_vectors, num_words: int):
    embedding_dim = keyed_vectors.vector_size
    matrix = np.random.normal(scale=0.02, size=(num_words, embedding_dim)).astype(np.float32)
    matrix[0] = 0.0

    for word, idx in word_index.items():
        if idx >= num_words:
            continue
        try:
            matrix[idx] = keyed_vectors.get_vector(word)
        except KeyError:
            pass
    return matrix


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", choices=["w2v", "ft"], default="ft")
    ap.add_argument("--trainable", action="store_true")
    ap.add_argument("--max_words", type=int, default=MAX_NUM_WORDS)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--master_path", type=str, default=str(PART1_PROCESSED_DIR / "part1_master_labeled_final.csv"))
    args = ap.parse_args()

    set_seed(args.seed)
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master_path)
    df = df[df["domain_5"].isin(DOMAINS)].copy()

    if args.embedding == "w2v":
        emb_model = Word2Vec.load(str(EMBEDDINGS_DIR / "w2v_combined.model"))
    else:
        emb_model = FastText.load(str(EMBEDDINGS_DIR / "ft_combined.model"))

    results = []
    for domain in DOMAINS:
        holdout = df[df["domain_5"] == domain]
        if holdout.empty:
            results.append({"held_out_domain": domain, "macro_f1": None, "n_test": 0})
            continue

        train_df = df[df["domain_5"] != domain]
        X_train, X_val, y_train, y_val = train_test_split(
            train_df["text"].astype(str).tolist(),
            train_df["label_id"].astype(int).values,
            test_size=0.1,
            random_state=args.seed,
            stratify=train_df["label_id"].astype(int).values,
        )

        tok = build_tokenizer(X_train, args.max_words)
        num_words = min(args.max_words, len(tok.word_index) + 1)
        emb_matrix = build_embedding_matrix(tok.word_index, emb_model.wv, num_words)

        X_train_pad = texts_to_padded(tok, X_train, args.max_len)
        X_val_pad = texts_to_padded(tok, X_val, args.max_len)
        X_test_pad = texts_to_padded(tok, holdout["text"].astype(str).tolist(), args.max_len)
        y_test = holdout["label_id"].astype(int).values

        model = build_gru_model(
            vocab_size=num_words,
            embedding_dim=emb_matrix.shape[1],
            max_len=args.max_len,
            embedding_matrix=emb_matrix,
            embedding_trainable=args.trainable,
        )
        model.fit(
            X_train_pad,
            y_train,
            validation_data=(X_val_pad, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )

        y_prob = model.predict(X_test_pad, batch_size=args.batch_size, verbose=0)
        y_pred = y_prob.argmax(axis=1)
        macro = f1_score(y_test, y_pred, average="macro")

        results.append({
            "held_out_domain": domain,
            "macro_f1": float(macro),
            "n_test": int(len(holdout)),
        })

    out_csv = ANALYTICS_DIR / "expB_lodo_macro_f1.csv"
    out_json = ANALYTICS_DIR / "expB_lodo_macro_f1.json"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()
