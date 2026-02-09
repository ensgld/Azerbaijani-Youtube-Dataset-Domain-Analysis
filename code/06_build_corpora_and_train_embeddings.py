"""Build combined corpus and train Word2Vec/FastText embeddings.
Outputs (default):
- data/youtube/corpora/combined_corpus.txt
- models/embeddings/w2v_combined.model
- models/embeddings/ft_combined.model
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List

from gensim.models import Word2Vec, FastText

from config import (
    PART1_RAW_DIR,
    YOUTUBE_COMMENTS_FILTERED_DIR,
    YOUTUBE_CORPORA_DIR,
    EMBEDDINGS_DIR,
)
from utils.text_normalize import normalize_text
from utils.io import load_jsonl

TOKEN_RE = re.compile(r"[a-zəöüğışç]+", re.IGNORECASE)

class CorpusIterator:
    def __init__(self, path: Path):
        self.path = path

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = TOKEN_RE.findall(line.lower())
                if tokens:
                    yield tokens


def iter_part1_corpus(corpus_path: Path) -> Iterable[str]:
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if " " in line:
                _, text = line.split(" ", 1)
            else:
                text = line
            text = normalize_text(text)
            if text:
                yield text


def iter_youtube_comments(filtered_dir: Path) -> Iterable[str]:
    for jsonl_path in filtered_dir.rglob("*.jsonl"):
        for record in load_jsonl(jsonl_path):
            text = record.get("comment", "")
            text = normalize_text(text)
            if text:
                yield text


def write_combined_corpus(output_path: Path, part1_lines: Iterable[str], yt_lines: Iterable[str]) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part1_count = 0
    yt_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for line in part1_lines:
            f.write(line + "\n")
            part1_count += 1
        for line in yt_lines:
            f.write(line + "\n")
            yt_count += 1
    return {
        "part1_lines": part1_count,
        "youtube_lines": yt_count,
        "combined_lines": part1_count + yt_count,
    }


def train_w2v(corpus_path: Path, out_path: Path, vector_size: int, window: int, min_count: int,
              epochs: int, workers: int, sg: int, seed: int) -> None:
    sentences = CorpusIterator(corpus_path)
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
    )
    model.build_vocab(sentences)
    model.train(CorpusIterator(corpus_path), total_examples=model.corpus_count, epochs=epochs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    model.wv.save(str(out_path.with_suffix(".kv")))


def train_ft(corpus_path: Path, out_path: Path, vector_size: int, window: int, min_count: int,
             epochs: int, workers: int, sg: int, seed: int) -> None:
    sentences = CorpusIterator(corpus_path)
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
    )
    model.build_vocab(sentences)
    model.train(CorpusIterator(corpus_path), total_examples=model.corpus_count, epochs=epochs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    model.wv.save(str(out_path.with_suffix(".kv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1_corpus", type=str, default=str(PART1_RAW_DIR / "corpus_all.txt"))
    ap.add_argument("--youtube_filtered_dir", type=str, default=str(YOUTUBE_COMMENTS_FILTERED_DIR))
    ap.add_argument("--output_corpus", type=str, default=str(YOUTUBE_CORPORA_DIR / "combined_corpus.txt"))
    ap.add_argument("--include_youtube", action="store_true", default=True)
    ap.add_argument("--no_youtube", action="store_false", dest="include_youtube")
    ap.add_argument("--train_part1_only", action="store_true")
    ap.add_argument("--vector_size", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sg", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    part1_path = Path(args.part1_corpus)
    yt_dir = Path(args.youtube_filtered_dir)
    out_corpus = Path(args.output_corpus)

    part1_iter = iter_part1_corpus(part1_path)
    yt_iter: Iterable[str] = []
    if args.include_youtube:
        yt_iter = iter_youtube_comments(yt_dir)

    counts = write_combined_corpus(out_corpus, part1_iter, yt_iter)
    stats = dict(counts)
    stats["output_corpus"] = str(out_corpus)
    stats_path = YOUTUBE_CORPORA_DIR / "combined_corpus_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Train combined embeddings
    w2v_path = EMBEDDINGS_DIR / "w2v_combined.model"
    ft_path = EMBEDDINGS_DIR / "ft_combined.model"
    train_w2v(out_corpus, w2v_path, args.vector_size, args.window, args.min_count,
              args.epochs, args.workers, args.sg, args.seed)
    train_ft(out_corpus, ft_path, args.vector_size, args.window, args.min_count,
             args.epochs, args.workers, args.sg, args.seed)

    # Optional: part1-only embeddings
    if args.train_part1_only:
        part1_only_path = YOUTUBE_CORPORA_DIR / "part1_only_corpus.txt"
        write_combined_corpus(part1_only_path, iter_part1_corpus(part1_path), [])
        w2v_p1 = EMBEDDINGS_DIR / "w2v_part1.model"
        ft_p1 = EMBEDDINGS_DIR / "ft_part1.model"
        train_w2v(part1_only_path, w2v_p1, args.vector_size, args.window, args.min_count,
                  args.epochs, args.workers, args.sg, args.seed)
        train_ft(part1_only_path, ft_p1, args.vector_size, args.window, args.min_count,
                 args.epochs, args.workers, args.sg, args.seed)

    print("Wrote corpus:", out_corpus)
    print("Wrote stats:", stats_path)
    print("Saved embeddings:", w2v_path, ft_path)


if __name__ == "__main__":
    main()
