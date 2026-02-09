"""Write the required Mermaid GRU diagram to repo root mermaid_gru_diagram.md."""

from __future__ import annotations

from pathlib import Path

DIAGRAM = """flowchart LR
  A[Input Text] --> B[Tokenizer and Padding]
  B --> C[Embedding Layer - Word2Vec or FastText]
  C --> D[GRU Layer]
  D --> E[Dropout]
  E --> F[Dense Layer]
  F --> G[Softmax Output: neg, neu, pos]
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "mermaid_gru_diagram.md"
    content = "# Mermaid GRU Diagram\n\n```mermaid\n" + DIAGRAM + "```\n"
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote mermaid_gru_diagram.md: {out_path}")


if __name__ == "__main__":
    main()
