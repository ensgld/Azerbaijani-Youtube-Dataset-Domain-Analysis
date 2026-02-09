"""Rule-based domain scoring over video metadata (title/desc/tags)."""

from __future__ import annotations
from typing import Dict, List, Tuple
from config import DOMAIN_KEYWORDS, TITLE_WEIGHT, TAGS_WEIGHT, DESC_WEIGHT, DOMAINS

def _count_hits(text: str, keywords: List[str]) -> int:
    text = (text or "").lower()
    return sum(1 for kw in keywords if kw.lower() in text)

def score_video(title: str, description: str, tags: List[str] | None) -> Dict[str, float]:
    tags_text = " ".join(tags or [])
    scores: Dict[str, float] = {}
    for d in DOMAINS:
        kws = DOMAIN_KEYWORDS.get(d, [])
        s = 0.0
        s += TITLE_WEIGHT * _count_hits(title, kws)
        s += DESC_WEIGHT * _count_hits(description, kws)
        s += TAGS_WEIGHT * _count_hits(tags_text, kws)
        scores[d] = s
    return scores

def assign_domain(title: str, description: str, tags: List[str] | None) -> Tuple[str, Dict[str, float]]:
    scores = score_video(title, description, tags)
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best, scores
