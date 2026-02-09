"""Azerbaijani-only filter (two-layer) as required by the assignment."""

import re
import unicodedata
from .text_normalize import normalize_text
from config import AZ_STRONG_CHARS, AZ_MARKERS, TR_MARKERS

TOKEN_RE = re.compile(r"[a-zəöüğışç]+", re.IGNORECASE)
LATIN_RE = re.compile(r"[a-zəöüğışç]", re.IGNORECASE)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")

def non_latin_letter_count(text: str) -> int:
    count = 0
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            continue
        count += 1
    return count

def analyze_text(text: str) -> dict:
    t = normalize_text(text)

    # Cyrillic detection for hard rejection
    cyrillic_count = len(CYRILLIC_RE.findall(t))
    latin_count = len(LATIN_RE.findall(t))
    total_letters = latin_count + cyrillic_count
    cyrillic_ratio = (cyrillic_count / total_letters) if total_letters > 0 else 0.0
    non_latin_count = non_latin_letter_count(t)
    non_latin_ratio = (non_latin_count / total_letters) if total_letters > 0 else 0.0

    # Layer A: character signals
    char_score = 0.0
    if any(ch in AZ_STRONG_CHARS for ch in t):
        char_score += 3.0

    qx_count = t.count("q") + t.count("x")
    if qx_count >= 2:
        char_score += 1.0

    # Layer B: marker lexicon signals
    tokens = set(TOKEN_RE.findall(t))
    az_hits = len(tokens & AZ_MARKERS)
    tr_hits = len(tokens & TR_MARKERS)

    lex_score = (az_hits * 1.0) - (tr_hits * 1.5)
    score = char_score + lex_score

    return {
        "score": score,
        "char_score": char_score,
        "lex_score": lex_score,
        "az_hits": az_hits,
        "tr_hits": tr_hits,
        "cyrillic_count": cyrillic_count,
        "cyrillic_ratio": cyrillic_ratio,
        "non_latin_count": non_latin_count,
        "non_latin_ratio": non_latin_ratio,
        "latin_count": latin_count,
        "total_letters": total_letters,
    }

def az_likelihood_score(text: str) -> float:
    return analyze_text(text)["score"]

def is_azerbaijani(text: str, threshold: float, return_debug: bool = False):
    stats = analyze_text(text)
    cyrillic_reject = (
        stats["cyrillic_count"] >= 3
        or stats["cyrillic_ratio"] >= 0.02
    )
    tr_reject = stats["tr_hits"] > 0
    non_latin_reject = stats["non_latin_count"] > 0
    passed = (
        (not cyrillic_reject)
        and (not tr_reject)
        and (not non_latin_reject)
        and (stats["score"] >= threshold)
    )
    if return_debug:
        stats = dict(stats)
        stats["cyrillic_reject"] = cyrillic_reject
        stats["tr_reject"] = tr_reject
        stats["non_latin_reject"] = non_latin_reject
        stats["threshold"] = threshold
        stats["passed"] = passed
        return passed, stats
    return passed
