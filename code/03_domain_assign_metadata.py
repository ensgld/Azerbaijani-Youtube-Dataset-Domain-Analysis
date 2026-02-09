"""Assign domain based on metadata (title/description/tags), not comments.
Outputs:
- data/youtube/videos_labeled/videos_with_domain.csv
- data/analytics/youtube_domain_assignment_counts.csv
- data/analytics/youtube_domain_assignment_crosstab.csv
- data/analytics/youtube_domain_assignment_samples.csv
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import config

from config import (
    DOMAIN_KEYWORDS,
    DOMAINS,
    YOUTUBE_VIDEO_CANDIDATES_DIR,
    YOUTUBE_VIDEOS_LABELED_DIR,
    ANALYTICS_DIR,
    AZ_STRONG_CHARS,
    AZ_MARKERS,
    TR_MARKERS,
    AZ_FILTER_DEFAULT_THRESHOLD,
)

TOKEN_RE = re.compile(r"[a-zəöüğışç]+", re.IGNORECASE)

def normalize_text(t: str) -> str:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    t = str(t).lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def safe_text(t: str) -> str:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    return str(t)

def build_keyword_patterns(domain_keywords: Dict[str, List[str]]) -> Dict[str, Dict[str, re.Pattern]]:
    patterns: Dict[str, Dict[str, re.Pattern]] = {}
    for domain, kws in domain_keywords.items():
        patterns[domain] = {}
        for kw in kws:
            kw_norm = kw.lower().strip()
            if not kw_norm:
                continue
            if re.search(r"\w", kw_norm):
                pat = re.compile(rf"\b{re.escape(kw_norm)}\b", re.IGNORECASE)
            else:
                pat = re.compile(re.escape(kw_norm), re.IGNORECASE)
            patterns[domain][kw_norm] = pat
    return patterns

def score_text_fields(
    title: str,
    description: str,
    tags_text: str,
    domain: str,
    patterns: Dict[str, Dict[str, re.Pattern]],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    score = 0.0
    kw_scores: Dict[str, float] = {}

    title_norm = normalize_text(title)
    desc_norm = normalize_text(description)
    if tags_text is None or (isinstance(tags_text, float) and pd.isna(tags_text)):
        tags_raw = ""
    else:
        tags_raw = str(tags_text)
    tags_norm = normalize_text(tags_raw.replace("|", " "))

    for kw, pat in patterns.get(domain, {}).items():
        hit_title = bool(pat.search(title_norm))
        hit_tags = bool(pat.search(tags_norm))
        hit_desc = bool(pat.search(desc_norm))
        kw_score = 0.0
        if hit_title:
            kw_score += weights["title"]
        if hit_tags:
            kw_score += weights["tags"]
        if hit_desc:
            kw_score += weights["description"]
        if kw_score > 0:
            kw_scores[kw] = kw_score
            score += kw_score

    return score, kw_scores

def pick_domain(scores: Dict[str, float], fallback: str) -> str:
    if not scores:
        return fallback
    best_domain = None
    best_score = None
    for d in DOMAINS:
        s = scores.get(d, 0.0)
        if best_score is None or s > best_score:
            best_score = s
            best_domain = d
    if best_score == 0:
        return fallback
    return best_domain or fallback

def az_meta_score(text: str) -> float:
    t = normalize_text(text)
    char_score = 0.0
    if any(ch in t for ch in AZ_STRONG_CHARS):
        char_score += 3.0
    qx_count = t.count("q") + t.count("x")
    if qx_count >= 2:
        char_score += 1.0
    tokens = set(TOKEN_RE.findall(t))
    az_hits = len(tokens & AZ_MARKERS)
    tr_hits = len(tokens & TR_MARKERS)
    lex_score = (az_hits * 1.0) - (tr_hits * 1.5)
    return char_score + lex_score

def pick_domain_with_tie_break(
    scores: Dict[str, float],
    seed_domain: str,
    az_score: float,
    az_threshold: float,
) -> str:
    if not scores:
        return seed_domain
    max_score = max(scores.values())
    if max_score == 0:
        return seed_domain
    top = [d for d, s in scores.items() if s == max_score]
    if len(top) == 1:
        return top[0]
    if az_score >= az_threshold:
        if seed_domain in top:
            return seed_domain
        return top[0]
    return seed_domain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=str(YOUTUBE_VIDEO_CANDIDATES_DIR / "videos_raw.csv"))
    ap.add_argument("--output", type=str, default=str(YOUTUBE_VIDEOS_LABELED_DIR / "videos_with_domain.csv"))
    ap.add_argument("--sample_n", type=int, default=20)
    args = ap.parse_args()

    weights = getattr(config, "DOMAIN_KEYWORD_WEIGHTS", {
        "title": config.TITLE_WEIGHT,
        "tags": config.TAGS_WEIGHT,
        "description": config.DESC_WEIGHT,
    })

    in_path = Path(args.input)
    df = pd.read_csv(in_path)
    patterns = build_keyword_patterns(DOMAIN_KEYWORDS)

    score_cols = [f"score_{d}" for d in DOMAINS]
    for col in score_cols:
        df[col] = 0.0

    assigned = []
    matched_keywords = []

    for _, row in df.iterrows():
        title = row.get("title", "")
        description = row.get("description", "")
        tags_text = row.get("tags", "")
        seed_domain = row.get("seed_domain", "")

        domain_scores: Dict[str, float] = {}
        domain_kw_scores: Dict[str, Dict[str, float]] = {}
        for domain in DOMAINS:
            s, kw_scores = score_text_fields(
                title,
                description,
                tags_text,
                domain,
                patterns,
                weights,
            )
            domain_scores[domain] = s
            domain_kw_scores[domain] = kw_scores

        combined_meta = f"{safe_text(title)} {safe_text(description)} {safe_text(tags_text)}"
        az_score = az_meta_score(combined_meta)
        az_flag = az_score >= AZ_FILTER_DEFAULT_THRESHOLD
        assigned_domain = pick_domain_with_tie_break(
            domain_scores,
            seed_domain=seed_domain,
            az_score=az_score,
            az_threshold=AZ_FILTER_DEFAULT_THRESHOLD,
        )
        assigned.append(assigned_domain)

        assigned_kw = domain_kw_scores.get(assigned_domain, {})
        top_kw = sorted(assigned_kw.items(), key=lambda x: (-x[1], x[0]))[:3]
        matched_keywords.append(", ".join([k for k, _ in top_kw]))

        for domain in DOMAINS:
            df.at[row.name, f"score_{domain}"] = domain_scores.get(domain, 0.0)

        df.at[row.name, "az_meta_score"] = round(az_score, 3)
        df.at[row.name, "az_meta_flag"] = az_flag

    df["domain_assigned"] = assigned

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    counts = df["domain_assigned"].value_counts().reindex(DOMAINS, fill_value=0)
    print("Counts by domain_assigned:")
    print(counts.to_string())

    ctab = pd.crosstab(df["seed_domain"], df["domain_assigned"]).reindex(
        index=DOMAINS, columns=DOMAINS, fill_value=0
    )
    print("Seed vs assigned crosstab:")
    print(ctab.to_string())

    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    counts.reset_index().rename(columns={"index": "domain_assigned", "domain_assigned": "count"}).to_csv(
        ANALYTICS_DIR / "youtube_domain_assignment_counts.csv",
        index=False,
        encoding="utf-8",
    )
    ctab.to_csv(ANALYTICS_DIR / "youtube_domain_assignment_crosstab.csv", encoding="utf-8")

    sample_cols = ["seed_domain", "domain_assigned", "title", "channelTitle"]
    sample = df.copy()
    sample["matched_keywords_top3"] = matched_keywords
    sample = sample[sample_cols + ["matched_keywords_top3"]]
    sample = sample.sample(n=min(args.sample_n, len(sample)), random_state=42)
    sample.to_csv(ANALYTICS_DIR / "youtube_domain_assignment_samples.csv", index=False, encoding="utf-8")

    print("Wrote:", out_path)
    print("Evidence files:", ANALYTICS_DIR / "youtube_domain_assignment_counts.csv")
    print("Evidence files:", ANALYTICS_DIR / "youtube_domain_assignment_crosstab.csv")
    print("Evidence files:", ANALYTICS_DIR / "youtube_domain_assignment_samples.csv")

if __name__ == "__main__":
    main()
