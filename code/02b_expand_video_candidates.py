"""Expand video candidates per domain using YouTube search + metadata.
Outputs:
- data/youtube/video_candidates/videos_raw_expanded.csv
"""

from __future__ import annotations
import argparse
import itertools
from pathlib import Path
from typing import Dict, List
import pandas as pd
from dotenv import load_dotenv

from config import (
    DOMAIN_KEYWORDS,
    DOMAIN_LOCAL_TERMS,
    AZ_QUERY_BOOST_TERMS,
    YOUTUBE_VIDEO_CANDIDATES_DIR,
    RETAIL_BIAS_TERMS,
    FINANCE_BIAS_TERMS,
    DOMAINS,
)
from utils.youtube_api import get_youtube_client, search_videos, fetch_video_metadata

def build_queries(
    keywords: List[str],
    local_terms: List[str],
    boost_terms: List[str],
    bias_terms: List[str],
    num_queries: int,
    max_pairs: int = 10,
) -> List[str]:
    keywords = [k.strip() for k in keywords if k and k.strip()]
    local_terms = [t.strip() for t in local_terms if t and t.strip()]
    boost_terms = [t.strip() for t in boost_terms if t and t.strip()]
    queries: List[str] = []

    if len(keywords) >= 2 and boost_terms:
        kw1, kw2 = keywords[0], keywords[1]
        for boost in boost_terms:
            queries.append(f"{kw1} {kw2} {boost}")

    for local in local_terms[:8]:
        for boost in boost_terms[:4]:
            queries.append(f"{local} {boost}")

    for local in local_terms[:8]:
        for kw in keywords[:6]:
            queries.append(f"{local} {kw} azərbaycan")

    for local in local_terms[:8]:
        for kw in keywords[:6]:
            queries.append(f"{local} {kw} azərbaycanca")

    for local in local_terms[:8]:
        for kw in keywords[:6]:
            queries.append(f"{local} {kw}")

    for kw in keywords[:8]:
        queries.append(f"{kw} azərbaycan")

    for kw in keywords[:6]:
        queries.append(f"{kw} bakı")

    for bias in bias_terms[:6]:
        for kw in keywords[:4]:
            queries.append(f"{bias} {kw} azərbaycan")

    for k in keywords:
        queries.append(k)

    for a, b in itertools.combinations(keywords[:10], 2):
        if len(queries) >= len(keywords) + max_pairs:
            break
        queries.append(f"{a} {b}")

    if len(keywords) >= 3:
        queries.append(" ".join(keywords[:3]))

    deduped: Dict[str, None] = {}
    for q in queries:
        deduped.setdefault(q, None)
    return list(deduped.keys())[:num_queries]

def normalize_tags(tags) -> str:
    if not tags:
        return ""
    if isinstance(tags, list):
        return "|".join(tags)
    return str(tags)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_videos_per_domain", type=int, default=800)
    ap.add_argument("--per_query", type=int, default=25)
    ap.add_argument("--num_queries_per_domain", type=int, default=10)
    ap.add_argument("--domains", type=str, default=None)
    ap.add_argument("--output", type=str, default=str(YOUTUBE_VIDEO_CANDIDATES_DIR / "videos_raw_expanded.csv"))
    ap.add_argument("--region_code", type=str, default="AZ")
    ap.add_argument("--relevance_language", type=str, default="az")
    args = ap.parse_args()

    load_dotenv()
    yt = get_youtube_client()

    rows = []
    seen_global = set()
    if args.domains:
        domain_list = [d.strip() for d in args.domains.split(",") if d.strip()]
        invalid = [d for d in domain_list if d not in DOMAINS]
        if invalid:
            raise ValueError(f"Invalid domain(s): {invalid}. Must be one of: {DOMAINS}")
    else:
        domain_list = DOMAINS

    for domain in domain_list:
        kws = DOMAIN_KEYWORDS.get(domain, [])
        local_terms = DOMAIN_LOCAL_TERMS.get(domain, [])
        if domain == "Retail & Lifestyle":
            bias_terms = RETAIL_BIAS_TERMS
        elif domain == "Finance & Business":
            bias_terms = FINANCE_BIAS_TERMS
        else:
            bias_terms = []
        queries = build_queries(kws, local_terms, AZ_QUERY_BOOST_TERMS, bias_terms, args.num_queries_per_domain)
        domain_ids: List[str] = []
        id_to_query: Dict[str, str] = {}
        for query in queries:
            if len(domain_ids) >= args.max_videos_per_domain:
                break
            results = search_videos(
                yt,
                query,
                max_results=args.per_query,
                region_code=args.region_code,
                relevance_language=args.relevance_language,
            )
            for item in results:
                vid = item.get("video_id")
                if not vid:
                    continue
                if vid in seen_global:
                    continue
                seen_global.add(vid)
                domain_ids.append(vid)
                id_to_query[vid] = query
                if len(domain_ids) >= args.max_videos_per_domain:
                    break

        metadata = fetch_video_metadata(yt, domain_ids)
        for vid in domain_ids:
            meta = metadata.get(vid)
            if not meta:
                continue
            rows.append({
                "seed_domain": domain,
                "query": id_to_query.get(vid, ""),
                "video_id": meta.get("video_id", ""),
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "tags": normalize_tags(meta.get("tags")),
                "channelTitle": meta.get("channelTitle", ""),
                "publishedAt": meta.get("publishedAt", ""),
                "categoryId": meta.get("categoryId", ""),
                "defaultLanguage": meta.get("defaultLanguage", ""),
                "defaultAudioLanguage": meta.get("defaultAudioLanguage", ""),
                "viewCount": meta.get("viewCount", 0),
                "likeCount": meta.get("likeCount", 0),
                "commentCount": meta.get("commentCount", 0),
            })

        print(f"{domain}: {len(domain_ids)} videos (queries: {len(queries)})")

    df = pd.DataFrame(rows)
    columns = [
        "seed_domain",
        "query",
        "video_id",
        "title",
        "description",
        "tags",
        "channelTitle",
        "publishedAt",
        "categoryId",
        "defaultLanguage",
        "defaultAudioLanguage",
        "viewCount",
        "likeCount",
        "commentCount",
    ]
    df = df.reindex(columns=columns)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote:", out_path, "rows:", len(df))

if __name__ == "__main__":
    main()
