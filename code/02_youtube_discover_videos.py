"""Discover videos per domain using keyword pools + YouTube search.
Outputs:
- data/youtube/video_candidates/videos_raw.csv
"""

from __future__ import annotations
import argparse
import itertools
from typing import Dict, List
import pandas as pd
from dotenv import load_dotenv
from config import DOMAIN_KEYWORDS, YOUTUBE_VIDEO_CANDIDATES_DIR
from utils.youtube_api import get_youtube_client, search_videos, fetch_video_metadata

def build_queries(keywords: List[str]) -> List[str]:
    keywords = [k.strip() for k in keywords if k and k.strip()]
    combos: List[str] = []

    if len(keywords) >= 2:
        combos.append(" ".join(keywords[:2]))
    if len(keywords) >= 3:
        combos.append(" ".join(keywords[:3]))

    for a, b in itertools.combinations(keywords[:5], 2):
        if len(combos) >= 4:
            break
        combos.append(f"{a} {b}")

    singles = keywords[:6]
    queries = combos + singles
    deduped: Dict[str, None] = {}
    for q in queries:
        deduped.setdefault(q, None)
    return list(deduped.keys())

def normalize_tags(tags) -> str:
    if not tags:
        return ""
    if isinstance(tags, list):
        return "|".join(tags)
    return str(tags)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_videos_per_domain", type=int, default=60)
    ap.add_argument("--per_query", type=int, default=20)
    args = ap.parse_args()

    load_dotenv()
    yt = get_youtube_client()

    rows = []
    seen_global = set()
    for domain, kws in DOMAIN_KEYWORDS.items():
        queries = build_queries(kws)
        domain_ids: List[str] = []
        id_to_query: Dict[str, str] = {}
        for query in queries:
            if len(domain_ids) >= args.max_videos_per_domain:
                break
            results = search_videos(yt, query, max_results=args.per_query)
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

    YOUTUBE_VIDEO_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = YOUTUBE_VIDEO_CANDIDATES_DIR / "videos_raw.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("Wrote:", out_path, "rows:", len(df))

if __name__ == "__main__":
    main()
