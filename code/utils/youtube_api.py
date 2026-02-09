"""YouTube Data API helpers (search + videos.list)."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Iterable, List
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json as _json

def get_youtube_client(api_key: str | None = None):
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv()
    load_dotenv(repo_root / ".env")
    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY. Set it in .env or environment.")
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)

def search_videos(
    yt,
    query: str,
    max_results: int,
    region_code: str = "AZ",
    relevance_language: str = "az",
) -> List[Dict[str, str]]:
    max_results = max(1, min(int(max_results), 50))
    req = yt.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        regionCode=region_code,
        relevanceLanguage=relevance_language,
    )
    try:
        resp = req.execute()
    except HttpError as exc:
        raise RuntimeError(f"YouTube search failed for query='{query}': {exc}") from exc

    items = resp.get("items", [])
    results: List[Dict[str, str]] = []
    for item in items:
        vid = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not vid:
            continue
        results.append({
            "video_id": vid,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "channelTitle": snippet.get("channelTitle", ""),
            "publishedAt": snippet.get("publishedAt", ""),
        })
    return results

def _chunk_ids(video_ids: Iterable[str], size: int = 50) -> List[List[str]]:
    chunk: List[str] = []
    chunks: List[List[str]] = []
    for vid in video_ids:
        chunk.append(vid)
        if len(chunk) >= size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks

def fetch_video_metadata(yt, video_ids: List[str]) -> Dict[str, Dict[str, object]]:
    if not video_ids:
        return {}
    results: Dict[str, Dict[str, object]] = {}
    for batch in _chunk_ids(video_ids, size=50):
        req = yt.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
            maxResults=len(batch),
        )
        try:
            resp = req.execute()
        except HttpError as exc:
            raise RuntimeError(f"YouTube videos.list failed: {exc}") from exc

        for item in resp.get("items", []):
            vid = item.get("id", "")
            snippet = item.get("snippet", {}) or {}
            stats = item.get("statistics", {}) or {}
            results[vid] = {
                "video_id": vid,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "tags": snippet.get("tags", []) or [],
                "channelTitle": snippet.get("channelTitle", ""),
                "publishedAt": snippet.get("publishedAt", ""),
                "categoryId": snippet.get("categoryId", ""),
                "defaultAudioLanguage": snippet.get("defaultAudioLanguage", ""),
                "defaultLanguage": snippet.get("defaultLanguage", ""),
                "viewCount": int(stats.get("viewCount", 0)) if stats.get("viewCount") is not None else 0,
                "likeCount": int(stats.get("likeCount", 0)) if stats.get("likeCount") is not None else 0,
                "commentCount": int(stats.get("commentCount", 0)) if stats.get("commentCount") is not None else 0,
            }
    return results

def _parse_http_error(exc: HttpError) -> dict:
    status = getattr(exc, "status_code", None)
    if status is None and hasattr(exc, "resp"):
        status = exc.resp.status
    reason = "httpError"
    message = str(exc)
    try:
        content = exc.content.decode("utf-8") if hasattr(exc, "content") else ""
        payload = _json.loads(content) if content else {}
        err = payload.get("error", {})
        if err.get("message"):
            message = err["message"]
        errors = err.get("errors") or []
        if errors and errors[0].get("reason"):
            reason = errors[0]["reason"]
    except Exception:
        pass
    return {"reason": reason, "message": message, "status": status}

def fetch_comments_sample(
    yt,
    video_id: str,
    max_comments: int = 100,
) -> tuple[list[str], dict | None]:
    comments: list[str] = []
    page_token = None
    while len(comments) < max_comments:
        max_results = min(100, max_comments - len(comments))
        req = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            pageToken=page_token,
            order="time",
        )
        try:
            resp = req.execute()
        except HttpError as exc:
            return [], _parse_http_error(exc)

        for item in resp.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textDisplay") or snippet.get("textOriginal") or ""
            if text:
                comments.append(text)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return comments, None

def fetch_comments_paged(
    yt,
    video_id: str,
    max_comments: int | None = None,
) -> tuple[list[str], dict | None]:
    comments: list[str] = []
    page_token = None
    while True:
        max_results = 100
        if max_comments is not None:
            remaining = max_comments - len(comments)
            if remaining <= 0:
                break
            max_results = min(100, remaining)

        req = yt.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText",
            pageToken=page_token,
            order="time",
        )
        try:
            resp = req.execute()
        except HttpError as exc:
            return [], _parse_http_error(exc)

        for item in resp.get("items", []):
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textDisplay") or snippet.get("textOriginal") or ""
            if text:
                comments.append(text)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return comments, None
