"""Fetch comments with pagination for each labeled video.
Outputs:
- data/youtube/comments_raw/<domain>/<video_id>.jsonl
"""

from __future__ import annotations
import argparse
import pandas as pd
from config import YOUTUBE_VIDEOS_LABELED_DIR, YOUTUBE_COMMENTS_RAW_DIR

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_comments_per_video", type=int, default=5000)
    args = ap.parse_args()

    videos_path = YOUTUBE_VIDEOS_LABELED_DIR / "videos_with_domain.csv"
    df = pd.read_csv(videos_path)

    # TODO: Implement API commentThreads.list pagination.
    for _, r in df.iterrows():
        domain = r["domain_assigned"]
        vid = r.get("video_id", "UNKNOWN_VIDEO")
        out_dir = YOUTUBE_COMMENTS_RAW_DIR / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{vid}.jsonl").write_text("", encoding="utf-8")

    print("Created placeholder raw comment files.")

if __name__ == "__main__":
    main()
