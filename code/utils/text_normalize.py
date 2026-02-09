"""Text normalization helpers (URLs, whitespace, lowercasing)."""

import re

URL_RE = re.compile(r"http\S+", re.IGNORECASE)

def normalize_text(t: str) -> str:
    t = (t or "").lower().strip()
    t = URL_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t
