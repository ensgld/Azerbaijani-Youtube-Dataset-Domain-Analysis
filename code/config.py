"""config.py
Merkezi konfigürasyon dosyası.
- Domain isimleri (birebir)
- Keyword pool'ları
- AZ-only filtre marker'ları
- Path'ler
- Model hiperparametreleri
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DELIVERABLES_DIR = PROJECT_ROOT / "deliverables"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
RUNS_DIR = PROJECT_ROOT / "runs"

PART1_RAW_DIR = DATA_DIR / "part1" / "raw"
PART1_PROCESSED_DIR = DATA_DIR / "part1" / "processed"

YOUTUBE_DIR = DATA_DIR / "youtube"
YOUTUBE_VIDEO_CANDIDATES_DIR = YOUTUBE_DIR / "video_candidates"
YOUTUBE_VIDEOS_LABELED_DIR = YOUTUBE_DIR / "videos_labeled"
YOUTUBE_COMMENTS_RAW_DIR = YOUTUBE_DIR / "comments_raw"
YOUTUBE_COMMENTS_FILTERED_DIR = YOUTUBE_DIR / "comments_filtered"
YOUTUBE_CORPORA_DIR = YOUTUBE_DIR / "corpora"

ANALYTICS_DIR = DATA_DIR / "analytics"

EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
TOKENIZERS_DIR = MODELS_DIR / "tokenizers"

# --- Domains (use verbatim) ---
DOMAINS = ['Technology & Digital Services', 'Finance & Business', 'Social Life & Entertainment', 'Retail & Lifestyle', 'Public Services']

# --- Keyword pools (Azerbaijani examples; extend as needed) ---
DOMAIN_KEYWORDS = {
    "Technology & Digital Services": ["telefon", "internet", "tətbiq", "oyun", "texnologiya", "rəqəmsal"],
    "Finance & Business": [
        "biznes", "bank", "kredit", "investisiya", "valyuta", "faiz", "sığorta",
        "maliyyə", "iqtisadiyyat", "depozit", "ipoteka", "məzənnə", "kredit kartı",
    ],
    "Social Life & Entertainment": ["musiqi", "film", "serial", "şou", "komediya", "məşhur", "vlog"],
    "Retail & Lifestyle": ["alış", "qiymət", "endirim", "market", "geyim", "məişət", "review"],
    "Public Services": ["dövlət", "bələdiyyə", "xidmət", "səhiyyə", "təhsil", "kommunal", "vergi"],
}

AZ_QUERY_BOOST_TERMS = ["azərbaycan", "azərbaycanca", "bakı", "baku"]

DOMAIN_LOCAL_TERMS = {
    "Technology & Digital Services": ["azercell", "bakcell", "nar", "wifi", "internet", "tətbiq", "telefon"],
    "Finance & Business": [
        "kapital bank", "kapitalbank", "abb", "paşa bank", "pasha bank", "xalq bank",
        "accessbank", "unibank", "turanbank", "birbank", "bank respublika",
        "bank of baku", "rabitəbank", "e-manat", "m10",
    ],
    "Retail & Lifestyle": ["bravo", "bazarstore", "endirim", "qiymət", "alış-veriş", "market"],
    "Public Services": ["asan xidmət", "vergi", "təhsil", "səhiyyə", "dövlət", "kommunal"],
    "Social Life & Entertainment": ["musiqi", "film", "serial", "şou", "vlog", "komediya"],
}

RETAIL_BIAS_TERMS = ["vlog", "review", "icmal", "alış-veriş vlog", "market alış-veriş", "unboxing"]
FINANCE_BIAS_TERMS = ["analiz", "icmal", "müsahibə", "intervyu", "müzakirə", "canlı", "proqnoz", "şərh", "rəy", "xəbər"]

# --- Domain scoring weights ---
TITLE_WEIGHT = 3.0
TAGS_WEIGHT = 2.0
DESC_WEIGHT = 1.0

# --- Azerbaijani-only filter (two-layer) ---
AZ_STRONG_CHARS = set("əƏ")
AZ_MARKERS = {
    "mən", "sən", "biz", "siz", "deyil", "hansı", "necə", "niyə", "üçün",
    "gərək", "bəlkə", "heç", "çox"
}
TR_MARKERS = {
    "değil", "için", "bence", "gerçekten", "abi", "kanka", "şey"
}
AZ_FILTER_DEFAULT_THRESHOLD = 2.5

# --- Modeling defaults ---
SEED = 42
NUM_CLASSES = 3  # neg/neu/pos
GRU_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 256
EPOCHS = 5
MAX_LEN = 80
MAX_NUM_WORDS = 50000

@dataclass(frozen=True)
class LabelMap:
    neg_value: float = 0.0
    neu_value: float = 0.5
    pos_value: float = 1.0
    neg_id: int = 0
    neu_id: int = 1
    pos_id: int = 2

LABEL_MAP = LabelMap()
