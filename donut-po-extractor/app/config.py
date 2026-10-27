"""Application settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DONUT_PO_DATA_DIR", ROOT_DIR / "data"))
FIELD_DEFINITION_PATH = Path(
    os.environ.get("DONUT_PO_FIELD_DEFINITION", ROOT_DIR / "config" / "po_field_definition.json")
)

# Pre-trained Donut checkpoint used as the starting point for fine-tuning (BUC 2).
BASE_MODEL = os.environ.get("DONUT_PO_BASE_MODEL", "naver-clova-ix/donut-base")

# Resolution used when rendering PDF pages to images (BUC 1 & BUC 3).
PDF_RENDER_DPI = int(os.environ.get("DONUT_PO_PDF_DPI", "200"))

# Donut input size (height, width). Smaller = faster training, larger = better on dense POs.
IMAGE_HEIGHT = int(os.environ.get("DONUT_PO_IMAGE_HEIGHT", "1280"))
IMAGE_WIDTH = int(os.environ.get("DONUT_PO_IMAGE_WIDTH", "960"))
MAX_TARGET_LENGTH = int(os.environ.get("DONUT_PO_MAX_LENGTH", "768"))

# Minimum completed ground truth entries per customer layout before a record can be filed.
# 2 = at least one page for training and one page for validation.
MIN_ENTRIES_PER_LAYOUT = int(os.environ.get("DONUT_PO_MIN_PER_LAYOUT", "2"))

ACCEPTED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ACCEPTED_EXT = ACCEPTED_IMAGE_EXT | {".pdf"}


def paths() -> dict[str, Path]:
    """Data folders, created on demand. Read at call time so tests can point DATA_DIR elsewhere."""
    base = Path(os.environ.get("DONUT_PO_DATA_DIR", DATA_DIR))
    p = {
        "samples": base / "ground_truth" / "samples",
        "entries": base / "ground_truth" / "entries",
        "gt_records": base / "ground_truth" / "records",
        "runs": base / "models" / "runs",
        "releases": base / "models" / "releases",
        "extractions": base / "extractions",
    }
    for d in p.values():
        d.mkdir(parents=True, exist_ok=True)
    return p
