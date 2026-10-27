"""Document intake: accept only PDF or image files and turn them into page images.

Business rules (BUC 1 & BUC 3):
  * only purchase orders in PDF or image format are accepted,
  * PDFs are converted / processed page by page as images.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from . import config
from .storage import BusinessRuleError


def detect_kind(filename: str, content: bytes) -> str:
    """Return 'pdf' or 'image'; reject anything else (checks the file content, not just the name)."""
    ext = Path(filename or "").suffix.lower()
    if ext not in config.ACCEPTED_EXT:
        raise BusinessRuleError(
            f"File type '{ext or 'unknown'}' is not accepted. Use PDF or an image "
            f"({', '.join(sorted(config.ACCEPTED_EXT))})."
        )
    if content[:5] == b"%PDF-":
        return "pdf"
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
        return "image"
    except Exception as exc:  # noqa: BLE001
        raise BusinessRuleError("The file is not a readable PDF or image.") from exc


def to_page_images(content: bytes, kind: str, out_dir: Path, dpi: int | None = None) -> list[Path]:
    """Save every page as an RGB PNG (page_001.png, page_002.png, ...) and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    if kind == "pdf":
        import fitz  # PyMuPDF

        dpi = dpi or config.PDF_RENDER_DPI
        with fitz.open(stream=content, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise BusinessRuleError("The PDF has no pages.")
            for i, page in enumerate(doc, 1):
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                p = out_dir / f"page_{i:03d}.png"
                pix.save(p)
                pages.append(p)
    else:
        with Image.open(io.BytesIO(content)) as img:
            frames = getattr(img, "n_frames", 1)  # multi-page TIFF
            for i in range(frames):
                img.seek(i)
                p = out_dir / f"page_{i + 1:03d}.png"
                img.convert("RGB").save(p)
                pages.append(p)
    return pages
