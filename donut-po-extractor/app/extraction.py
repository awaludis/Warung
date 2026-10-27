"""BUC 3 — Extract Customer PO Data to JSON (actor: Order Processing Officer).

Flow: receive PO (PDF/image) -> submit to the application -> Donut reads every page image directly and
identifies the defined fields -> officer obtains the JSON in the sheet's structure -> officer saves the
Structured PO Data Record for further order processing.
"""
from __future__ import annotations

import threading
from pathlib import Path

from . import config, documents, training
from .donut_model import DonutExtractor, merge_pages
from .field_definition import load_definition
from .storage import BusinessRuleError, NotFoundError, new_id, now_iso, read_json, safe_id, write_json

_cache: dict = {"release_id": None, "extractor": None}
_cache_lock = threading.Lock()


def get_extractor() -> tuple[DonutExtractor, dict]:
    rel = training.current_release()
    if rel is None:
        raise BusinessRuleError("No extraction model has been released yet (see BUC 2: Train and Validate).")
    with _cache_lock:
        if _cache["release_id"] != rel["release_id"]:
            _cache["extractor"] = DonutExtractor(training.released_model_dir(rel))
            _cache["release_id"] = rel["release_id"]
        return _cache["extractor"], rel


def _dir(extraction_id: str) -> Path:
    return config.paths()["extractions"] / safe_id(extraction_id)


def extract(filename: str, content: bytes, customer_layout: str = "", user: str = "") -> dict:
    fd = load_definition()
    kind = documents.detect_kind(filename, content)  # rule: PDF or image only
    rel = training.current_release()
    if rel is None:
        raise BusinessRuleError("No extraction model has been released yet (see BUC 2: Train and Validate).")
    if rel["field_definition_hash"] != fd.hash:
        raise BusinessRuleError(
            "The PO Data Field Definition Sheet changed after the released model was trained. "
            "The model must be retrained before extraction (BUC 2)."
        )
    extractor, rel = get_extractor()

    extraction_id = new_id("EXT")
    d = _dir(extraction_id)
    d.mkdir(parents=True)
    (d / f"source{Path(filename).suffix.lower()}").write_bytes(content)
    page_paths = documents.to_page_images(content, kind, d / "pages")  # rule: PDF page by page

    pages = []
    for i, img in enumerate(page_paths, 1):
        raw, seq = extractor.read_page(img)
        pages.append({"page": i, "image": f"pages/{img.name}", "data": fd.normalize_output(raw), "raw_sequence": seq})
    data = fd.normalize_output(merge_pages(fd, [p["data"] for p in pages]))

    layout = " ".join((customer_layout or "").split())
    warnings = []
    if not layout:
        warnings.append("Customer layout not specified — check the result carefully.")
    elif layout.casefold() not in {l.casefold() for l in rel["trained_layouts"]}:
        warnings.append(
            f"Layout '{layout}' is not in the released model's training samples; extraction may be unreliable. "
            "Ask the Data Annotator to add samples for it and retrain."
        )
    missing = fd.missing_required(data)
    if missing:
        warnings.append(f"Required fields not found: {', '.join(missing)}")

    result = {
        "extraction_id": extraction_id,
        "status": "extracted",
        "filename": filename,
        "kind": kind,
        "customer_layout": layout,
        "release_id": rel["release_id"],
        "field_definition_hash": fd.hash,
        "extracted_at": now_iso(),
        "extracted_by": user,
        "pages": pages,
        "data": data,
        "warnings": warnings,
    }
    write_json(d / "extraction.json", result)
    return result


def save(extraction_id: str, data: dict | None = None, user: str = "") -> dict:
    """Step 5: save the (optionally corrected) structured PO data as the filed record."""
    ext = get_extraction(extraction_id)
    fd = load_definition()
    if data is not None:
        fd.validate_annotation(_as_text(data))  # reject fields outside the definition sheet
        ext["data"] = fd.normalize_output(data)
        ext["corrected"] = True
    ext.update(status="saved", saved_at=now_iso(), saved_by=user)
    write_json(_dir(extraction_id) / "extraction.json", ext)
    write_json(_dir(extraction_id) / "po_data.json", ext["data"])
    return ext


def _as_text(data: dict) -> dict:
    """Numbers are fine in output JSON; turn them into text only for the structure check."""
    def conv(v):
        if isinstance(v, list):
            return [{k: conv(x) for k, x in r.items()} if isinstance(r, dict) else r for r in v]
        return "" if v is None else v if isinstance(v, (dict, str)) else str(v)

    if not isinstance(data, dict):
        raise BusinessRuleError("PO data must be a JSON object")
    return {k: conv(v) for k, v in data.items()}


def get_extraction(extraction_id: str) -> dict:
    return read_json(_dir(extraction_id) / "extraction.json")


def list_extractions() -> list[dict]:
    out = []
    for f in sorted(config.paths()["extractions"].glob("*/extraction.json"), reverse=True):
        e = read_json(f)
        out.append(
            {k: e.get(k) for k in ("extraction_id", "status", "filename", "customer_layout", "extracted_at", "saved_at", "release_id", "warnings")}
            | {"po_number": (e.get("data") or {}).get("po_number"), "pages": len(e.get("pages", []))}
        )
    return out


def page_image(extraction_id: str, rel: str) -> Path:
    base = _dir(extraction_id).resolve()
    full = (base / rel).resolve()
    if base not in full.parents or not full.exists():
        raise NotFoundError("image not found")
    return full
