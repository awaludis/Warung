"""BUC 1 — Prepare Annotated Customer PO Samples (actor: Data Annotator).

Flow: collect sample PO (PDF/image) -> convert to page images -> record the correct value of each
defined field per page -> save page + values as a ground truth entry -> file the completed entries
as a PO Ground Truth Record for model training.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from . import config, documents
from .field_definition import load_definition
from .storage import BusinessRuleError, NotFoundError, new_id, now_iso, read_json, safe_id, write_json

DRAFT, COMPLETED = "draft", "completed"


def _norm_layout(layout: str) -> str:
    layout = " ".join((layout or "").split())
    if not layout:
        raise BusinessRuleError("Customer layout is required (e.g. the customer name / PO template).")
    return layout


# ---------------- samples ----------------
def create_sample(filename: str, content: bytes, customer_layout: str, user: str = "") -> dict:
    """Steps 1-2: store a collected sample and convert it into page images."""
    layout = _norm_layout(customer_layout)
    kind = documents.detect_kind(filename, content)
    p = config.paths()
    sample_id = new_id("SMP")
    sdir = p["samples"] / sample_id
    sdir.mkdir(parents=True)
    (sdir / f"source{Path(filename).suffix.lower()}").write_bytes(content)
    try:
        pages = documents.to_page_images(content, kind, sdir / "pages")
    except Exception:
        shutil.rmtree(sdir, ignore_errors=True)
        raise
    fd = load_definition()
    sample = {
        "sample_id": sample_id,
        "filename": filename,
        "kind": kind,
        "customer_layout": layout,
        "created_at": now_iso(),
        "created_by": user,
        "pages": [],
    }
    for i, img in enumerate(pages, 1):
        entry_id = f"{sample_id}-P{i:03d}"
        write_json(
            p["entries"] / f"{entry_id}.json",
            {
                "entry_id": entry_id,
                "sample_id": sample_id,
                "page": i,
                "image": str(img.relative_to(p["samples"])),
                "customer_layout": layout,
                "status": DRAFT,
                "values": fd.empty_record(),
                "field_definition_hash": fd.hash,
                "updated_at": now_iso(),
                "updated_by": user,
            },
        )
        sample["pages"].append({"page": i, "entry_id": entry_id})
    write_json(sdir / "sample.json", sample)
    return sample


def list_samples() -> list[dict]:
    out = []
    for f in sorted(config.paths()["samples"].glob("*/sample.json"), reverse=True):
        s = read_json(f)
        for pg in s["pages"]:
            pg["status"] = get_entry(pg["entry_id"])["status"]
        s["completed_pages"] = sum(pg["status"] == COMPLETED for pg in s["pages"])
        out.append(s)
    return out


def get_sample(sample_id: str) -> dict:
    return read_json(config.paths()["samples"] / safe_id(sample_id) / "sample.json")


def delete_sample(sample_id: str) -> None:
    s = get_sample(sample_id)
    p = config.paths()
    for pg in s["pages"]:
        (p["entries"] / f"{pg['entry_id']}.json").unlink(missing_ok=True)
    shutil.rmtree(p["samples"] / s["sample_id"])


def image_path(rel: str) -> Path:
    base = config.paths()["samples"].resolve()
    full = (base / rel).resolve()
    if base not in full.parents or not full.exists():
        raise NotFoundError("image not found")
    return full


# ---------------- entries ----------------
def get_entry(entry_id: str) -> dict:
    return read_json(config.paths()["entries"] / f"{safe_id(entry_id)}.json")


def list_entries(status: str | None = None) -> list[dict]:
    out = [read_json(f) for f in sorted(config.paths()["entries"].glob("*.json"))]
    return [e for e in out if status is None or e["status"] == status]


def save_entry(entry_id: str, values: dict, complete: bool, user: str = "") -> dict:
    """Steps 4-5: record field values for a page and save it as a ground truth entry."""
    entry = get_entry(entry_id)
    fd = load_definition()
    clean = fd.validate_annotation(values)  # rule: only defined fields, defined structure
    if complete:
        missing = fd.missing_required(clean)
        if len(get_sample(entry["sample_id"])["pages"]) > 1:
            # in a multi-page PO the header and totals sit on different pages: only line-item rules apply per page
            missing = [m for m in missing if "[" in m]
        if missing:
            raise BusinessRuleError(f"Required fields are empty: {', '.join(missing)}")
    entry.update(
        values=clean,
        status=COMPLETED if complete else DRAFT,
        field_definition_hash=fd.hash,
        updated_at=now_iso(),
        updated_by=user,
    )
    write_json(config.paths()["entries"] / f"{entry_id}.json", entry)
    return entry


def coverage() -> dict:
    """Completed / draft entries per customer layout — shows whether every layout is represented."""
    fd = load_definition()
    layouts: dict[str, dict] = {}
    for e in list_entries():
        row = layouts.setdefault(e["customer_layout"], {"completed": 0, "draft": 0, "outdated": 0})
        row[e["status"]] += 1
        if e["status"] == COMPLETED and e.get("field_definition_hash") != fd.hash:
            row["outdated"] += 1
    for row in layouts.values():
        row["ready"] = row["completed"] - row["outdated"] >= config.MIN_ENTRIES_PER_LAYOUT and row["outdated"] == 0
    return {"min_per_layout": config.MIN_ENTRIES_PER_LAYOUT, "field_definition_hash": fd.hash, "layouts": layouts}


# ---------------- filing the ground truth record ----------------
def file_record(user: str = "", note: str = "") -> dict:
    """Step 6: file all completed entries as an immutable PO Ground Truth Record [Completed]."""
    fd = load_definition()
    cov = coverage()
    if not cov["layouts"]:
        raise BusinessRuleError("There are no ground truth entries to file.")
    problems = []
    for layout, row in cov["layouts"].items():
        if row["outdated"]:
            problems.append(f"'{layout}': {row['outdated']} entries were annotated with an older field definition — re-save them")
        elif row["completed"] < config.MIN_ENTRIES_PER_LAYOUT:
            problems.append(
                f"'{layout}': {row['completed']} completed entries, at least {config.MIN_ENTRIES_PER_LAYOUT} needed"
            )
    if problems:
        raise BusinessRuleError("Every customer PO layout must be represented. " + "; ".join(problems))

    p = config.paths()
    record_id = new_id("GTR")
    rdir = p["gt_records"] / record_id
    (rdir / "images").mkdir(parents=True)
    completed = list_entries(COMPLETED)
    entries, lines = [], []
    for e in completed:
        name = f"{e['entry_id']}.png"
        shutil.copy2(image_path(e["image"]), rdir / "images" / name)
        entries.append(
            {"entry_id": e["entry_id"], "image": f"images/{name}", "customer_layout": e["customer_layout"], "values": e["values"]}
        )
        # Hugging Face / Donut dataset format, so the record can also be used with standard Donut scripts.
        lines.append(json.dumps({"file_name": name, "ground_truth": json.dumps({"gt_parse": e["values"]}, ensure_ascii=False)}, ensure_ascii=False))
    (rdir / "images" / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = {
        "record_id": record_id,
        "status": "completed",
        "filed_at": now_iso(),
        "filed_by": user,
        "note": note,
        "field_definition_version": fd.version,
        "field_definition_hash": fd.hash,
        "layouts": dict(Counter(e["customer_layout"] for e in entries)),
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(rdir / "record.json", record)
    return record


def list_records() -> list[dict]:
    out = []
    for f in sorted(config.paths()["gt_records"].glob("*/record.json"), reverse=True):
        r = read_json(f)
        r.pop("entries", None)
        out.append(r)
    return out


def get_record(record_id: str) -> dict:
    return read_json(record_dir(record_id) / "record.json")


def record_dir(record_id: str) -> Path:
    return config.paths()["gt_records"] / safe_id(record_id)


# ---------------- practice documents ----------------
PRACTICE_DIR = config.ROOT_DIR / "practice_docs" / "training"


def import_answer_key(folder: Path | None = None, complete: bool = True, user: str = "") -> dict:
    """Load every document listed in answer_key.json with its correct values (practice shortcut for BUC 1)."""
    folder = Path(folder or PRACTICE_DIR)
    key_file = folder / "answer_key.json"
    if not key_file.exists():
        raise BusinessRuleError(f"answer_key.json not found in {folder}")
    key = json.loads(key_file.read_text(encoding="utf-8"))
    existing = {(s["filename"], s["customer_layout"]) for s in list_samples()}
    dupes = [d["file"] for d in key if (Path(d["file"]).name, d["customer_layout"]) in existing]
    if dupes:
        raise BusinessRuleError(
            f"{len(dupes)} of these documents were already imported (e.g. {dupes[0]}), so nothing was imported again. "
            "Your existing samples are fine; there is no need to delete anything."
        )
    pages = 0
    for doc in key:
        path = folder / doc["file"]
        sample = create_sample(path.name, path.read_bytes(), doc["customer_layout"], user)
        if len(sample["pages"]) != len(doc["pages"]):
            raise BusinessRuleError(f"{doc['file']}: {len(sample['pages'])} pages but the answer key has {len(doc['pages'])}")
        for pg, values in zip(sample["pages"], doc["pages"]):
            save_entry(pg["entry_id"], values, complete, user)
            pages += 1
    return {"documents": len(key), "pages": pages, "status": COMPLETED if complete else DRAFT, "coverage": coverage()}
