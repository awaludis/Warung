import json

import pytest

from app import documents, ground_truth, training
from app.storage import BusinessRuleError

from .conftest import pdf_bytes, png_bytes, po_image, sample_values


def test_only_pdf_or_image_accepted():
    with pytest.raises(BusinessRuleError, match="not accepted"):
        documents.detect_kind("po.docx", b"PK\x03\x04")
    with pytest.raises(BusinessRuleError, match="not a readable"):
        documents.detect_kind("po.png", b"not really a png")
    assert documents.detect_kind("po.pdf", pdf_bytes([["x"]])) == "pdf"
    assert documents.detect_kind("po.jpg", png_bytes(po_image(["x"]))) == "image"


def test_pdf_converted_page_by_page():
    s = ground_truth.create_sample("po.pdf", pdf_bytes([["PO-1 page 1"], ["page 2"], ["page 3"]]), "PT Sinar", "ann")
    assert [p["page"] for p in s["pages"]] == [1, 2, 3]
    e = ground_truth.get_entry(s["pages"][1]["entry_id"])
    assert e["status"] == "draft" and ground_truth.image_path(e["image"]).suffix == ".png"


def _annotated(layout: str, n: int, start: int = 1):
    for i in range(start, start + n):
        s = ground_truth.create_sample(f"{layout}-{i}.png", png_bytes(po_image([f"PO-{i:03d}"])), layout)
        ground_truth.save_entry(s["pages"][0]["entry_id"], sample_values(i, layout), complete=True)


def test_complete_requires_required_fields():
    s = ground_truth.create_sample("a.png", png_bytes(po_image(["x"])), "Toko Maju")
    with pytest.raises(BusinessRuleError, match="Required fields"):
        ground_truth.save_entry(s["pages"][0]["entry_id"], {"po_number": "1"}, complete=True)
    e = ground_truth.save_entry(s["pages"][0]["entry_id"], {"po_number": "1"}, complete=False)
    assert e["status"] == "draft"


def test_filing_requires_every_layout_represented():
    _annotated("Toko Maju", 2)
    ground_truth.create_sample("new.png", png_bytes(po_image(["x"])), "PT Baru")  # only a draft
    with pytest.raises(BusinessRuleError, match="PT Baru"):
        ground_truth.file_record("ann")
    _annotated("PT Baru", 2, start=10)
    # the draft of PT Baru stays a draft; layout now has 2 completed entries
    rec = ground_truth.file_record("ann", "first batch")
    assert rec["entry_count"] == 4 and rec["layouts"] == {"Toko Maju": 2, "PT Baru": 2}
    lines = (ground_truth.record_dir(rec["record_id"]) / "images" / "metadata.jsonl").read_text().splitlines()
    gt = json.loads(json.loads(lines[0])["ground_truth"])["gt_parse"]
    assert gt["po_number"].startswith("PO-")


def test_split_is_disjoint_and_stratified():
    entries = [{"entry_id": f"{l}-{i}", "customer_layout": l} for l in "ABC" for i in range(5)]
    train, val = training.split_entries(entries, 0.2, seed=1)
    assert not set(train) & set(val)
    assert len(train) + len(val) == 15
    assert {v.split("-")[0] for v in val} == {"A", "B", "C"}
    assert {t.split("-")[0] for t in train} == {"A", "B", "C"}
    assert training.split_entries(entries, 0.2, seed=1) == (train, val)  # reproducible


def test_release_needs_validation():
    _annotated("Toko Maju", 3)
    rec = ground_truth.file_record()
    run = training.create_run(rec["record_id"])
    with pytest.raises(BusinessRuleError, match="Validation results must be recorded"):
        training.release(run["run_id"])
    with pytest.raises(BusinessRuleError, match="must be trained"):
        training.validate(run["run_id"])


def test_compare_counts_rows(monkeypatch):
    from app.field_definition import load_definition

    fd = load_definition()
    exp = fd.normalize_output({"po_number": "PO-1", "items": [{"description": "A", "quantity": "2"}, {"description": "B", "quantity": "1"}]})
    pred = fd.normalize_output({"po_number": "po-1", "items": [{"description": "A", "quantity": "2.0"}]})
    c = training.compare(fd, exp, pred)
    assert c["po_number"] == {"correct": 1, "total": 1}
    assert c["items.description"] == {"correct": 1, "total": 2}
    assert c["items.quantity"] == {"correct": 1, "total": 2}
    assert c["items.row_count"]["correct"] == 0
