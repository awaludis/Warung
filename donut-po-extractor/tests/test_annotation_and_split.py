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
    pick = lambda d: {k: d[k] for k in ("correct", "total")}  # noqa: E731
    assert pick(c["po_number"]) == {"correct": 1, "total": 1}
    assert pick(c["items.description"]) == {"correct": 1, "total": 2}
    assert pick(c["items.quantity"]) == {"correct": 1, "total": 2}
    assert c["items.row_count"]["correct"] == 0
    assert c["po_date"]["total"] == 0  # empty on both sides: not counted


def test_similarity():
    assert training.similarity("PT Sinar", "pt sinar") == 1.0
    assert 0.8 < training.similarity("Gula Pasir 1 kg", "Gula Pasr 1 kg") < 1.0
    assert training.similarity("abc", None) == 0.0


def test_bulk_delete_samples_keeps_filed_records():
    from fastapi.testclient import TestClient

    from app.main import app

    _annotated("Toko Maju", 3)
    rec = ground_truth.file_record()
    ids = [s["sample_id"] for s in ground_truth.list_samples()]
    c = TestClient(app)
    r = c.post("/api/samples/delete", json={"sample_ids": ids[:2]})
    assert r.json() == {"deleted": 2} and len(ground_truth.list_samples()) == 1
    assert c.post("/api/samples/delete", json={"sample_ids": ["SMP-nope"]}).status_code == 404
    assert len(ground_truth.list_samples()) == 1  # nothing deleted when an id is invalid
    assert c.post("/api/samples/delete", json={}).status_code == 422
    assert c.post("/api/samples/delete", json={"all": True}).json() == {"deleted": 1}
    assert ground_truth.list_samples() == [] and ground_truth.list_entries() == []
    assert ground_truth.get_record(rec["record_id"])["entry_count"] == 3  # filed record untouched


def test_split_details_name_the_documents():
    _annotated("Toko Maju", 2)
    rec = ground_truth.file_record()
    run = training.create_run(rec["record_id"])
    rows = training.split_details(run)
    assert sorted(r["set"] for r in rows) == ["training", "validation"]
    assert {r["source_file"] for r in rows} == {"Toko Maju-1.png", "Toko Maju-2.png"}
    assert all(r["page"] == 1 for r in rows)


def test_manual_split_by_document(tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app

    from .conftest import pdf_bytes

    _annotated("Toko Maju", 3)
    _annotated("PT Sinar", 2, start=10)
    two_page = ground_truth.create_sample("long.pdf", pdf_bytes([["p1"], ["p2"]]), "PT Sinar")
    for pg in two_page["pages"]:
        ground_truth.save_entry(pg["entry_id"], {"po_number": "9", "items": [{"description": "x", "quantity": "1"}]}, True)
    rec = ground_truth.file_record()
    c = TestClient(app)

    docs = c.get(f"/api/gt-records/{rec['record_id']}/documents").json()
    assert len(docs) == 6 and {d["pages"] for d in docs} == {1, 2}
    assert any(d["suggested"] == "validation" for d in docs)
    by_name = {d["source_file"]: d for d in docs}
    img = c.get(f"/api/gt-records/{rec['record_id']}/images/{docs[0]['entry_ids'][0]}.png")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    # every layout needs at least one training and one validation document
    r = c.post("/api/runs", json={"record_id": rec["record_id"], "start": False,
                                  "validation_docs": [by_name["Toko Maju-1.png"]["doc_id"]]})
    assert r.status_code == 422 and "PT Sinar" in r.json()["detail"]
    all_maju = [d["doc_id"] for d in docs if d["customer_layout"] == "Toko Maju"]
    r = c.post("/api/runs", json={"record_id": rec["record_id"], "start": False,
                                  "validation_docs": all_maju + [two_page["sample_id"]]})
    assert r.status_code == 422 and "Toko Maju" in r.json()["detail"]
    assert c.post("/api/runs", json={"record_id": rec["record_id"], "start": False,
                                     "validation_docs": ["SMP-nope"]}).status_code == 422

    # valid: the 2-page PO goes to validation as a whole
    r = c.post("/api/runs", json={"record_id": rec["record_id"], "start": False,
                                  "validation_docs": [by_name["Toko Maju-2.png"]["doc_id"], two_page["sample_id"]]})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["params"]["split_mode"] == "manual"
    assert sorted(run["split"]["validation"]) == sorted([by_name["Toko Maju-2.png"]["entry_ids"][0]] + [p["entry_id"] for p in two_page["pages"]])
    assert not set(run["split"]["train"]) & set(run["split"]["validation"])
    assert run["layouts"] == {"Toko Maju": {"train": 2, "validation": 1}, "PT Sinar": {"train": 2, "validation": 2}}

    # random mode still works and respects the seed
    a = c.post("/api/runs", json={"record_id": rec["record_id"], "start": False, "params": {"seed": 1}}).json()
    b = c.post("/api/runs", json={"record_id": rec["record_id"], "start": False, "params": {"seed": 1}}).json()
    assert a["params"]["split_mode"] == "random" and a["split"] == b["split"]
