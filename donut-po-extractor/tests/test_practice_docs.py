import json

from app import cli, ground_truth
from tools.generate_practice_docs import generate


def test_generated_docs_import_and_file(tmp_path, capsys):
    out = tmp_path / "docs"
    generate(out, per_layout_train=4, per_layout_test=1, seed=7)
    key = json.loads((out / "training" / "answer_key.json").read_text())
    assert {d["customer_layout"] for d in key} == {
        "PT Sumber Makmur Sejahtera", "CV Berkah Jaya Abadi", "Koperasi Karyawan Sentosa"
    }
    assert any(len(d["pages"]) == 2 for d in key)  # a multi-page PO is included
    assert any(d["file"].endswith(".jpg") for d in key)  # and a scanned image

    cli.import_samples(out / "training", complete=True, user="t")
    rec = ground_truth.file_record("t")
    assert rec["entry_count"] == sum(len(d["pages"]) for d in key)
    test_key = json.loads((out / "extraction_test" / "answer_key.json").read_text())
    assert "Toko Serba Ada Mulia" in {d["customer_layout"] for d in test_key}


def test_multi_page_po_pages_can_be_completed():
    from .conftest import pdf_bytes

    s = ground_truth.create_sample("long.pdf", pdf_bytes([["page 1"], ["page 2"]]), "CV X")
    # page 1: header + items, no total; page 2: items + total only
    ground_truth.save_entry(s["pages"][0]["entry_id"], {"po_number": "1", "items": [{"description": "A", "quantity": "1"}]}, True)
    ground_truth.save_entry(s["pages"][1]["entry_id"], {"items": [{"description": "B", "quantity": "2"}], "total_amount": "3"}, True)


def test_practice_import_api_refuses_duplicates(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    out = tmp_path / "docs"
    generate(out, per_layout_train=2, per_layout_test=1, seed=3)
    monkeypatch.setattr(ground_truth, "PRACTICE_DIR", out / "training")
    c = TestClient(app)
    assert c.get("/api/practice").json()["available"]
    r = c.post("/api/practice/import", json={})
    assert r.status_code == 200 and r.json()["documents"] == 6
    r = c.post("/api/practice/import", json={})
    assert r.status_code == 422 and "already imported" in r.json()["detail"]
