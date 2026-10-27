"""Full BUC 1 -> 2 -> 3 flow through the HTTP API with a tiny Donut model."""
import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from app import training  # noqa: E402
from app.field_definition import load_definition  # noqa: E402
from app.main import app  # noqa: E402

from .conftest import pdf_bytes, png_bytes, po_image, sample_values  # noqa: E402

H = {"X-User": "tester"}


def test_full_flow(tiny_settings):
    c = TestClient(app)
    fd = load_definition()

    # extraction before any release is refused
    r = c.post("/api/extractions", files={"file": ("po.png", png_bytes(po_image(["x"])), "image/png")}, headers=H)
    assert r.status_code == 422 and "No extraction model" in r.json()["detail"]

    # ---- BUC 1 ----
    for layout in ("Toko Maju", "PT Sinar"):
        for i in range(1, 4):
            r = c.post(
                "/api/samples",
                files={"file": (f"{i}.png", png_bytes(po_image([f"PO-{i:03d}", layout])), "image/png")},
                data={"customer_layout": layout},
                headers=H,
            )
            assert r.status_code == 200, r.text
            entry_id = r.json()["pages"][0]["entry_id"]
            r = c.put(f"/api/entries/{entry_id}", json={"values": sample_values(i, layout), "complete": True}, headers=H)
            assert r.status_code == 200, r.text
    assert c.get(f"/api/entries/{entry_id}/image").status_code == 200
    r = c.post("/api/gt-records", json={"note": "batch 1"}, headers=H)
    assert r.status_code == 200, r.text
    record_id = r.json()["record_id"]

    # ---- BUC 2 ----
    r = c.post(
        "/api/runs",
        json={"record_id": record_id, "start": False, "params": {"epochs": 2, "learning_rate": 1e-3, "auto_validate": True}},
        headers=H,
    )
    assert r.status_code == 200, r.text
    run = r.json()
    assert not set(run["split"]["train"]) & set(run["split"]["validation"])
    run = training.start_training(run["run_id"], "tester", blocking=True)
    assert run["status"] == "validated", training.read_log(run["run_id"])

    sheet = c.get(f"/api/runs/{run['run_id']}/validation-sheet").json()
    assert sheet["validation_pages"] == len(run["split"]["validation"])
    assert set(sheet["per_layout"]) == {"Toko Maju", "PT Sinar"}
    assert "po_number" in sheet["per_field"] and "items.quantity" in sheet["per_field"]
    csv = c.get(f"/api/runs/{run['run_id']}/validation-sheet?format=csv")
    assert csv.status_code == 200 and "customer_layout,field" in csv.text

    r = c.post(f"/api/runs/{run['run_id']}/release", json={"note": "ok"}, headers=H)
    assert r.status_code == 200, r.text
    rel = r.json()
    assert rel["trained_layouts"] == ["PT Sinar", "Toko Maju"]
    assert c.get("/api/status").json()["model"]["retraining_required"] is False

    # ---- BUC 3 ----
    r = c.post(
        "/api/extractions",
        files={"file": ("po.pdf", pdf_bytes([["PO-001 page 1"], ["page 2"]]), "application/pdf")},
        data={"customer_layout": "Unknown Customer"},
        headers=H,
    )
    assert r.status_code == 200, r.text
    ext = r.json()
    assert len(ext["pages"]) == 2  # processed page by page
    assert set(ext["data"]) == {f.name for f in fd.fields}  # follows the definition sheet
    assert any("not in the released model" in w for w in ext["warnings"])
    assert c.get(f"/api/extractions/{ext['extraction_id']}/{ext['pages'][0]['image']}").status_code == 200
    assert c.get(f"/api/extractions/{ext['extraction_id']}/json").status_code == 409  # not saved yet

    bad = c.post(f"/api/extractions/{ext['extraction_id']}/save", json={"data": {"po_number": "1", "foo": "x"}}, headers=H)
    assert bad.status_code == 422
    corrected = {**ext["data"], "po_number": "PO-777", "total_amount": 15000}
    r = c.post(f"/api/extractions/{ext['extraction_id']}/save", json={"data": corrected}, headers=H)
    assert r.status_code == 200 and r.json()["status"] == "saved"
    out = c.get(f"/api/extractions/{ext['extraction_id']}/json")
    assert out.json()["po_number"] == "PO-777" and out.json()["total_amount"] == 15000
    assert "PO_PO-777.json" in out.headers["content-disposition"]

    # unsupported file type
    r = c.post("/api/extractions", files={"file": ("po.docx", b"PK\x03\x04", "application/octet-stream")}, headers=H)
    assert r.status_code == 422

    # a new layout makes the released model need retraining
    c.post("/api/samples", files={"file": ("n.png", png_bytes(po_image(["x"])), "image/png")}, data={"customer_layout": "PT Baru"}, headers=H)
    status = c.get("/api/status").json()["model"]
    assert status["retraining_required"] and "PT Baru" in status["reasons"][0]

    assert c.get("/").status_code == 200
