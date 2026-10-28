"""HTTP API + web UI. Run with:  uvicorn app.main:app --reload"""
from __future__ import annotations

import json
from typing import Any

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, extraction, ground_truth, training
from .field_definition import load_definition
from .storage import BusinessRuleError, NotFoundError

app = FastAPI(title="Customer PO Extraction (Donut)", version="1.0")

MAX_UPLOAD = 50 * 1024 * 1024


@app.exception_handler(BusinessRuleError)
async def _rule(_: Request, exc: BusinessRuleError):
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.exception_handler(NotFoundError)
async def _nf(_: Request, exc: NotFoundError):
    return JSONResponse({"detail": str(exc)}, status_code=404)


async def _read(file: UploadFile) -> bytes:
    content = await file.read()
    if len(content) > MAX_UPLOAD:
        raise BusinessRuleError("File is larger than 50 MB.")
    if not content:
        raise BusinessRuleError("The uploaded file is empty.")
    return content


def _user(x_user: str | None) -> str:
    return (x_user or "").strip()[:80]


# ---------------- shared ----------------
@app.get("/api/field-definition")
def field_definition():
    return load_definition().to_public()


@app.get("/api/status")
def status():
    p = config.paths()
    return {
        "model": training.model_status(),
        "coverage": ground_truth.coverage(),
        "base_model": config.BASE_MODEL,
        "data_dir": str(p["runs"].parent.parent),
        "releases_dir": str(p["releases"]),
    }


# ---------------- BUC 1: annotation ----------------
@app.post("/api/samples")
async def upload_sample(
    file: UploadFile = File(...), customer_layout: str = Form(...), x_user: str | None = Header(None)
):
    return ground_truth.create_sample(file.filename or "", await _read(file), customer_layout, _user(x_user))


@app.get("/api/samples")
def samples():
    return ground_truth.list_samples()


@app.get("/api/samples/{sample_id}")
def sample(sample_id: str):
    return ground_truth.get_sample(sample_id)


@app.delete("/api/samples/{sample_id}")
def delete_sample(sample_id: str):
    ground_truth.delete_sample(sample_id)
    return {"ok": True}


@app.get("/api/entries/{entry_id}")
def entry(entry_id: str):
    return ground_truth.get_entry(entry_id)


@app.get("/api/entries/{entry_id}/image")
def entry_image(entry_id: str):
    return FileResponse(ground_truth.image_path(ground_truth.get_entry(entry_id)["image"]))


@app.put("/api/entries/{entry_id}")
def save_entry(entry_id: str, body: dict = Body(...), x_user: str | None = Header(None)):
    return ground_truth.save_entry(entry_id, body.get("values") or {}, bool(body.get("complete")), _user(x_user))


@app.get("/api/practice")
def practice_available():
    return {"available": (ground_truth.PRACTICE_DIR / "answer_key.json").exists()}


@app.post("/api/practice/import")
def practice_import(body: dict = Body(default={}), x_user: str | None = Header(None)):
    return ground_truth.import_answer_key(None, not body.get("draft"), _user(x_user))


@app.get("/api/coverage")
def coverage():
    return ground_truth.coverage()


@app.post("/api/gt-records")
def file_record(body: dict = Body(default={}), x_user: str | None = Header(None)):
    return ground_truth.file_record(_user(x_user), body.get("note", ""))


@app.get("/api/gt-records")
def records():
    return ground_truth.list_records()


@app.get("/api/gt-records/{record_id}")
def record(record_id: str):
    return ground_truth.get_record(record_id)


# ---------------- BUC 2: training & validation ----------------
@app.get("/api/runs")
def runs():
    return training.list_runs()


@app.post("/api/runs")
def create_run(body: dict = Body(...), x_user: str | None = Header(None)):
    run = training.create_run(body["record_id"], body.get("params") or {}, _user(x_user))
    if body.get("start", True):
        run = training.start_training(run["run_id"], _user(x_user))
    return run


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    r = training.get_run(run_id)
    return {**r, "log": training.read_log(run_id)}


@app.post("/api/runs/{run_id}/train")
def train(run_id: str, x_user: str | None = Header(None)):
    return training.start_training(run_id, _user(x_user))


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    return training.cancel(run_id)


@app.post("/api/runs/{run_id}/validate")
def validate(run_id: str, x_user: str | None = Header(None)):
    return training.validate(run_id, _user(x_user), blocking=False)


@app.get("/api/runs/{run_id}/validation-sheet")
def validation_sheet(run_id: str, format: str = "json"):
    if format == "csv":
        p = training.run_dir(run_id) / "validation_sheet.csv"
        if not p.exists():
            raise NotFoundError("validation sheet not found")
        return PlainTextResponse(
            p.read_text(encoding="utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="validation_sheet_{run_id}.csv"'},
        )
    return training.get_validation_sheet(run_id)


@app.post("/api/runs/{run_id}/release")
def release(run_id: str, body: dict = Body(default={}), x_user: str | None = Header(None)):
    return training.release(run_id, _user(x_user), body.get("note", ""), body.get("min_field_accuracy"))


@app.get("/api/releases")
def releases():
    return training.list_releases()


# ---------------- BUC 3: extraction ----------------
@app.post("/api/extractions")
async def extract(
    file: UploadFile = File(...), customer_layout: str = Form(""), x_user: str | None = Header(None)
):
    return extraction.extract(file.filename or "", await _read(file), customer_layout, _user(x_user))


@app.get("/api/extractions")
def extractions():
    return extraction.list_extractions()


@app.get("/api/extractions/{extraction_id}")
def get_extraction(extraction_id: str):
    return extraction.get_extraction(extraction_id)


@app.get("/api/extractions/{extraction_id}/pages/{name}")
def extraction_page(extraction_id: str, name: str):
    return FileResponse(extraction.page_image(extraction_id, f"pages/{name}"))


@app.post("/api/extractions/{extraction_id}/save")
def save_extraction(extraction_id: str, body: dict[str, Any] = Body(default={}), x_user: str | None = Header(None)):
    return extraction.save(extraction_id, body.get("data"), _user(x_user))


@app.get("/api/extractions/{extraction_id}/json")
def download_json(extraction_id: str):
    e = extraction.get_extraction(extraction_id)
    if e["status"] != "saved":
        raise HTTPException(409, "Save the extraction first.")
    name = str(e["data"].get("po_number") or extraction_id).replace('"', "").replace("/", "-")
    return Response(
        json.dumps(e["data"], ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="PO_{name}.json"'},
    )


app.mount("/", StaticFiles(directory=config.ROOT_DIR / "static", html=True), name="static")
