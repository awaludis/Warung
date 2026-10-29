"""BUC 2 — Train and Validate PO Extraction Model (actor: AI Developer).

Flow: receive a filed PO Ground Truth Record -> split into training / validation sets -> fine-tune
Donut on the training set -> run the model on the validation set and compare with the correct values
-> record validation results per field and per customer layout -> release the validated model.

Business rules:
  * validation uses only samples that were not used for training (disjoint split, checked again
    before validating),
  * validation results must be recorded before the model can be released,
  * a new customer layout or a changed field definition sheet requires new samples and retraining.
"""
from __future__ import annotations

import csv
import os
import random
import threading
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from . import config, ground_truth
from .field_definition import FieldDefinition, load_definition, parse_number
from .storage import BusinessRuleError, NotFoundError, new_id, now_iso, read_json, safe_id, write_json

# run status values
CREATED, TRAINING, TRAINED, VALIDATING, VALIDATED, FAILED, CANCELLED = (
    "created", "training", "trained", "validating", "validated", "failed", "cancelled",
)
_BUSY = {TRAINING, VALIDATING}

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

DEFAULT_PARAMS = {
    "validation_ratio": 0.2,
    "seed": 42,
    "epochs": 30,
    "learning_rate": 3e-5,
    "batch_size": 1,
    "grad_accum": 1,
    "image_height": None,
    "image_width": None,
    "max_length": None,
    "base_model": None,
    "gradient_checkpointing": False,
    "auto_validate": True,
}


# ---------------- paths / persistence ----------------
def run_dir(run_id: str) -> Path:
    return config.paths()["runs"] / safe_id(run_id)


def get_run(run_id: str) -> dict:
    run = read_json(run_dir(run_id) / "run.json")
    if run["status"] in _BUSY and not _job_alive(run_id, run.get("pid")):  # app restarted mid-job
        run.update(status=FAILED, error="Interrupted (application restarted)")
        _save(run)
    return run


def _job_alive(run_id: str, pid: int | None) -> bool:
    """A job may run in this process (web app) or another one (CLI) sharing the data folder."""
    if pid is None or pid == os.getpid():
        return run_id in _jobs
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _save(run: dict) -> None:
    write_json(run_dir(run["run_id"]) / "run.json", run)


def _update(run_id: str, **kw) -> dict:
    run = read_json(run_dir(run_id) / "run.json")
    run.update(kw)
    _save(run)
    return run


def _log(run_id: str, msg: str) -> None:
    with open(run_dir(run_id) / "train.log", "a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {msg}\n")


def _sample_name(sample_id: str) -> str:
    """Filename of a sample (records filed before source_file existed); the id if it was deleted."""
    try:
        return ground_truth.get_sample(sample_id)["filename"]
    except NotFoundError:
        return f"{sample_id} (sampel sudah dihapus)"


def split_details(run: dict) -> list[dict]:
    """Which document / page went to the training set and which to the validation set."""
    try:
        record = ground_truth.get_record(run["record_id"])
    except NotFoundError:
        return []
    by_id = {e["entry_id"]: e for e in record["entries"]}
    rows = []
    for which, label in (("train", "training"), ("validation", "validation")):
        for eid in run.get("split", {}).get(which, []):
            e = by_id.get(eid, {})
            rows.append({
                "set": label,
                "entry_id": eid,
                "customer_layout": e.get("customer_layout", ""),
                "source_file": e.get("source_file") or _sample_name(eid.rsplit("-P", 1)[0]),
                "page": e.get("page") or int(eid.rsplit("-P", 1)[-1]) if "-P" in eid else e.get("page"),
            })
    return sorted(rows, key=lambda r: (r["customer_layout"], r["set"] != "validation", r["source_file"], r["page"] or 0))


def read_log(run_id: str, tail: int = 200) -> str:
    p = run_dir(run_id) / "train.log"
    return "\n".join(p.read_text(encoding="utf-8").splitlines()[-tail:]) if p.exists() else ""


def list_runs() -> list[dict]:
    out = []
    for f in sorted(config.paths()["runs"].glob("*/run.json"), reverse=True):
        r = get_run(f.parent.name)
        r.pop("split", None)
        out.append(r)
    return out


# ---------------- step 2: split ----------------
def split_entries(entries: list[dict], validation_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Stratified per customer layout so every layout appears in both sets when it has >= 2 pages."""
    if not 0 < validation_ratio < 1:
        raise BusinessRuleError("validation_ratio must be between 0 and 1")
    by_layout: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_layout[e["customer_layout"]].append(e["entry_id"])
    rng = random.Random(seed)
    train, val = [], []
    for layout in sorted(by_layout):
        ids = sorted(by_layout[layout])
        rng.shuffle(ids)
        n_val = 0 if len(ids) < 2 else min(len(ids) - 1, max(1, round(len(ids) * validation_ratio)))
        val += ids[:n_val]
        train += ids[n_val:]
    if not val:
        raise BusinessRuleError("Not enough samples to build a validation set (need >= 2 pages for a layout).")
    return train, val


def create_run(record_id: str, params: dict | None = None, user: str = "") -> dict:
    """Steps 1-2: receive the ground truth record and separate training / validation sets."""
    record = ground_truth.get_record(record_id)
    fd = load_definition()
    if record["field_definition_hash"] != fd.hash:
        raise BusinessRuleError(
            "This ground truth record was annotated with a different PO Data Field Definition Sheet. "
            "Re-annotate the samples and file a new record."
        )
    unknown = set(params or {}) - set(DEFAULT_PARAMS)
    if unknown:
        raise BusinessRuleError(f"Unknown training parameters: {', '.join(sorted(unknown))}")
    p = {**DEFAULT_PARAMS, **{k: v for k, v in (params or {}).items() if v is not None}}
    train_ids, val_ids = split_entries(record["entries"], float(p["validation_ratio"]), int(p["seed"]))
    layouts = defaultdict(lambda: {"train": 0, "validation": 0})
    by_id = {e["entry_id"]: e for e in record["entries"]}
    for i in train_ids:
        layouts[by_id[i]["customer_layout"]]["train"] += 1
    for i in val_ids:
        layouts[by_id[i]["customer_layout"]]["validation"] += 1
    run_id = new_id("RUN")
    run_dir(run_id).mkdir(parents=True)
    run = {
        "run_id": run_id,
        "record_id": record_id,
        "status": CREATED,
        "created_at": now_iso(),
        "created_by": user,
        "params": p,
        "field_definition_hash": fd.hash,
        "field_definition_version": fd.version,
        "layouts": dict(layouts),
        "trained_layouts": sorted(l for l, c in layouts.items() if c["train"]),
        "split": {"train": train_ids, "validation": val_ids},
        "progress": 0.0,
        "released": False,
    }
    _save(run)
    _log(run_id, f"Split {len(train_ids)} training / {len(val_ids)} validation pages from {record_id}")
    return run


def _items(run: dict, which: str):
    from .donut_model import TrainItem

    record = ground_truth.get_record(run["record_id"])
    rdir = ground_truth.record_dir(run["record_id"])
    by_id = {e["entry_id"]: e for e in record["entries"]}
    return [(by_id[i], TrainItem(image=rdir / by_id[i]["image"], values=by_id[i]["values"])) for i in run["split"][which]]


# ---------------- step 3: train ----------------
def start_training(run_id: str, user: str = "", blocking: bool = False) -> dict:
    run = get_run(run_id)
    if run["status"] not in (CREATED, FAILED, CANCELLED):
        raise BusinessRuleError(f"Run is '{run['status']}', it cannot be (re)trained.")
    with _jobs_lock:
        if any(j["thread"].is_alive() for j in _jobs.values()):
            raise BusinessRuleError("Another training / validation job is running. Wait for it to finish.")
        stop = threading.Event()
        _update(run_id, status=TRAINING, started_at=now_iso(), started_by=user, progress=0.0, error=None, pid=os.getpid())
        t = threading.Thread(target=_train_job, args=(run_id, stop, user), daemon=True)
        _jobs[run_id] = {"thread": t, "stop": stop}
    t.start()
    if blocking:
        t.join()
    return get_run(run_id)


def _train_job(run_id: str, stop: threading.Event, user: str) -> None:
    from . import donut_model

    try:
        run = get_run(run_id)
        p = run["params"]
        items = [it for _, it in _items(run, "train")]
        size = None
        if p.get("image_height") and p.get("image_width"):
            size = (int(p["image_height"]), int(p["image_width"]))
        donut_model.train(
            items,
            load_definition(),
            run_dir(run_id) / "model",
            base_model=p.get("base_model"),
            epochs=int(p["epochs"]),
            learning_rate=float(p["learning_rate"]),
            batch_size=int(p["batch_size"]),
            grad_accum=int(p["grad_accum"]),
            image_size=size,
            max_length=p.get("max_length"),
            seed=int(p["seed"]),
            gradient_checkpointing=bool(p.get("gradient_checkpointing")),
            log=lambda m: _log(run_id, m),
            progress=lambda x: _update(run_id, progress=round(x, 3)),
            stop_event=stop,
        )
        _update(run_id, status=TRAINED, trained_at=now_iso(), progress=1.0)
        if p.get("auto_validate"):
            _validate(run_id, user)
    except InterruptedError:
        _log(run_id, "Training cancelled")
        _update(run_id, status=CANCELLED)
    except Exception as exc:  # noqa: BLE001
        _log(run_id, "ERROR " + traceback.format_exc())
        _update(run_id, status=FAILED, error=str(exc))
    finally:
        with _jobs_lock:
            _jobs.pop(run_id, None)


def cancel(run_id: str) -> dict:
    job = _jobs.get(run_id)
    if not job:
        raise BusinessRuleError("This run has no active job.")
    job["stop"].set()
    return get_run(run_id)


# ---------------- steps 4-5: validate ----------------
def _norm_text(v: Any) -> str:
    return " ".join(str(v).split()).casefold() if v not in (None, "") else ""


def values_match(expected: Any, predicted: Any, ftype: str) -> bool:
    if ftype == "number":
        def num(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
            return parse_number(str(v)) if v not in (None, "") else None

        e, p = num(expected), num(predicted)
        if e is not None or p is not None:
            return e is not None and p is not None and abs(e - p) < 1e-6
    return _norm_text(expected) == _norm_text(predicted)


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def similarity(expected: Any, predicted: Any) -> float:
    """1 - normalised edit distance between the two values as text (1.0 = identical)."""
    a, b = _norm_text(expected), _norm_text(predicted)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return max(0.0, 1 - prev[-1] / max(len(a), len(b)))


def _score(expected: Any, predicted: Any, ftype: str) -> dict:
    """Pairs where both sides are empty are not counted (the field is simply not on this page)."""
    if _empty(expected) and _empty(predicted):
        return {"correct": 0, "total": 0, "sim": 0.0}
    ok = values_match(expected, predicted, ftype)
    return {"correct": int(ok), "total": 1, "sim": 1.0 if ok else similarity(expected, predicted)}


def _add(into: dict, c: dict) -> None:
    for k in ("correct", "total", "sim"):
        into[k] = into.get(k, 0) + c[k]


def compare(fd: FieldDefinition, expected: dict, predicted: dict) -> dict[str, dict]:
    """Per field path -> {"correct", "total", "sim"}. List sub-fields are compared row by row."""
    result: dict[str, dict] = {}
    for f in fd.fields:
        if f.is_list:
            exp_rows, pred_rows = expected.get(f.name) or [], predicted.get(f.name) or []
            rows = max(len(exp_rows), len(pred_rows))
            for s in f.fields:
                acc = {"correct": 0, "total": 0, "sim": 0.0}
                for i in range(rows):
                    _add(acc, _score(
                        exp_rows[i].get(s.name) if i < len(exp_rows) else None,
                        pred_rows[i].get(s.name) if i < len(pred_rows) else None,
                        s.type,
                    ))
                result[f"{f.name}.{s.name}"] = acc
            same = int(len(exp_rows) == len(pred_rows))
            result[f"{f.name}.row_count"] = {"correct": same, "total": 1, "sim": float(same)}
        else:
            result[f.name] = _score(expected.get(f.name), predicted.get(f.name), f.type)
    return result


def _stats(c: dict) -> dict:
    t = c.get("total", 0)
    return {
        "correct": c.get("correct", 0),
        "total": t,
        "accuracy": round(c["correct"] / t, 4) if t else None,
        "similarity": round(c.get("sim", 0) / t, 4) if t else None,
    }


def validate(run_id: str, user: str = "", blocking: bool = True) -> dict:
    run = get_run(run_id)
    if run["status"] not in (TRAINED, VALIDATED):
        raise BusinessRuleError(f"Run is '{run['status']}'; it must be trained before validation.")
    if not blocking:
        with _jobs_lock:
            if any(j["thread"].is_alive() for j in _jobs.values()):
                raise BusinessRuleError("Another job is running.")
            t = threading.Thread(target=_validate_job, args=(run_id, user), daemon=True)
            _jobs[run_id] = {"thread": t, "stop": threading.Event()}
        _update(run_id, status=VALIDATING, pid=os.getpid())
        t.start()
        return get_run(run_id)
    return _validate(run_id, user)


def _validate_job(run_id: str, user: str) -> None:
    try:
        _validate(run_id, user)
    except Exception as exc:  # noqa: BLE001
        _log(run_id, "ERROR " + traceback.format_exc())
        _update(run_id, status=TRAINED, error=f"Validation failed: {exc}")
    finally:
        with _jobs_lock:
            _jobs.pop(run_id, None)


def _validate(run_id: str, user: str, extractor_factory: Callable | None = None) -> dict:
    from .donut_model import DonutExtractor

    run = get_run(run_id)
    overlap = set(run["split"]["train"]) & set(run["split"]["validation"])
    if overlap:  # business rule: validation never uses training samples
        raise BusinessRuleError(f"Validation set contains training samples: {sorted(overlap)}")
    _update(run_id, status=VALIDATING, pid=os.getpid())
    _log(run_id, f"Validating on {len(run['split']['validation'])} pages")
    fd = load_definition()
    extractor = (extractor_factory or DonutExtractor)(run_dir(run_id) / "model")

    per_field: dict[str, dict] = defaultdict(dict)
    per_layout: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    layout_docs: dict[str, dict] = defaultdict(lambda: {"exact": 0, "total": 0})
    details = []
    for entry, item in _items(run, "validation"):
        raw, seq = extractor.read_page(item.image)
        predicted = fd.normalize_output(raw)
        expected = fd.normalize_output(entry["values"])
        cmp = compare(fd, expected, predicted)
        exact = all(c["correct"] == c["total"] for c in cmp.values())
        layout = entry["customer_layout"]
        for k, c in cmp.items():
            _add(per_field[k], c)
            _add(per_layout[layout][k], c)
        layout_docs[layout]["exact"] += int(exact)
        layout_docs[layout]["total"] += 1
        details.append(
            {
                "entry_id": entry["entry_id"],
                "customer_layout": layout,
                "exact_match": exact,
                "mismatched_fields": [k for k, c in cmp.items() if c["correct"] != c["total"]],
                "expected": expected,
                "predicted": predicted,
                "raw_sequence": seq,
            }
        )

    def summary(counts: dict) -> dict:
        tot: dict = {}
        for c in counts.values():
            _add(tot, c)
        st = _stats(tot)
        return {"field_accuracy": st["accuracy"], "similarity": st["similarity"], "correct": st["correct"], "total": st["total"]}

    docs_total = sum(d["total"] for d in layout_docs.values())
    sheet = {
        "sheet": "Model Validation Sheet",
        "status": "completed",
        "run_id": run_id,
        "record_id": run["record_id"],
        "validated_at": now_iso(),
        "validated_by": user,
        "field_definition_hash": fd.hash,
        "validation_pages": len(details),
        "overall": {
            **summary(per_field),
            "document_exact_match": round(sum(d["exact"] for d in layout_docs.values()) / docs_total, 4) if docs_total else None,
        },
        "per_field": {k: _stats(c) for k, c in per_field.items()},
        "per_layout": {
            layout: {
                **summary(counts),
                "pages": layout_docs[layout]["total"],
                "document_exact_match": round(layout_docs[layout]["exact"] / layout_docs[layout]["total"], 4),
                "fields": {k: _stats(c) for k, c in counts.items()},
            }
            for layout, counts in per_layout.items()
        },
        "details": details,
    }
    d = run_dir(run_id)
    write_json(d / "validation_sheet.json", sheet)
    _write_sheet_csv(d / "validation_sheet.csv", sheet)
    _log(run_id, f"Validation recorded: field accuracy {sheet['overall']['field_accuracy']}, "
                 f"character similarity {sheet['overall']['similarity']}")
    weakest = sorted((c["accuracy"], k) for k, c in sheet["per_field"].items() if c["accuracy"] is not None)[:3]
    if weakest:
        _log(run_id, "Weakest fields: " + ", ".join(f"{k} {a:.0%}" for a, k in weakest))
    _update(run_id, status=VALIDATED, validated_at=sheet["validated_at"], validation_summary=sheet["overall"], error=None)
    return sheet


def _write_sheet_csv(path: Path, sheet: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["customer_layout", "field", "correct", "total", "accuracy", "similarity"])
        for k, c in sheet["per_field"].items():
            w.writerow(["(all layouts)", k, c["correct"], c["total"], c["accuracy"], c["similarity"]])
        for layout, data in sheet["per_layout"].items():
            for k, c in data["fields"].items():
                w.writerow([layout, k, c["correct"], c["total"], c["accuracy"], c["similarity"]])


def get_validation_sheet(run_id: str) -> dict:
    return read_json(run_dir(run_id) / "validation_sheet.json")


# ---------------- step 6: release ----------------
def release(run_id: str, user: str = "", note: str = "", min_field_accuracy: float | None = None) -> dict:
    run = get_run(run_id)
    if run["status"] != VALIDATED or not (run_dir(run_id) / "validation_sheet.json").exists():
        raise BusinessRuleError("Validation results must be recorded before the model can be released.")
    fd = load_definition()
    if run["field_definition_hash"] != fd.hash:
        raise BusinessRuleError("The PO Data Field Definition Sheet changed after this model was trained — retrain first.")
    sheet = get_validation_sheet(run_id)
    acc = sheet["overall"]["field_accuracy"] or 0
    if min_field_accuracy is not None and acc < min_field_accuracy:
        raise BusinessRuleError(f"Field accuracy {acc:.2%} is below the required {min_field_accuracy:.2%}.")
    rel_dir = config.paths()["releases"]
    previous = current_release()
    release_id = new_id("REL")
    rec = {
        "record": "Model Release Record",
        "release_id": release_id,
        "run_id": run_id,
        "model_path": str((run_dir(run_id) / "model").relative_to(config.paths()["runs"].parent)),
        "ground_truth_record_id": run["record_id"],
        "field_definition_hash": run["field_definition_hash"],
        "field_definition_version": run["field_definition_version"],
        "trained_layouts": run["trained_layouts"],
        "validation_summary": sheet["overall"],
        "validation_per_layout": {k: v["field_accuracy"] for k, v in sheet["per_layout"].items()},
        "released_at": now_iso(),
        "released_by": user,
        "note": note,
        "supersedes": previous["release_id"] if previous else None,
    }
    write_json(rel_dir / f"{release_id}.json", rec)
    write_json(rel_dir / "current.json", {"release_id": release_id})
    _update(run_id, released=True, release_id=release_id)
    return rec


def current_release() -> dict | None:
    rel_dir = config.paths()["releases"]
    try:
        cur = read_json(rel_dir / "current.json")
        return read_json(rel_dir / f"{safe_id(cur['release_id'])}.json")
    except NotFoundError:
        # current.json missing (e.g. only the release record was copied): use the newest release record
        records = sorted(rel_dir.glob("REL-*.json"))
        return read_json(records[-1]) if records else None


def list_releases() -> list[dict]:
    return [read_json(f) for f in sorted(config.paths()["releases"].glob("REL-*.json"), reverse=True)]


def released_model_dir(rel: dict) -> Path:
    return config.paths()["runs"].parent / rel["model_path"]


def model_status() -> dict:
    """Is the released model still valid for the current sheet and known layouts? (BUC 2, rule 3)."""
    fd = load_definition()
    rel = current_release()
    reasons = []
    model_missing = False
    if rel is None:
        reasons.append("No model has been released yet.")
    elif not (released_model_dir(rel) / "config.json").exists():
        model_missing = True
        reasons.append(f"Model files of {rel['release_id']} not found at {released_model_dir(rel)}.")
    else:
        if rel["field_definition_hash"] != fd.hash:
            reasons.append("The PO Data Field Definition Sheet changed after the model was trained.")
        known = set(rel["trained_layouts"])
        new_layouts = sorted({e["customer_layout"] for e in ground_truth.list_entries()} - known)
        if new_layouts:
            reasons.append(f"New customer layouts not in the released model: {', '.join(new_layouts)}.")
    return {
        "release": rel,
        "model_missing": model_missing,
        "retraining_required": bool(rel) and not model_missing and bool(reasons),
        "reasons": reasons,
    }
