"""Command line for long jobs (e.g. training on a GPU server without the web UI).

  python -m app.cli records
  python -m app.cli import-samples practice_docs/training [--draft]
  python -m app.cli train --record GTR-... [--epochs 30] [--release]
  python -m app.cli validate --run RUN-...
  python -m app.cli release --run RUN-... [--min-accuracy 0.9]
  python -m app.cli extract path/to/po.pdf [--layout "PT Maju"] [--save]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import extraction, ground_truth, training

INT_PARAMS = ("epochs", "batch_size", "grad_accum", "image_height", "image_width", "max_length", "seed")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.cli")
    ap.add_argument("--user", default="cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("records")
    im = sub.add_parser("import-samples", help="load PO files + answer_key.json as ground truth entries")
    im.add_argument("folder", type=Path)
    im.add_argument("--draft", action="store_true", help="import values as drafts to review in the UI")
    t = sub.add_parser("train")
    t.add_argument("--record", required=True)
    for k in INT_PARAMS:
        t.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int)
    t.add_argument("--learning-rate", type=float)
    t.add_argument("--validation-ratio", type=float)
    t.add_argument("--base-model")
    t.add_argument("--release", action="store_true", help="release right after validation")
    v = sub.add_parser("validate")
    v.add_argument("--run", required=True)
    r = sub.add_parser("release")
    r.add_argument("--run", required=True)
    r.add_argument("--min-accuracy", type=float)
    r.add_argument("--note", default="")
    e = sub.add_parser("extract")
    e.add_argument("file", type=Path)
    e.add_argument("--layout", default="")
    e.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "records":
        print(json.dumps(ground_truth.list_records(), indent=2))
    elif a.cmd == "import-samples":
        import_samples(a.folder, not a.draft, a.user)
    elif a.cmd == "train":
        keys = INT_PARAMS + ("learning_rate", "validation_ratio", "base_model")
        params = {k: getattr(a, k) for k in keys if getattr(a, k) is not None}
        run = training.create_run(a.record, params, a.user)
        print(f"Run {run['run_id']}: {len(run['split']['train'])} train / {len(run['split']['validation'])} validation pages")
        run = training.start_training(run["run_id"], a.user, blocking=True)
        print(training.read_log(run["run_id"]))
        print(json.dumps({k: run.get(k) for k in ("run_id", "status", "validation_summary", "error")}, indent=2))
        if a.release and run["status"] == training.VALIDATED:
            print(json.dumps(training.release(run["run_id"], a.user), indent=2))
    elif a.cmd == "validate":
        sheet = training.validate(a.run, a.user)
        print(json.dumps({"overall": sheet["overall"], "per_field": sheet["per_field"]}, indent=2))
    elif a.cmd == "release":
        print(json.dumps(training.release(a.run, a.user, a.note, a.min_accuracy), indent=2))
    elif a.cmd == "extract":
        res = extraction.extract(a.file.name, a.file.read_bytes(), a.layout, a.user)
        if a.save:
            res = extraction.save(res["extraction_id"], None, a.user)
        for w in res["warnings"]:
            print("WARNING:", w)
        print(json.dumps(res["data"], indent=2, ensure_ascii=False))


def import_samples(folder: Path, complete: bool, user: str) -> None:
    """Upload every document listed in answer_key.json and record its answers (BUC 1 shortcut for practice)."""
    key = json.loads((folder / "answer_key.json").read_text(encoding="utf-8"))
    pages = 0
    for doc in key:
        path = folder / doc["file"]
        sample = ground_truth.create_sample(path.name, path.read_bytes(), doc["customer_layout"], user)
        if len(sample["pages"]) != len(doc["pages"]):
            raise SystemExit(f"{doc['file']}: {len(sample['pages'])} pages but answer key has {len(doc['pages'])}")
        for pg, values in zip(sample["pages"], doc["pages"]):
            ground_truth.save_entry(pg["entry_id"], values, complete, user)
            pages += 1
    print(f"Imported {len(key)} documents / {pages} pages as {'completed' if complete else 'draft'} entries")
    for layout, row in ground_truth.coverage()["layouts"].items():
        print(f"  {layout}: {row['completed']} completed, {row['draft']} draft")


if __name__ == "__main__":
    main()
