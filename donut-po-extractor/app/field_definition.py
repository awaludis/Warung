"""PO Data Field Definition Sheet.

The sheet (config/po_field_definition.json) is the single source of truth for:
  * which fields the Data Annotator may record (BUC 1),
  * the Donut special tokens and target sequence used in training (BUC 2),
  * the JSON structure returned by extraction (BUC 3).

Field types: "string", "date", "number" (leaf values) and "list" (repeating rows with sub-fields).
Ground truth values are recorded as text exactly as printed on the PO; the extraction output
converts "number" fields to JSON numbers.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import config
from .storage import BusinessRuleError

LEAF_TYPES = {"string", "date", "number"}
SEP_TOKEN = "<sep/>"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    type: str = "string"
    required: bool = False
    fields: tuple["Field", ...] = field(default_factory=tuple)

    @property
    def is_list(self) -> bool:
        return self.type == "list"


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    version: str
    task_token: str
    fields: tuple[Field, ...]
    raw: dict
    hash: str

    # ---------- Donut tokens ----------
    def special_tokens(self) -> list[str]:
        tokens = [self.task_token, SEP_TOKEN]
        for f in self.fields:
            tokens += [f"<s_{f.name}>", f"</s_{f.name}>"]
            for sub in f.fields:
                tokens += [f"<s_{sub.name}>", f"</s_{sub.name}>"]
        # keep order, drop duplicates (a sub-field may share a name with a top-level field)
        return list(dict.fromkeys(tokens))

    def to_token_sequence(self, values: dict) -> str:
        """Ground truth dict -> Donut target sequence (without task token / eos)."""
        out = []
        for f in self.fields:
            v = values.get(f.name)
            if f.is_list:
                rows = v or []
                if not rows:
                    continue
                body = SEP_TOKEN.join(
                    "".join(
                        f"<s_{s.name}>{row.get(s.name, '')}</s_{s.name}>"
                        for s in f.fields
                        if row.get(s.name, "") != ""
                    )
                    for row in rows
                )
                out.append(f"<s_{f.name}>{body}</s_{f.name}>")
            elif v not in (None, ""):
                out.append(f"<s_{f.name}>{v}</s_{f.name}>")
        return "".join(out)

    # ---------- BUC 1: strict validation of annotator input ----------
    def validate_annotation(self, values: dict) -> dict:
        """Return a clean copy of the annotation or raise BusinessRuleError.

        Rule: only fields listed in the sheet may be annotated, in the sheet's structure.
        """
        if not isinstance(values, dict):
            raise BusinessRuleError("Field values must be an object")
        known = {f.name: f for f in self.fields}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise BusinessRuleError(f"Fields not in the PO Data Field Definition Sheet: {', '.join(unknown)}")
        clean: dict[str, Any] = {}
        for f in self.fields:
            v = values.get(f.name)
            if f.is_list:
                rows = v or []
                if not isinstance(rows, list):
                    raise BusinessRuleError(f"'{f.name}' must be a list of rows")
                sub_known = {s.name for s in f.fields}
                clean_rows = []
                for i, row in enumerate(rows, 1):
                    if not isinstance(row, dict):
                        raise BusinessRuleError(f"'{f.name}' row {i} must be an object")
                    extra = sorted(set(row) - sub_known)
                    if extra:
                        raise BusinessRuleError(
                            f"'{f.name}' row {i} has fields not in the definition sheet: {', '.join(extra)}"
                        )
                    clean_row = {s.name: _clean_leaf(row.get(s.name), s, f"{f.name}[{i}].{s.name}") for s in f.fields}
                    if any(clean_row.values()):
                        clean_rows.append(clean_row)
                clean[f.name] = clean_rows
            else:
                clean[f.name] = _clean_leaf(v, f, f.name)
        return clean

    def missing_required(self, values: dict) -> list[str]:
        missing = []
        for f in self.fields:
            v = values.get(f.name)
            if f.is_list:
                if f.required and not v:
                    missing.append(f.name)
                for i, row in enumerate(v or [], 1):
                    missing += [f"{f.name}[{i}].{s.name}" for s in f.fields if s.required and not row.get(s.name)]
            elif f.required and not v:
                missing.append(f.name)
        return missing

    # ---------- BUC 3: lenient normalisation of model output ----------
    def normalize_output(self, data: Any) -> dict:
        """Coerce whatever Donut produced into exactly the sheet's JSON structure.

        Unknown keys are dropped, missing fields become null / [], numbers become JSON numbers.
        """
        data = data if isinstance(data, dict) else {}
        out: dict[str, Any] = {}
        for f in self.fields:
            v = data.get(f.name)
            if f.is_list:
                if isinstance(v, dict):
                    v = [v]
                rows = [r for r in (v or []) if isinstance(r, dict)] if isinstance(v, list) else []
                out[f.name] = [{s.name: _typed_leaf(r.get(s.name), s) for s in f.fields} for r in rows]
            else:
                out[f.name] = _typed_leaf(v, f)
        return out

    def empty_record(self) -> dict:
        return {f.name: ([] if f.is_list else None) for f in self.fields}

    def to_public(self) -> dict:
        return {**self.raw, "hash": self.hash}


def _clean_leaf(value: Any, f: Field, path: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        raise BusinessRuleError(f"'{path}' must be a single value")
    text = " ".join(str(value).split())
    if text and f.type == "number" and parse_number(text) is None:
        raise BusinessRuleError(f"'{path}' must be a number (got {text!r})")
    return text


def _typed_leaf(value: Any, f: Field) -> Any:
    if isinstance(value, list):  # Donut may split a leaf on <sep/>
        value = " ".join(str(x) for x in value)
    if isinstance(value, dict) or value is None:
        return None
    text = " ".join(str(value).split())
    if text == "":
        return None
    if f.type == "number":
        num = parse_number(text)
        return num if num is not None else text
    return text


def parse_number(text: str) -> float | int | None:
    """Parse numbers written as 1,234.56 / 1.234,56 / 1234 / Rp 12.500 / (100)."""
    s = str(text).strip()
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^0-9.,]", "", s)
    if not s or not any(c.isdigit() for c in s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        s = s.replace(thou, "").replace(dec, ".")
    else:
        sep = "," if "," in s else "." if "." in s else None
        if sep:
            parts = s.split(sep)
            if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and parts[0] not in ("", "0")):
                s = s.replace(sep, "")  # thousands separator
            else:
                s = s.replace(sep, ".")
    try:
        num = float(s)
    except ValueError:
        return None
    num = -num if negative else num
    return int(num) if num.is_integer() else num


def _parse_field(d: dict, nested: bool = False) -> Field:
    name = d.get("name", "")
    if not _NAME_RE.match(name):
        raise ValueError(f"Invalid field name {name!r}: use lowercase letters, digits and _")
    ftype = d.get("type", "string")
    if ftype not in LEAF_TYPES | {"list"}:
        raise ValueError(f"Field {name!r}: unknown type {ftype!r}")
    subs: tuple[Field, ...] = ()
    if ftype == "list":
        if nested:
            raise ValueError(f"Field {name!r}: lists cannot be nested")
        subs = tuple(_parse_field(s, nested=True) for s in d.get("fields", []))
        if not subs:
            raise ValueError(f"List field {name!r} needs sub-fields")
    return Field(name=name, label=d.get("label", name), type=ftype, required=bool(d.get("required")), fields=subs)


def parse_definition(raw: dict) -> FieldDefinition:
    fields = tuple(_parse_field(f) for f in raw.get("fields", []))
    if not fields:
        raise ValueError("Field definition sheet has no fields")
    names = [f.name for f in fields]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate field names in definition sheet")
    canonical = json.dumps({"task_token": raw.get("task_token"), "fields": raw.get("fields")}, sort_keys=True)
    return FieldDefinition(
        name=raw.get("name", "PO Data Field Definition Sheet"),
        version=str(raw.get("version", "1")),
        task_token=raw.get("task_token", "<s_po>"),
        fields=fields,
        raw=raw,
        hash=hashlib.sha256(canonical.encode()).hexdigest()[:16],
    )


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> FieldDefinition:
    with open(path, encoding="utf-8") as f:
        return parse_definition(json.load(f))


def load_definition(path: Path | None = None) -> FieldDefinition:
    p = Path(path or config.FIELD_DEFINITION_PATH)
    return _load_cached(str(p), p.stat().st_mtime)
