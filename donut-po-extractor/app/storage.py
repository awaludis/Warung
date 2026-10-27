"""Small file-based storage helpers (JSON documents on disk, no database needed)."""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BusinessRuleError(ValueError):
    """Raised when a request would break a business rule from the BUC."""


class NotFoundError(LookupError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise NotFoundError(f"{path.name} not found")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_id(value: str) -> str:
    """Reject ids that could escape the data folder."""
    if not value or not all(c.isalnum() or c in "-_." for c in value) or ".." in value:
        raise NotFoundError(f"invalid id: {value!r}")
    return value
