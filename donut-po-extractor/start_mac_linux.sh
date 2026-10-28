#!/usr/bin/env bash
# Customer PO Extraction (Donut) - start on macOS / Linux.
# First run: creates .venv and installs libraries (10-20 minutes). Next runs: starts in seconds.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null; then
  echo "Python 3 belum terpasang. Install Python 3.11 atau 3.12 dari https://www.python.org/downloads/" >&2
  exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "[1/2] Membuat virtual environment..."
  "$PY" -m venv .venv
  echo "[2/2] Memasang library, mohon tunggu..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

export DONUT_PO_DATA_DIR=${DONUT_PO_DATA_DIR:-"$PWD/data"}
PORT=${PORT:-8000}
echo
echo "Folder data : $DONUT_PO_DATA_DIR"
echo "Aplikasi    : http://localhost:$PORT   (Ctrl+C untuk berhenti)"
echo
(sleep 3; (command -v open >/dev/null && open "http://localhost:$PORT") || (command -v xdg-open >/dev/null && xdg-open "http://localhost:$PORT") || true) >/dev/null 2>&1 &
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
