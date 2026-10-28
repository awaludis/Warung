@echo off
REM Customer PO Extraction (Donut) - start on Windows.
REM First run: creates .venv and installs libraries (10-20 minutes). Next runs: starts in seconds.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python belum terpasang. Install Python 3.11 atau 3.12 dari https://www.python.org/downloads/
  echo dan centang "Add python.exe to PATH" saat instalasi.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Membuat virtual environment...
  python -m venv .venv || goto :error
  echo [2/2] Memasang library, mohon tunggu...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
) else (
  echo Memeriksa library...
  ".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt || goto :error
)

if "%DONUT_PO_DATA_DIR%"=="" set "DONUT_PO_DATA_DIR=%~dp0data"
if "%PORT%"=="" set "PORT=8000"
echo.
echo Folder data : %DONUT_PO_DATA_DIR%
echo Aplikasi    : http://localhost:%PORT%   (tutup jendela ini untuk menghentikan)
echo.
start "" "http://localhost:%PORT%"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
goto :eof

:error
echo.
echo Instalasi gagal. Salin pesan error di atas untuk dicek.
pause
exit /b 1
