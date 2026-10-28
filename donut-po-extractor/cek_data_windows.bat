@echo off
REM Cek folder data: apakah model hasil Colab sudah di tempat yang benar.
cd /d "%~dp0"
if "%DONUT_PO_DATA_DIR%"=="" set "DONUT_PO_DATA_DIR=%~dp0data"
if not exist ".venv\Scripts\python.exe" (
  echo Jalankan start_windows.bat dulu untuk memasang aplikasi.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.cli doctor
echo.
pause
