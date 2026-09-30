@echo off
rem ============================================================
rem  IMPORTANT: keep this file ASCII-only.
rem  Chinese text inside a .bat breaks cmd.exe parsing
rem  (it may even execute the text as a command).
rem  All Chinese instructions live in README.md instead.
rem
rem  NOTE: this file must keep CRLF line endings.
rem  .gitattributes pins "*.bat text eol=crlf" so git never
rem  converts it to LF (LF-only .bat files break cmd parsing).
rem ============================================================
cd /d "%~dp0"
title RAG Knowledge Base Demo

echo.
echo  [1/2] Checking Python environment...
echo.

set "PY="

rem Prefer the project's .venv, but only if dependencies are really installed there
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import fastapi, uvicorn" >nul 2>&1 && set "PY=.venv\Scripts\python.exe"
)

rem Otherwise fall back to the python on PATH
if not defined PY (
    python -c "import fastapi, uvicorn" >nul 2>&1 && set "PY=python"
)

if not defined PY (
    echo.
    echo  [ERROR] No Python with dependencies was found.
    echo.
    echo  Please run this once in this folder:
    echo      pip install -r requirements.txt
    echo.
    echo  If you are using the .venv environment, activate it first:
    echo      .venv\Scripts\activate
    echo      pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo  [2/2] Python OK: %PY%
echo.
echo  ==========================================================
echo    Web page : http://127.0.0.1:8000
echo    API docs : http://127.0.0.1:8000/docs
echo    Stop     : press Ctrl + C
echo  ==========================================================
echo.

%PY% -m uvicorn app.main:app --port 8000

echo.
echo  Server stopped.
pause
