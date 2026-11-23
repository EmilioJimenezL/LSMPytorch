@echo off
rem lsm - launcher for the LSMPytorch orchestrator CLI (Windows cmd.exe).
rem Prefers .venv\Scripts\python.exe, falls back to python / py -3. Run: lsm --help
setlocal
set "ROOT=%~dp0"

if not exist "%ROOT%Orchestrator\lsm_cli\__main__.py" (
  echo lsm: cannot find Orchestrator\lsm_cli under %ROOT% 1>&2
  exit /b 1
)

set "PYTHONPATH=%ROOT%Orchestrator;%PYTHONPATH%"
set "PYTHONIOENCODING=utf-8"

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" -m lsm_cli %*
  exit /b
)
where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python -m lsm_cli %*
  exit /b
)
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 -m lsm_cli %*
  exit /b
)
echo lsm: no Python found. Install Python 3.10+ and re-run (then: lsm setup). 1>&2
exit /b 1
