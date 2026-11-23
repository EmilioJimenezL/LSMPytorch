# lsm - launcher for the LSMPytorch orchestrator CLI (Windows PowerShell).
# Prefers .venv\Scripts\python.exe, falls back to python / py -3. Run `.\lsm.ps1 --help`.
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $Root 'Orchestrator\lsm_cli'))) {
    [Console]::Error.WriteLine("lsm: cannot find Orchestrator\lsm_cli under $Root")
    exit 1
}

$VenvPy = Join-Path $Root '.venv\Scripts\python.exe'
$PyArgs = @()
if (Test-Path $VenvPy) {
    $Py = $VenvPy
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Py = (Get-Command python).Source
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $Py = (Get-Command py).Source
    $PyArgs = @('-3')
} else {
    [Console]::Error.WriteLine('lsm: no Python found. Install Python 3.10+ and re-run (then: .\lsm.ps1 setup).')
    exit 1
}

$Orch = Join-Path $Root 'Orchestrator'
if ($env:PYTHONPATH) { $env:PYTHONPATH = "$Orch;$($env:PYTHONPATH)" } else { $env:PYTHONPATH = $Orch }
$env:PYTHONIOENCODING = 'utf-8'

& $Py @PyArgs -m lsm_cli @args
exit $LASTEXITCODE
