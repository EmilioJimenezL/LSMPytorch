"""`lsm doctor`: environment report (read-only)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import paths
from .errors import CliError
from .runner import Context
from .ui import ui

# Runs inside the target interpreter (.venv python) so we report *its* packages.
_PROBE = r'''
import json, sys, platform
info = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    info["cuda_name"] = torch.cuda.get_device_name(0) if info["cuda"] else None
    mps = getattr(torch.backends, "mps", None)
    info["mps"] = bool(mps is not None and mps.is_available())
except Exception as e:
    info["torch_error"] = repr(e)
try:
    import coremltools as ct
    info["coremltools"] = ct.__version__
    info["ct_convert"] = hasattr(ct, "convert")
    native = False
    try:
        from coremltools.models.model import _MLModelProxy
        native = _MLModelProxy is not None
    except Exception:
        native = False
    is_mac = False
    try:
        from coremltools.models.utils import _is_macos
        is_mac = bool(_is_macos())
    except Exception:
        is_mac = sys.platform == "darwin"
    info["ct_predict"] = bool(native and is_mac)
    info["ct_native"] = bool(native)
except Exception as e:
    info["coremltools_error"] = repr(e)
for mod in ("numpy", "pytest", "rich"):
    try:
        m = __import__(mod)
        info[mod] = getattr(m, "__version__", "installed")
    except Exception:
        info[mod] = None
print("LSMDOCTOR:" + json.dumps(info))
'''

OK, WARN, MISS, INFO = "ok", "WARN", "MISSING", "info"


def _which_python() -> Tuple[Optional[str], str]:
    """Interpreter to probe and a description of why."""
    if paths.venv_exists():
        return str(paths.venv_python()), ".venv"
    if importlib.util.find_spec("torch") is not None:
        return sys.executable, "current interpreter (no .venv)"
    return None, "none"


def _run(argv: List[str], timeout: int = 60) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            timeout=timeout, text=True, errors="replace",
        )
        return proc.returncode, proc.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def _probe(python: str) -> dict:
    rc, out = _run([python, "-c", _PROBE], timeout=180)
    for line in out.splitlines():
        if line.startswith("LSMDOCTOR:"):
            try:
                return json.loads(line[len("LSMDOCTOR:"):])
            except ValueError:
                break
    return {"probe_error": "\n".join(out.strip().splitlines()[-3:]) or f"exit code {rc}"}


def _lfs_pointer_models() -> List[str]:
    bad = []
    if paths.MODELS_PYTORCH.is_dir():
        for pt in sorted(paths.MODELS_PYTORCH.glob("*.pt")):
            try:
                if pt.stat().st_size < 1024 and pt.read_bytes().startswith(b"version https://git-lfs"):
                    bad.append(pt.name)
            except OSError:
                pass
    return bad


def planned_checks() -> List[str]:
    return [
        "repo layout (TrainingPipeline/, CoreMLConversion/)",
        "python + .venv",
        "torch version and compute device (cuda / mps / cpu)",
        "coremltools: convert and predict capability",
        "pytest, numpy, rich",
        "xcodebuild + LSMExtractorCLI binary (macOS)",
        "git-lfs and Models/*.pt pointer files",
    ]


def run_doctor(ctx: Context) -> int:
    rows: List[Tuple[str, str, str]] = []  # (check, status, detail)
    fixes: List[str] = []

    missing_layout = paths.check_repo_layout()
    if missing_layout:
        raise CliError(
            "Repository layout looks wrong; missing: " + ", ".join(paths.rel(Path(p)) for p in missing_layout),
            hint=f"Run lsm from the LSMPytorch repo root (detected: {paths.REPO_ROOT}).",
        )
    rows.append(("repo", OK, str(paths.REPO_ROOT)))
    rows.append(("workspace", OK if paths.WORKSPACE.exists() else INFO,
                 f"{paths.rel(paths.WORKSPACE)}" + ("" if paths.WORKSPACE.exists() else " (created on first use)")))
    rows.append(("orchestrator python", INFO, f"{sys.version.split()[0]} ({sys.executable})"))

    python, why = _which_python()
    if paths.venv_exists():
        rows.append((".venv", OK, str(paths.venv_python())))
    else:
        rows.append((".venv", MISS if python is None else WARN, "not found"))
        fixes.append("lsm setup            # create .venv and install PyTorch + requirements")

    info: dict = {}
    if python:
        ui.info(f"Probing {why} ...", "dim")
        info = _probe(python)

    if "probe_error" in info:
        rows.append(("python probe", WARN, info["probe_error"]))
    elif info:
        rows.append(("python (env)", OK, f"{info.get('python')}"))
        if info.get("torch"):
            rows.append(("torch", OK, info["torch"]))
            if info.get("cuda"):
                device = f"cuda ({info.get('cuda_name')})"
            elif info.get("mps"):
                device = "mps (Apple GPU)"
            else:
                device = "cpu"
            status = OK if device != "cpu" else WARN
            rows.append(("device", status, device))
            if device == "cpu":
                if paths.is_macos():
                    fixes.append("Training will run on CPU (MPS not available: Intel Mac, old macOS, or sandboxed shell).")
                else:
                    fixes.append("Training will run on CPU. On NVIDIA: `lsm setup --cuda 12.4` (reinstalls torch).")
        else:
            rows.append(("torch", MISS, info.get("torch_error", "not installed")))
            fixes.append("lsm setup            # installs torch")
        if info.get("coremltools"):
            rows.append(("coremltools", OK, info["coremltools"]))
            rows.append(("  convert", OK if info.get("ct_convert") and info.get("torch") else WARN,
                         "available" if info.get("ct_convert") else "ct.convert missing"))
            if info.get("ct_predict"):
                rows.append(("  predict", OK, "available (parity checks supported)"))
            else:
                reason = "macOS only" if not paths.is_macos() else "native libs not loaded"
                rows.append(("  predict", WARN, f"unavailable ({reason}); `lsm convert` will skip parity"))
        else:
            rows.append(("coremltools", MISS, info.get("coremltools_error", "not installed")))
            fixes.append("lsm setup            # installs coremltools")
        for mod in ("pytest", "numpy"):
            rows.append((mod, OK if info.get(mod) else MISS, info.get(mod) or "not installed"))
        rows.append(("rich", OK if info.get("rich") else INFO,
                     info.get("rich") or "not installed (plain output; optional)"))
    else:
        rows.append(("torch", MISS, "no environment to probe"))

    # --- macOS toolchain -------------------------------------------------
    if paths.is_macos():
        xb = shutil.which("xcodebuild")
        if not xb:
            rows.append(("xcodebuild", MISS, "not on PATH"))
            fixes.append("Install Xcode from the App Store (needed to build LSMExtractorCLI / the iOS app).")
        else:
            rc, out = _run([xb, "-version"], timeout=30)
            if rc == 0:
                rows.append(("xcodebuild", OK, out.strip().splitlines()[0]))
            else:
                rows.append(("xcodebuild", WARN, "found, but full Xcode is not selected (command line tools only)"))
                fixes.append("sudo xcode-select -s /Applications/Xcode.app && sudo xcodebuild -license accept")
        from .tasks import find_extractor_binary  # local import: avoid cycle

        binary = find_extractor_binary()
        if binary:
            rows.append(("LSMExtractorCLI", OK, str(binary)))
        else:
            rows.append(("LSMExtractorCLI", WARN, "not built yet (`lsm extract` builds it with xcodebuild)"))
    else:
        rows.append(("xcodebuild", INFO, "n/a on this OS (landmark extraction + iOS testing need macOS)"))

    # --- git-lfs ---------------------------------------------------------
    git = shutil.which("git")
    if not git:
        rows.append(("git", WARN, "not found"))
    else:
        rc, out = _run([git, "lfs", "version"], timeout=20)
        if rc == 0:
            rows.append(("git-lfs", OK, out.strip().splitlines()[0]))
        else:
            rows.append(("git-lfs", WARN, "not installed (Models/*.pt and *.mlpackage are stored in LFS)"))
            fixes.append("brew install git-lfs && git lfs install   (or see https://git-lfs.com)")
    pointers = _lfs_pointer_models()
    if pointers:
        rows.append(("model weights", WARN, f"{len(pointers)} LFS pointer file(s), e.g. {pointers[0]}"))
        fixes.append("git lfs pull          # download the real Models/Pytorch/*.pt files")

    ui.out.print()
    ui.out.table(["check", "status", "detail"], [(a, b, ui_shorten(c)) for a, b, c in rows], title="lsm doctor",
                 flex_last=True)
    if fixes:
        ui.out.print()
        ui.out.print("Suggested fixes:", "bold")
        for fx in dict.fromkeys(fixes):
            ui.out.print(f"  {fx}")
    n_problems = sum(1 for _, s, _ in rows if s in (WARN, MISS))
    ui.out.print()
    if n_problems:
        ui.out.print(f"{n_problems} item(s) need attention (see above); doctor always exits 0 unless the repo is broken.", "warn")
    else:
        ui.out.print("Everything looks good. Next: lsm pipeline --help", "ok")
    return 0


def ui_shorten(text: str) -> str:
    from .ui import shorten

    return shorten(text)
