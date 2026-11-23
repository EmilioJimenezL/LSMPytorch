"""Task builders: resolve options into concrete commands, then execute them.

Every subcommand builds a ``Task`` (resolved plan + commands + post-processing).
``execute_single`` prints the plan, honours --dry-run, confirms overwrites and
runs it; ``execute_pipeline`` does the same for several tasks with [i/N] banners.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import paths
from .errors import CliError, StepError, UserError
from .runner import Cmd, CommandLog, Context, Result, run_cmd
from .ui import Row, format_cmd, shorten, ui

DEFAULT_LAYOUT = "v2_162"
LAYOUTS = ("v2_162", "v1_135")
MODELS = ("cnn", "tcn", "3dcnn")
FAMILIES = ("cnn", "tcn", "cnn3d", "fingerprint", "legacy")
MODEL_TO_FAMILY = {"cnn": "cnn", "tcn": "tcn", "3dcnn": "cnn3d"}
CUDA_CHOICES = ("11.8", "12.1", "12.4", "12.6")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------
@dataclass
class Task:
    key: str  # subcommand name, used in the plan header ("train")
    title: str  # human title for stage banners
    rows: List[Row]  # resolved plan (key/value)
    cmds: List[Cmd]
    equiv: str = ""  # equivalent standalone `lsm ...` command (retry hint)
    writes: List[Path] = field(default_factory=list)  # existing paths here need overwrite confirmation
    notes: List[str] = field(default_factory=list)
    post: Optional[Callable[[List[Result]], List[Row]]] = None
    outputs: List[Row] = field(default_factory=list)  # summary rows when no post hook
    next_cmd: Optional[str] = None
    done_title: str = ""


def lsm(*parts: object) -> str:
    """Render an equivalent `lsm ...` command line (for hints / next steps)."""
    return "lsm " + format_cmd([str(p) for p in parts])


def py_script_cmd(python: str, script: Path, args: Sequence[object], label: str, **kw) -> Cmd:
    return Cmd([python, str(script), *[str(a) for a in args]], label, cwd=paths.REPO_ROOT, **kw)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "dataset"


def layout_short(layout: str) -> str:
    m = re.match(r"v(\d+)", layout)
    return f"v{m.group(1)}" if m else slugify(layout)


def dataset_label(path: Path) -> str:
    base = path.name
    if base.lower() in {"landmarks", "landmark", "output", "outputs", "json"} and path.parent.name:
        base = path.parent.name
        if base.lower() in {"workspace", ""}:
            base = "dataset"
    return slugify(base)


def validate_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise UserError(
            f"Invalid name '{name}'.",
            hint="Use letters, digits, '.', '_' or '-' (no spaces or slashes), e.g. --name lab_v2_cnn.",
        )
    return name


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sidecar_for(cache: Path) -> Path:
    return cache.with_suffix(".split.json")


def cache_layout(cache: Path) -> Optional[str]:
    data = read_json(sidecar_for(cache))
    if isinstance(data, dict):
        settings = data.get("settings")
        if isinstance(settings, dict):
            return settings.get("layout")
    return None


def need(
    ctx: Context, path: Path, what: str, hint: str, notes: List[str], *, kind: str = "any", skip: bool = False
) -> None:
    """Require ``path`` to exist. In --dry-run a missing path becomes a plan note."""
    if skip:
        return
    ok = path.is_dir() if kind == "dir" else path.is_file() if kind == "file" else path.exists()
    if ok:
        return
    msg = f"{what} not found: {shorten(path)}"
    if ctx.dry_run:
        notes.append(f"{msg} (the real run would fail)")
        return
    raise UserError(msg, hint=hint)


def python_for_scripts(ctx: Context, notes: Optional[List[str]] = None) -> str:
    """Interpreter for wrapped Python scripts: .venv if present, else one that has torch."""
    if paths.venv_exists():
        return str(paths.venv_python())
    if importlib.util.find_spec("torch") is not None:
        return sys.executable
    if ctx.dry_run:
        if notes is not None:
            notes.append("No .venv found; the real run would fail. Run `lsm setup` first.")
        return str(paths.venv_python())
    raise UserError(
        f"No Python environment with PyTorch found ({paths.rel(paths.VENV)} does not exist).",
        hint="Run `lsm setup` to create .venv and install dependencies, then `lsm doctor`.",
    )


def list_runs() -> List[str]:
    if not paths.RUNS.is_dir():
        return []
    return sorted(p.name for p in paths.RUNS.iterdir() if (p / "best_model.pt").is_file())


def resolve_run_name(run: Optional[str], *, flag: str = "--run") -> str:
    if run:
        return validate_name(Path(run.rstrip("/\\")).name)
    return ui.choose("Pick a trained run (Runs/<name>/best_model.pt)", list_runs(), flag=flag)


def find_cache_for(name: str) -> Optional[Path]:
    """Guess the prepared cache for a run by longest matching prefix of its name."""
    best: Optional[Path] = None
    if paths.PREPARED_DIR.is_dir():
        for npz in paths.PREPARED_DIR.glob("*.npz"):
            if name == npz.stem or name.startswith(npz.stem + "_"):
                if best is None or len(npz.stem) > len(best.stem):
                    best = npz
    return best


def fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f}%"


def metrics_summary(metrics: Optional[dict]) -> Optional[str]:
    if not metrics:
        return None
    val = metrics.get("val") or {}
    parts = [f"val top1 {fmt_pct(val.get('top1', metrics.get('best_val_acc')))}"]
    if val.get("top5") is not None:
        parts.append(f"top5 {fmt_pct(val.get('top5'))}")
    extra = metrics.get("extra_test") or {}
    if extra.get("top1") is not None:
        parts.append(f"| extra-test top1 {fmt_pct(extra['top1'])}")
    return " ".join(parts)


def overwrite_notes(task: Task) -> List[str]:
    return [f"will overwrite existing: {shorten(p)}" for p in task.writes if p.exists()]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def confirm_overwrites(ctx: Context, tasks: Sequence[Task]) -> None:
    existing = [p for t in tasks for p in t.writes if p.exists()]
    if not existing:
        return
    ui.warn("These outputs already exist and will be overwritten:")
    for p in existing:
        ui.err.print(f"    {shorten(p)}", "warn")
    if not ui.confirm("Overwrite?", default=False):
        raise UserError(
            "Refusing to overwrite existing outputs.",
            hint="Pass --yes to overwrite, or choose a different --name / --output.",
        )


def run_task(task: Task, ctx: Context, log: CommandLog) -> List[Row]:
    results: List[Result] = []
    for cmd in task.cmds:
        results.append(run_cmd(cmd, log=log, dry_run=ctx.dry_run))
    if task.post:
        return task.post(results)
    return task.outputs


def execute_single(ctx: Context, task: Task) -> int:
    notes = list(task.notes) + overwrite_notes(task)
    ui.print_plan(
        f"lsm {task.key}", task.rows, [c.argv for c in task.cmds], notes=notes, dry_run=ctx.dry_run
    )
    if ctx.dry_run:
        return 0
    confirm_overwrites(ctx, [task])
    log = CommandLog(ctx.command)
    try:
        rows = run_task(task, ctx, log)
    except CliError as err:
        if err.log_path is None and log.exists:
            err.log_path = log.path
        raise
    finally:
        log.close()
    ui.print_summary(
        task.done_title or task.title,
        rows,
        next_cmd=task.next_cmd,
        log_path=log.path if log.exists else None,
        elapsed=time.time() - ctx.started,
    )
    return 0


def execute_pipeline(ctx: Context, header_rows: List[Row], tasks: List[Task], notes: List[str]) -> int:
    total = len(tasks)
    rows = list(header_rows) + [(f"[{i}/{total}]", t.title) for i, t in enumerate(tasks, 1)]
    all_notes = list(notes)
    for t in tasks:
        all_notes += t.notes + overwrite_notes(t)
    ui.print_plan(
        "lsm pipeline", rows, [c.argv for t in tasks for c in t.cmds], notes=all_notes, dry_run=ctx.dry_run
    )
    if ctx.dry_run:
        return 0
    confirm_overwrites(ctx, tasks)
    log = CommandLog(ctx.command)
    summary: List[Row] = []
    try:
        for i, task in enumerate(tasks, 1):
            ui.print_stage(i, total, task.title)
            try:
                got = run_task(task, ctx, log)
            except CliError as err:
                if err.log_path is None and log.exists:
                    err.log_path = log.path
                if task.equiv:
                    retry = f"Fix the cause, then re-run just this stage: {task.equiv}"
                    err.hint = f"{err.hint}  |  {retry}" if err.hint else retry
                err.message = f"Pipeline stopped at stage [{i}/{total}] {task.title}: {err.message}"
                raise
            summary += [(f"[{i}] {k}", v) for k, v in got]
    finally:
        log.close()
    last_next = next((t.next_cmd for t in reversed(tasks) if t.next_cmd), None)
    ui.print_summary(
        "pipeline",
        summary,
        next_cmd=last_next,
        log_path=log.path if log.exists else None,
        elapsed=time.time() - ctx.started,
    )
    return 0


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def build_setup(ctx: Context, cuda: Optional[str], cpu: bool) -> Task:
    if sys.version_info < (3, 9):
        raise UserError(
            f"Python {sys.version.split()[0]} is too old (need 3.9+; 3.10-3.12 recommended).",
            hint="Install a newer Python and run it explicitly: `python3.12 Orchestrator/lsm_cli setup`.",
        )
    notes: List[str] = []
    if sys.version_info >= (3, 13):
        notes.append(
            f"Python {sys.version.split()[0]}: PyTorch/coremltools wheels may be missing; 3.10-3.12 is safest."
        )
    venv_py = str(paths.venv_python())
    cmds: List[Cmd] = []
    existed = paths.venv_exists()
    if not existed:
        cmds.append(Cmd([sys.executable, "-m", "venv", str(paths.VENV)], "Create .venv"))
    pip = [venv_py, "-m", "pip"]
    cmds.append(Cmd([*pip, "install", "--upgrade", "pip", "-q", "--disable-pip-version-check"], "Upgrade pip"))

    if paths.is_macos():
        torch_note = "default PyPI wheel (Apple GPU / MPS support built in)"
        torch_cmd = [*pip, "install", "torch", "torchvision"]
        if cuda or cpu:
            notes.append("--cuda/--cpu are ignored on macOS (the default wheel already supports MPS).")
    elif cpu:
        torch_note = "CPU-only wheel (download.pytorch.org/whl/cpu)"
        torch_cmd = [*pip, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]
    elif cuda:
        tag = "cu" + cuda.replace(".", "")
        torch_note = f"CUDA {cuda} wheel (download.pytorch.org/whl/{tag})"
        torch_cmd = [*pip, "install", "torch", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{tag}"]
    else:
        torch_note = "default PyPI wheel (pass --cuda 12.4 for an explicit CUDA build, or --cpu)"
        torch_cmd = [*pip, "install", "torch", "torchvision"]
    cmds.append(Cmd(torch_cmd, "Install PyTorch"))
    cmds.append(
        Cmd([*pip, "install", "-r", str(paths.REQUIREMENTS)], "Install requirements", cwd=paths.REPO_ROOT)
    )

    if not paths.REQUIREMENTS.exists():
        raise UserError(f"Missing {paths.rel(paths.REQUIREMENTS)}.", hint="Run from the LSMPytorch repo root.")

    rows: List[Row] = [
        ("Virtualenv", f"{paths.rel(paths.VENV)} ({'exists, reusing' if existed else 'will be created'})"),
        ("Base python", f"{sys.executable} ({sys.version.split()[0]})"),
        ("PyTorch", torch_note),
        ("Requirements", paths.rel(paths.REQUIREMENTS)),
    ]
    return Task(
        key="setup",
        title="Set up Python environment",
        rows=rows,
        cmds=cmds,
        notes=notes,
        outputs=[("Virtualenv", paths.rel(paths.VENV)), ("Python", paths.rel(paths.venv_python()))],
        next_cmd="lsm doctor",
        done_title="environment ready",
    )


# ---------------------------------------------------------------------------
# extract (macOS)
# ---------------------------------------------------------------------------
def find_extractor_binary() -> Optional[Path]:
    """Locate a built LSMExtractorCLI: repo build/ dir first, then Xcode DerivedData."""
    name = "LSMExtractorCLI"
    candidates: List[Path] = [paths.EXTRACTOR_BUILD / "Build" / "Products" / "Release" / name]
    dd = Path.home() / "Library" / "Developer" / "Xcode" / "DerivedData"
    derived_rel, derived_dbg = [], []
    if dd.is_dir():
        derived_rel = sorted(dd.glob(f"{name}-*/Build/Products/Release/{name}"), key=_mtime, reverse=True)
        derived_dbg = sorted(dd.glob(f"{name}-*/Build/Products/Debug/{name}"), key=_mtime, reverse=True)
    candidates += derived_rel
    candidates.append(paths.EXTRACTOR_BUILD / "Build" / "Products" / "Debug" / name)
    candidates += derived_dbg
    for c in candidates:
        if c.is_file():
            return c
    return None


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def build_extract(
    ctx: Context,
    videos: Path,
    output: Path,
    overlay: bool,
    skip_existing: bool,
    manifest: Optional[Path],
    rebuild: bool,
    *,
    videos_will_exist: bool = False,
) -> Task:
    if not paths.is_macos():
        raise UserError(
            f"`lsm extract` needs macOS (it uses Apple's Vision framework); this machine is {sys.platform}.",
            hint="Extract landmarks on a Mac, copy the folder here, then run "
            "`lsm prepare --landmarks DIR` (or `lsm pipeline --landmarks DIR`).",
        )
    notes: List[str] = []
    need(ctx, videos, "Videos folder", "Pass --videos DIR with {category}/{word}/*.mp4|mov|m4v files.",
         notes, kind="dir", skip=videos_will_exist)
    if manifest:
        need(ctx, manifest, "Manifest", "Pass an existing dataset_manifest.json or omit --manifest.", notes, kind="file")
    if not paths.EXTRACTOR_PROJECT.exists():
        raise UserError(
            f"Xcode project not found: {paths.rel(paths.EXTRACTOR_PROJECT)}",
            hint="Run from the LSMPytorch repo root (layout: LandmarkExtraction/LSMExtractorCLI/...).",
        )

    binary = find_extractor_binary()
    default_bin = paths.EXTRACTOR_BUILD / "Build" / "Products" / "Release" / "LSMExtractorCLI"
    need_build = rebuild or binary is None
    cmds: List[Cmd] = []
    if need_build:
        build_argv = [
            "xcodebuild",
            "-project", str(paths.EXTRACTOR_PROJECT),
            "-scheme", "LSMExtractorCLI",
            "-configuration", "Release",
            "-derivedDataPath", str(paths.EXTRACTOR_BUILD),
        ]
        if not ui.verbose:
            build_argv.append("-quiet")
        cmds.append(Cmd(build_argv, "Build LSMExtractorCLI (xcodebuild)", cwd=paths.REPO_ROOT))
        exe = default_bin
        if shutil.which("xcodebuild") is None:
            notes.append("xcodebuild not found on PATH; install Xcode (see `lsm doctor`).")
    else:
        exe = binary  # type: ignore[assignment]

    def _check_binary() -> None:
        found = find_extractor_binary()
        if found is None:
            raise StepError(
                "xcodebuild finished but the LSMExtractorCLI binary was not found.",
                hint=f"Expected {paths.rel(default_bin)}; build it from Xcode (Release) and retry.",
            )

    run_argv = [str(exe), "--input", str(videos), "--output", str(output)]
    if manifest:
        run_argv += ["--manifest", str(manifest)]
    if overlay:
        run_argv.append("--overlay")
    if skip_existing:
        run_argv.append("--skip-existing")
    cmds.append(Cmd(run_argv, "Extract landmarks", cwd=paths.REPO_ROOT, ok_codes=(0, 2), before=_check_binary))

    def post(results: List[Result]) -> List[Row]:
        n_json = sum(1 for _ in output.rglob("*_landmarks.json")) if output.is_dir() else 0
        rows: List[Row] = [("Landmarks", f"{paths.rel(output)} ({n_json} *_landmarks.json files)")]
        if (output / "dataset_manifest.json").exists():
            rows.append(("Manifest", paths.rel(output / "dataset_manifest.json")))
        if (output / "_batch_report.json").exists():
            rows.append(("Batch report", paths.rel(output / "_batch_report.json")))
        if results and results[-1].returncode == 2:
            ui.warn(
                "At least one video failed to process (extractor exit code 2). "
                "See _batch_report.json in the output folder; re-run with --skip-existing after fixing."
            )
        return rows

    rows = [
        ("Videos", videos),
        ("Output", output),
        ("Overlay videos", "yes" if overlay else "no (JSON only)"),
        ("Skip existing", "yes" if skip_existing else "no"),
        ("Binary", f"{exe} ({'will be built' if need_build else 'found'})"),
    ]
    if manifest:
        rows.append(("Manifest", manifest))
    equiv = ["extract", "--videos", videos, "--output", output] + (["--overlay"] if overlay else []) + (
        ["--skip-existing"] if skip_existing else []
    )
    return Task(
        key="extract",
        title="Extract landmarks from videos",
        rows=rows,
        cmds=cmds,
        equiv=lsm(*equiv),
        notes=notes,
        post=post,
        next_cmd=lsm("prepare", "--landmarks", output),
        done_title="landmarks extracted",
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def build_prepare(
    ctx: Context,
    landmarks: Path,
    layout: str,
    min_videos: int,
    cap: int,
    val_split: float,
    seed: int,
    name: Optional[str],
    no_shoulder_norm: bool,
    output: Optional[Path] = None,
    *,
    landmarks_will_exist: bool = False,
) -> Task:
    notes: List[str] = []
    need(ctx, landmarks, "Landmarks folder",
         "Pass --landmarks DIR (output of `lsm extract`: {category}/{word}/*_landmarks.json).",
         notes, kind="dir", skip=landmarks_will_exist)
    if name:
        validate_name(name)
    if output is None:
        stem = name or f"{dataset_label(landmarks)}_{layout_short(layout)}_cap{cap}"
        output = paths.PREPARED_DIR / f"{stem}.npz"
    elif output.suffix != ".npz":
        output = output.with_suffix(".npz")
    sidecar = sidecar_for(output)
    python = python_for_scripts(ctx, notes)

    args = [
        "--dataset", landmarks,
        "--output", output,
        "--layout", layout,
        "--min-videos", min_videos,
        "--max-videos-per-class", cap,
        "--val-split", val_split,
        "--seed", seed,
    ]
    if no_shoulder_norm:
        args.append("--no-shoulder-norm")
    cmd = py_script_cmd(python, paths.PREPARE_SCRIPT, args, "Prepare dataset cache")

    rows: List[Row] = [
        ("Landmarks", landmarks),
        ("Layout", layout),
        ("Min videos/word", min_videos),
        ("Cap videos/word", cap),
        ("Val split", val_split),
        ("Seed", seed),
        ("Shoulder norm", "off (v1 only)" if no_shoulder_norm else "on"),
        ("Output", output),
    ]
    rows = [(k, str(v)) for k, v in rows]
    equiv = ["prepare", "--landmarks", landmarks, "--layout", layout, "--min-videos", min_videos, "--cap", cap,
             "--val-split", val_split, "--seed", seed, "--output", output]

    def post(_: List[Result]) -> List[Row]:
        if not output.exists():
            raise StepError("prepare_dataset.py finished but no cache was written.", hint="See the log for details.")
        size = output.stat().st_size / 1e6
        return [("Cache", f"{paths.rel(output)} ({size:.1f} MB)"), ("Split manifest", paths.rel(sidecar))]

    return Task(
        key="prepare",
        title=f"Prepare dataset ({layout}, cap {cap})",
        rows=rows,
        cmds=[cmd],
        equiv=lsm(*equiv),
        writes=[output, sidecar],
        notes=notes,
        post=post,
        next_cmd=lsm("train", "--model", "cnn", "--cache", output),
        done_title="dataset cache ready",
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@dataclass
class TrainOpts:
    model: str = "cnn"
    landmarks: Optional[Path] = None
    cache: Optional[Path] = None
    name: Optional[str] = None
    layout: str = DEFAULT_LAYOUT
    epochs: int = 100
    batch: int = 64
    lr: float = 1e-3
    dropout: float = 0.3
    val_split: float = 0.15
    augment: int = 10
    min_videos: int = 2
    cap: Optional[int] = None
    causal: int = 1
    pool: str = "avg"
    joint_dropout: float = 0.1
    warmup: int = 10
    patience: int = 15
    legacy_augment: bool = False
    no_shoulder_norm: bool = False
    resume: Optional[str] = None
    workers: Optional[int] = None


def default_run_name(opts: TrainOpts, layout: str) -> str:
    return f"{layout_short(layout)}_{opts.model}"


def build_train(ctx: Context, o: TrainOpts, *, data_will_exist: bool = False) -> Task:
    if o.model not in MODELS:
        raise UserError(f"Unknown model '{o.model}'.", hint=f"Choose one of: {', '.join(MODELS)}.")
    if bool(o.landmarks) == bool(o.cache):
        raise UserError(
            "Training needs exactly one data source: --landmarks DIR or --cache FILE.npz.",
            hint="Use `lsm prepare` first and pass --cache (faster, reproducible), or pass raw --landmarks.",
        )
    notes: List[str] = []
    layout = o.layout
    if o.cache:
        need(ctx, o.cache, "Prepared cache", "Create it with `lsm prepare --landmarks DIR`.", notes,
             kind="file", skip=data_will_exist)
        cl = cache_layout(o.cache)
        if cl:
            if cl != o.layout:
                notes.append(f"Layout taken from cache sidecar: {cl} (ignoring --layout {o.layout}).")
            layout = cl
    else:
        assert o.landmarks is not None
        need(ctx, o.landmarks, "Landmarks folder", "Pass --landmarks DIR.", notes, kind="dir", skip=data_will_exist)

    name = validate_name(o.name) if o.name else default_run_name(o, layout)
    run_dir = paths.RUNS / name
    model_copy = paths.MODELS_PYTORCH / f"{name}.pt"
    python = python_for_scripts(ctx, notes)

    resume_path: Optional[Path] = None
    if o.resume:
        resume_path = run_dir / "last.pt" if o.resume == "auto" else paths.resolve_user_path(o.resume)
        need(ctx, resume_path, "Resume checkpoint", "Check the path, or omit --resume to start fresh.", notes, kind="file")

    args: List[object] = ["--model", o.model]
    args += ["--cache", o.cache] if o.cache else ["--dataset", o.landmarks]
    args += [
        "--output", run_dir,
        "--epochs", o.epochs,
        "--batch", o.batch,
        "--lr", o.lr,
        "--dropout", o.dropout,
        "--val_split", o.val_split,
        "--augment", o.augment,
        "--min_videos", o.min_videos,
        "--layout", layout,
        "--pool", o.pool,
        "--joint_dropout", o.joint_dropout,
        "--warmup", o.warmup,
        "--patience", o.patience,
    ]
    if o.model == "tcn":
        args += ["--causal", o.causal]
    if o.cap is not None:
        args += ["--max_videos_per_class", o.cap]
    if o.legacy_augment:
        args.append("--legacy-augment")
    if o.no_shoulder_norm:
        args.append("--no-shoulder-norm")
    if resume_path:
        args += ["--resume", resume_path]
    if o.workers is not None:
        args += ["--workers", o.workers]
    cmd = py_script_cmd(python, paths.TRAIN_SCRIPT, args, f"Train {o.model}")

    cfg = [f"epochs={o.epochs}", f"batch={o.batch}", f"lr={o.lr:g}", f"dropout={o.dropout:g}",
           f"augment={o.augment}{' (legacy CPU)' if o.legacy_augment else ''}", f"patience={o.patience}",
           f"warmup={o.warmup}", f"pool={o.pool}"]
    if o.model == "tcn":
        cfg.append(f"causal={o.causal}")
    rows: List[Row] = [
        ("Model", o.model),
        ("Data", f"cache {o.cache}" if o.cache else f"landmarks {o.landmarks}"),
        ("Layout", layout),
        ("Hyperparameters", ", ".join(cfg)),
    ]
    if o.cap is not None:
        rows.append(("Cap videos/word", str(o.cap)))
    if resume_path:
        rows.append(("Resume from", str(resume_path)))
    rows += [
        ("Run name", name),
        ("Run output", str(run_dir)),
        ("Checkpoint copy", str(model_copy)),
    ]

    equiv_parts: List[object] = ["train", "--model", o.model]
    equiv_parts += ["--cache", o.cache] if o.cache else ["--landmarks", o.landmarks]
    equiv_parts += ["--name", name, "--epochs", o.epochs, "--layout", layout]
    equiv = lsm(*equiv_parts)

    writes = [model_copy]
    if not resume_path:
        writes.insert(0, run_dir / "best_model.pt")

    def post(_: List[Result]) -> List[Row]:
        best = run_dir / "best_model.pt"
        if not best.is_file():
            raise StepError(
                f"Training finished but {paths.rel(best)} was not produced.",
                hint="See the log; training may have exited early.",
            )
        paths.MODELS_PYTORCH.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, model_copy)
        metrics = read_json(run_dir / "metrics.json")
        readme = run_dir / "README.md"
        rows_out: List[Row] = [("Run", paths.rel(run_dir)), ("Best checkpoint", paths.rel(model_copy))]
        if readme.exists():
            rows_out.append(("README", f"{paths.rel(readme)} (kept existing)"))
        else:
            write_run_readme(readme, name, o, layout, metrics, equiv)
            rows_out.append(("README", f"{paths.rel(readme)} (skeleton written)"))
        summary = metrics_summary(metrics)
        if summary:
            rows_out.append(("Metrics", summary))
        return rows_out

    convert_next = ["convert", "--run", name] + (["--cache", o.cache] if o.cache else [])
    return Task(
        key="train",
        title=f"Train {o.model} -> Runs/{name}",
        rows=rows,
        cmds=[cmd],
        equiv=equiv,
        writes=writes,
        notes=notes,
        post=post,
        next_cmd=lsm(*convert_next),
        done_title=f"trained {name}",
    )


def write_run_readme(
    path: Path, name: str, o: TrainOpts, layout: str, metrics: Optional[dict], equiv: str
) -> None:
    m = metrics or {}
    val = m.get("val") or {}
    extra = m.get("extra_test") or {}
    cfg = m.get("model_config") or {}

    def row(label: str, split: dict) -> str:
        if not split:
            return f"| {label} | n/a | n/a | n/a |"
        return f"| {label} | {fmt_pct(split.get('top1'))} | {fmt_pct(split.get('top5'))} | {split.get('n', 'n/a')} |"

    lines = [
        f"# {name}",
        "",
        f"> Skeleton generated by `lsm train` on {datetime.now():%Y-%m-%d %H:%M}. Add your analysis under *Notes*.",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | `{m.get('model', o.model)}` |",
        f"| Layout | `{m.get('layout', layout)}` |",
        f"| Classes | {m.get('n_classes', 'n/a')} |",
        f"| Epochs run | {m.get('epochs_run', 'n/a')} / {o.epochs} (early stop: {m.get('stopped_early', 'n/a')}) |",
        f"| Data | `{shorten(o.cache or o.landmarks)}` |",
        f"| Model config | `{json.dumps(cfg, ensure_ascii=False)}` |" if cfg else "| Model config | n/a |",
        "",
        "## Results",
        "",
        "| Split | Top-1 | Top-5 | n |",
        "| --- | --- | --- | --- |",
        row("Validation", val),
        row("Extra test", extra),
        "",
        "## Reproduce",
        "",
        "```bash",
        equiv,
        "```",
        "",
        "## Artifacts",
        "",
        f"- `Runs/{name}/best_model.pt` (copied to `Models/Pytorch/{name}.pt`)",
        "- `classes.json`, `split_manifest.json`, `history.json`, `metrics.json`, `training_curves.png`",
        "",
        "## Notes",
        "",
        "- TODO: observations, failure cases, comparison with other runs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------
def build_convert(
    ctx: Context,
    run: Optional[str],
    checkpoint: Optional[Path],
    name: Optional[str],
    output: Optional[Path],
    cache: Optional[Path],
    parity: Optional[bool],
    fixture: Optional[bool],
    *,
    ckpt_will_exist: bool = False,
) -> Task:
    notes: List[str] = []
    run_dir: Optional[Path] = None
    if checkpoint:
        ckpt = checkpoint
        if ckpt.name == "best_model.pt":
            run_dir = ckpt.parent
        derived = ckpt.parent.name if ckpt.name == "best_model.pt" else ckpt.stem
        nm = validate_name(name or derived)
    else:
        nm = validate_name(name or resolve_run_name(run, flag="--run"))
        run_dir = paths.RUNS / nm
        ckpt = run_dir / "best_model.pt"
        if not ckpt.exists() and (paths.MODELS_PYTORCH / f"{nm}.pt").exists():
            ckpt = paths.MODELS_PYTORCH / f"{nm}.pt"
    need(ctx, ckpt, "Checkpoint", f"Train first (`lsm train ...`) or pass --checkpoint FILE. Known runs: "
         f"{', '.join(list_runs()) or '(none)'}", notes, kind="file", skip=ckpt_will_exist)

    pkg = output if output else paths.MODELS_COREML / f"{nm}.mlpackage"
    if pkg.suffix != ".mlpackage":
        pkg = pkg.parent / (pkg.name + ".mlpackage")
    stem = pkg.name[: -len(".mlpackage")]
    classes_dest = pkg.parent / f"{stem}_classes.json"
    fixture_path = pkg.parent / f"{stem}_fixture.json"

    classes_src: Optional[Path] = None
    for cand in ([run_dir / "classes.json"] if run_dir else []) + [ckpt.parent / "classes.json"]:
        if cand.is_file():
            classes_src = cand
            break
    if classes_src is None and ckpt_will_exist and run_dir is not None:
        classes_src = run_dir / "classes.json"  # expected location once training is done
    if classes_src is None:
        notes.append("No classes.json found next to the checkpoint; it will not be copied (needed by `lsm deploy`).")

    # Cache / parity / fixture resolution
    cache_auto = False
    if cache is None:
        cache = find_cache_for(nm)
        cache_auto = cache is not None
    if cache is not None:
        need(ctx, cache, "Prepared cache", "Pass --cache FILE.npz (from `lsm prepare`).", notes, kind="file",
             skip=ckpt_will_exist)

    do_parity = bool(parity) if parity is not None else cache is not None
    if do_parity and not paths.is_macos():
        notes.append("Skipping Core ML parity check: coremltools predict only works on macOS "
                     "(run `lsm convert --parity` on a Mac to verify).")
        do_parity = False
    if do_parity and cache is None:
        if parity:
            raise UserError("--parity needs validation tensors.", hint="Pass --cache FILE.npz (from `lsm prepare`).")
        do_parity = False
    do_fixture = bool(fixture) if fixture is not None else cache is not None
    if do_fixture and cache is None:
        if fixture:
            raise UserError("--fixture needs a prepared cache.", hint="Pass --cache FILE.npz.")
        do_fixture = False
    if cache is None:
        notes.append("No prepared cache found: parity check and golden fixture are skipped (pass --cache FILE.npz).")

    python = python_for_scripts(ctx, notes)
    args: List[object] = ["--checkpoint", ckpt, "--output", pkg]
    if cache is not None:
        args += ["--cache", cache]
        args.append("--parity" if do_parity else "--no-parity")
    if do_fixture:
        args += ["--fixture", fixture_path]

    def _clean_existing() -> None:
        if pkg.exists():
            shutil.rmtree(pkg) if pkg.is_dir() else pkg.unlink()
        pkg.parent.mkdir(parents=True, exist_ok=True)

    cmd = py_script_cmd(python, paths.CONVERT_SCRIPT, args, "Convert to Core ML", before=_clean_existing)

    rows: List[Row] = [
        ("Checkpoint", str(ckpt)),
        ("Name", nm),
        ("Output", str(pkg)),
        ("Classes", f"{classes_src} -> {classes_dest}" if classes_src else "none"),
        ("Cache", f"{cache}{' (auto-detected)' if cache_auto else ''}" if cache else "none"),
        ("Parity check", "yes" if do_parity else "no"),
        ("Golden fixture", str(fixture_path) if do_fixture else "no"),
    ]
    equiv_parts: List[object] = ["convert", "--checkpoint", ckpt, "--name", nm, "--output", pkg]
    if cache:
        equiv_parts += ["--cache", cache]

    def post(_: List[Result]) -> List[Row]:
        if not pkg.exists():
            raise StepError(f"Conversion finished but {paths.rel(pkg)} was not produced.", hint="See the log.")
        out: List[Row] = [("Core ML package", paths.rel(pkg))]
        if classes_src and classes_src.is_file():
            shutil.copy2(classes_src, classes_dest)
            out.append(("Classes", paths.rel(classes_dest)))
        if do_fixture and fixture_path.exists():
            out.append(("Fixture", paths.rel(fixture_path)))
        out.append(("Parity", "passed" if do_parity else "skipped"))
        return out

    writes = [pkg, classes_dest] + ([fixture_path] if do_fixture else [])
    return Task(
        key="convert",
        title=f"Convert {nm} to Core ML",
        rows=rows,
        cmds=[cmd],
        equiv=lsm(*equiv_parts),
        writes=writes,
        notes=notes,
        post=post,
        next_cmd=lsm("deploy", "--run", stem),
        done_title=f"converted {nm}",
    )


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------
def infer_family(name: str, run_dir: Path) -> str:
    metrics = read_json(run_dir / "metrics.json") or {}
    model = metrics.get("model")
    if model in MODEL_TO_FAMILY:
        return MODEL_TO_FAMILY[model]
    low = name.lower()
    if "3dcnn" in low:
        return "cnn3d"
    if "tcn" in low:
        return "tcn"
    return "cnn"


def build_deploy(
    ctx: Context,
    run: Optional[str],
    sign_set: str,
    display_name: Optional[str],
    family: Optional[str],
    description: Optional[str],
    *,
    will_exist: bool = False,
    fixture_expected: bool = False,
) -> Task:
    notes: List[str] = []
    nm = resolve_run_name(run, flag="--run") if not run else validate_name(Path(run.rstrip("/\\")).name)
    pkg = paths.MODELS_COREML / f"{nm}.mlpackage"
    classes = paths.MODELS_COREML / f"{nm}_classes.json"
    if not classes.exists() and (paths.RUNS / nm / "classes.json").exists():
        classes = paths.RUNS / nm / "classes.json"
    fixture = paths.MODELS_COREML / f"{nm}_fixture.json"
    use_fixture = fixture.exists() or fixture_expected

    need(ctx, pkg, "Core ML package", f"Run `lsm convert --run {nm}` first.", notes, kind="dir", skip=will_exist)
    need(ctx, classes, "classes.json", f"Run `lsm convert --run {nm}` (copies classes.json), or retrain.",
         notes, kind="file", skip=will_exist)
    if not paths.EXPORT_SCRIPT.exists():
        raise UserError(f"Missing {paths.rel(paths.EXPORT_SCRIPT)}.", hint="Run from the repo root.")

    fam = family or infer_family(nm, paths.RUNS / nm)
    python = python_for_scripts(ctx, notes)
    args: List[object] = ["--mlpackage", pkg, "--classes", classes, "--run-stem", nm, "--family", fam,
                          "--sign-set", sign_set]
    if use_fixture:
        args += ["--fixture", fixture]
    if display_name:
        args += ["--name", display_name]
    if description:
        args += ["--description", description]
    cmd = py_script_cmd(python, paths.EXPORT_SCRIPT, args, "Copy model into iOS app")

    app_bundle_dir = paths.IOS_APP_DIR / "LSMMobileModelTesting"
    dest_pkg = app_bundle_dir / pkg.name
    rows: List[Row] = [
        ("Core ML package", str(pkg)),
        ("Classes", str(classes)),
        ("Fixture", str(fixture) if use_fixture else "none"),
        ("Family / sign set", f"{fam} / {sign_set}"),
        ("Display name", display_name or nm),
        ("iOS app folder", str(app_bundle_dir)),
    ]
    if not paths.is_macos():
        notes.append("Deploy only copies files, but building/running the iOS app needs macOS + Xcode.")

    def post(_: List[Result]) -> List[Row]:
        return [
            ("iOS bundle", paths.rel(dest_pkg)),
            ("Next step", "paste the AVAILABLE_MODELS snippet (see log / output above) into LandmarkPipeline.swift"),
        ]

    return Task(
        key="deploy",
        title=f"Deploy {nm} to the iOS test app",
        rows=rows,
        cmds=[cmd],
        equiv=lsm("deploy", "--run", nm),
        writes=[dest_pkg],
        notes=notes,
        post=post,
        next_cmd="Open TestingTools/LSMMobileModelTesting in Xcode, register the model, build & run on device",
        done_title=f"deployed {nm}",
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def build_evaluate(
    ctx: Context, run: Optional[str], dataset: Optional[Path], min_videos: int, output: Optional[Path]
) -> Task:
    notes: List[str] = []
    if not paths.EVALUATE_SCRIPT.exists():
        raise UserError(
            f"{paths.rel(paths.EVALUATE_SCRIPT)} does not exist in this checkout.",
            hint="Use `lsm report --runs NAME` for metrics.json summaries instead.",
        )
    nm = resolve_run_name(run, flag="--run")
    ckpt = paths.RUNS / nm / "best_model.pt"
    need(ctx, ckpt, "Checkpoint", f"Train first: `lsm train ... --name {nm}`.", notes, kind="file")
    if dataset is None:
        default = paths.LANDMARKS_DIR if paths.LANDMARKS_DIR.is_dir() else None
        dataset = ui.prompt_path("Landmarks folder to evaluate on", flag="--dataset", default=default)
    need(ctx, dataset, "Dataset", "Pass --dataset DIR (landmark JSON folder).", notes, kind="dir")
    out = output or (paths.RUNS / nm / "evaluation.json")
    python = python_for_scripts(ctx, notes)
    cmd = py_script_cmd(
        python, paths.EVALUATE_SCRIPT,
        ["--checkpoint", ckpt, "--dataset", dataset, "--min-videos", min_videos, "--output", out],
        "Evaluate checkpoint",
    )
    rows: List[Row] = [("Checkpoint", str(ckpt)), ("Dataset", str(dataset)), ("Min videos", str(min_videos)),
                       ("Report", str(out))]
    return Task(
        key="evaluate",
        title=f"Evaluate {nm}",
        rows=rows,
        cmds=[cmd],
        equiv=lsm("evaluate", "--run", nm, "--dataset", dataset),
        writes=[out],
        notes=notes,
        outputs=[("Evaluation JSON", paths.rel(out))],
        next_cmd=lsm("report", "--runs", nm),
        done_title=f"evaluated {nm}",
    )


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
def build_test(ctx: Context, quick: bool, extra: Sequence[str]) -> Task:
    notes: List[str] = []
    python = python_for_scripts(ctx, notes)
    argv = [python, "-m", "pytest"]
    argv += ["-q", "-m", "not slow"] if quick else ["-ra"]
    argv += [a for a in extra if a != "--"]
    cmd = Cmd(argv, "pytest", cwd=paths.TRAINING)
    return Task(
        key="test",
        title="Run TrainingPipeline tests",
        rows=[("Directory", str(paths.TRAINING)), ("Mode", "quick (-m 'not slow')" if quick else "full")],
        cmds=[cmd],
        notes=notes,
        outputs=[("Tests", "passed")],
        next_cmd=lsm("pipeline", "--help"),
        done_title="tests passed",
    )


# ---------------------------------------------------------------------------
# report (in-process: combine metrics.json files)
# ---------------------------------------------------------------------------
def run_report(ctx: Context, runs: Sequence[str], output: Optional[Path]) -> int:
    names = [validate_name(Path(r.rstrip("/\\")).name) for r in runs]
    table: List[List[str]] = []
    missing: List[str] = []
    data: Dict[str, dict] = {}
    for nm in names:
        metrics = read_json(paths.RUNS / nm / "metrics.json")
        if metrics is None:
            missing.append(nm)
            continue
        data[nm] = metrics
        val = metrics.get("val") or {}
        extra = metrics.get("extra_test") or {}
        cfg = metrics.get("model_config") or {}
        table.append([
            nm, str(metrics.get("model", "?")), str(metrics.get("layout", "?")), str(metrics.get("n_classes", "?")),
            str(metrics.get("epochs_run", "?")), str(cfg.get("pool", "?")),
            fmt_pct(val.get("top1", metrics.get("best_val_acc"))), fmt_pct(val.get("top5")),
            fmt_pct(extra.get("top1")), fmt_pct(extra.get("top5")),
        ])
    if missing:
        raise UserError(
            f"No metrics.json for: {', '.join(missing)}.",
            hint=f"Known runs: {', '.join(sorted(p.name for p in paths.RUNS.iterdir() if p.is_dir())) if paths.RUNS.is_dir() else '(none)'}. "
                 "Runs trained with `lsm train` write Runs/<name>/metrics.json.",
        )
    headers = ["run", "model", "layout", "classes", "epochs", "pool", "val@1", "val@5", "extra@1", "extra@5"]
    out = output or (paths.EXPORTS_DIR / f"report_{datetime.now():%Y%m%d-%H%M%S}.md")
    plan_rows: List[Row] = [("Runs", ", ".join(names)), ("Source", "Runs/<name>/metrics.json"), ("Report file", str(out))]
    ui.print_plan("lsm report", plan_rows, [], dry_run=ctx.dry_run)
    if ctx.dry_run:
        return 0
    if out.exists() and not ui.confirm(f"Overwrite {shorten(out)}?", default=False):
        raise UserError("Refusing to overwrite existing report.", hint="Pass --yes or choose another --output.")
    if not ui.quiet:
        ui.out.table(headers, table, title="Run comparison")
    md = ["# Run comparison", "", f"_Generated by `lsm report` on {datetime.now():%Y-%m-%d %H:%M}._", "",
          "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in table]
    best = max(data, key=lambda k: ((data[k].get("val") or {}).get("top1") or data[k].get("best_val_acc") or 0))
    md += ["", f"Best validation top-1: **{best}**", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    ui.print_summary(
        "report written",
        [("Report", paths.rel(out)), ("Best val top-1", best)],
        next_cmd=lsm("convert", "--run", best),
        elapsed=time.time() - ctx.started,
    )
    return 0
