"""`lsm` command line: argparse definition and command dispatch."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import paths
from .doctor import planned_checks, run_doctor
from .errors import EXIT_INTERRUPT, CliError, UserError
from .runner import Context
from .tasks import (
    CUDA_CHOICES,
    DEFAULT_LAYOUT,
    FAMILIES,
    LAYOUTS,
    MODEL_TO_FAMILY,
    MODELS,
    TrainOpts,
    build_convert,
    build_deploy,
    build_evaluate,
    build_extract,
    build_prepare,
    build_setup,
    build_test,
    build_train,
    dataset_label,
    execute_pipeline,
    execute_single,
    layout_short,
    run_report,
    validate_name,
)
from .ui import NORMAL, QUIET, VERBOSE, ui

PROG = "lsm"

# Workflow order (also used for the overview printed by plain `lsm`).
COMMANDS = [
    ("setup", "Create .venv and install PyTorch + requirements"),
    ("doctor", "Check Python/torch/device, Core ML, Xcode and git-lfs"),
    ("extract", "Videos -> landmark JSON (macOS, Swift + Apple Vision)"),
    ("prepare", "Landmark JSON -> prepared .npz cache + split manifest"),
    ("train", "Train a cnn / tcn / 3dcnn model into Runs/<name>"),
    ("convert", "PyTorch checkpoint -> Core ML .mlpackage (+ parity, fixture)"),
    ("deploy", "Copy a Core ML model into the iOS test app"),
    ("evaluate", "Evaluate a trained run on a landmark dataset"),
    ("report", "Compare metrics.json of several runs"),
    ("test", "Run the TrainingPipeline pytest suite"),
    ("pipeline", "End-to-end: extract -> prepare -> train -> convert -> deploy"),
]


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults (but not 'None'/'False'), keep epilog/description line breaks."""

    def __init__(self, prog: str) -> None:
        width = min(shutil.get_terminal_size((100, 24)).columns - 2, 100)
        super().__init__(prog, max_help_position=34, width=width)

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or ""
        if "%(default)" in text:
            return text
        if action.default is None or action.default is False or action.default is argparse.SUPPRESS:
            return text
        if not action.option_strings and action.nargs not in (argparse.OPTIONAL, argparse.ZERO_OR_MORE):
            return text
        return text + " (default: %(default)s)"


class Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors exit with 1 (2 is reserved for step failures)."""

    def error(self, message: str) -> "None":  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\nTry `{self.prog} --help` for usage and examples.\n")


def _epilog(outputs: Sequence[str], examples: Sequence[str], extra: str = "") -> str:
    out = "Outputs:\n" + "\n".join(f"  {line}" for line in outputs)
    ex = "Examples:\n" + "\n".join(f"  {line}" for line in examples)
    return f"{out}\n\n{ex}" + (f"\n\n{extra}" if extra else "")


def _common_parent() -> argparse.ArgumentParser:
    """Global flags, accepted both before and after the subcommand."""
    parent = Parser(add_help=False)
    group = parent.add_argument_group("global options")
    verbosity = group.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS,
                           help="Verbose: echo every command and detail")
    verbosity.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                           help="Quiet: only errors, warnings and result paths (full output still logged)")
    group.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                       help="Print the resolved plan and exit without running anything")
    group.add_argument("-y", "--yes", action="store_true", default=argparse.SUPPRESS,
                       help="Assume yes: overwrite existing outputs without asking")
    group.add_argument("--workspace", metavar="DIR", default=argparse.SUPPRESS,
                       help="Workspace dir for logs/prepared/landmarks/exports (default: <repo>/Workspace)")
    return parent


def build_parser() -> argparse.ArgumentParser:
    common = _common_parent()
    parser = Parser(
        prog=PROG,
        parents=[common],
        formatter_class=HelpFormatter,
        description="lsm - one CLI for the LSM (Mexican Sign Language) pipeline:\n"
                    "videos -> landmarks -> dataset -> training -> Core ML -> iOS app.",
        epilog="Start here:   lsm setup && lsm doctor\n"
               "Typical flow: lsm extract -> prepare -> train -> convert -> deploy   (or just: lsm pipeline)\n\n"
               "Every command prints its resolved plan first; add --dry-run to stop there.\n"
               "Logs of every run: Workspace/logs/<timestamp>_<command>.log\n"
               "Exit codes: 0 ok, 1 usage/environment problem, 2 a step failed.\n\n"
               "Run `lsm <command> --help` for flags, defaults and examples.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", title="commands")

    def add(name: str, summary: str, purpose: str, outputs: Sequence[str], examples: Sequence[str],
            extra: str = "") -> argparse.ArgumentParser:
        return sub.add_parser(
            name, help=summary, parents=[common], formatter_class=HelpFormatter,
            description=purpose, epilog=_epilog(outputs, examples, extra),
        )

    summaries = dict(COMMANDS)

    # ---- setup -----------------------------------------------------------
    p = add("setup", summaries["setup"],
            "Create the .venv at the repo root (if missing) and install PyTorch + TrainingPipeline/requirements.txt.",
            [".venv/                 virtual environment used by every other command"],
            ["lsm setup                 # macOS: default torch wheel (MPS); Linux/Windows: pip default",
             "lsm setup --cuda 12.4     # Linux/Windows: NVIDIA GPU wheel (cu124)",
             "lsm setup --cpu           # Linux/Windows: CPU-only wheel"],
            "On macOS --cuda/--cpu are ignored (the default wheel already uses the Apple GPU).")
    g = p.add_argument_group("PyTorch build (Linux/Windows)")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--cuda", nargs="?", const="12.4", choices=CUDA_CHOICES, metavar="VERSION",
                    help="Install the CUDA wheel from download.pytorch.org (VERSION one of "
                         + ", ".join(CUDA_CHOICES) + "; bare --cuda means 12.4)")
    mx.add_argument("--cpu", action="store_true", help="Install the CPU-only wheel")

    # ---- doctor ----------------------------------------------------------
    add("doctor", summaries["doctor"],
        "Report python/torch versions, compute device (cuda/mps/cpu), coremltools convert/predict support, "
        "xcodebuild and git-lfs. Read-only; exits 0 unless the repo layout is broken.",
        ["A status table plus suggested fixes (nothing is modified)"],
        ["lsm doctor", "lsm doctor -q"])

    # ---- extract ---------------------------------------------------------
    p = add("extract", summaries["extract"],
            "Extract hand/pose landmarks from videos with the Swift LSMExtractorCLI (macOS only; builds it with "
            "xcodebuild if no binary is found).",
            ["<output>/{category}/{word}/*_landmarks.json   one JSON per video",
             "<output>/dataset_manifest.json, _batch_report.json"],
            ["lsm extract --videos ~/Videos/LSM_LAB",
             "lsm extract --videos ~/Videos/LSM_LAB --output Workspace/landmarks --skip-existing",
             "lsm extract --videos ./videos --overlay --dry-run"],
            "Videos must be laid out as {videos}/{category}/{word}/*.mp4|mov|m4v.")
    g = p.add_argument_group("input / output")
    g.add_argument("--videos", metavar="DIR", help="Root folder of videos (prompted if omitted in a terminal)")
    g.add_argument("--output", metavar="DIR", help="Where to write landmark JSONs (default: <workspace>/landmarks)")
    g.add_argument("--manifest", metavar="FILE", help="Existing dataset_manifest.json to load and update")
    g = p.add_argument_group("behaviour")
    g.add_argument("--overlay", action="store_true", help="Also render {name}_overlay.mp4 debug videos")
    g.add_argument("--skip-existing", action="store_true", help="Skip videos whose *_landmarks.json already exists")
    g.add_argument("--rebuild", action="store_true", help="Rebuild LSMExtractorCLI even if a binary exists")

    # ---- prepare ---------------------------------------------------------
    p = add("prepare", summaries["prepare"],
            "Parse landmark JSONs once and write a prepared cache (.npz) with a fixed train/val/extra split "
            "(wraps TrainingPipeline/prepare_dataset.py).",
            ["<workspace>/prepared/<name>.npz           tensors + split + settings hash",
             "<workspace>/prepared/<name>.split.json    human-readable split manifest"],
            ["lsm prepare --landmarks Workspace/landmarks",
             "lsm prepare --landmarks Workspace/landmarks --layout v1_135 --cap 10 --name lab_v1_cap10",
             "lsm prepare --landmarks ./landmarks --min-videos 3 --val-split 0.15 --dry-run"])
    g = p.add_argument_group("input / output")
    g.add_argument("--landmarks", metavar="DIR", help="Folder with *_landmarks.json (default prompt: <workspace>/landmarks)")
    g.add_argument("--name", metavar="NAME", help="Cache name (default: <dataset>_<layout>_cap<N>)")
    g.add_argument("--output", metavar="FILE", help="Explicit .npz path (overrides --name)")
    g = p.add_argument_group("dataset options")
    g.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT,
                   help="Feature layout: v2_162 (current) or v1_135 (legacy shoulder-normalised)")
    g.add_argument("--min-videos", "--min_videos", dest="min_videos", type=int, default=5,
                   metavar="N", help="Drop words with fewer than N videos")
    g.add_argument("--cap", "--max-videos-per-class", "--max_videos_per_class", dest="cap", type=int, default=5,
                   metavar="N", help="Keep at most N videos per word for train+val (rest become extra test)")
    g.add_argument("--val-split", "--val_split", dest="val_split", type=float, default=0.2,
                   metavar="FRAC", help="Validation fraction of the capped videos")
    g.add_argument("--seed", type=int, default=42, help="Split seed")
    g.add_argument("--no-shoulder-norm", action="store_true", help="Disable shoulder normalisation (v1 layout only)")

    # ---- train -----------------------------------------------------------
    p = add("train", summaries["train"],
            "Train a CNN, TCN or 3D-CNN on landmark sequences (wraps TrainingPipeline/train.py). "
            "On success the best checkpoint is copied to Models/Pytorch/<name>.pt.",
            ["Runs/<name>/best_model.pt, last.pt, classes.json, metrics.json, history.json, training_curves.png",
             "Runs/<name>/README.md                      skeleton from metrics.json (never overwrites yours)",
             "Models/Pytorch/<name>.pt                   copy of the best checkpoint"],
            ["lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz",
             "lsm train --model tcn --cache Workspace/prepared/lab_v2_cap5.npz --pool attn --causal 0 --name lab_tcn_attn",
             "lsm train --model cnn --landmarks Workspace/landmarks --cap 5 --epochs 60 --batch 32"],
            "Default run name: <layout_short>_<model>, e.g. v2_cnn. Resume with --resume (uses Runs/<name>/last.pt).")
    g = p.add_argument_group("data (one required; prompted in a terminal)")
    g.add_argument("--landmarks", metavar="DIR", help="Raw landmark JSON folder (parsed on the fly)")
    g.add_argument("--cache", metavar="FILE", help="Prepared .npz from `lsm prepare` (recommended)")
    g.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT,
                   help="Feature layout for --landmarks (a --cache dictates its own)")
    g.add_argument("--min-videos", "--min_videos", dest="min_videos", type=int, default=2, metavar="N",
                   help="--landmarks only: drop words with fewer than N videos")
    g.add_argument("--cap", "--max-videos-per-class", "--max_videos_per_class", dest="cap", type=int, default=None,
                   metavar="N", help="--landmarks only: cap videos per word (omit for no cap)")
    g.add_argument("--val-split", "--val_split", dest="val_split", type=float, default=0.15, metavar="FRAC",
                   help="--landmarks only: validation fraction")
    g.add_argument("--no-shoulder-norm", action="store_true", help="--landmarks only: disable shoulder norm (v1)")
    g = p.add_argument_group("model")
    g.add_argument("--model", choices=MODELS, default="cnn", help="Architecture")
    g.add_argument("--pool", choices=["avg", "attn"], default="avg", help="Temporal pooling")
    g.add_argument("--causal", type=int, choices=[0, 1], default=1, help="TCN only: 1 causal padding, 0 symmetric")
    g.add_argument("--dropout", type=float, default=0.3, help="Dropout before the classifier")
    g.add_argument("--joint-dropout", "--joint_dropout", dest="joint_dropout", type=float, default=0.1,
                   metavar="P", help="Probability of dropping whole joints during training")
    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    g.add_argument("--batch", type=int, default=64, help="Batch size (lower it on GPU out-of-memory)")
    g.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (AdamW + cosine)")
    g.add_argument("--warmup", type=int, default=10, help="Linear warm-up epochs")
    g.add_argument("--patience", type=int, default=15, help="Early-stopping patience in epochs (0 disables)")
    g = p.add_argument_group("augmentation")
    g.add_argument("--augment", type=int, default=10,
                   help="Augmented views per train video per epoch (static factor with --legacy-augment)")
    g.add_argument("--legacy-augment", action="store_true", help="Old static CPU augmentation instead of on-GPU")
    g = p.add_argument_group("run control")
    g.add_argument("--name", metavar="NAME", help="Run name -> Runs/<name> (default: <layout_short>_<model>)")
    g.add_argument("--resume", nargs="?", const="auto", metavar="CKPT",
                   help="Resume from CKPT; bare --resume uses Runs/<name>/last.pt")
    g.add_argument("--workers", type=int, default=None, metavar="N", help="DataLoader workers (train.py picks if omitted)")

    # ---- convert ---------------------------------------------------------
    p = add("convert", summaries["convert"],
            "Convert a trained PyTorch checkpoint to a Core ML .mlpackage (wraps CoreMLConversion/convert_to_coreml.py), "
            "optionally verifying Core ML == PyTorch and writing a golden fixture for the iOS tests.",
            ["Models/CoreML/<name>.mlpackage",
             "Models/CoreML/<name>_classes.json     (copied from the run; needed by `lsm deploy`)",
             "Models/CoreML/<name>_fixture.json     (with --cache; first validation sample + logits)"],
            ["lsm convert --run v2_cnn --cache Workspace/prepared/lab_v2_cap5.npz",
             "lsm convert --checkpoint Models/Pytorch/lab_cap5_cnn.pt --no-parity",
             "lsm convert --run v2_tcn --output Models/CoreML/tcn_demo.mlpackage"],
            "Parity needs macOS (coremltools predict); elsewhere it is skipped automatically with a warning.")
    g = p.add_argument_group("source (one required; run picker shown in a terminal)")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--run", metavar="NAME", help="Use Runs/<NAME>/best_model.pt")
    mx.add_argument("--checkpoint", metavar="FILE", help="Explicit .pt checkpoint")
    g.add_argument("--name", metavar="NAME", help="Output stem (default: run name / checkpoint name)")
    g = p.add_argument_group("output and checks")
    g.add_argument("--output", metavar="PATH", help="Output .mlpackage path (default: Models/CoreML/<name>.mlpackage)")
    g.add_argument("--cache", metavar="FILE", help="Prepared .npz for parity + fixture (auto-detected by name if possible)")
    g.add_argument("--parity", action=argparse.BooleanOptionalAction, default=None,
                   help="Fail if Core ML differs from PyTorch (--no-parity to skip); default: on when a cache is known (macOS)")
    g.add_argument("--fixture", action=argparse.BooleanOptionalAction, default=None,
                   help="Write the golden fixture JSON (--no-fixture to skip); default: on when a cache is known")

    # ---- deploy ----------------------------------------------------------
    p = add("deploy", summaries["deploy"],
            "Copy Models/CoreML/<run>.mlpackage + classes (+ fixture) into the iOS testing app and print the "
            "AVAILABLE_MODELS snippet (wraps CoreMLConversion/export_to_app.py).",
            ["TestingTools/LSMMobileModelTesting/LSMMobileModelTesting/<run>.mlpackage",
             "..._classes.json and ..._phase2_fixture.json next to it, plus a Swift snippet to paste"],
            ["lsm deploy --run v2_cnn",
             "lsm deploy --run v2_tcn --sign-set labCap5 --display-name \"LAB TCN\"",
             "lsm deploy --run v2_cnn --dry-run"])
    g = p.add_argument_group("model")
    g.add_argument("--run", metavar="NAME", help="Run/export name (picker shown in a terminal)")
    g.add_argument("--family", choices=FAMILIES, default=None,
                   help="Swift model family (auto: from the run's metrics.json / name)")
    g.add_argument("--sign-set", default="labCap5", metavar="SET", help="Swift SignSet case")
    g.add_argument("--display-name", "--name", dest="display_name", metavar="TEXT",
                   help="Display name in the app (default: run name)")
    g.add_argument("--description", metavar="TEXT", help="Description for the Swift snippet (default: class count)")

    # ---- evaluate --------------------------------------------------------
    p = add("evaluate", summaries["evaluate"],
            "Evaluate Runs/<run>/best_model.pt on a landmark dataset (wraps TrainingPipeline/evaluation/evaluate_nn.py).",
            ["Runs/<name>/evaluation.json   (or --output)"],
            ["lsm evaluate --run v2_cnn --dataset Workspace/landmarks",
             "lsm evaluate --run v2_tcn --dataset ~/LSM_EXTRA/landmarks --min-videos 1"])
    g = p.add_argument_group("inputs")
    g.add_argument("--run", metavar="NAME", help="Run to evaluate (picker shown in a terminal)")
    g.add_argument("--dataset", metavar="DIR", help="Landmark JSON folder (prompted in a terminal)")
    g.add_argument("--min-videos", "--min_videos", dest="min_videos", type=int, default=2, metavar="N",
                   help="Drop words with fewer than N videos")
    g.add_argument("--output", metavar="FILE", help="Report JSON path (default: Runs/<name>/evaluation.json)")

    # ---- report ----------------------------------------------------------
    p = add("report", summaries["report"],
            "Print and save a side-by-side summary of several runs from their Runs/<name>/metrics.json.",
            ["<workspace>/exports/report_<timestamp>.md   (or --output)"],
            ["lsm report --runs v2_cnn v2_tcn",
             "lsm report --runs lab_cap5_cnn lab_cap5_tcn --output Workspace/exports/lab.md"])
    p.add_argument("--runs", nargs="+", required=True, metavar="NAME", help="Run names to compare")
    p.add_argument("--output", metavar="FILE", help="Markdown file to write (default: <workspace>/exports/report_<ts>.md)")

    # ---- test ------------------------------------------------------------
    p = add("test", summaries["test"],
            "Run pytest in TrainingPipeline/ using the project .venv.",
            ["pytest output; exit code 2 if any test fails"],
            ["lsm test", "lsm test --quick", "lsm test -- -k augment -x"],
            "Anything after `--` is passed to pytest.")
    p.add_argument("--quick", action="store_true", help="Quiet run that skips tests marked slow (-q -m 'not slow')")
    p.add_argument("pytest_args", nargs="*", metavar="PYTEST_ARG", help="Extra arguments for pytest (after --)")

    # ---- pipeline --------------------------------------------------------
    p = add("pipeline", summaries["pipeline"],
            "Run the whole workflow: extract (unless skipped) -> prepare -> train each model -> convert -> deploy.",
            ["Workspace/landmarks/, Workspace/prepared/<prefix>.npz",
             "Runs/<prefix>_<model>/ and Models/Pytorch/<prefix>_<model>.pt for each model",
             "Models/CoreML/<prefix>_<model>.mlpackage (+ classes, fixture) and the iOS app bundle copy"],
            ["lsm pipeline --videos ~/Videos/LSM_LAB --models cnn,tcn --epochs 80",
             "lsm pipeline --landmarks Workspace/landmarks --models tcn --no-deploy --name lab",
             "lsm pipeline --videos ./videos --skip-extract --dry-run"],
            "Stages are numbered [i/N]; if one fails you get the equivalent single-stage `lsm ...` command to retry.")
    g = p.add_argument_group("source (at least one; prompted in a terminal)")
    g.add_argument("--videos", metavar="DIR", help="Video root to extract landmarks from (macOS)")
    g.add_argument("--landmarks", metavar="DIR",
                   help="Existing landmarks folder; with --videos it is where extraction writes (default: <workspace>/landmarks)")
    g.add_argument("--skip-extract", action="store_true", help="Do not extract; use existing landmarks")
    g.add_argument("--overlay", action="store_true", help="Extraction: also render overlay videos")
    g.add_argument("--skip-existing", action="store_true", help="Extraction: skip videos that already have landmarks")
    g = p.add_argument_group("models and naming")
    g.add_argument("--models", default="cnn", metavar="LIST", help="Comma-separated: " + ", ".join(MODELS))
    g.add_argument("--name", metavar="PREFIX", help="Name prefix (default: <dataset>_<layout>_cap<N>)")
    g.add_argument("--no-deploy", action="store_true", help="Stop after convert (do not touch the iOS app)")
    g.add_argument("--sign-set", default="labCap5", metavar="SET", help="Swift SignSet case for deploy")
    g = p.add_argument_group("dataset")
    g.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT, help="Feature layout")
    g.add_argument("--cap", "--max-videos-per-class", "--max_videos_per_class", dest="cap", type=int, default=5,
                   metavar="N", help="Max videos per word for train+val")
    g.add_argument("--min-videos", "--min_videos", dest="min_videos", type=int, default=5, metavar="N",
                   help="Drop words with fewer than N videos")
    g.add_argument("--val-split", "--val_split", dest="val_split", type=float, default=0.2, metavar="FRAC",
                   help="Validation fraction")
    g.add_argument("--seed", type=int, default=42, help="Split seed")
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100, help="Maximum epochs per model")
    g.add_argument("--batch", type=int, default=64, help="Batch size")
    g.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    g.add_argument("--pool", choices=["avg", "attn"], default="avg", help="Temporal pooling")
    g.add_argument("--patience", type=int, default=15, help="Early-stopping patience (0 disables)")
    g.add_argument("--workers", type=int, default=None, metavar="N", help="DataLoader workers")

    return parser


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _p(value: Optional[str]) -> Optional[Path]:
    return paths.resolve_user_path(value) if value else None


def cmd_setup(args: argparse.Namespace, ctx: Context) -> int:
    return execute_single(ctx, build_setup(ctx, args.cuda, args.cpu))


def cmd_doctor(args: argparse.Namespace, ctx: Context) -> int:
    if ctx.dry_run:
        ui.print_plan("lsm doctor", [("Would check", "; ".join(planned_checks()))], dry_run=True)
        return 0
    return run_doctor(ctx)


def cmd_extract(args: argparse.Namespace, ctx: Context) -> int:
    if not paths.is_macos():  # fail before prompting for anything
        build_extract(ctx, Path("."), Path("."), False, False, None, False)
    videos = _p(args.videos) or ui.prompt_path("Folder with videos ({category}/{word}/*.mp4)", flag="--videos")
    output = _p(args.output) or paths.LANDMARKS_DIR
    task = build_extract(ctx, videos, output, args.overlay, args.skip_existing, _p(args.manifest), args.rebuild)
    return execute_single(ctx, task)


def _default_landmarks() -> Optional[Path]:
    return paths.LANDMARKS_DIR if paths.LANDMARKS_DIR.is_dir() else None


def cmd_prepare(args: argparse.Namespace, ctx: Context) -> int:
    landmarks = _p(args.landmarks) or ui.prompt_path(
        "Landmarks folder (output of `lsm extract`)", flag="--landmarks", default=_default_landmarks())
    task = build_prepare(
        ctx, landmarks, args.layout, args.min_videos, args.cap, args.val_split, args.seed,
        args.name, args.no_shoulder_norm, _p(args.output),
    )
    return execute_single(ctx, task)


def cmd_train(args: argparse.Namespace, ctx: Context) -> int:
    landmarks, cache = _p(args.landmarks), _p(args.cache)
    if not landmarks and not cache:
        chosen = ui.prompt_path("Prepared cache (.npz) or landmarks folder", flag="--cache/--landmarks",
                                kind="any", default=_default_landmarks())
        if chosen.is_file():
            cache = chosen
        else:
            landmarks = chosen
    opts = TrainOpts(
        model=args.model, landmarks=landmarks, cache=cache, name=args.name, layout=args.layout,
        epochs=args.epochs, batch=args.batch, lr=args.lr, dropout=args.dropout, val_split=args.val_split,
        augment=args.augment, min_videos=args.min_videos, cap=args.cap, causal=args.causal, pool=args.pool,
        joint_dropout=args.joint_dropout, warmup=args.warmup, patience=args.patience,
        legacy_augment=args.legacy_augment, no_shoulder_norm=args.no_shoulder_norm,
        resume=args.resume, workers=args.workers,
    )
    return execute_single(ctx, build_train(ctx, opts))


def cmd_convert(args: argparse.Namespace, ctx: Context) -> int:
    task = build_convert(
        ctx, args.run, _p(args.checkpoint), args.name, _p(args.output), _p(args.cache), args.parity, args.fixture,
    )
    return execute_single(ctx, task)


def cmd_deploy(args: argparse.Namespace, ctx: Context) -> int:
    task = build_deploy(ctx, args.run, args.sign_set, args.display_name, args.family, args.description)
    return execute_single(ctx, task)


def cmd_evaluate(args: argparse.Namespace, ctx: Context) -> int:
    task = build_evaluate(ctx, args.run, _p(args.dataset), args.min_videos, _p(args.output))
    return execute_single(ctx, task)


def cmd_report(args: argparse.Namespace, ctx: Context) -> int:
    return run_report(ctx, args.runs, _p(args.output))


def cmd_test(args: argparse.Namespace, ctx: Context) -> int:
    return execute_single(ctx, build_test(ctx, args.quick, args.pytest_args))


def cmd_pipeline(args: argparse.Namespace, ctx: Context) -> int:
    models: List[str] = []
    for m in [s.strip().lower() for s in args.models.split(",") if s.strip()]:
        if m not in MODELS:
            raise UserError(f"Unknown model '{m}' in --models.", hint=f"Choose from: {', '.join(MODELS)} (comma-separated).")
        if m not in models:
            models.append(m)
    if not models:
        raise UserError("--models is empty.", hint="Example: --models cnn,tcn")

    videos, landmarks = _p(args.videos), _p(args.landmarks)
    if not videos and not landmarks:
        landmarks = ui.prompt_path(
            "Landmarks folder (or re-run with --videos DIR to extract first)", flag="--videos or --landmarks",
            default=_default_landmarks())
    if args.name:
        validate_name(args.name)

    do_extract = videos is not None and not args.skip_extract
    lm_dir = landmarks or paths.LANDMARKS_DIR
    notes: List[str] = []
    if (args.overlay or args.skip_existing) and not do_extract:
        notes.append("--overlay/--skip-existing only apply to extraction, which is skipped.")
    if videos is not None and args.skip_extract and landmarks is None and not lm_dir.is_dir():
        notes.append(f"--skip-extract but {lm_dir} does not exist yet; the real run would fail.")

    prefix = args.name or f"{dataset_label(videos or lm_dir)}_{layout_short(args.layout)}_cap{args.cap}"
    cache = paths.PREPARED_DIR / f"{prefix}.npz"

    tasks = []
    if do_extract:
        assert videos is not None
        tasks.append(build_extract(ctx, videos, lm_dir, args.overlay, args.skip_existing, None, False))
    tasks.append(build_prepare(
        ctx, lm_dir, args.layout, args.min_videos, args.cap, args.val_split, args.seed, prefix,
        False, None, landmarks_will_exist=do_extract,
    ))
    for m in models:
        run_name = f"{prefix}_{m}"
        opts = TrainOpts(
            model=m, cache=cache, name=run_name, layout=args.layout, epochs=args.epochs, batch=args.batch,
            lr=args.lr, pool=args.pool, patience=args.patience, workers=args.workers,
        )
        tasks.append(build_train(ctx, opts, data_will_exist=True))
        tasks.append(build_convert(ctx, run_name, None, None, None, cache, None, None, ckpt_will_exist=True))
        if not args.no_deploy:
            tasks.append(build_deploy(
                ctx, run_name, args.sign_set, None, MODEL_TO_FAMILY[m], None,
                will_exist=True, fixture_expected=True,
            ))

    header = [
        ("Source", f"videos {videos}" if do_extract else f"landmarks {lm_dir}"),
        ("Landmarks", str(lm_dir)),
        ("Prefix", prefix),
        ("Models", ", ".join(models)),
        ("Layout / cap", f"{args.layout} / {args.cap}"),
        ("Epochs", str(args.epochs)),
        ("Deploy to iOS app", "no" if args.no_deploy else "yes"),
    ]
    return execute_pipeline(ctx, header, tasks, notes)


HANDLERS: Dict[str, Callable[[argparse.Namespace, Context], int]] = {
    "setup": cmd_setup,
    "doctor": cmd_doctor,
    "extract": cmd_extract,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "convert": cmd_convert,
    "deploy": cmd_deploy,
    "evaluate": cmd_evaluate,
    "report": cmd_report,
    "test": cmd_test,
    "pipeline": cmd_pipeline,
}


def print_overview() -> None:
    ui.out.print("lsm - LSM sign-language pipeline orchestrator", "head")
    ui.out.print()
    ui.out.print("Commands (in workflow order):", "bold")
    width = max(len(n) for n, _ in COMMANDS)
    for name, desc in COMMANDS:
        ui.out.print(f"  {name.ljust(width)}  {desc}")
    ui.out.print()
    ui.out.print("Global flags: -v/--verbose  -q/--quiet  --dry-run  -y/--yes  --workspace DIR", "dim")
    ui.out.print()
    ui.out.print("Start here:  lsm setup && lsm doctor", "bold")
    ui.out.print("All-in-one:  lsm pipeline --videos DIR   (see `lsm pipeline --help`)")
    ui.out.print("Details:     lsm <command> --help")


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "quiet", False):
        verbosity = QUIET
    elif getattr(args, "verbose", 0):
        verbosity = VERBOSE
    else:
        verbosity = NORMAL
    yes = bool(getattr(args, "yes", False))
    dry_run = bool(getattr(args, "dry_run", False))
    ui.configure(verbosity=verbosity, assume_yes=yes)

    if args.command is None:
        print_overview()
        return 0

    try:
        paths.set_workspace(getattr(args, "workspace", None))
        ctx = Context(command=args.command, dry_run=dry_run, yes=yes)
        return HANDLERS[args.command](args, ctx)
    except CliError as err:
        ui.print_error(err)
        return err.exit_code
    except KeyboardInterrupt:
        ui.err.print("Interrupted.", "warn")
        return EXIT_INTERRUPT


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
