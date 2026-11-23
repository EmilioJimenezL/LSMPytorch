"""Console helpers: rich when available, plain text otherwise.

Honours ``NO_COLOR``, ``TERM=dumb`` and non-TTY output (no escape codes are
written when stdout/stderr is redirected). Set ``LSM_NO_RICH=1`` to force the
plain fallback.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from . import paths
from .errors import CliError, UserError

QUIET, NORMAL, VERBOSE = 0, 1, 2

try:  # rich is optional
    if os.environ.get("LSM_NO_RICH"):
        raise ImportError("rich disabled by LSM_NO_RICH")
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText

    HAVE_RICH = True
except Exception:  # pragma: no cover - depends on environment
    HAVE_RICH = False

Row = Tuple[str, str]

_RICH_STYLES = {
    "ok": "green",
    "warn": "yellow",
    "err": "bold red",
    "dim": "dim",
    "head": "bold cyan",
    "bold": "bold",
    "cmd": "cyan",
}
_ANSI = {"ok": "32", "warn": "33", "err": "1;31", "dim": "2", "head": "1;36", "bold": "1", "cmd": "36"}


def _stream_supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None and os.environ.get("NO_COLOR") != "":
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def is_interactive() -> bool:
    """True when we can ask the user questions (stdin and stdout are TTYs)."""
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def shorten(text: object) -> str:
    """Display helper: strip the repo root prefix from absolute paths."""
    s = str(text)
    root = str(paths.REPO_ROOT) + os.sep
    return s.replace(root, "")


def format_cmd(argv: Sequence[object]) -> str:
    parts = [shorten(a) for a in argv]
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    import shlex

    return " ".join(shlex.quote(p) for p in parts)


class Console:
    """Tiny console wrapper: rich if installed, ANSI/plain text otherwise."""

    def __init__(self, *, stderr: bool = False) -> None:
        self.stderr = stderr
        self.stream = sys.stderr if stderr else sys.stdout
        self.color = _stream_supports_color(self.stream)
        self._rich = None
        if HAVE_RICH:
            if self.color:
                self._rich = _RichConsole(stderr=stderr, highlight=False, soft_wrap=True)
            else:
                # Not a TTY, NO_COLOR set or TERM=dumb: emit no escape codes at all
                # (even if FORCE_COLOR leaks in from the environment).
                self._rich = _RichConsole(
                    stderr=stderr, highlight=False, soft_wrap=True, force_terminal=False,
                    color_system=None, no_color=True, force_interactive=False,
                )

    @property
    def width(self) -> int:
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except Exception:
            return 100

    def print(self, text: str = "", style: Optional[str] = None) -> None:
        if self._rich is not None:
            self._rich.print(_RichText(text, style=_RICH_STYLES.get(style or "", "")), soft_wrap=True)
            return
        if style and self.color and text:
            text = f"\033[{_ANSI[style]}m{text}\033[0m"
        print(text, file=self.stream, flush=True)

    def rule(self, title: str = "") -> None:
        if self._rich is not None:
            self._rich.rule(_RichText(title, style=_RICH_STYLES["head"]), style="cyan")
            return
        width = min(self.width, 100)
        if title:
            line = f"== {title} " + "=" * max(3, width - len(title) - 4)
        else:
            line = "=" * width
        self.print(line, "head")

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        title: Optional[str] = None,
        *,
        flex_last: bool = False,
    ) -> None:
        """Print a table. With ``flex_last`` the last column may wrap; others never shrink."""
        if self._rich is not None:
            table = _RichTable(title=title, header_style="bold", show_edge=False, pad_edge=False)
            for i, h in enumerate(headers):
                if flex_last and i == len(headers) - 1:
                    table.add_column(h, overflow="fold")
                else:
                    width = max([len(h)] + [len(str(r[i])) for r in rows])
                    table.add_column(h, overflow="fold", no_wrap=True, min_width=width)
            for row in rows:
                table.add_row(*[_RichText(str(c)) for c in row])
            self._rich.print(table)
            return
        if title:
            self.print(title, "bold")
        cells = [[str(c) for c in r] for r in rows]
        widths = [len(h) for h in headers]
        for r in cells:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(c))
        fmt = "  ".join("{:<%d}" % w for w in widths)
        self.print(fmt.format(*headers).rstrip(), "bold")
        self.print(fmt.format(*["-" * w for w in widths]), "dim")
        for r in cells:
            self.print(fmt.format(*r).rstrip())

    def kv(self, rows: Iterable[Row], indent: int = 2) -> None:
        rows = [(str(k), shorten(v)) for k, v in rows]
        if not rows:
            return
        width = max(len(k) for k, _ in rows)
        for k, v in rows:
            self.print(f"{' ' * indent}{k.ljust(width)}  {v}")


class UI:
    """Verbosity-aware front end used by all commands."""

    def __init__(self) -> None:
        self.verbosity = NORMAL
        self.assume_yes = False
        self.out = Console()
        self.err = Console(stderr=True)

    def configure(self, *, verbosity: int = NORMAL, assume_yes: bool = False) -> None:
        self.verbosity = verbosity
        self.assume_yes = assume_yes

    # ---- basic output -----------------------------------------------------
    @property
    def quiet(self) -> bool:
        return self.verbosity <= QUIET

    @property
    def verbose(self) -> bool:
        return self.verbosity >= VERBOSE

    def info(self, msg: str = "", style: Optional[str] = None) -> None:
        if self.verbosity >= NORMAL:
            self.out.print(shorten(msg), style)

    def detail(self, msg: str = "") -> None:
        if self.verbosity >= VERBOSE:
            self.out.print(shorten(msg), "dim")

    def ok(self, msg: str) -> None:
        self.info(f"[ok] {msg}", "ok")

    def warn(self, msg: str) -> None:
        # Warnings are shown unless quiet; they go to stderr like errors.
        if self.verbosity >= NORMAL:
            self.err.print(f"[warn] {shorten(msg)}", "warn")

    # ---- structured output ------------------------------------------------
    def print_plan(
        self,
        title: str,
        rows: Sequence[Row],
        commands: Sequence[Sequence[object]] = (),
        *,
        notes: Sequence[str] = (),
        dry_run: bool = False,
    ) -> None:
        if self.quiet and not dry_run:
            return
        self.out.print()
        self.out.print(f"Plan: {title}" + ("  (dry run)" if dry_run else ""), "head")
        self.out.kv(rows)
        if commands:
            self.out.print("  Commands:", "bold")
            for cmd in commands:
                self.out.print(f"    $ {format_cmd(cmd)}", "cmd")
        for note in notes:
            self.out.print(f"  note: {shorten(note)}", "warn")
        self.out.print()

    def print_stage(self, index: int, total: int, title: str) -> None:
        if self.quiet:
            return
        self.out.print()
        self.out.rule(f"[{index}/{total}] {title}")

    def print_summary(
        self,
        title: str,
        rows: Sequence[Row],
        *,
        next_cmd: Optional[str] = None,
        log_path: Optional[Path] = None,
        elapsed: Optional[float] = None,
    ) -> None:
        if self.quiet:
            for _, value in rows:
                self.out.print(shorten(value))
            return
        self.out.print()
        suffix = f"  ({_fmt_elapsed(elapsed)})" if elapsed is not None else ""
        self.out.print(f"Done: {title}{suffix}", "ok")
        self.out.kv(rows)
        if log_path:
            self.out.print(f"  Log  {shorten(log_path)}", "dim")
        if next_cmd:
            self.out.print()
            self.out.print(f"Next: {next_cmd}", "bold")

    def print_error(self, err: BaseException) -> None:
        if isinstance(err, CliError):
            self.err.print(f"error: {shorten(err.message)}", "err")
            if err.details:
                for line in err.details.rstrip().splitlines():
                    self.err.print(f"  | {shorten(line)}", "dim")
            if err.hint:
                self.err.print(f"  fix: {shorten(err.hint)}", "warn")
            if err.log_path:
                self.err.print(f"  log: {shorten(err.log_path)}", "dim")
        else:
            self.err.print(f"error: {err}", "err")

    # ---- interaction ------------------------------------------------------
    def confirm(self, prompt: str, default: bool = False) -> bool:
        """Ask y/N. ``--yes`` answers yes; non-interactive sessions get ``default``."""
        if self.assume_yes:
            return True
        if not is_interactive():
            return default
        suffix = " [Y/n] " if default else " [y/N] "
        try:
            answer = input(prompt + suffix).strip().lower()
        except EOFError:
            return default
        if not answer:
            return default
        return answer in ("y", "yes", "s", "si", "sí")

    def prompt_path(
        self,
        label: str,
        *,
        flag: str,
        kind: str = "dir",
        must_exist: bool = True,
        default: Optional[Path] = None,
    ) -> Path:
        """Ask for a path when stdin is a TTY, otherwise fail with the flag to use."""
        if not is_interactive():
            raise UserError(
                f"Missing required option {flag} ({label}).",
                hint=f"Pass {flag} PATH, or run in an interactive terminal to be prompted.",
            )
        shown = f" [{shorten(default)}]" if default else ""
        for _ in range(5):
            try:
                raw = input(f"{label}{shown}: ").strip()
            except EOFError:
                break
            if not raw and default is not None:
                raw = str(default)
            if not raw:
                continue
            p = paths.resolve_user_path(raw)
            if must_exist and not p.exists():
                self.err.print(f"  not found: {p}", "warn")
                continue
            if must_exist and kind == "dir" and not p.is_dir():
                self.err.print(f"  not a directory: {p}", "warn")
                continue
            if must_exist and kind == "file" and not p.is_file():
                self.err.print(f"  not a file: {p}", "warn")
                continue
            return p
        raise UserError(f"No valid path given for {label}.", hint=f"Pass {flag} PATH.")

    def choose(self, label: str, options: Sequence[str], *, flag: str) -> str:
        """Pick one of ``options`` by number or name (interactive only)."""
        if not is_interactive():
            raise UserError(
                f"Missing required option {flag} ({label}).",
                hint=f"Pass {flag} NAME. Available: {', '.join(options) if options else '(none yet)'}",
            )
        if not options:
            raise UserError(f"No choices available for {label}.", hint="Train something first: `lsm train ...`.")
        self.out.print(f"{label}:", "bold")
        for i, opt in enumerate(options, 1):
            self.out.print(f"  {i:>2}. {opt}")
        for _ in range(5):
            try:
                raw = input("Choose number or name: ").strip()
            except EOFError:
                break
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            if raw in options:
                return raw
        raise UserError(f"No valid choice for {label}.", hint=f"Pass {flag} NAME.")


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# Module-level singleton + function aliases (spec'd helper names).
ui = UI()
print_plan = ui.print_plan
print_stage = ui.print_stage
print_summary = ui.print_summary
print_error = ui.print_error
confirm = ui.confirm
prompt_path = ui.prompt_path
