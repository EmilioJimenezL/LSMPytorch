"""Subprocess runner: live output, per-command log files, dry-run support."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import paths
from .errors import EXIT_INTERRUPT, CliError, StepError, UserError, explain_failure
from .ui import format_cmd, ui

_TAIL_BYTES = 24_000


@dataclass
class Context:
    """Global flags shared by every command."""

    command: str = "lsm"
    dry_run: bool = False
    yes: bool = False
    started: float = field(default_factory=time.time)


@dataclass
class Cmd:
    """One external command to run as part of a task."""

    argv: List[str]
    label: str
    cwd: Optional[Path] = None
    env: Optional[Mapping[str, str]] = None
    ok_codes: Tuple[int, ...] = (0,)
    # Extra hint shown when the command fails and the output is not a known failure.
    fail_hint: Optional[str] = None
    # Called right before the command runs (not in dry-run); may raise CliError.
    before: Optional[Callable[[], None]] = None


@dataclass
class Result:
    returncode: int
    tail: str
    elapsed: float


class CommandLog:
    """Lazily-created log file: Workspace/logs/<timestamp>_<command>.log."""

    def __init__(self, command: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", command).strip("-") or "lsm"
        self.path = paths.LOGS_DIR / f"{stamp}_{safe}.log"
        self._fh = None
        self.command = command

    def _open(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "ab", buffering=0)
            self.write_text(f"# lsm {self.command}\n# started {datetime.now().isoformat(timespec='seconds')}\n"
                            f"# python {sys.version.split()[0]} on {sys.platform}\n")
        return self._fh

    def write_text(self, text: str) -> None:
        self._open().write(text.encode("utf-8", errors="replace"))

    def write_bytes(self, data: bytes) -> None:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        self._open().write(data)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def exists(self) -> bool:
        return self._fh is not None or self.path.exists()


def child_env(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # The orchestrator package dir on PYTHONPATH is irrelevant to wrapped scripts.
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def run_cmd(
    cmd: Cmd,
    *,
    log: Optional[CommandLog],
    dry_run: bool = False,
    check: bool = True,
) -> Result:
    """Run ``cmd``, streaming output (unless --quiet) and logging everything.

    * dry-run: prints the command and returns immediately with rc 0
    * failure: raises StepError (exit code 2) with a mapped, actionable message
    """
    printable = format_cmd(cmd.argv)
    if dry_run:
        ui.info(f"[dry-run] $ {printable}", "dim")
        return Result(0, "", 0.0)

    if cmd.before:
        cmd.before()

    ui.detail(f"$ {printable}" + (f"   (cwd={paths.rel(cmd.cwd)})" if cmd.cwd else ""))
    if log:
        log.write_text(f"\n$ {printable}\n# cwd={cmd.cwd or os.getcwd()}  label={cmd.label}\n")

    stream = not ui.quiet
    started = time.time()
    try:
        proc = subprocess.Popen(
            [str(a) for a in cmd.argv],
            cwd=str(cmd.cwd) if cmd.cwd else None,
            env=child_env(cmd.env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        raise UserError(
            f"Cannot run '{cmd.argv[0]}': executable not found.",
            hint="Run `lsm doctor` to see which tools are missing (or `lsm setup` for the Python env).",
            log_path=log.path if log and log.exists else None,
        )
    except OSError as exc:
        raise UserError(f"Cannot run '{cmd.argv[0]}': {exc}")

    tail = b""
    out = getattr(sys.stdout, "buffer", None)
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read1(65536)  # type: ignore[attr-defined]
            if not chunk:
                break
            if log:
                log.write_bytes(chunk)
            tail = (tail + chunk)[-_TAIL_BYTES:]
            if stream:
                if out is not None:
                    out.write(chunk)
                    out.flush()
                else:  # pragma: no cover - exotic stdout replacements
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log:
            log.write_text("\n# interrupted by user\n")
        raise CliError(
            "Interrupted (Ctrl-C).",
            hint="Partial outputs may exist. Re-run the same command (use --resume for `lsm train`).",
            exit_code=EXIT_INTERRUPT,
            log_path=log.path if log and log.exists else None,
        )

    elapsed = time.time() - started
    text = tail.decode("utf-8", errors="replace")
    if log:
        log.write_text(f"\n# exit code {rc} after {elapsed:.1f}s\n")

    if rc in cmd.ok_codes or not check:
        return Result(rc, text, elapsed)

    known = explain_failure(text)
    message = f"{cmd.label} failed (exit code {rc})."
    hint = cmd.fail_hint
    if known:
        message = f"{cmd.label} failed: {known[0]}"
        hint = known[1]
    # When output was not streamed (quiet), show the tail so the failure is diagnosable.
    details = None
    if not stream:
        lines = [ln for ln in text.splitlines() if ln.strip()][-12:]
        details = "\n".join(lines) if lines else None
    raise StepError(
        message,
        hint=hint or "See the log for the full output.",
        details=details,
        log_path=log.path if log else None,
    )
