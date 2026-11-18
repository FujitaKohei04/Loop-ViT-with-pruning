"""Save everything a script prints to logs/<script>_<YYYYmmdd_HHMMSS>[_<tag>].log.

    from run_logging import start_run_log, link_run_log, run_timestamp
    start_run_log(__file__, tag="bvit_s_cifar100", config=asdict(config))
    ...
    out_dir = Path("outputs") / f"name_{run_timestamp()}"   # same timestamp as the log
    link_run_log(out_dir)                                   # out_dir/logs/<log name> -> logs/<log name>

How it works: sys.stdout and sys.stderr are replaced by a Tee that writes to the
terminal as before and also to the log file, so existing print / tqdm / warnings
/ tracebacks are recorded without changing any code. The file gets
    header: start time, the command, cwd, git commit, environment, config
    body:   the output; tqdm's \r progress redraws are dropped, only finished lines are kept
    footer: end time, elapsed time, status (ok / exception / interrupted)
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

_current: RunLog | None = None


class Tee:
    """File-like object writing to the terminal and to the log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self.terminal = terminal
        self.log_file = log_file
        self.buffer = ""

    def write(self, text: str) -> int:
        self.terminal.write(text)
        if not self.log_file.closed:
            self.buffer += text
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                self.log_file.write(line.split("\r")[-1] + "\n")  # keep only the last redraw of the line
            self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        if not self.log_file.closed:
            self.log_file.flush()

    def flush_buffer(self) -> None:
        """Write an unfinished last line (no trailing newline)."""
        tail = self.buffer.split("\r")[-1]
        self.buffer = ""
        if tail and not self.log_file.closed:
            self.log_file.write(tail + "\n")

    def __getattr__(self, name: str):
        # isatty, fileno, encoding, ... behave like the terminal (tqdm relies on these)
        return getattr(self.terminal, name)


class RunLog:
    def __init__(self, path: Path, timestamp: str) -> None:
        self.path = path
        self.timestamp = timestamp
        self.started = time.time()
        self.status = "ok"
        self.pid = os.getpid()
        self.copies: list[Path] = []  # link targets where a symlink was not possible
        self.file = open(path, "w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        self.previous_excepthook = sys.excepthook

    def install(self) -> None:
        sys.stdout = Tee(self.stdout, self.file)
        sys.stderr = Tee(self.stderr, self.file)
        sys.excepthook = self._excepthook
        atexit.register(self.close)

    def _excepthook(self, exc_type, exc, tb) -> None:
        self.status = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else f"exception ({exc_type.__name__})"
        self.previous_excepthook(exc_type, exc, tb)  # prints the traceback through the Tee

    def link_into(self, directory: str | Path) -> Path:
        link_dir = Path(directory) / "logs"
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / self.path.name
        if not link.exists():
            try:
                link.symlink_to(os.path.relpath(self.path.resolve(), link_dir.resolve()))
            except OSError:
                self.copies.append(link)  # copied when the run ends
        return link

    def close(self) -> None:
        if self.file.closed or os.getpid() != self.pid:  # DataLoader workers must not close it
            return
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, Tee):
                stream.flush_buffer()
        elapsed = int(time.time() - self.started)
        self.file.write(f"===== end: {datetime.now():%Y-%m-%d %H:%M:%S} "
                        f"({elapsed // 3600}h{elapsed % 3600 // 60:02d}m{elapsed % 60:02d}s), "
                        f"status={self.status} =====\n")
        sys.stdout, sys.stderr, sys.excepthook = self.stdout, self.stderr, self.previous_excepthook
        self.file.close()
        for copy in self.copies:
            shutil.copy(self.path, copy)


def run_timestamp() -> str:
    """The timestamp of the current run (log file name); now if no log was started."""
    return _current.timestamp if _current is not None else datetime.now().strftime(TIMESTAMP_FORMAT)


def start_run_log(script: str, *, tag: str | None = None, log_dir: str | Path = "./logs",
                  config: dict | None = None, enabled: bool = True) -> RunLog | None:
    """Start logging this process. script: __file__ or a name. Returns None when disabled."""
    global _current
    if not enabled:
        return None
    if _current is not None:
        return _current

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    stem = f"{Path(script).stem}_{timestamp}" + (f"_{_safe(tag)}" if tag else "")
    path = log_dir / f"{stem}.log"
    suffix = 2
    while path.exists():  # two runs started in the same second
        path = log_dir / f"{stem}_{suffix}.log"
        suffix += 1

    run_log = RunLog(path, timestamp)
    run_log.file.write(_header(config))
    run_log.file.flush()
    run_log.install()
    _current = run_log
    print(f"log: {path}")
    return run_log


def link_run_log(directory: str | Path) -> None:
    """Make the current log reachable from an output directory (no-op without a log)."""
    if _current is not None:
        _current.link_into(directory)


def add_log_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-dir", default="./logs", help="where log files are written")
    parser.add_argument("--no-log", action="store_true", help="do not write a log file")


# * ===== Header =====

def _header(config: dict | None) -> str:
    lines = [
        "===== run log =====",
        f"start     : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"command   : {shlex.join([sys.executable, *sys.argv])}",
        f"cwd       : {os.getcwd()}",
        f"git       : {_git()}",
        f"host      : {socket.gethostname()} ({platform.platform()})",
        f"env       : {_environment()}",
    ]
    if config is not None:
        lines.append(f"config    : {json.dumps(config, default=str, ensure_ascii=False)}")
    lines.append("===================")
    return "\n".join(lines) + "\n"


def _git() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                 check=True).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return f"{commit} (uncommitted changes: {len(changed)} files)" if changed else f"{commit} (clean)"


def _environment() -> str:
    parts = [f"python {platform.python_version()}"]
    try:
        import torch
        parts.append(f"torch {torch.__version__}")
        parts.append(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")
    except ImportError:
        pass
    try:
        import timm
        parts.append(f"timm {timm.__version__}")
    except ImportError:
        pass
    return " / ".join(parts)


def _safe(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.+" else "_" for c in tag)
