"""Utilities for logging."""

import logging
import platform
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

try:
    from mpi4py import MPI

    _rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _rank = 0

_TIME_FORMAT: str = "%d.%m.%Y %H:%M:%S"


def _write_header(file_path: Path) -> None:
    now = datetime.now().strftime(_TIME_FORMAT)
    pyver = platform.python_version()
    hostname = platform.node()
    size = 1
    try:
        size = MPI.COMM_WORLD.Get_size()
    except Exception:
        pass

    header_lines = [
        f"# Session start: {now}",
        f"# Python: {pyver}",
        f"# Host: {hostname}",
        f"# MPI ranks: {size}\n",
        "",
    ]
    file_path.write_text("\n".join(header_lines))


def setup_logging(verbose: bool, *, output_path: Path | None = None) -> None:
    """Set up console + file logging. Only rank 0 emits INFO/DEBUG; others WARN+."""
    root = logging.getLogger()
    if root.handlers:
        return

    level = logging.DEBUG if verbose else logging.INFO

    console = Console(force_terminal=True, color_system="auto")
    console_handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    console_handler.setLevel(level)
    root.setLevel(level)
    root.addHandler(console_handler)

    log_file = output_path or Path(".") / "log.log"
    _write_header(log_file)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt=_TIME_FORMAT,
        )
    )
    root.addHandler(file_handler)


def log_global(logger: logging.Logger, level: int, msg: str, *args, **kwargs) -> None:
    """Log globally (in parallel runs, this means logging only on rank 0)."""
    if _rank == 0:
        logger.log(level, msg, *args, stacklevel=2, **kwargs)


def log_rank(logger: logging.Logger, level: int, msg: str, *args, **kwargs) -> None:
    """Log on all ranks, prefixing the message with the rank."""
    logger.log(level, f"[{_rank:d}] {msg}", *args, stacklevel=2, **kwargs)
