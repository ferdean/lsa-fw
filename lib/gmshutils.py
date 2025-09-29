"""Utilities for gmsh."""

import contextlib
import logging
import os
import re
import tempfile
from typing import Generator

from lib.loggingutils import log_global

try:
    import gmsh  # type: ignore[import-untyped]
except Exception:
    gmsh = None

_GMSH_PREFIX = re.compile(r"^(Info|Warning|Error)\s*:?\s+")


@contextlib.contextmanager
def _capture_c_streams_to_logger(
    logger: logging.Logger | None,
    level: int,
) -> Generator[None, None, None]:
    """Capture C-level stdout/stderr and (optionally) re-emit to logger."""
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)

    with tempfile.TemporaryFile(mode="w+b") as tmp_out, tempfile.TemporaryFile(
        mode="w+b"
    ) as tmp_err:
        try:
            os.dup2(tmp_out.fileno(), 1)
            os.dup2(tmp_err.fileno(), 2)
            yield
        finally:
            # Restore fds first so our own logging doesn't get captured
            os.dup2(orig_stdout_fd, 1)
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)

            if logger is not None:
                # Read and re-emit
                for f, _ in ((tmp_out, "stdout"), (tmp_err, "stderr")):
                    try:
                        f.flush()
                        f.seek(0)
                        data = f.read().decode(errors="replace")
                        if data:
                            for line in data.splitlines():
                                clean = _GMSH_PREFIX.sub("", line).strip()
                                if clean:
                                    log_global(logger, level, f"GMSH log: {clean}")
                    except Exception:
                        # Never fail teardown due to logging
                        pass


@contextlib.contextmanager
def gmsh_quiet(
    logger: logging.Logger | None = None,
    *,
    verbosity: int = 0,
    reemit_level: int = logging.INFO,
) -> Generator[None, None, None]:
    """Mute Gmsh terminal spam and (optionally) forward it to `logger`."""
    was_init: bool = False
    old_term: float | None = None
    old_verb: float | None = None

    if gmsh is not None:
        try:
            was_init = gmsh.isInitialized()
        except Exception:
            was_init = False

    with _capture_c_streams_to_logger(logger, reemit_level):
        # If gmsh was already initialized, mute at the source as well
        if was_init:
            try:
                old_term = gmsh.option.getNumber("General.Terminal")
            except Exception:
                old_term = None
            try:
                old_verb = gmsh.option.getNumber("General.Verbosity")
            except Exception:
                old_verb = None
            try:
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.option.setNumber("General.Verbosity", verbosity)
            except Exception:
                pass

        try:
            yield
        finally:
            if gmsh is not None:
                try:
                    if gmsh.isInitialized():
                        try:
                            if old_term is not None:
                                gmsh.option.setNumber("General.Terminal", old_term)
                        except Exception:
                            pass
                        try:
                            if old_verb is not None:
                                gmsh.option.setNumber("General.Verbosity", old_verb)
                        except Exception:
                            pass
                except Exception:
                    pass
