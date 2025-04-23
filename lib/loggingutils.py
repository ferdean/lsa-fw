"""Utilities for logging."""

import logging
from rich.logging import RichHandler


def setup_logging(verbose: bool) -> None:
    """Set up logging for the module."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
