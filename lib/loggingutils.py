import logging
from rich.logging import RichHandler
from rich.console import Console

try:
    from mpi4py import MPI

    _rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    _rank = 0


def setup_logging(verbose: bool) -> None:
    """Set up logging so that only rank 0 emits INFO/DEBUG, others only WARNING+."""
    root = logging.getLogger()
    if root.handlers:
        return

    if _rank == 0:
        level = logging.DEBUG if verbose else logging.INFO
        console = Console(force_terminal=True, color_system="auto")
        handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
    else:
        level = logging.WARNING
        handler = RichHandler(rich_tracebacks=False, markup=False)
        handler.setLevel(level)

    root.setLevel(level)
    root.addHandler(handler)
