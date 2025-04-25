"""Initialize LSA-FW Meshing."""

__author__ = "Ferran de Andres <ferran.de-andres-vert@campus.tu-berlin.de>"

from lib.loggingutils import setup_logging

from .core import Mesher
from .geometries import get_geometry
from .plot import PlotMode, plot_mesh, get_mesh_summary
from .utils import Format, Geometry, Shape, iCellType

setup_logging(verbose=False)

__all__ = [
    "Shape",
    "Format",
    "Mesher",
    "Geometry",
    "PlotMode",
    "plot_mesh",
    "iCellType",
    "get_geometry",
    "setup_logging",
    "get_mesh_summary",
]
