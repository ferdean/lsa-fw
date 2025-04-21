"""Initialize LSA-FW Meshing."""

__author__ = "Ferran de Andres <ferran.de-andres-vert@campus.tu-berlin.de>"

from .core import Mesher
from .geometries import get_geometry
from .plot import PlotMode, plot_mesh, setup_logging
from .utils import Format, Geometry, Shape, iCellType

setup_logging(verbose=False)

__all__ = [
    "Shape",
    "Format",
    "Mesher",
    "Geometry",
    "iCellType",
    "get_geometry",
    "plot_mesh",
    "PlotMode",
    "setup_logging",
]
