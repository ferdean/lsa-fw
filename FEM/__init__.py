"""Initialize LSA-FW FEM."""

__author__ = "Ferran de Andres <ferran.de-andres-vert@campus.tu-berlin.de>"

from lib.loggingutils import setup_logging

from .bcs import (
    BoundaryCondition,
    BoundaryConditionType,
    BoundaryConditions,
    define_bcs,
)
from .operators import LinearizedNavierStokesAssembler, VariationalForms
from .spaces import FunctionSpaceType, FunctionSpaces, define_spaces
from .utils import iElementFamily, iMeasure, iPETScMatrix, iPETScVector

setup_logging(verbose=True)

__all__ = [
    "iMeasure",
    "setup_logging",
    "define_bcs",
    "define_spaces",
    "iPETScMatrix",
    "iPETScVector",
    "FunctionSpaces",
    "VariationalForms",
    "iElementFamily",
    "FunctionSpaceType",
    "BoundaryCondition",
    "BoundaryConditions",
    "BoundaryConditionType",
    "LinearizedNavierStokesAssembler",
]
