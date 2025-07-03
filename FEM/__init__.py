"""Initialize LSA-FW FEM."""

from .bcs import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryConditionType,
    define_bcs,
)
from .operators import LinearizedNavierStokesAssembler, VariationalForms
from .spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from .utils import iElementFamily, iMeasure, iPETScMatrix, iPETScVector

__author__ = "Ferran de Andres <ferran.de-andres-vert@campus.tu-berlin.de>"
__all__ = [
    "iMeasure",
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
