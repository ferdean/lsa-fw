"""Utilities for meshing."""

from dolfinx.mesh import CellType
from enum import Enum, auto, StrEnum
from typing import Self


class iCellType(Enum):
    """Internal cell type.

    Implementation note:
      This is necessary because CellType is an IntEnum, and argparse cannot directly handle
      enum types that don't inherit from str.
      Additionally, this mapping improves IDE support (autocomplete, syntax highlighting,
      and "go to definition") in CLI code.
    """

    POINT = 1
    INTERVAL = 2
    TRIANGLE = 3
    QUADRILATERAL = -4
    TETRAHEDRON = 4
    PYRAMID = -5
    PRISM = -6
    HEXAHEDRON = -8

    def to_dolfinx(self) -> CellType:
        """Convert internal type to dolfinx CellType."""
        return CellType(self.value)

    @classmethod
    def from_dolfinx(cls, cell_type: CellType) -> Self:
        """Create internal type from a dolfinx CellType."""
        return cls(cell_type.value)

    @classmethod
    def from_string(cls, cell_type: str) -> Self:
        """Create internal type from a string."""
        try:
            return cls[cell_type.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid cell type: {cell_type}. Choose from {list(cls.__members__.keys())}."
            )


class Shape(StrEnum):
    """Supported shapes."""

    UNIT_INTERVAL = auto()
    """One-dimensional interval domain [0, 1]."""
    UNIT_SQUARE = auto()
    """Two-dimensional square domain [0, 1]^2."""
    UNIT_CUBE = auto()
    """Three-dimensional cube domain [0, 1]^3."""
    BOX = auto()
    """Arbitrary box domain with user-defined resolution."""
    CUSTOM_XDMF = auto()
    """Imported mesh from a custom XDMF (.xdmf) file."""
    CUSTOM_MSH = auto()
    """Imported mesh from a GMSH (.msh) file."""


class Format(StrEnum):
    """Supported formats (export)."""

    XDMF = auto()
    """Exported as XDMF file."""
    GMSH = auto()
    """Exported as GMSH model."""
    VTK = auto()
    """Exported as VTK (requires `meshio`)."""
