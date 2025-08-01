"""Utilities for LSA-FW Meshing."""

from enum import Enum, StrEnum, auto
from typing import Self

from basix import CellType as BasixCellType
from dolfinx.mesh import CellType as DolfinxCellType


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

    def to_dolfinx(self) -> DolfinxCellType:
        """Convert internal type to dolfinx CellType."""
        return DolfinxCellType(self.value)

    def to_basix(self) -> BasixCellType:
        """Convert internal type to basix CellType."""
        return BasixCellType[self.to_dolfinx().name.lower()]

    @classmethod
    def from_dolfinx(cls, cell_type: DolfinxCellType) -> Self:
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
    PREDEFINED = auto()
    """Imported mesh. Used for pre-defined shapes (refer to Geometry-enum) or dolfinx.mesh.Mesh objects."""


class Format(StrEnum):
    """Supported formats (export)."""

    XDMF = auto()
    """Exported as XDMF file."""
    GMSH = auto()
    """Exported as GMSH model."""
    VTK = auto()
    """Exported as VTK."""


class Geometry(StrEnum):
    """Supported geometries."""

    CYLINDER_FLOW = auto()
    """Cylinder flow geometry."""
    STEP_FLOW = auto()
    """Step flow geometry."""
