"""Utilities for meshing."""

from dolfinx.mesh import CellType

CELL_TYPE_MAP: dict[str, CellType] = {
    "point": CellType.point,
    "interval": CellType.interval,
    "triangle": CellType.triangle,
    "quadrilateral": CellType.quadrilateral,
    "tetrahedron": CellType.tetrahedron,
    "pyramid": CellType.pyramid,
    "prism": CellType.prism,
    "hexahedron": CellType.hexahedron,
}
"""Mapping from string keys to `dolfinx.mesh.CellType` enum members.

Implementation note:
  This is necessary because CellType is an IntEnum, and argparse cannot directly handle
  enum types that don't inherit from str.
  Additionally, this mapping improves IDE support (autocomplete, syntax highlighting,
  and "go to definition") in CLI code.
"""
