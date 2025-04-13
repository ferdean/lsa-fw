"""LSA-FW mesh core."""

import pathlib
import numpy as np
from enum import StrEnum, auto
from typing import assert_never

from mpi4py import MPI
import dolfinx.mesh as dmesh
import dolfinx.io as dio
from dolfinx.io import gmshio
import meshio


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


class Mesher:
    """Generate and export meshes for FEniCSx using built-in or custom geometries."""

    def __init__(
        self,
        shape: Shape,
        n: tuple[int, ...] = (10,),
        cell_type: dmesh.CellType = dmesh.CellType.interval,
        custom_file: pathlib.Path | None = None,
    ):
        if shape in {Shape.CUSTOM_XDMF, Shape.CUSTOM_MSH} and not custom_file:
            raise ValueError(f"{shape} requires a custom file.")
        if not (1 <= len(n) <= 3):
            raise ValueError("Number of dimensions must be between 1 and 3.")

        self._shape = shape
        self._n = n
        self._cell_type = cell_type
        self._custom_file = custom_file
        self._mesh: dmesh.Mesh | None = None

    @property
    def mesh(self) -> dmesh.Mesh:
        """Get generated mesh."""
        if self._mesh is None:
            raise RuntimeError("Mesh not yet generated.")
        return self._mesh

    def generate(self) -> None:
        """Generate the mesh according to the specified shape and parameters."""
        match self._shape:
            case Shape.UNIT_INTERVAL:
                self._mesh = dmesh.create_unit_interval(
                    MPI.COMM_WORLD, self._n[0], self._cell_type
                )
            case Shape.UNIT_SQUARE:
                self._mesh = dmesh.create_unit_square(
                    MPI.COMM_WORLD, self._n[0], self._n[1], self._cell_type
                )
            case Shape.UNIT_CUBE:
                self._mesh = dmesh.create_unit_cube(
                    MPI.COMM_WORLD, self._n[0], self._n[1], self._n[2], self._cell_type
                )
            case Shape.BOX:
                points = np.array(
                    [[0.0] * len(self._n), [1.0] * len(self._n)], dtype=np.float64
                )
                self._mesh = dmesh.create_box(
                    MPI.COMM_WORLD, points, self._n, self._cell_type
                )
            case Shape.CUSTOM_XDMF:
                with dio.XDMFFile(MPI.COMM_WORLD, str(self._custom_file), "r") as xdmf:
                    self._mesh = xdmf.read_mesh(name="Grid")
            case Shape.CUSTOM_MSH:
                self._mesh, _ = gmshio.read_from_msh(
                    str(self._custom_file), MPI.COMM_WORLD, gdim=3
                )
            case _:
                assert_never(self._shape)

    def export(self, path: pathlib.Path, format: Format) -> None:
        """Export the generated mesh to the given file in the specified format."""
        if self._mesh is None:
            raise RuntimeError("Mesh must be generated before exporting.")

        path.parent.mkdir(parents=True, exist_ok=True)

        match format:
            case Format.XDMF:
                with dio.XDMFFile(MPI.COMM_WORLD, str(path), "w") as file:
                    file.write_mesh(self.mesh)
            case Format.GMSH:
                gmsh_model = gmshio.model_from_mesh(self.mesh)
                gmsh_model.write(str(path))
            case Format.VTK:
                self.mesh.topology.create_connectivity(
                    self.mesh.topology.dim, self.mesh.topology.dim
                )
                topology = self.mesh.topology.connectivity(self.mesh.topology.dim).array
                geometry = self.mesh.geometry.x
                cells = [(self.mesh.topology.cell_name(), topology)]
                meshio.Mesh(points=geometry, cells=cells).write(str(path))
            case _:
                assert_never(format)
