"""LSA-FW mesh core."""

import pathlib
import numpy as np
from typing import assert_never, Self

from mpi4py import MPI
import dolfinx.mesh as dmesh
import dolfinx.io as dio
from dolfinx.io import gmshio
import meshio  # type: ignore[import-untyped]

from .types import Format, Shape, iCellType

_CUSTOM_FILES = {Shape.CUSTOM_XDMF, Shape.CUSTOM_MSH}


class Mesher:
    """Generate and export meshes for FEniCSx using built-in or custom geometries."""

    def __init__(
        self,
        shape: Shape,
        n: tuple[int, ...] = (10,),
        cell_type: iCellType = iCellType.INTERVAL,
        domain: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        gdim: int | None = None,
        custom_file: pathlib.Path | None = None,
    ):
        """Initialize.

        Args:
            shape: The geometric shape to generate or import.
            n (optional): Number of cells per dimension (e.g., (nx,), (nx, ny), or (nx, ny, nz)).
            domain: Domain bounding box as ((x0, y0, z0), (x1, y1, z1)). Defaults to unit box.
            cell_type (optional): The type of finite element cell (interval, triangle, hexahedron, etc.).
            gdim (optional): Geometric dimension. Defaults to len(n).
            custom_file (optional): Path to a mesh file if using a custom shape (XDMF or MSH).
        """
        if shape in _CUSTOM_FILES and not custom_file:
            raise ValueError(f"{shape} requires a custom file.")
        if not (1 <= len(n) <= 3):
            raise ValueError("Number of dimensions must be between 1 and 3.")

        self._shape = shape
        self._n = n
        self._cell_type = cell_type.to_dolfinx()
        self._custom_file = custom_file
        self._gdim = gdim or len(n)
        self._mesh: dmesh.Mesh | None = None
        self._domain = domain or (
            tuple(0.0 for _ in range(len(n))),
            tuple(1.0 for _ in range(len(n))),
        )

    def __repr__(self) -> str:
        return (
            f"Mesher(shape={self._shape}, n={self._n}, cell_type={self._cell_type}, "
            f"domain={self._domain}, custom_file={self._custom_file})"
        )

    @property
    def mesh(self) -> dmesh.Mesh:
        """Get the generated mesh."""
        if self._mesh is None:
            raise RuntimeError("Mesh not yet generated.")
        return self._mesh

    @classmethod
    def from_file(cls, path: pathlib.Path, shape: Shape, gdim: int = 3) -> Self:
        """Create a Mesher instance from a custom mesh file."""
        if shape not in _CUSTOM_FILES:
            raise ValueError(f"Shape {shape} is not a supported import type.")
        mesh = cls(shape=shape, custom_file=path, gdim=gdim)
        _ = mesh.generate()
        return mesh

    def generate(self) -> dmesh.Mesh:
        """Generate the mesh according to shape."""
        comm = MPI.COMM_WORLD

        match self._shape:
            case Shape.UNIT_INTERVAL:
                self._mesh = dmesh.create_unit_interval(comm, self._n[0], np.float64)

            case Shape.UNIT_SQUARE:
                self._mesh = dmesh.create_unit_square(
                    comm, self._n[0], self._n[1], self._cell_type
                )

            case Shape.UNIT_CUBE:
                self._mesh = dmesh.create_unit_cube(
                    comm, self._n[0], self._n[1], self._n[2], self._cell_type
                )

            case Shape.BOX:
                self._mesh = self._create_box(comm)

            case Shape.CUSTOM_XDMF:
                self._mesh = self._read_custom_xdmf(comm)

            case Shape.CUSTOM_MSH:
                self._mesh = self._read_custom_msh(comm)

            case _:
                assert_never(self._shape)

        return self._mesh

    def export(self, path: pathlib.Path, format: Format) -> None:
        """Export the generated mesh to the given file."""
        if self._mesh is None:
            raise RuntimeError("Mesh must be generated before exporting.")
        path.parent.mkdir(parents=True, exist_ok=True)

        match format:
            case Format.XDMF:
                with dio.XDMFFile(MPI.COMM_WORLD, str(path), "w") as file:
                    file.write_mesh(self.mesh)

            case Format.VTK | Format.GMSH:
                # Ensure connectivity from cells to vertices exists
                tdim = self.mesh.topology.dim
                self.mesh.topology.create_connectivity(tdim, 0)

                # Get cell connectivity array
                conn = self.mesh.topology.connectivity(tdim, 0).array
                num_cells = self.mesh.topology.index_map(tdim).size_local
                num_vertices = conn.size // num_cells
                cells = conn.reshape((num_cells, num_vertices))

                # Convert dolfinx cell type to meshio format
                cell_type = (
                    self.mesh.topology.cell_name()
                )  # e.g., "triangle", "tetra", etc.

                # Extract coordinates
                points = self.mesh.geometry.x

                # Construct and export with meshio
                mesh = meshio.Mesh(points=points, cells=[(cell_type, cells)])

                ext = {
                    Format.VTK: ".vtk",
                    Format.GMSH: ".msh",
                }[format]
                corrected_path = path.with_suffix(ext)

                mesh.write(corrected_path)
            case _:
                assert_never(format)

    def _create_box(self, comm) -> dmesh.Mesh:
        if self._gdim == 2:
            return dmesh.create_rectangle(
                comm,
                [self._domain[0], self._domain[1]],
                list(self._n),
                self._cell_type,
                dtype=np.float64,
            )
        if self._gdim == 3:
            return dmesh.create_box(
                comm,
                [self._domain[0], self._domain[1]],
                list(self._n),
                self._cell_type,
                dtype=np.float64,
            )
        raise ValueError("BOX shape requires 2 or 3 dimensions.")

    def _read_custom_xdmf(self, comm) -> dmesh.Mesh:
        with dio.XDMFFile(comm, str(self._custom_file), "r") as xdmf:
            return xdmf.read_mesh(name="Grid")

    def _read_custom_msh(self, comm) -> dmesh.Mesh:
        mesh, _, _ = gmshio.read_from_msh(str(self._custom_file), comm, gdim=self._gdim)
        return mesh
