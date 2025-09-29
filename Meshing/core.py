"""LSA-FW Meshing core."""

import logging
import pathlib
from typing import Callable, Self, assert_never

import dolfinx.io as dio
import dolfinx.mesh as dmesh
import numpy as np
from dolfinx.io import gmshio
from dolfinx.mesh import MeshTags, compute_midpoints, locate_entities_boundary, meshtags
from mpi4py import MPI

from lib.cache import CacheStore
from lib.loggingutils import log_global

from lib.gmshutils import gmsh_quiet

from .geometries import GeometryConfig, get_geometry
from .utils import Format, Geometry, Shape, iCellType

logger = logging.getLogger(__name__)

_CUSTOM_FILES: set[Shape] = {Shape.CUSTOM_XDMF, Shape.CUSTOM_MSH}
_COMM: MPI.Intracomm = MPI.COMM_WORLD


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
    ) -> None:
        """Initialize mesher."""
        if shape in _CUSTOM_FILES and not custom_file:
            raise ValueError(f"{shape} requires a custom file.")
        if not (1 <= len(n) <= 3):
            raise ValueError("Number of dimensions must be between 1 and 3.")
        if any(x <= 0 for x in n):
            raise ValueError("All resolution values in 'n' must be greater than zero.")

        self._shape = shape
        self._n = n
        self._cell_type = cell_type.to_dolfinx()
        self._custom_file = custom_file.resolve() if custom_file else None
        self._gdim = gdim or len(n)
        self._mesh: dmesh.Mesh | None = None
        self._domain = domain or (
            tuple(0.0 for _ in range(len(n))),
            tuple(1.0 for _ in range(len(n))),
        )

        self._facet_tags: MeshTags | None = None
        self._cell_tags: MeshTags | None = None

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

    @property
    def facet_tags(self) -> MeshTags | None:
        """Get the facet tags of the mesh."""
        return self._facet_tags

    @property
    def cell_tags(self) -> MeshTags | None:
        """Get the cell tags of the mesh."""
        return self._cell_tags

    @property
    def domain(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Get the meshing domain."""
        return self._domain

    @property
    def gdim(self) -> int:
        """Get the geometric dimensions."""
        return self._gdim

    @classmethod
    def from_file(
        cls, path: pathlib.Path, gdim: int = 3, comm: MPI.Intracomm = _COMM
    ) -> Self:
        """Create a Mesher instance from a custom mesh file."""
        try:
            shape = Shape.from_path(path)
            mesh = cls(shape=shape, custom_file=path, gdim=gdim)
            _ = mesh.generate(comm)
            return mesh
        except ValueError as e:
            raise ValueError("Shape is not a supported import type.") from e

    @classmethod
    def from_mesh(
        cls,
        mesh: dmesh.Mesh,
        facet_tags: MeshTags | None = None,
        cell_tags: MeshTags | None = None,
    ) -> Self:
        """Wrap an existing dolfinx Mesh into a Mesher."""
        obj = cls.__new__(cls)

        obj._mesh = mesh
        obj._shape = Shape.PREDEFINED
        obj._facet_tags = facet_tags
        obj._cell_tags = cell_tags
        obj._gdim = mesh.topology.dim

        coords = mesh.geometry.x
        obj._domain = (
            tuple(coords.min(axis=0).tolist()),
            tuple(coords.max(axis=0).tolist()),
        )

        obj._cell_type = mesh.topology.cell_type
        obj._n = tuple()  # only meaningful for structured meshes
        obj._custom_file = None

        return obj

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        config: GeometryConfig,
        comm: MPI.Intracomm = _COMM,
        *,
        cache: CacheStore | None = None,
        key: str | None = None,
    ) -> Self:
        """Generate via one of the prebuilt generators, then wrap in a Mesher."""
        log_global(logger, logging.INFO, "Started GMSH-based mesher.")
        if cache is not None and key is not None:
            cached = cache.load_mesh(key, comm)
            if cached is not None:
                mesh, facet, cell = cached
                return cls.from_mesh(mesh, facet_tags=facet, cell_tags=cell)

        with gmsh_quiet(logger, verbosity=1, reemit_level=logging.DEBUG):
            mesh = get_geometry(geometry, config, comm)

        obj = cls.from_mesh(mesh)
        if cache is not None and key is not None:
            cache.save_mesh(key, obj.mesh, obj.facet_tags, obj.cell_tags, comm)
        return obj

    def generate(
        self,
        comm: MPI.Intracomm = _COMM,
        *,
        cache: CacheStore | None = None,
        key: str | None = None,
    ) -> dmesh.Mesh:
        """Generate the mesh according to shape.

        If `cache` and `key` are provided, the method will attempt to load the mesh from disk before generating it.
        The newly generated mesh is also written back to the cache.
        """
        if cache is not None and key is not None:
            cached = cache.load_mesh(key, comm)
            if cached is not None:
                self._mesh, self._facet_tags, self._cell_tags = cached
                return self._mesh
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

            case Shape.PREDEFINED:
                raise ValueError(
                    "Predefined meshes generation should be managed externally."
                )

            case _:
                assert_never(self._shape)

        if cache is not None and key is not None:
            cache.save_mesh(key, self._mesh, self._facet_tags, self._cell_tags, comm)

        return self._mesh

    def export(self, path: pathlib.Path, comm: MPI.Intracomm = _COMM) -> None:
        """Export the generated mesh to the given file.

        XDMF is generally preferred over GMSH for exporting FEniCSx meshes because it supports larger, more complex
        meshes efficiently, is more flexible with regards to multi-block data, and is better suited for parallel I/O.
        GMSH can still be useful for visualization and some other purposes, but XDMF is often the better choice for
        high-performance simulations and large datasets.
        """
        if self._mesh is None:
            raise RuntimeError("Mesh must be generated before exporting.")

        try:
            format = Format.from_path(path)
        except ValueError as e:
            raise ValueError("Format is not a supported import type.") from e

        path.parent.mkdir(parents=True, exist_ok=True)

        match format:
            case Format.XDMF:
                with dio.XDMFFile(comm, str(path), "w") as file:
                    file.write_mesh(self.mesh)

                    if self._facet_tags is not None:
                        file.write_meshtags(self._facet_tags, self.mesh.geometry)
                    else:
                        log_global(logger, logging.WARNING, "No facet tags to export.")

                    if self._cell_tags is not None:
                        file.write_meshtags(self._cell_tags, self.mesh.geometry)
                    else:
                        log_global(logger, logging.WARNING, "No cell tags to export.")

            case Format.VTK:
                with dio.VTKFile(comm, str(path), "w") as vtk_file:
                    vtk_file.write_mesh(self.mesh)

            case Format.GMSH:
                # Implementation note: In order to export meshes as GMSH, it is required to convert the FEniCSx mesh
                # topology and geometry into `meshio` format GMSH doesn't always preserve all metadata that XDMF or VTK
                # can, then, it is recommended to use XDMF or VTK for exporting meshes.
                # Then, GMSH export is not yet supported.
                raise NotImplementedError(
                    "GMSH export is not yet implemented. XDMF is recommended instead."
                )

            case _:
                assert_never(format)

    def mark_boundary_facets(self, marker_fn: Callable[[np.ndarray], int]) -> None:
        """Mark boundary facets of the mesh using a user-defined function.

        This method allows for custom marker logic, such as:
          `mesher.mark_boundary_facets(lambda x: 1 if x[0] < 1e-12 else 2)`
        """
        facet_dim = self.mesh.topology.dim - 1

        # Identify unique exterior facets
        facets = np.unique(
            locate_entities_boundary(
                self.mesh, facet_dim, lambda x: np.full(x.shape[1], True)
            )
        )
        if len(facets) == 0:
            raise RuntimeError("No boundary facets found.")

        # Compute midpoint of each facet
        midpoints = compute_midpoints(self.mesh, facet_dim, facets)
        if len(midpoints) != len(facets):
            raise RuntimeError("Mismatch between facets and midpoints count.")

        # Generate integer markers from user-defined function
        try:
            markers = np.array([marker_fn(p) for p in midpoints], dtype=np.int32)
            self._facet_tags = meshtags(self.mesh, facet_dim, facets, markers)
            self._facet_tags.name = "facet_tags"
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate marker function: {e}")

    def _create_box(self, comm: MPI.Intracomm) -> dmesh.Mesh:
        if self._gdim == 2:
            return dmesh.create_rectangle(
                comm,
                [list(self._domain[0]), list(self._domain[1])],
                list(self._n),
                self._cell_type,
            )
        elif self._gdim == 3:
            return dmesh.create_box(
                comm,
                [list(self._domain[0]), list(self._domain[1])],
                list(self._n),
                self._cell_type,
            )
        else:
            raise ValueError("BOX shape requires 2 or 3 dimensions.")

    def _read_custom_xdmf(self, comm: MPI.Intracomm) -> dmesh.Mesh:
        with dio.XDMFFile(comm, str(self._custom_file), "r") as xdmf:
            try:
                mesh = xdmf.read_mesh(name="mesh")
                mesh.topology.create_connectivity(
                    mesh.topology.dim, mesh.topology.dim - 1
                )
                mesh.topology.create_connectivity(
                    mesh.topology.dim - 1, mesh.topology.dim
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read mesh from {self._custom_file}: {e}")

            try:
                self._facet_tags = xdmf.read_meshtags(mesh, name="facet_tags")
            except Exception:
                log_global(logger, logging.WARNING, "No facet tags found in XDMF.")

            try:
                self._cell_tags = xdmf.read_meshtags(mesh, name="cell_tags")
            except Exception:
                log_global(logger, logging.WARNING, "No cell tags found in XDMF.")

        return mesh

    def _read_custom_msh(self, comm: MPI.Intracomm) -> dmesh.Mesh:
        if not self._custom_file or not self._custom_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {self._custom_file}")
        mesh, self._cell_tags, self._facet_tags = gmshio.read_from_msh(
            str(self._custom_file), comm, gdim=self._gdim
        )
        return mesh
