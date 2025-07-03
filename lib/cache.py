"""Simple XDMF/HDF5 caching utilities."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Tuple

import dolfinx.fem as dfem
import dolfinx.io as dio
import dolfinx.mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc

from lib.loggingutils import log_global

logger = logging.getLogger(__name__)


class CacheStore:
    """Disk-based cache using XDMF/HDF5 files."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _path(self, key: str, suffix: str) -> Path:
        return self.cache_dir / f"{self._hash(key)}.{suffix}"

    def load_mesh(
        self, key: str, comm: MPI.Intracomm = MPI.COMM_WORLD
    ) -> Tuple[dmesh.Mesh, dmesh.MeshTags | None, dmesh.MeshTags | None] | None:
        path = self._path(key, "xdmf")
        if not path.exists():
            return None
        with dio.XDMFFile(comm, str(path), "r") as xdmf:
            mesh = xdmf.read_mesh(name="mesh")
            mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            facet = None
            cell = None
            try:
                facet = xdmf.read_meshtags(mesh, name="facet_tags")
            except Exception:  # noqa: BLE001
                log_global(
                    logger, logging.DEBUG, "No facet tags in cache file %s", path
                )
            try:
                cell = xdmf.read_meshtags(mesh, name="cell_tags")
            except Exception:  # noqa: BLE001
                log_global(logger, logging.DEBUG, "No cell tags in cache file %s", path)
        return mesh, facet, cell

    def save_mesh(
        self,
        key: str,
        mesh: dmesh.Mesh,
        facet: dmesh.MeshTags | None,
        cell: dmesh.MeshTags | None,
        comm: MPI.Intracomm = MPI.COMM_WORLD,
    ) -> None:
        path = self._path(key, "xdmf")
        with dio.XDMFFile(comm, str(path), "w") as xdmf:
            xdmf.write_mesh(mesh)
            if facet is not None:
                xdmf.write_meshtags(facet, mesh.geometry)
            if cell is not None:
                xdmf.write_meshtags(cell, mesh.geometry)

    def load_function(
        self, key: str, V: dfem.FunctionSpace, comm: MPI.Intracomm = MPI.COMM_WORLD
    ) -> dfem.Function | None:
        path = self._path(key, "xdmf")
        if not path.exists():
            return None
        fn = dfem.Function(V)
        with dio.XDMFFile(comm, str(path), "r") as xdmf:
            xdmf.read_function(fn, name="function")
        return fn

    def save_function(
        self, key: str, fn: dfem.Function, comm: MPI.Intracomm = MPI.COMM_WORLD
    ) -> None:
        path = self._path(key, "xdmf")
        with dio.XDMFFile(comm, str(path), "w") as xdmf:
            xdmf.write_function(fn, name="function")

    def load_matrix(
        self, key: str, comm: MPI.Comm = PETSc.COMM_WORLD
    ) -> PETSc.Mat | None:
        path = self._path(key, "h5")
        if not path.exists():
            return None
        viewer = PETSc.Viewer().createHDF5(str(path), "r", comm=comm)
        mat = PETSc.Mat().create(comm=comm)
        mat.load(viewer)
        viewer.destroy()
        return mat

    def save_matrix(
        self, key: str, mat: PETSc.Mat, comm: MPI.Comm = PETSc.COMM_WORLD
    ) -> None:
        path = self._path(key, "h5")
        viewer = PETSc.Viewer().createHDF5(str(path), "w", comm=comm)
        mat.save(viewer)
        viewer.destroy()
