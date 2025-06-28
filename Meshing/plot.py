"""LSA-FW Meshing plotter.

Provides basic interactive or static visualization of meshes using PyVista. Supports both 2D and 3D mesh rendering.

All routines are MPI-aware. When run in parallel, the mesh (and any provided MeshTags) are gathered onto rank 0 via
XDMF round-trip, ensuring correct global indexing for plotting. Only rank 0 will perform the actual rendering or
screenshot save.
"""

import logging
from enum import StrEnum, auto
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyvista as pv
from mpi4py import MPI
from dolfinx.mesh import Mesh, MeshTags
from dolfinx.plot import vtk_mesh
import dolfinx.io as dio
from lib.loggingutils import log_global

logger = logging.getLogger(__name__)


class PlotMode(StrEnum):
    """Available plot modes."""

    INTERACTIVE = auto()
    STATIC = auto()


def plot_mesh(
    mesh: Mesh,
    *,
    mode: PlotMode = PlotMode.INTERACTIVE,
    show_edges: bool = True,
    color: str = "white",
    background: str = "transparent",
    window_size: tuple[int, int] = (800, 600),
    screenshot_path: Path | None = None,
    tags: MeshTags | None = None,
) -> None:
    """Render the given mesh using PyVista.

    Args:
        mesh: The mesh to visualize.
        mode: Display mode. Refer to PlotMode enum.
        show_edges: Whether to show cell edges.
        color: Mesh surface color.
        background: Background color of the scene.
        window_size: Size of the render window.
        screenshot_path: If provided, saves a screenshot to this path.
        tags: Mesh tags for colouring boundary conditions.
    """
    mpi_comm = mesh.comm
    mesh_to_plot, tags_to_plot = mesh, tags

    if mpi_comm.size > 1:
        mesh_to_plot, tags_to_plot = _gather_mesh_from_ranks(mesh, tags)
        if mpi_comm.rank != 0:
            return

    if mode is not PlotMode.INTERACTIVE and not screenshot_path:
        log_global(
            logger,
            logging.WARNING,
            "Non-interactive mode with no export: nothing will be displayed or saved.",
        )
        return

    topo, cell_types, points = vtk_mesh(mesh_to_plot)
    grid = pv.UnstructuredGrid(topo, cell_types, points)
    off_screen = bool(screenshot_path)
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

    if background != "transparent":
        plotter.set_background(background)
    elif not off_screen:
        log_global(
            logger,
            logging.WARNING,
            "Transparent background only supported off-screen; using default background.",
        )

    if tags_to_plot is None:
        _add_plain_mesh(plotter, grid, color, show_edges)
    else:
        dim = mesh_to_plot.topology.dim
        if tags_to_plot.dim == dim:
            _add_cell_tags(plotter, grid, tags_to_plot, show_edges)
        elif tags_to_plot.dim == dim - 1:
            _add_facet_tags(plotter, mesh_to_plot, grid, tags_to_plot, show_edges)
        else:
            log_global(logger, logging.WARNING, "MeshTags dimension %d not supported.", tags.dim)

    if screenshot_path:
        ext = screenshot_path.suffix.lower()
        if ext == ".svg":
            plotter.save_graphic(str(screenshot_path))
            log_global(logger, logging.INFO, "Exported mesh to SVG: %s", screenshot_path)
        elif ext == ".png":
            plotter.screenshot(
                str(screenshot_path),
                transparent_background=(background == "transparent"),
            )
        else:
            raise ValueError(f"Unsupported export format: '{ext}'")

    if mode is PlotMode.INTERACTIVE:
        plotter.show()


def get_mesh_summary(mesh: Mesh) -> dict[str, int | tuple[np.ndarray, np.ndarray]]:
    """Return basic mesh info for diagnostics/logging."""
    dim = mesh.topology.dim
    return {
        "dimension": dim,
        "num_cells": mesh.topology.index_map(dim).size_local,
        "num_vertices": mesh.geometry.x.shape[0],
        "bounding_box": _compute_bounding_box(mesh),
    }


def _compute_bounding_box(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    coords = mesh.geometry.x
    return coords.min(axis=0), coords.max(axis=0)


def _add_plain_mesh(
    plotter: pv.Plotter, grid: pv.UnstructuredGrid, color: str, show_edges: bool
) -> None:
    plotter.add_mesh(grid, color=color, show_edges=show_edges)


def _add_cell_tags(
    plotter: pv.Plotter, grid: pv.UnstructuredGrid, tags: MeshTags, show_edges: bool
) -> None:
    scalars = np.full(grid.n_cells, -1, dtype=np.int32)
    scalars[tags.indices] = tags.values
    masked = np.where(scalars != -1, scalars, np.nan)
    unique = np.unique(tags.values)

    plotter.add_mesh(
        grid,
        scalars=masked,
        show_edges=show_edges,
        cmap="tab10",
        categories=True,
        clim=[int(unique.min()), int(unique.max())],
        nan_color="lightgray",
        scalar_bar_args={"title": "Cell tags", "n_labels": len(unique), "fmt": "%.0f"},
    )


def _add_facet_tags(
    plotter: pv.Plotter,
    mesh: Mesh,
    grid: pv.UnstructuredGrid,
    tags: MeshTags,
    show_edges: bool,
) -> None:
    dim = mesh.topology.dim
    plotter.add_mesh(grid, color="lightgray", opacity=0.5, show_edges=show_edges)

    if dim == 2:
        mesh.topology.create_connectivity(1, 0)
        conn = mesh.topology.connectivity(1, 0)
        lines, vals = [], []
        for f, t in zip(tags.indices, tags.values):
            vids = conn.links(f)
            if vids.size == 2:
                lines.extend([2, *vids.tolist()])
                vals.append(int(t))
        if not vals:
            log_global(logger, logging.WARNING, "No 2D facets tagged, skipping facet plot.")
            return
        cells = np.array(lines, np.int64)
        types = np.full(len(vals), pv.CellType.LINE, np.uint8)
        edge = pv.UnstructuredGrid(cells, types, mesh.geometry.x)
        edge.cell_data["facet tags"] = np.array(vals, np.int32)
        unique = np.unique(vals)
        plotter.add_mesh(
            edge,
            scalars="facet tags",
            show_edges=show_edges,
            cmap="tab10",
            categories=True,
            clim=[int(unique.min()), int(unique.max())],
            line_width=2.5,
            scalar_bar_args={
                "title": "Facet tags",
                "n_labels": len(unique),
                "fmt": "%.0f",
            },
        )

    elif dim == 3:
        fd = tags.dim
        mesh.topology.create_connectivity(fd, 0)
        conn = mesh.topology.connectivity(fd, 0)
        all_vids = np.unique(np.hstack([conn.links(f) for f in tags.indices]))
        coords = mesh.geometry.x[all_vids]
        vid_map = {v: i for i, v in enumerate(all_vids)}
        faces, vals = [], []
        for f, t in zip(tags.indices, tags.values):
            vids = conn.links(f)
            local = [vid_map[v] for v in vids]
            if len(local) >= 3:
                faces.extend([len(local), *local])
                vals.append(int(t))
        if not vals:
            log_global(logger, logging.WARNING, "No 3D facets tagged, skipping facet plot.")
            return
        poly = pv.PolyData(coords, np.array(faces, np.int64))
        poly.cell_data["facet tags"] = np.array(vals, np.int32)
        unique = np.unique(vals)
        plotter.add_mesh(
            poly,
            scalars="facet tags",
            show_edges=show_edges,
            cmap="tab10",
            categories=True,
            clim=[int(unique.min()), int(unique.max())],
            scalar_bar_args={
                "title": "Facet tags",
                "n_labels": len(unique),
                "fmt": "%.0f",
            },
        )
    else:
        log_global(logger, logging.WARNING, "Mesh dimension %d not supported for facets.", dim)


def _gather_mesh_from_ranks(
    mesh: Mesh, tags: MeshTags | None
) -> tuple[Mesh, MeshTags | None]:
    """Gather mesh and tags to rank 0 via XDMF round-trip, preserving tag names and dims."""
    comm = mesh.comm
    tag_name = (
        "cell_tags"
        if tags and tags.dim == mesh.topology.dim
        else "facet_tags" if tags else None
    )
    tf = NamedTemporaryFile(suffix=".xdmf", delete=False)
    path0 = Path(tf.name) if comm.rank == 0 else None
    tf.close()
    path = Path(comm.bcast(str(path0), root=0))

    with dio.XDMFFile(comm, str(path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        if tags:
            xdmf.write_meshtags(tags, mesh.geometry)

    if comm.rank == 0:
        with dio.XDMFFile(MPI.COMM_SELF, str(path), "r") as xdmf:
            full = xdmf.read_mesh()
            full_tags = None
            if tags:
                full.topology.create_entities(tags.dim)
                full.topology.create_connectivity(tags.dim, 0)
                full_tags = xdmf.read_meshtags(full, name=tag_name)
        path.unlink()
        path.with_suffix(".h5").unlink(missing_ok=True)
        return full, full_tags
    return None, None
