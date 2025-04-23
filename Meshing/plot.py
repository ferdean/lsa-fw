"""LSA-FW Meshing plotter.

Provides basic interactive or static visualization of meshes using PyVista.
Supports both 2D and 3D mesh rendering.
"""

from dolfinx.mesh import Mesh, MeshTags
from dolfinx.plot import vtk_mesh
import pyvista as pv
import numpy as np
from enum import StrEnum, auto
from pathlib import Path
import logging


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
    if not screenshot_path and mode is not PlotMode.INTERACTIVE:
        logger.warning(
            "Non-interactive mode with no export: nothing will be displayed or saved."
        )
        return

    topology, cell_types, points = vtk_mesh(mesh)
    grid = pv.UnstructuredGrid(topology, cell_types, points)

    plotter = pv.Plotter(window_size=window_size, off_screen=bool(screenshot_path))

    transparent_background = background == "transparent"

    if tags is not None:
        scalars = np.zeros(grid.n_cells, dtype=np.int32)
        scalars[tags.indices] = tags.values
        plotter.add_mesh(
            grid,
            scalars=scalars,
            show_edges=show_edges,
            cmap="viridis",
            scalar_bar_args={"title": "Tags"},
        )
    else:
        plotter.add_mesh(grid, show_edges=show_edges, color=color)

    match (
        export_format := screenshot_path.suffix.lower() if screenshot_path else None
    ):
        case ".svg":
            plotter.save_graphic(str(screenshot_path))
            logger.info("Exported mesh to SVG: %s", screenshot_path)
        case ".png":
            plotter.screenshot(
                str(screenshot_path), transparent_background=transparent_background
            )
            logger.info("Saved mesh screenshot: %s", screenshot_path)
        case None:
            pass  # No export requested
        case _:
            raise ValueError(f"Unsupported export format: '{export_format}'")

    if mode is PlotMode.INTERACTIVE:
        plotter.show()


def get_mesh_summary(mesh: Mesh) -> dict[str, int | tuple[np.ndarray, np.ndarray]]:
    """Return basic mesh info for diagnostics/logging."""
    dim = mesh.topology.dim
    num_cells = mesh.topology.index_map(dim).size_local
    num_vertices = mesh.geometry.x.shape[0]
    bbox = _compute_bounding_box(mesh)

    return {
        "dimension": dim,
        "num_cells": num_cells,
        "num_vertices": num_vertices,
        "bounding_box": bbox,
    }


def _compute_bounding_box(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    coords = mesh.geometry.x
    return coords.min(axis=0), coords.max(axis=0)
