"""LSA-FW mesh plotter.

First draft.

Provides basic interactive or static visualization of meshes using PyVista.
Supports both 2D and 3D mesh rendering.
"""

from dolfinx.mesh import Mesh
from dolfinx.plot import vtk_mesh
import pyvista as pv
import numpy as np
from enum import StrEnum


class PlotMode(StrEnum):
    """Available plot modes."""

    INTERACTIVE = "interactive"
    STATIC = "static"


def plot_mesh(
    mesh: Mesh,
    *,
    mode: PlotMode = PlotMode.INTERACTIVE,
    show_edges: bool = True,
    color: str = "white",
    background: str = "black",
    window_size: tuple[int, int] = (800, 600),
    screenshot_path: str | None = None,
) -> None:
    """Render the given mesh using PyVista.

    Args:
        mesh: The mesh to visualize.
        mode: Display mode (interactive or static).
        show_edges: Whether to show cell edges.
        color: Mesh surface color.
        background: Background color of the scene.
        window_size: Size of the render window.
        screenshot_path: If provided, saves a screenshot to this path.
    """
    topology, cell_types, points = vtk_mesh(mesh)

    grid = pv.UnstructuredGrid(topology, cell_types, points)

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(grid, show_edges=show_edges, color=color)

    if screenshot_path:
        plotter.screenshot(screenshot_path)

    if mode == PlotMode.INTERACTIVE:
        plotter.show()


def get_mesh_summary(mesh: Mesh) -> dict:
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
