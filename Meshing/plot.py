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
    if mode is not PlotMode.INTERACTIVE and screenshot_path is None:
        logger.warning(
            "Non-interactive mode with no export: nothing will be displayed or saved."
        )
        return

    topo, cell_types, points = vtk_mesh(mesh)
    grid = pv.UnstructuredGrid(topo, cell_types, points)

    off_screen = screenshot_path is not None
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    if background != "transparent":
        plotter.set_background(background)
    elif not off_screen:
        logger.warning(
            "Transparent background only supported for off-screen export; "
            "interactive view will use default background."
        )

    if tags is None:
        _add_plain_mesh(plotter, grid, color, show_edges)
    else:
        dim = mesh.topology.dim
        if tags.dim == dim:
            _add_cell_tags(plotter, grid, tags, show_edges)
        elif tags.dim == dim - 1:
            _add_facet_tags(plotter, mesh, grid, tags, show_edges)
        else:
            logger.warning("MeshTags dimension %d not supported.", tags.dim)

    if screenshot_path is not None:
        fmt = screenshot_path.suffix.lower()
        if fmt == ".svg":
            plotter.save_graphic(str(screenshot_path))
            logger.info("Exported mesh to SVG: %s", screenshot_path)
        elif fmt == ".png":
            plotter.screenshot(
                str(screenshot_path),
                transparent_background=(background == "transparent"),
            )
        else:
            raise ValueError(f"Unsupported export format: '{fmt}'")

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


def _add_plain_mesh(
    plotter: pv.Plotter, grid: pv.UnstructuredGrid, color: str, show_edges: bool
):
    plotter.add_mesh(grid, color=color, show_edges=show_edges)


def _add_cell_tags(
    plotter: pv.Plotter, grid: pv.UnstructuredGrid, tags: MeshTags, show_edges: bool
):
    scalars = np.full(grid.n_cells, fill_value=-1, dtype=np.int32)
    scalars[tags.indices] = tags.values

    # Mask untagged cells
    tagged_mask = scalars != -1
    tagged_scalars = np.where(tagged_mask, scalars, np.nan)

    unique_tags = np.unique(tags.values)
    n_tags = len(unique_tags)

    plotter.add_mesh(
        grid,
        scalars=tagged_scalars,
        show_edges=show_edges,
        cmap="tab10",
        categories=True,
        clim=[int(unique_tags.min()), int(unique_tags.max())],
        nan_color="lightgray",
        scalar_bar_args={
            "title": "Cell tags",
            "n_labels": n_tags,
            "fmt": "%.0f",
        },
    )


def _add_facet_tags(
    plotter: pv.Plotter,
    mesh: Mesh,
    grid: pv.UnstructuredGrid,
    tags: MeshTags,
    show_edges: bool,
):
    dim = mesh.topology.dim
    plotter.add_mesh(grid, color="lightgray", opacity=0.5, show_edges=show_edges)

    # 2D: edges only
    if dim == 2:
        mesh.topology.create_connectivity(1, 0)
        conn = mesh.topology.connectivity(1, 0)

        lines = []
        tag_vals = []
        for facet, tag in zip(tags.indices, tags.values):
            vids = conn.links(facet)
            if vids.size == 2:
                lines.append(2)
                lines.extend(vids.tolist())
                tag_vals.append(int(tag))

        if not tag_vals:
            logger.warning("No 2D facets tagged, skipping facet plot.")
            return

        cells = np.array(lines, dtype=np.int64)
        types = np.full(len(tag_vals), pv.CellType.LINE, dtype=np.uint8)
        edge_grid = pv.UnstructuredGrid(cells, types, mesh.geometry.x)
        edge_grid.cell_data["facet tags"] = np.array(tag_vals, dtype=np.int32)

        unique = np.unique(tag_vals)
        plotter.add_mesh(
            edge_grid,
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

    # 3D: polygon facets
    elif dim == 3:
        facet_dim = tags.dim
        mesh.topology.create_connectivity(facet_dim, 0)
        conn = mesh.topology.connectivity(facet_dim, 0)

        all_vids = np.unique(np.concatenate([conn.links(f) for f in tags.indices]))
        coords = mesh.geometry.x[all_vids]
        vid_map = {old: new for new, old in enumerate(all_vids)}

        faces = []
        tag_vals = []
        for facet, tag in zip(tags.indices, tags.values):
            vids = conn.links(facet)
            local = [vid_map[v] for v in vids]
            if len(local) >= 3:
                faces.append(len(local))
                faces.extend(local)
                tag_vals.append(int(tag))

        if not tag_vals:
            logger.warning("No 3D facets tagged, skipping facet plot.")
            return

        faces = np.array(faces, dtype=np.int64)
        poly = pv.PolyData(coords, faces)
        poly.cell_data["facet tags"] = np.array(tag_vals, dtype=np.int32)

        unique = np.unique(tag_vals)
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
        logger.warning("Mesh dimension %d not supported for facets.", dim)
