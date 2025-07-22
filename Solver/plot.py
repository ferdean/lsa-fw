"""LSA-FW Solution plotter."""

import logging
from enum import Enum, StrEnum, auto

import dolfinx.fem as dfem
import numpy as np
import pyvista as pv
from dolfinx.plot import vtk_mesh
from mpi4py import MPI


logger = logging.getLogger(__name__)

# Global pyvista settings
pv.OFF_SCREEN = True
pv.global_theme.font.family = "times"
pv.global_theme.font.size = 18
pv.global_theme.font.label_size = 16


class PlotMode(StrEnum):
    """Available plot modes."""

    INTERACTIVE = auto()
    STATIC = auto()


class PlotType(Enum):
    """Available plot types."""

    VEL_X = auto()
    VEL_Y = auto()
    VEL_Z = auto()
    PRESSURE = auto()
    VELOCITY = auto()


def plot_streamlines(
    mixed_function: dfem.Function,
    factor: float = 0.1,
    n_points: int = 20,
    off_screen: bool = False,
    screenshot: str | None = None,
    transparent: bool = True,
    figsize: int = 800,
) -> None:
    """Visualize velocity streamlines from a mixed (velocity, pressure) function."""
    # Extract velocity component
    u_c = mixed_function.sub(0).collapse()

    # Mesh geometry
    cells, types, x = vtk_mesh(u_c.function_space)
    grid = pv.UnstructuredGrid(cells, types, x)

    # Pad to 3D
    gdim = u_c.function_space.mesh.geometry.dim
    vals = u_c.x.array.real.reshape((-1, gdim))
    vecs = np.zeros((vals.shape[0], 3), dtype=float)
    vecs[:, :gdim] = vals
    grid["vectors"] = vecs
    grid.set_active_vectors("vectors")

    # Generate seed points in the bounding box
    bounds = grid.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    xs = np.linspace(bounds[0], bounds[1], n_points)
    ys = np.linspace(bounds[2], bounds[3], n_points)
    if gdim == 3:
        zs = np.linspace(bounds[4], bounds[5], n_points)
    else:
        zs = [0.0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    seed_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    seeds = pv.PolyData(seed_points)

    # Streamlines
    stream = grid.streamlines_from_source(
        seeds,
        integrator_type=45,
        integration_direction="both",
        max_time=1.0,
        initial_step_length=0.01,
        terminal_speed=1e-10,
    ).tube(radius=factor)

    # Plotter
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_text(
        "u streamlines", position="upper_edge", font_size=20, color="black"
    )
    plotter.add_mesh(grid, show_edges=False)
    plotter.add_mesh(stream)
    plotter.view_xy()

    if screenshot or off_screen:
        fname = screenshot or f"streamlines_{MPI.COMM_WORLD.rank}.png"
        plotter.show(
            screenshot=fname,
            transparent_background=transparent,
            window_size=[figsize, figsize],
        )
    else:
        plotter.show()
