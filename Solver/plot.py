"""LSA-FW Solution plotter."""

import logging
import pyvista as pv
import numpy as np
from enum import StrEnum, auto, Enum
from pathlib import Path

from dolfinx.plot import vtk_mesh

from Meshing import Mesher
from FEM.utils import iComplexPETScVector
from FEM.spaces import FunctionSpaces

logger = logging.getLogger(__name__)

# Global pyvista settings
pv.OFF_SCREEN = True
pv.start_xvfb()
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


def plot_eigenvector(
    mesher: Mesher,
    spaces: FunctionSpaces,
    eigenvector: iComplexPETScVector,
    eigenvalue: float | complex,
    *,
    mode: PlotMode = PlotMode.INTERACTIVE,
    type: PlotType = PlotType.VEL_X,
    show_edges: bool = False,
    color: str = "white",
    background: str = "transparent",
    window_size: tuple[int, int] = (800, 600),
    screenshot_path: Path | None = None,
) -> None:
    """Plot eigenvector."""
    topo, cell_types, points = vtk_mesh(mesher.mesh)
    grid = pv.UnstructuredGrid(topo, cell_types, points)

    num_vel_dof, _ = spaces.velocity_dofs
    num_pre_dof, _ = spaces.pressure_dofs

    if type is PlotType.PRESSURE:
        dof = eigenvector.real.as_array()[num_vel_dof:]

    else:
        # Velocity plot
        dof = eigenvector.real.as_array()[:num_vel_dof].reshape((-1, mesher.gdim))

        if type is PlotType.VELOCITY:
            dof = np.linalg.norm(dof, axis=1)
        else:
            dof = dof[:, type.value]

    grid.point_data[type.name.lower] = dof

    off_screen = screenshot_path is not None
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)
    if background != "transparent":
        plotter.set_background(background)
    elif not off_screen:
        logger.warning(
            "Transparent background only supported for off-screen export; "
            "interactive view will use default background."
        )

    plotter.add_mesh(
        grid, scalars=dof, color=color, show_edges=show_edges, lighting=False
    )
    title = f"Eigenvector for λ = {eigenvalue.real:.3f}" + (
        f" + {eigenvalue.imag:.3f}i" if abs(eigenvalue.imag) > 1e-8 else ""
    )
    plotter.add_text(title, font="times", font_size=20)

    if mode is PlotMode.INTERACTIVE:
        plotter.show()
