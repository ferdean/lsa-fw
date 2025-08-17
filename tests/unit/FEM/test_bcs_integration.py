"""Boundary conditions integration tests."""

import dolfinx.fem as dfem
import numpy as np
import ufl

from config import BoundaryConditionsConfig
from FEM.bcs import (
    BoundaryConditions,
    BoundaryConditionType,
    define_bcs,
)
from FEM.operators import StokesAssembler
from FEM.spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from Meshing.core import Mesher
from Meshing.utils import Shape, iCellType
from Meshing.plot import plot_mesh
from Solver.linear import LinearSolver


def _get_mesher(*, show_plot: bool = True) -> Mesher:
    """Get problem mesher."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(16, 16), cell_type=iCellType.TRIANGLE)
    _ = mesher.generate()

    def _facet_tags(x: np.ndarray) -> int:
        if np.isclose(x[0], 0):
            # Left
            return 1
        if np.isclose(x[0], 1):
            # Right
            return 2
        if np.isclose(x[1], 0):
            # Bottom
            return 3
        if np.isclose(x[1], 1):
            # Top
            return 4
        raise ValueError

    mesher.mark_boundary_facets(_facet_tags)

    if show_plot:
        plot_mesh(mesher.mesh, tags=mesher.facet_tags)

    return mesher


def _get_neumann_bcs(mesher: Mesher, spaces: FunctionSpaces) -> BoundaryConditions:
    """Get Dirichlet and Neumann boundary conditions."""
    bc_cfgs = [
        BoundaryConditionsConfig(
            marker=1, type=BoundaryConditionType.NEUMANN_VELOCITY, value=(0.0, 0.0)
        ),
        BoundaryConditionsConfig(
            marker=1, type=BoundaryConditionType.DIRICHLET_PRESSURE, value=0.0
        ),
        BoundaryConditionsConfig(
            marker=2, type=BoundaryConditionType.NEUMANN_VELOCITY, value=(0.0, 0.0)
        ),
        BoundaryConditionsConfig(
            marker=3, type=BoundaryConditionType.DIRICHLET_VELOCITY, value=(0.0, 0.0)
        ),
        BoundaryConditionsConfig(
            marker=4, type=BoundaryConditionType.DIRICHLET_VELOCITY, value=(1.0, 0.0)
        ),
    ]

    return define_bcs(mesher, spaces, bc_cfgs)


def _solve(
    spaces: FunctionSpaces, bcs: BoundaryConditions, *, show_plot: bool = True
) -> dfem.Function:
    """Simple LU solver."""
    solver = LinearSolver(StokesAssembler(spaces, bcs))
    return solver.direct_lu_solve(show_plot=show_plot)


def test_stokes_bcs_integration() -> None:
    """This integration test verifies the implementation of mixed Dirichlet and Neumann boundary conditions in the
    steady Stokes solver on the unit square with a Taylor-Hood functions space.

    The exact solution, given the BCs, is u(x,y) = (y, 0), p(x,y) = 0.
    """
    mesher = _get_mesher(show_plot=False)
    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
    bcs = _get_neumann_bcs(mesher, spaces)
    sol = _solve(spaces, bcs, show_plot=False)

    u_h, p_h = sol.split()

    u_exact = dfem.Function(spaces.velocity)
    u_exact.interpolate(lambda x: np.vstack((x[1], np.zeros_like(x[0]))))

    err_u = dfem.assemble_scalar(
        dfem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
    )
    assert err_u**0.5 < 1e-12, f"Velocity L2-error too large: {err_u:.3e}"

    p_exact = dfem.Function(spaces.pressure)
    p_exact.interpolate(lambda x: np.zeros_like(x[0]))

    err_p = dfem.assemble_scalar(
        dfem.form(ufl.inner(p_h - p_exact, p_h - p_exact) * ufl.dx)
    )
    assert err_p**0.5 < 1e-12, f"Pressure L2-error too large: {err_p:.3e}"
