"""Poiseuille flow benchmark.

Solve the linear stability (Orr-Sommerfeld) eigenproblem for plane Poiseuille flow using a finite-element
discretization and compare with the classical results in Schmid & Henningson.

This script reuses meshing, BC, and solver components from LSA-FW as-is, with minimal adaptation to produce the complex
phase speeds c = c_r + i c_i.
"""

from pathlib import Path
from typing import Final
import numpy as np
from math import pi

import dolfinx.fem as dfem
from ufl import SpatialCoordinate, as_vector, inner, dx

from Meshing import Mesher, Shape, iCellType, plot_mesh
from FEM.spaces import define_spaces, FunctionSpaces, FunctionSpaceType
from FEM.bcs import define_bcs, BoundaryConditions
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.plot import spy
from FEM.utils import iPETScMatrix
from Solver.eigen import EigensolverConfig, EigenSolver
from Solver.utils import iEpsProblemType, iEpsWhich
from config import load_facet_config, load_bc_config

_CONFIG_DIR: Final[Path] = Path(__file__).parent / "poiseuille_flow"
_LX: Final[float] = 4 * pi
_LY: Final[float] = 1.0
_RE: Final[float] = 2000
_NUM_EIG: Final[int] = 10


def get_mesher(nxy: tuple[int, int], *, show_plot: bool = False) -> Mesher:
    """Generate a rectangular mesh with LSA-FW Mesher and apply boundary tags."""
    mesher = Mesher(
        shape=Shape.BOX,
        n=nxy,
        cell_type=iCellType.TRIANGLE,
        domain=((-_LX, -_LY), (_LX, _LY)),
    )
    mesh = mesher.generate()
    tags = load_facet_config(_CONFIG_DIR / "mesh_tags.toml")
    mesher.mark_boundary_facets(tags)
    if show_plot:
        plot_mesh(mesh, tags=mesher.facet_tags)
    return mesher


def get_spaces(mesher: Mesher) -> FunctionSpaces:
    """Get function spaces with LSA-FW FEM."""
    return define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)


def get_bcs(mesher: Mesher, spaces: FunctionSpaces) -> BoundaryConditions:
    """Get boundary conditions with LSA-FW FEM."""
    return define_bcs(mesher, spaces, load_bc_config(_CONFIG_DIR / "bcs.toml"))


def get_base_flow(
    spaces: FunctionSpaces,
    *,
    max_velocity: float = 1.0,
    show_debug: bool = False,
    tolerance: float = 1e-8,
) -> dfem.Function:
    """Get the 2D Poiseuille base flow.

    U(y) = max_velocity * (1 - (y/L)**2) in the 2-component velocity space, U_y = 0.
    """
    V = spaces.velocity
    mesh = V.mesh

    def _u_poiseuille(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = x[1]
        u_x = max_velocity * (1.0 - (y / _LY) ** 2)
        return u_x, np.zeros_like(u_x)

    u = dfem.Function(V, name="base_flow")
    u.interpolate(_u_poiseuille)

    if show_debug:
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        v_dofs, _ = spaces.velocity_dofs
        p_dofs, _ = spaces.pressure_dofs
        print(
            f"    [DEBUG] Mesh has {n_cells} cells; {v_dofs} velocity dofs and {p_dofs}"
        )
        print(
            f"    [DEBUG] Analytic base flow: Ux = {max_velocity}*(1 - (y/{_LY})**2), Uy = 0"
        )
        print(
            f"    [DEBUG] Theoretical extremal values:  min=0.000, max={max_velocity:.3f}"
        )

        x = SpatialCoordinate(mesh)
        u_true = as_vector(
            (
                max_velocity * (1 - (x[1] / _LY) ** 2),
                0.0,
            )
        )

        error_form = inner(u - u_true, u - u_true) * dx

        L2_sq = dfem.assemble_scalar(dfem.form(error_form))
        L2 = np.sqrt(L2_sq)

        print(f"    [DEBUG] Poiseuille base flow interpolation error (L2): {L2:.3e}")
        if L2 >= tolerance:
            raise AssertionError(
                f"Poiseuille interpolation error {L2:.3e} exceeds tolerance {tolerance:.1e}"
            )

    return u


def get_matrices(
    base_flow: dfem.Function,
    spaces: FunctionSpaces,
    bcs: BoundaryConditions,
    *,
    show_plot: bool = False,
) -> tuple[iPETScMatrix, iPETScMatrix]:
    """Get the assembled FEM matrices with LSA-FW FEM."""
    assembler = LinearizedNavierStokesAssembler(base_flow, spaces, re=_RE, bcs=bcs)
    A, M = assembler.assemble_eigensystem()
    if show_plot:
        spy(A, _CONFIG_DIR / "A.png")
        spy(M, _CONFIG_DIR / "M.png")
    return A, M


if __name__ == "__main__":
    mesher = get_mesher((32, 8), show_plot=False)
    spaces = get_spaces(mesher)
    bcs = get_bcs(mesher, spaces)
    base_flow = get_base_flow(spaces, show_debug=False)
    A, M = get_matrices(base_flow, spaces, bcs, show_plot=True)

    es = EigenSolver(
        cfg=EigensolverConfig(
            num_eig=_NUM_EIG,
            problem_type=iEpsProblemType.GNHEP,
            atol=1e-4,
            max_it=1000,
        ),
        A=A,
        M=M,
    )
    es.solver.set_which_eigenpairs(iEpsWhich.SMALLEST_MAGNITUDE)
    numerical = es.solve()

    # Extract eigenvalues and convert λ → phase speed
    lams, _ = zip(*numerical)
    lams = np.array(lams, dtype=complex)
    alpha = 1.0
    c_vals = 1j * lams / alpha
    c_r = c_vals.real
    c_i = c_vals.imag

    header = f"\nPoiseuille flow stability (alpha=1, beta=0, Re={_RE:.0f}):\n\n"
    header += f"{'mode':>4}{'c_r':>12}  {'c_i':>12}"
    print(header)
    for idx, (cr, ci) in enumerate(zip(c_r, c_i), start=1):
        print(f"{idx:4d}   {cr:12.8f}   {ci:12.8f}")
    print()
