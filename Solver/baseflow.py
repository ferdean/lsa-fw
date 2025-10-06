"""LSA-FW Baseflow Solver.

This module provides the BaseFlowSolver class to compute the steady-state (base) flow solution of the incompressible
Navier-Stokes equations. Under the hood, it:

  - Assembles and solves a Stokes problem to generate a robust initial guess.
  - Optionally ramps the Reynolds number from 1.0 to the user-specified target over multiple steps to improve
  convergence.
  - Solves the stationary Navier-Stokes equations using Newton's method with an adjustable damping factor.
  - Can plot the resulting mixed velocity-pressure field for visualization.

The computed baseflow can then be used as the background state for linear stability analyses or perturbation solvers.

Example:
```python
# Upstream: build mesh, define FunctionSpaces and BoundaryConditions...
solver = BaseFlowSolver(spaces, bcs)

# Compute baseflow at Re = 100 with 5-step ramp and 0.8 damping, and show plot
baseflow = solver.solve(
    re=100.0,
    ramp=True,
    steps=5,
    damping_factor=0.8,
    show_plot=True,
)
"""

import logging
from pathlib import Path

from FEM.utils import Scalar
import dolfinx.fem as dfem
from dolfinx.mesh import MeshTags
from dolfinx import la
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx.io import XDMFFile

from ufl import Identity, FacetNormal, as_vector, sym, grad, dot, ds

from FEM.bcs import BoundaryConditions
from FEM.operators import StationaryNavierStokesAssembler, StokesAssembler
from FEM.plot import plot_mixed_function, PlotMode
from FEM.spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from lib.cache import CacheStore
from lib.loggingutils import log_global

from .linear import LinearSolver
from .nonlinear2 import NewtonSolver

logger = logging.getLogger(__name__)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num < 2:
        return [stop]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


class BaseFlowSolver:
    """Solves for the base (stationary) flow."""

    def __init__(
        self,
        spaces: FunctionSpaces,
        *,
        re: float,
        bcs: BoundaryConditions | None = None,
        tags: MeshTags | None = None,
        use_sponge: bool = True,
    ) -> None:
        """Initialize."""
        self._spaces = spaces
        self._re = re
        self._bcs = bcs
        self._initial_guess: dfem.Function | None = None
        self._tags = tags
        self._use_sponge = use_sponge

    def _solve_stokes_flow(self) -> dfem.Function:
        """Assemble and solve the stokes flow, to be used as initial guess for the stationary NS flow."""
        log_global(
            logger,
            logging.INFO,
            "Assembling and solving Stokes flow, to be used as Newton's initial guess.",
        )
        stokes_assembler = StokesAssembler(self._spaces, re=self._re, bcs=self._bcs)
        stokes_solver = LinearSolver(stokes_assembler)
        return stokes_solver.gmres_solve(show_plot=False)

    def solve(
        self,
        *,
        ramp: bool = False,
        steps: int = 3,
        max_it: int = 1_000,
        tol: float = 1e-6,
        damping_factor: float = 1.0,
        show_plot: bool = False,
        plot_scale: float = 0.01,
        cache: CacheStore | None = None,
        key: str | None = None,
    ) -> dfem.Function:
        """Assemble and solve the stationary Navier-Stokes equations for a given Reynolds number.

        If `cache` and `key` are provided, the solver attempts to read a cached
        solution before computing. After solving, the result is stored in the cache.
        """
        if cache is not None and key is not None:
            cached = cache.load_function(key, self._spaces.mixed)
            if cached is not None:
                self._initial_guess = cached
                return cached
        if self._initial_guess is None:
            # Use Stokes flow as initial guess for the Newton solver
            self._initial_guess = self._solve_stokes_flow()
            self._initial_guess.x.scatter_forward()

        if ramp and steps > 1:
            re_ramp = _linspace(1.0, self._re, steps)
        else:
            re_ramp = [self._re]

        sol = self._initial_guess
        for re in re_ramp:
            log_global(
                logger, logging.INFO, "Solving stationary Navier-Stokes at Re=%.2f", re
            )
            ns_assembler = StationaryNavierStokesAssembler(
                self._spaces,
                re=re,
                bcs=self._bcs,
                initial_guess=sol,
                tags=self._tags,
                use_sponge=self._use_sponge,
            )
            newton = NewtonSolver(ns_assembler, damping=damping_factor)

            sol.x.scatter_reverse(mode=la.InsertMode.add)
            sol = newton.solve(max_it=max_it, tol=tol)
            sol.x.scatter_forward()

        if show_plot:
            plot_mixed_function(
                sol,
                PlotMode.INTERACTIVE,
                scale=plot_scale,
                title=f"Baseflow (Re = {re:.2f})",
            )

        if cache is not None and key is not None:
            cache.save_function(key, sol)

        return sol


def compute_recirculation_length(
    baseflow: dfem.Function,
    *,
    restrict_to_centreline: bool = False,
    centreline_tol: float = 1e-6,
) -> float:
    """Compute the recirculation length behind a cylinder or object in a channel flow.

    Assumes that the object axis is located at the coordinate origin.
    This function defines the recirculation length as the maximum x-coordinate where u_x < 0.
    As per definition, if the computation is restricted to the center line, only nodes with abs(y) <= tol are
    considered.
    """
    u = baseflow.sub(0).collapse()
    gdim = u.function_space.mesh.geometry.dim
    with u.x.petsc_vec.localForm() as u_local:
        uv = u_local.getArray().reshape((-1, gdim))  # (u_x, u_y)

    coords = u.function_space.tabulate_dof_coordinates()[:, :gdim]

    # Find all dofs where u_x < 0
    mask = uv[:, 0] < 0.0
    if restrict_to_centreline:
        mask &= np.abs(coords[:, 1]) <= centreline_tol

    if not np.any(mask):
        raise RuntimeError("No negative u_x found; no recirculation detected.")

    return float(max(coords[mask, 0]))


def compute_drag(
    baseflow: dfem.Function,
    *,
    re: float,
    facet_tags: MeshTags,
    cylinder_marker: int,
) -> float:
    """Compute non-dimensional drag Fx over the cylinder boundary."""
    mesh = baseflow.function_space.mesh
    gdim = mesh.geometry.dim
    if gdim not in (2, 3):
        raise ValueError("Only 2D/3D supported.")

    u, p = baseflow.split()

    n = FacetNormal(mesh)
    identity = Identity(gdim)
    sym_grad = sym(grad(u))
    sigma = -p * identity + (2.0 / re) * sym_grad  # Cauchy stress (nondimensional)
    traction = dot(sigma, n)

    ex = as_vector([1.0] + [0.0] * (gdim - 1))
    Fx_form = dot(traction, ex) * ds(subdomain_data=facet_tags)(cylinder_marker)

    Fx = dfem.assemble_scalar(dfem.form(Fx_form, dtype=Scalar))
    return abs(float(Fx))


def export_function(
    function: dfem.Function,
    output_folder: Path,
    *,
    linear_velocity_ok: bool = False,
    name: str = "baseflow",
) -> None:
    """Export baseflow (u, p) as real numpy arrays + subspace DOF maps, safe across real/complex builds."""
    if MPI.COMM_WORLD.size > 1:
        log_global(
            logger,
            logging.WARNING,
            "Function export is not supported in parallel (MPI size = %d); please run in serial.",
            MPI.COMM_WORLD.size,
        )
    output_folder.mkdir(parents=True, exist_ok=True)
    mesh = function.function_space.mesh

    # Split and (optionally) down-interpolate velocity to P1 if your pipeline assumes linear vectors.
    u_p2, p = function.split()
    if u_p2.function_space.ufl_element().degree > 1 and linear_velocity_ok:
        linear_spaces = define_spaces(mesh, type=FunctionSpaceType.SIMPLE)
        u = dfem.Function(linear_spaces.velocity)
        u.interpolate(u_p2)
        log_global(
            logger,
            logging.WARNING,
            "Interpolated P2 velocity to P1 for export; expect some precision loss.",
        )
    else:
        u = u_p2

    # DOF maps from mixed space to subspaces
    _, dofs_u = function.function_space.sub(0).collapse()
    _, dofs_p = function.function_space.sub(1).collapse()

    # Extract real arrays in subspace ordering
    arr_mixed = np.asarray(function.x.array, dtype=np.float64)
    u_data = (
        np.asarray(u.x.array, dtype=np.float64)
        if u is not u_p2
        else arr_mixed[dofs_u].copy()
    )
    p_data = arr_mixed[dofs_p].copy().astype(np.float64)

    # Persist as NumPy; scalar-type agnostic
    np.savez(
        output_folder / f"{name}_npz.npz",
        u=u_data,
        p=p_data,
        dofs_u=dofs_u,
        dofs_p=dofs_p,
    )

    with XDMFFile(mesh.comm, str(output_folder / "mesh.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
    log_global(
        logger, logging.INFO, "Function '%s' exported to '%s'", name, output_folder
    )


def load_function(
    input_folder: Path, spaces: FunctionSpaces, *, name: str = "baseflow"
) -> dfem.Function:
    """Load baseflow into a mixed complex space by filling the real part and zeroing imag."""
    if MPI.COMM_WORLD.size > 1:
        log_global(
            logger,
            logging.WARNING,
            "Function load is not supported in parallel (MPI size = %d); please run in serial.",
            MPI.COMM_WORLD.size,
        )
    if not input_folder.is_dir():
        raise ValueError(f"Input path {input_folder!r} is not a valid folder.")

    function = dfem.Function(spaces.mixed)
    z = function.x.array

    data = np.load(input_folder / f"{name}_npz.npz", allow_pickle=False)
    u_data: np.ndarray = data["u"]
    p_data: np.ndarray = data["p"]
    dofs_u: np.ndarray = data["dofs_u"]
    dofs_p: np.ndarray = data["dofs_p"]

    # Write into real part; zero imag if present
    if np.iscomplexobj(z):
        zr = z.real
        zi = z.imag
        zr[dofs_u] = u_data
        zr[dofs_p] = p_data
        zi[dofs_u] = 0.0
        zi[dofs_p] = 0.0
        # Ensure PETSc Vec sees updates
        function.x.petsc_vec.assemblyBegin()
        function.x.petsc_vec.assemblyEnd()
    else:
        z[dofs_u] = u_data
        z[dofs_p] = p_data

    function.x.petsc_vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    log_global(
        logger, logging.INFO, "Function '%s' loaded from '%s'", name, input_folder
    )
    return function
