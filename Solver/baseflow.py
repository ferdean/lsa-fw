"""LSA-FW Baseflow Solver.

This module provides the BaseFlowSolver class to compute the steady-state (base) flow solution of the incompressible
Navier-Stokes equations. Under the hood, it:

  - Assembles and solves a Stokes problem to generate a robust initial guess.
  - Optionally ramps the Reynolds number from 1.0 to the user-specified target over multiple steps to improve
  convergence.
  - Solves the stationary Navier-Stokes equations using Newton's method with an adjustable damping factor.
  - Can plot the resulting mixed velocity-pressure field for visualization.

The computed base flow can then be used as the background state for linear stability analyses or perturbation solvers.

Example:
```python
# Upstream: build mesh, define FunctionSpaces and BoundaryConditions...
solver = BaseFlowSolver(spaces, bcs)

# Compute base flow at Re = 100 with 5-step ramp and 0.8 damping, and show plot
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

import dolfinx.fem as dfem
from petsc4py import PETSc

from FEM.bcs import BoundaryConditions
from FEM.operators import StokesAssembler, StationaryNavierStokesAssembler
from lib.cache import CacheStore
from FEM.plot import plot_mixed_function
from FEM.spaces import FunctionSpaces, define_spaces, FunctionSpaceType
from lib.loggingutils import log_global

from .linear import LinearSolver
from .nonlinear import NewtonSolver


logger = logging.getLogger(__name__)


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num < 2:
        return [stop]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


class BaseFlowSolver:
    """Solves for the base (stationary) flow."""

    def __init__(
        self, spaces: FunctionSpaces, *, bcs: BoundaryConditions | None = None
    ) -> None:
        """Initialize."""
        self._spaces = spaces
        self._bcs = bcs
        self._initial_guess: dfem.Function | None = None

    def _solve_stokes_flow(self) -> dfem.Function:
        """Assemble and solve the stokes flow, to be used as initial guess for the stationary NS flow."""
        log_global(
            logger,
            logging.INFO,
            "Assembling and solving Stokes flow, to be used as Newton's initial guess.",
        )
        stokes_assembler = StokesAssembler(self._spaces, bcs=self._bcs)
        stokes_solver = LinearSolver(stokes_assembler)
        return stokes_solver.gmres_solve(show_plot=False)

    def solve(
        self,
        re: float,
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

        If ``cache`` and ``key`` are provided, the solver attempts to read a cached
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

        if ramp and steps > 1:
            re_ramp = _linspace(1.0, re, steps)
        else:
            re_ramp = [re]

        sol = self._initial_guess
        for re in re_ramp:
            log_global(logger, logging.INFO, "Solving stationary Navier-Stokes at Re=%.2f", re)
            ns_assembler = StationaryNavierStokesAssembler(
                self._spaces, re=re, bcs=self._bcs, initial_guess=sol
            )
            newton = NewtonSolver(ns_assembler)
            sol = newton.solve(max_it=max_it, atol=tol, damping_factor=damping_factor)

        if show_plot:
            plot_mixed_function(
                sol, scale=plot_scale, title=f"Baseflow (Re = {re:.2f})"
            )

        if cache is not None and key is not None:
            cache.save_function(key, sol)

        return sol


# HACK: In theory, the dolfinx API should be sufficient to export/import function objects. However, up to now,
# writing or reading a mixed-function (e.g., a Taylor–Hood mixed P2/P1) directly can lead to low-level crashes
# or mismatches because PETSc binary views and XDMF I/O expect a single continuous Lagrange field of matching
# polynomial degree. Mixed spaces must be split into their subcomponents (velocity and pressure), and even
# then the degree of the output must match the mesh degree to avoid I/O failures. As a workaround, we optionally
# interpolate the P2 velocity to a P1 vector space for safe export. Importing back into the original mixed
# space may still fail or lose fidelity if the exported data does not exactly match the mixed DOF layout.


def export_baseflow(
    baseflow: dfem.Function, output_folder: Path, *, linear_velocity_ok: bool = False
) -> None:
    """Export baseflow function."""
    output_folder.mkdir(parents=True, exist_ok=True)
    mesh = baseflow.function_space.mesh

    if baseflow.function_space.sub(0).ufl_element().degree > 1 and linear_velocity_ok:
        u_p2, p = baseflow.split()
        linear_spaces = define_spaces(mesh, type=FunctionSpaceType.SIMPLE)
        u = dfem.Function(linear_spaces.velocity)
        u.interpolate(u_p2)
        log_global(
            logger,
            logging.WARNING,
            "Interpolated P2 velocity to P1 vector space for safe export. Exported baseflow may lose precision.",
        )
    else:
        u, p = baseflow.split()
        log_global(
            logger,
            logging.WARNING,
            "Exporting full mixed function vector. Import back into mixed space may fail or lose fidelity"
            " due to mixed DOF layout limitations.",
        )

    for function, function_name in ((u, "velocity"), (p, "pressure")):
        path = output_folder / f"{function_name}.xdmf"
        viewer = PETSc.Viewer().createBinary(str(path), mode=PETSc.Viewer.Mode.WRITE)
        try:
            function.x.petsc_vec.view(viewer)
            log_global(logger, logging.INFO, "Baseflow %s properly exported to '%s'", function_name, path)
        except Exception as e:
            log_global(logger, logging.ERROR, "Baseflow %s could not be exported to disk.", function_name)
            raise e
        finally:
            viewer.destroy()


def load_baseflow(input_folder: Path, spaces: FunctionSpaces) -> dfem.Function:
    """Import baseflow function."""
    if not input_folder.exists() or not input_folder.is_dir():
        raise ValueError(f"Input path {input_folder!r} is not a valid folder.")

    log_global(
        logger,
        logging.WARNING,
        "Importing full mixed function vector may cause lose fidelity, as the function spaces may have been "
        "linearized during the export process. Refer to `export_baseflow` for further details.",
    )

    _, dofs_u = spaces.mixed.sub(0).collapse()
    _, dofs_p = spaces.mixed.sub(1).collapse()

    baseflow = dfem.Function(spaces.mixed)
    u = dfem.Function(spaces.mixed)
    p = dfem.Function(spaces.mixed)

    for function, function_name in ((u, "velocity"), (p, "pressure")):
        path = input_folder / f"{function_name}.xdmf"
        viewer = PETSc.Viewer().createBinary(str(path), mode=PETSc.Viewer.Mode.READ)
        try:
            function.x.petsc_vec.load(viewer)
            log_global(logger, logging.INFO, "Baseflow %s properly imported from '%s'", function_name, path)
        except Exception as e:
            log_global(
                logger,
                logging.ERROR,
                "Baseflow %s could not be imported from %s.",
                function_name,
                path,
            )
            raise e
        finally:
            viewer.destroy()

        function.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.FORWARD
        )

    baseflow.x.array[dofs_u] = u.x.array[dofs_u]
    baseflow.x.array[dofs_p] = p.x.array[dofs_p]

    return baseflow
