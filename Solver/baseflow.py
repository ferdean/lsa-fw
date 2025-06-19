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

import dolfinx.fem as dfem

from FEM.bcs import BoundaryConditions
from FEM.operators import StokesAssembler, StationaryNavierStokesAssembler
from FEM.plot import plot_mixed_function
from FEM.spaces import FunctionSpaces

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
        logger.info(
            "Assembling and solving Stokes flow, to be used as Newton's initial guess."
        )
        stokes_assembler = StokesAssembler(self._spaces, bcs=self._bcs)
        stokes_solver = LinearSolver(stokes_assembler)
        return stokes_solver.direct_lu_solve(show_plot=False)

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
    ) -> dfem.Function:
        """Assemble and solve the stationary Navier-Stokes equations for a given Reynolds number."""
        if self._initial_guess is None:
            # Use Stokes flow as initial guess for the Newton solver
            self._initial_guess = self._solve_stokes_flow()

        if ramp and steps > 1:
            re_ramp = _linspace(1.0, re, steps)
        else:
            re_ramp = [re]

        sol = self._initial_guess
        for re in re_ramp:
            logger.info("Solving stationary Navier-Stokes at Re=%.2f", re)
            ns_assembler = StationaryNavierStokesAssembler(
                self._spaces, re=re, bcs=self._bcs, initial_guess=sol
            )
            newton = NewtonSolver(ns_assembler)
            sol = newton.solve(max_it=max_it, atol=tol, damping_factor=damping_factor)

        if show_plot:
            plot_mixed_function(
                sol, scale=plot_scale, title=f"Baseflow (Re = {re:.2f})"
            )

        return sol
