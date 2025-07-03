"""LSA-FW Nonlinear Solver 2.

This module re-implements the nonlinear solvers for PDEs (e.g., stationary Navier-Stokes) via PETSc's SNES interface.
Under the hood, it:

  - Pre-allocates and reuses distributed PETSc matrix and vector for efficiency.
  - Defines user-supplied residual and Jacobian callbacks.
  - Leverages PETSc SNES for robust line-search, convergence monitoring, and MPI-aware parallel solves.
  - Captures residual history for post-solve diagnostics and plotting.

By migrating to SNES, this solver gains C-level performance, built-in parallelization, and flexible tolerance/
line-search options—all while retaining compatibility with LSA-FW's `BaseAssembler` and BC infrastructure.
"""

import logging
from typing import Any
import matplotlib.pyplot as plt

import dolfinx.fem as dfem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
from mpi4py import MPI

from FEM.operators import BaseAssembler
from lib.loggingutils import log_global, log_rank

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
    }
)
for name in ("matplotlib.ticker", "matplotlib.font_manager", "matplotlib.pyplot"):
    logging.getLogger(name).disabled = True


class NewtonSolver:
    """Newton solver implementation."""

    def __init__(self, assembler: BaseAssembler, *, damping: float = 1.0) -> None:
        """Initialize."""
        if not (0.0 < damping <= 1.0):
            log_global(
                logger,
                logging.WARNING,
                "Damping factor %.2f out of (0,1]; using 1.0 (undamped).",
                damping,
            )
            damping = 1.0

        self._configure_snes_opts(damping)

        self._assembler = assembler
        self._residual_history: list[float] = []
        self._A, self._b = assembler.get_matrix_forms()
        self._snes = self._get_snes(assembler.spaces.mixed.mesh.comm)

    @staticmethod
    def _configure_snes_opts(damping: float) -> None:
        opts = PETSc.Options()
        opts["ksp_type"] = "gmres"
        # FIXME: This might be a bottleneck for larger-scale problems (e.g., 3D)
        opts["pc_type"] = "lu"
        opts["pc_factor_mat_solver_type"] = "mumps"
        opts["snes_type"] = "newtonls"
        opts["snes_linesearch_type"] = "basic"  # Basic backtracking LS
        opts["snes_linesearch_damping"] = damping

    @staticmethod
    def _get_snes(comm: MPI.Intercomm) -> PETSc.SNES:
        snes = PETSc.SNES().create(comm=comm)
        snes.setFromOptions()
        return snes

    def solve(self, *, max_it: int = 500, tol: float = 1e-8) -> dfem.Function:
        """Solve the nonlinear problem."""

        log_global(logger, logging.DEBUG, "Newton solver started.")
        self._residual_history.clear()

        def _form_function(_: Any, x: PETSc.Vec, f: PETSc.Vec) -> None:
            sol_vec = self._assembler.sol.x
            with x.localForm() as x_local:
                sol_vec.array[:] = x_local[:]
            sol_vec.scatter_forward()

            with f.localForm() as loc:
                loc.set(0)  # Zero-out

            assemble_vector(f, self._assembler.residual)

            f.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            apply_lifting(
                f,
                [self._assembler.jacobian],
                [list(self._assembler.bcs)],
                x0=[sol_vec.petsc_vec],
                alpha=1.0,
            )
            set_bc(f, [*self._assembler.bcs], sol_vec.petsc_vec, 1.0)
            f.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )

        def _form_jacobian(_: Any, x: PETSc.Vec, J: PETSc.Mat, *args) -> None:
            sol_vec = self._assembler.sol.x
            with x.localForm() as x_local:
                sol_vec.array[:] = x_local[:]
            sol_vec.scatter_forward()
            J.zeroEntries()
            assemble_matrix(J, self._assembler.jacobian, bcs=list(self._assembler.bcs))
            J.assemble()

        self._snes.setFunction(_form_function, self._b.raw)
        self._snes.setJacobian(_form_jacobian, self._A.raw, self._A.raw)
        self._snes.setTolerances(atol=tol, rtol=tol, max_it=max_it)

        def _monitor(_, its: int, norm: float) -> None:
            log_rank(logger, logging.DEBUG, "Iteration %d - residual = %.2e", its, norm)
            self._residual_history.append(norm)

        self._snes.setMonitor(_monitor)

        x = self._assembler.sol.x
        self._snes.solve(None, x.petsc_vec)

        iters = self._snes.getIterationNumber()
        reason = self._snes.getConvergedReason()
        converged = reason > 0
        last_r = self._residual_history[-1] if self._residual_history else float("nan")
        if not converged:
            log_global(
                logger,
                logging.WARNING,
                "Newton did not converge after %d iterations (last residual: %.6f, reason code: %d)",
                iters,
                last_r,
                reason,
            )
        else:
            log_global(
                logger, logging.INFO, "Newton converged after %d iterations.", iters
            )

        return self._assembler.sol
