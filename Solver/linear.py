"""LSA-FW Linear Solver.

Provides a linear solver interface that can be extended for different linear solvers.
"""

import logging
from pathlib import Path

import numpy as np

import dolfinx.fem as dfem
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from matplotlib.ticker import MaxNLocator
from mpi4py import MPI

from FEM.operators import BaseAssembler
from FEM.plot import plot_mixed_function
from FEM.utils import iPETScMatrix, iPETScVector
from lib.loggingutils import log_global, log_rank
from Solver.utils import KSPType, PreconditionerType, iKSP

logger = logging.getLogger(__name__)
plt.rcParams.update({"font.family": "serif", "font.size": 10})


class LinearSolver:
    """Container for different linear-solving algorithms."""

    def __init__(self, assembler: BaseAssembler) -> None:
        """Initialize."""
        self.assembler = assembler
        self._ab_cache: dict[str, tuple[iPETScMatrix, iPETScVector, iPETScVector]] = {}
        self._ksp_cache: dict[str, iKSP] = {}
        self._lu_cache: dict[str, scipy.sparse.linalg.SuperLU] = {}
        self._res_hist: dict[str, list[float]] = {}

    @staticmethod
    def solve(
        A: iPETScMatrix,
        b: iPETScVector,
        *,
        ksp_type: KSPType,
        tol: float = 1e-12,
        rtol: float = 1e-8,
        max_it: int = 1_000,
    ) -> iPETScVector:
        """Solve the linear system Ax = b using the specified KSP type.

        Note that this function is static and does not have access to the instance variables of the LinearSolver class.
        It is designed to offer an assembler-free option to solve linear system if the system is generated from a
        different source.
        """
        if ksp_type not in (KSPType.PREONLY, KSPType.GMRES):
            raise ValueError("KSP type not supported.")

        log_rank(logger, logging.DEBUG, "Configuring iKSP for %s", ksp_type.name)
        solver = iKSP(A)
        solver.set_type(ksp_type)
        solver.set_preconditioner(
            PreconditionerType.LU
            if ksp_type is KSPType.PREONLY
            else PreconditionerType.NONE
        )

        if ksp_type is not KSPType.PREONLY:
            # Only set tolerances for iterative method
            solver.set_tolerances(tol=tol, rtol=rtol, max_it=max_it)
        else:
            solver.raw.setGMRESRestart(True)

        solver.set_from_options()

        # Solve and time
        sol = iPETScVector.from_array(np.zeros((b.size,)))
        t0 = MPI.Wtime()
        solver.solve(b, sol)
        t1 = MPI.Wtime()

        log_global(
            logger, logging.INFO, "%s solve time: %.3f s", ksp_type.name, t1 - t0
        )

        # sol.raw.scatter_forward()

        log_global(logger, logging.INFO, f"{ksp_type.name} solve completed.")
        return sol

    def direct_lu_solve(
        self,
        *,
        show_plot: bool = False,
        plot_scale: float = 0.1,
        use_scipy: bool = True,
        key: int | str | None = None,
    ) -> dfem.Function:
        """Direct solve using PETSc LU (parallel) or SciPy splu (serial)."""
        if use_scipy:
            if MPI.COMM_WORLD.size > 1:
                log_global(
                    logger,
                    logging.WARNING,
                    "Requested SciPy direct solver on %d MPI ranks: "
                    "this will run only on rank 0 without parallelism.",
                    MPI.COMM_WORLD.size,
                )
            return self.direct_scipy_solve(
                show_plot=show_plot, plot_scale=plot_scale, key=key
            )

        return self._solve(
            ksp_type=KSPType.PREONLY,
            pc_type=PreconditionerType.LU,
            tol=0.0,  # Not used for PREONLY
            rtol=0.0,  # Not used for PREONLY
            max_it=1,  # A single factor solve
            show_plot=show_plot,
            plot_scale=plot_scale,
            key=key,
        )

    def direct_scipy_solve(
        self,
        *,
        show_plot: bool = False,
        plot_scale: float = 0.1,
        key: int | str | None = None,
    ) -> dfem.Function:
        """Serial direct solve using scipy."""
        log_global(logger, logging.INFO, "Serial SciPy LU solve started.")
        cache_key = key or "direct_scipy"

        # Assemble and cache A, b in SciPy form
        if cache_key not in self._ab_cache:
            log_rank(
                logger, logging.DEBUG, "Assembling system matrix and RHS for SciPy LU"
            )
            A_petsc, b_petsc = self.assembler.get_matrix_forms()
            A_petsc.assemble()
            b_petsc.ghost_update()
            As = A_petsc.as_scipy_array().tocsc()
            bs = b_petsc.raw.array.copy()
            self._ab_cache[cache_key] = (As, bs)
        else:
            As, bs = self._ab_cache[cache_key]

        # Factor and cache
        if cache_key not in self._lu_cache:
            log_rank(logger, logging.DEBUG, "Factoring (splu) system matrix.")
            self._lu_cache[cache_key] = scipy.sparse.linalg.splu(As)
        lu = self._lu_cache[cache_key]

        # Solve
        try:
            xs = lu.solve(bs)
        except Exception as e:
            raise RuntimeError("SciPy LU solve failed") from e

        # Scatter solution back into dolfinx
        sol = self.assembler.sol
        sol.x.array[:] = xs
        sol.x.scatter_forward()

        if show_plot and MPI.COMM_WORLD.rank == 0:
            plot_mixed_function(sol, scale=plot_scale)

        log_global(logger, logging.INFO, "Serial SciPy LU solve completed.")
        return sol

    def cg_solve(
        self,
        *,
        tol: float = 1e-12,
        rtol: float = 1e-8,
        max_it: int = 1000,
        show_plot: bool = False,
        plot_scale: float = 0.1,
        key: int | str | None = None,
        enable_monitor: bool = True,
    ) -> dfem.Function:
        """Iterative solve using Conjugate Gradient (CG)."""
        return self._solve(
            ksp_type=KSPType.CG,
            pc_type=PreconditionerType.NONE,
            tol=tol,
            rtol=rtol,
            max_it=max_it,
            show_plot=show_plot,
            plot_scale=plot_scale,
            key=key,
            enable_monitor=enable_monitor,
        )

    def gmres_solve(
        self,
        *,
        tol: float = 1e-12,
        rtol: float = 1e-8,
        max_it: int = 1000,
        restart: int = 30,
        show_plot: bool = False,
        plot_scale: float = 0.1,
        key: int | str | None = None,
        enable_monitor: bool = True,
    ) -> dfem.Function:
        """Iterative solve using GMRES."""
        return self._solve(
            ksp_type=KSPType.GMRES,
            pc_type=PreconditionerType.NONE,
            tol=tol,
            rtol=rtol,
            max_it=max_it,
            restart=restart,
            show_plot=show_plot,
            plot_scale=plot_scale,
            key=key,
            enable_monitor=enable_monitor,
        )

    def _solve(
        self,
        ksp_type: KSPType,
        pc_type: PreconditionerType,
        *,
        tol: float,
        rtol: float,
        max_it: int,
        show_plot: bool,
        plot_scale: float,
        key: int | str | None,
        restart: int | None = None,
        enable_monitor: bool = True,
    ) -> dfem.Function:
        log_global(logger, logging.INFO, f"{ksp_type.name} solve started.")
        cache_key = key or f"{ksp_type.name.lower()}_{id(self)}"

        # Assemble and cache system matrix, RHS, and solution vector
        if cache_key not in self._ab_cache:
            log_rank(
                logger,
                logging.DEBUG,
                "Assembling system matrix and RHS for %s solve",
                ksp_type.name,
            )
            A, b = self.assembler.get_matrix_forms()
            A.assemble()
            b.ghost_update()

            x = iPETScVector(self.assembler.sol.x.petsc_vec)
            x.raw.set(0.0)
            self._ab_cache[cache_key] = (A, b, x)
        else:
            log_rank(
                logger,
                logging.DEBUG,
                "Using cached A and b for %s solve",
                ksp_type.name,
            )
            A, b, x = self._ab_cache[cache_key]
            x.raw.set(0.0)

        # For direct LU, force rebuild of the KSP so it sees the nullspace
        if ksp_type is KSPType.PREONLY:
            self._ksp_cache.pop(cache_key, None)

        # Configure or retrieve KSP solver
        if cache_key not in self._ksp_cache:
            log_rank(logger, logging.DEBUG, "Configuring iKSP for %s", ksp_type.name)
            solver = iKSP(A)
            solver.set_type(ksp_type)
            solver.set_preconditioner(pc_type)

            # Only set tolerances for iterative methods
            if ksp_type is not KSPType.PREONLY:
                solver.set_tolerances(tol=tol, rtol=rtol, max_it=max_it)

            if restart is not None and ksp_type is KSPType.GMRES:
                solver.raw.setGMRESRestart(restart)

            if enable_monitor and ksp_type is not KSPType.PREONLY:
                self._res_hist[cache_key] = []  # Reset history

                def _monitor(ksp, its, rnorm):
                    # Note: PETSc reports 'preconditioned residual norm' by default
                    self._res_hist[cache_key].append(float(rnorm))

                solver.raw.setMonitor(_monitor)

            solver.set_from_options(prefix=f"{cache_key}_")
            self._ksp_cache[cache_key] = solver
        else:
            log_rank(
                logger, logging.DEBUG, "Using cached iKSP solver for %s", ksp_type.name
            )
            solver = self._ksp_cache[cache_key]

            if enable_monitor and ksp_type is not KSPType.PREONLY:
                # Ensure history reset when reusing the solver
                self._res_hist[cache_key] = []

                def _monitor(ksp, its, rnorm):
                    self._res_hist[cache_key].append(float(rnorm))

                solver.raw.setMonitor(_monitor)

        # Solve and time
        t0 = MPI.Wtime()
        solver.solve(b, x)
        t1 = MPI.Wtime()
        log_global(
            logger, logging.INFO, "%s solve time: %.3f s", ksp_type.name, t1 - t0
        )

        # Scatter and optionally plot
        self.assembler.sol.x.scatter_forward()
        if show_plot and A.comm.rank == 0:
            plot_mixed_function(self.assembler.sol, scale=plot_scale)
            A.comm.barrier()

        log_global(logger, logging.INFO, f"{ksp_type.name} solve completed.")
        return self.assembler.sol

    def get_residual_history(self, *, key: int | str | None = None) -> list[float]:
        """Return the stored residual history for the last solve with this key."""
        cache_key = key or next(reversed(self._res_hist), "")
        return list(self._res_hist.get(cache_key, []))

    def plot_residuals(
        self,
        *,
        key: int | str | None = None,
        output_path: Path = Path(".") / "ksp_residuals.png",
        title: str | None = None,
    ) -> None:
        """Plot KSP residual history (semi-log) and save to disk."""
        cache_key = key or next(reversed(self._res_hist), None)
        if cache_key is None or cache_key not in self._res_hist:
            log_global(logger, logging.WARNING, "No residual history to plot.")
            return

        history = self._res_hist[cache_key]
        if not history:
            log_global(logger, logging.WARNING, "Residual history is empty.")
            return

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.semilogy(history, label="KSP residuals", color="k", linewidth=1.5)
        ax.set_xlabel(r"Iteration, $k$ (-)")
        ax.set_ylabel(r"$\|\mathbf{r}_k\|_{L2}$ (-)")
        if title:
            ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(which="major", linestyle="-", linewidth=0.8)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", color="gray", linewidth=0.5)
        ax.tick_params(which="major", direction="in", length=3, width=0.5)
        ax.tick_params(which="minor", direction="in", length=1.25, width=0.5)
        fig.tight_layout()
        fig.savefig(output_path, dpi=330)
        plt.close(fig)

        log_global(logger, logging.INFO, "KSP residual plot saved to %s", output_path)


# NOTE: More solver interfaces (BiCGSTAB, etc.) can be added as needed
