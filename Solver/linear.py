"""LSA-FW Linear Solver.

Provides a linear solver interface that can be extended for different linear solvers.
"""

import logging
import scipy.sparse.linalg

import dolfinx.fem as dfem

from FEM.operators import BaseAssembler
from FEM.plot import plot_mixed_function

logger = logging.getLogger(__name__)


class LinearSolver:
    """Container for different linear-solving algorithms."""

    def __init__(self, assembler: BaseAssembler) -> None:
        """Initialize."""
        self.assembler = assembler
        self._ab_cache: dict[str, tuple] = {}
        self._lu_cache: dict[str, scipy.sparse.linalg.SuperLU] = {}

    def direct_lu_solve(
        self, *, show_plot: bool = True, plot_scale: float = 0.1
    ) -> dfem.Function:
        """Direct solve using sparse LU (splu)."""
        logger.info("Direct LU solve started.")
        key = "direct_lu"
        if key not in self._ab_cache:
            logger.debug("Assembling system matrix and RHS for direct LU")
            A, b = self.assembler.get_matrix_forms()
            self._ab_cache[key] = (A, b)
        else:
            logger.debug("Using cached A and b for direct LU")
            A, b = self._ab_cache[key]

        if key not in self._lu_cache:
            logger.debug("Factoring (LU) system matrix.")
            self._lu_cache[key] = scipy.sparse.linalg.splu(A.as_scipy_array().tocsc())
        else:
            logger.debug("Using cached LU factorization")
        lu = self._lu_cache[key]

        try:
            x = lu.solve(b.raw.array)
        except Exception as e:
            raise RuntimeError("LU solve failed") from e

        wh = self.assembler.sol
        wh.x.array[:] = x
        wh.x.scatter_forward()

        if show_plot:
            plot_mixed_function(wh, plot_scale)

        logger.info("Direct LU solve completed.")
        return wh


# TODO: Extend with iterative solvers like GMRES, CG, etc.
# TODO: Integrate PETSc built-in solvers for better performance
