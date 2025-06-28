"""LSA-FW Eigensolver.

Implements a solver for eigenvalue problems based on SLEPc. Special interest is paid to the generalized eigenvalue
problem Ax = λMx.

Example usage (Shift and invert with pre-conditioning):
    ```python
    cfg = EigensolverConfig(
        num_eig=6,
        problem_type=iEpsProblemType.GHEP,
        atol=1e-8,
        max_it=500
    )
    es = EigenSolver(cfg, A, M)

    # Target interior eigenvalues near sigma=0.5
    es.solver.set_st_type(iSTType.SINVERT)
    es.solver.set_target(0.5)

    # Attach an incomplete Cholesky preconditioner to the spectral transform
    es.solver.set_st_pc_type(PreconditionerType.ICC)
    eigenpairs = es.solve()
    ```

Note that you can combine shift-invert, Cayley, polynomial filters, and other spectral transforms on the same
iEpsSolver object.
"""

import logging
import time
from dataclasses import dataclass

from FEM.utils import iPETScMatrix, iComplexPETScVector
from lib.loggingutils import log_global

from .utils import iEpsProblemType, iEpsSolver

logger = logging.getLogger(__name__)


_HERMITIAN_TYPES: set[iEpsProblemType] = {
    iEpsProblemType.HEP,
    iEpsProblemType.GHEP,
    iEpsProblemType.GHIEP,
}


@dataclass(frozen=True)
class EigensolverConfig:
    """Eigensolver configuration."""

    num_eig: int
    """Number of computed eigenpairs."""
    problem_type: iEpsProblemType
    """Problem type."""
    atol: float
    """Absolute tolerance."""
    max_it: int
    """Maximum number of iterations."""


class EigenSolver:
    """Solver for the generalized eigenvalue problem Ax = λMx, based on SLEPc."""

    def __init__(
        self, cfg: EigensolverConfig, A: iPETScMatrix, M: iPETScMatrix | None = None
    ) -> None:
        """Initialize eigensolver."""
        nrows, ncols = A.shape
        if nrows != ncols:
            raise ValueError(f"Operator A must be square, got shape ({nrows}, {ncols})")

        if M is not None:
            mrows, mcols = M.shape
            if (mrows, mcols) != (nrows, ncols):
                raise ValueError(
                    f"Operator M shape {M.shape} does not match A's shape {A.shape}"
                )
        if cfg.problem_type in _HERMITIAN_TYPES:
            if not A.is_numerically_hermitian():
                log_global(
                    logger,
                    logging.WARNING,
                    f"Problem type '{cfg.problem_type.name}' assumes Hermitian A," " but A is not (numerically) Hermitian.",
                )
            if (
                M is not None
                and cfg.problem_type in {iEpsProblemType.GHEP, iEpsProblemType.GHIEP}
                and not M.is_numerically_hermitian()
            ):

                log_global(
                    logger,
                    logging.WARNING,
                    f"Problem type '{cfg.problem_type.name}' assumes Hermitian M," " but M is not (numerically) Hermitian.",
                )

        self._cfg = cfg
        self._solver = iEpsSolver(A, M)

        self._solver.set_problem_type(cfg.problem_type)
        self._solver.set_dimensions(cfg.num_eig)
        self._solver.set_tolerances(cfg.atol, cfg.max_it)

    @property
    def solver(self) -> iEpsSolver:
        """Get the solver object."""
        return self._solver

    @property
    def config(self) -> EigensolverConfig:
        """Get the solver configuration."""
        return self._cfg

    def solve(self) -> list[tuple[float | complex, iComplexPETScVector]]:
        """Run the solver and return eigenpairs."""
        log_global(
            logger,
            logging.INFO,
            f"Starting eigenvalue solve: type={self._cfg.problem_type.name}, "
            f"nev={self._cfg.num_eig}, "
            f"tol={self._cfg.atol}, max_it={self._cfg.max_it}",
        )

        t0 = time.time()
        self._solver.solve()
        elapsed = time.time() - t0

        nconv = self._solver.get_num_converged()
        try:
            # Attempt to retrieve number of iterations from PETSc if exposed
            its = self._solver.raw.getST().getKSP().getIterationNumber()
        except Exception:
            its = None

        log_global(
            logger,
            logging.INFO,
            f"Solve completed in {elapsed:.2f} s; converged {nconv} eigenpairs"
            + (f"; iterations={its}" if its is not None else ""),
        )

        pairs = list(self._solver.get_all_eigenpairs_up_to(self._cfg.num_eig))
        log_global(logger, logging.INFO, f"Retrieved {len(pairs)} eigenpairs")
        return pairs
