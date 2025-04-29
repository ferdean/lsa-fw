"""LSA-FW eigensolver."""

import logging
from dataclasses import dataclass

from slepc4py import SLEPc

from FEM.utils import iPETScMatrix, iPETScVector

from .utils import iEPSProblemType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EigensolverConfig:
    """Eigensolver configuration."""

    num_eig: int
    """Number of computed eigenpairs."""
    problem_type: iEPSProblemType
    """Problem type."""
    atol: float
    """Absolute tolerance."""
    max_it: int
    """Maximum number of iterations."""


class EigenSolver:
    """Solver for the generalized eigenvalue problem Ax = λMx, based on SLEPc."""

    def __init__(
        self, cfg: EigensolverConfig, A: iPETScMatrix, M: iPETScMatrix
    ) -> None:
        """Initialize eigensolver."""
        if cfg.problem_type is not iEPSProblemType.GNHEP:
            logger.warning("...")
        self._cfg = cfg

        # herm_types = {
        #     iEPSProblemType.HEP,
        #     iEPSProblemType.GHEP,
        #     iEPSProblemType.GHIEP,
        # }
        # if cfg.problem_type in herm_types:
        #     if not A.is_hermitian(herm_tol):
        #         logger.warning(
        #             f"Problem type '{cfg.problem_type.name}' assumes Hermitian A,"
        #             " but A.is_hermitian({herm_tol}) returned False."
        #         )
        #     if cfg.problem_type in {iEPSProblemType.GHEP, iEPSProblemType.GHIEP} and B is not None:
        #         if not B.is_hermitian(herm_tol):
        #             logger.warning(
        #                 f"Problem type '{cfg.problem_type.name}' assumes Hermitian B,"
        #                 " but B.is_hermitian({herm_tol}) returned False."
        #             )
