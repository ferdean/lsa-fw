"""LSA-FW Eigensolver 2.

This module re-implements a solver for eigenvalue problems completely decoupled from SLEPc.
Special interest is paid to the generalized eigenvalue problem Ax = λMx.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, LinearOperator
from petsc4py import PETSc

from lib.loggingutils import log_global
from FEM.utils import Scalar, iPETScMatrix
from Solver.utils import iEpsWhich

logger = logging.getLogger(__name__)


def _complexify(A: sp.spmatrix, *, should_be_complex: bool) -> sp.spmatrix:
    if should_be_complex and A.dtype.kind != "c":
        return A.astype(np.complex128, copy=False)
    return A


def _sort_indices(lam: np.ndarray, which: str) -> np.ndarray:
    match which:
        case "LR":
            key = lam.real
        case "LI":
            key = lam.imag
        case "SR":
            key = -lam.real
        case "SI":
            key = -lam.imag
        case "LM_abs":
            key = np.abs(lam)
        case _:
            raise ValueError(f"Unknown which_sort = {which!r}")
    return np.argsort(-key)


def _compute_residuals(
    A: sp.spmatrix, M: sp.spmatrix, lam: np.ndarray, V: np.ndarray
) -> np.ndarray:
    Av = A @ V
    Mv = M @ V
    R = Av - Mv * lam[np.newaxis, :]
    num = np.linalg.norm(R, axis=0)
    den = np.linalg.norm(Av, axis=0) + np.abs(lam) * np.linalg.norm(Mv, axis=0) + 1e-16
    return num / den


@dataclass
class ShiftInvertConfig:
    """Shift-invert eigensolver configuration."""

    sigma: complex = 0.0
    k: int = 20
    tol: float = 1e-6
    maxiter: int = 500
    ncv: int | None = None
    which_sort: iEpsWhich = iEpsWhich.LARGEST_REAL


class ArpackEigenSolver:
    """Solver for Ax = λMx based on ARPACK."""

    def __init__(
        self,
        cfg: ShiftInvertConfig,
        A: iPETScMatrix,
        M: iPETScMatrix,
        *,
        dofs_u: np.ndarray,
        dofs_p: np.ndarray,
    ) -> None:
        nrows, ncols = A.shape
        mrows, mcols = M.shape
        if nrows != ncols or mrows != mcols or (mrows, mcols) != (nrows, ncols):
            raise ValueError(
                "Operators must be square and have the same shape. "
                f"Got A shape {A.shape}; and M shape {M.shape}"
            )
        is_complex = (cfg.sigma != 0) and (
            np.iscomplex(cfg.sigma) or np.iscomplexobj(cfg.sigma)
        )

        self._A = _complexify(A.as_scipy_array(), should_be_complex=is_complex)
        self._M = _complexify(M.as_scipy_array(), should_be_complex=is_complex)
        self._cfg = cfg
        self._n = nrows

        self._petsc_A = A.raw
        self._petsc_M = M.raw
        self._comm = self._petsc_A.getComm()

        # Guard against using a complex shift with real PETSc matrices
        if is_complex and self._petsc_A.getValuesCSR()[2].dtype.kind != "c":
            raise ValueError(
                "Complex sigma requires complex PETSc matrices for shift-invert."
            )

        # Build shifted operator C = A - sigma * M
        C = self._petsc_A.duplicate()
        C.axpy(-Scalar(cfg.sigma), self._petsc_M)

        # Inherit and keep a handle to the nullspace (constant-pressure at least)
        ns = self._petsc_A.getNullSpace()
        if ns is not None:
            C.setNullSpace(ns)
        self._ns = C.getNullSpace()

        # KSP: robust direct solve for the shift-invert apply
        self._C = C
        self._ksp = PETSc.KSP().create(comm=self._comm)
        self._ksp.setOperators(self._C)
        self._ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)

        # Prefer a robust LU backend and set safe defaults
        try:
            pc.setFactorSolverType("mumps")
            # Scoped options via prefix
            prefix = "si_"
            self._ksp.setOptionsPrefix(prefix)
            opts = PETSc.Options()
            # Help MUMPS with near-singular pivots
            opts[f"{prefix}mat_mumps_icntl_24"] = 1  # null-pivot detection
            opts[f"{prefix}mat_mumps_cntl_3"] = 1e-12  # pivot perturbation
        except Exception:
            # Fall back to SuperLU with equilibration and symmetric column permutations
            try:
                pc.setFactorSolverType("superlu")
                prefix = "si_"
                self._ksp.setOptionsPrefix(prefix)
                opts = PETSc.Options()
                opts[f"{prefix}mat_superlu_equil"] = 1
                opts[f"{prefix}mat_superlu_colperm"] = "MMD_AT_PLUS_A"
            except Exception:
                # Keep PETSc default if neither is available
                pass

        self._ksp.setFromOptions()
        self._ksp.setUp()

        # Work vectors
        self._x = self._petsc_A.createVecRight()
        self._b = self._petsc_A.createVecRight()
        self._y = self._petsc_A.createVecRight()

        # Always drive ARPACK's complex path (handles complex pairs cleanly)
        self._lop_dtype = np.complex128

        self._dofs_u = np.asarray(dofs_u, dtype=np.int32)
        self._dofs_p = np.asarray(dofs_p, dtype=np.int32)

        def _apply_real_vec(x_real: np.ndarray) -> np.ndarray:
            # Project input into velocity subspace
            x_real = x_real.copy()
            x_real[self._dofs_p] = 0.0
            with self._x.localForm() as xl:
                xl.array[:] = x_real

            if self._ns is not None:
                self._ns.remove(self._x)  # harmless after zeroing p, keeps things clean

            self._petsc_M.mult(self._x, self._b)
            if self._ns is not None:
                self._ns.remove(self._b)  # make RHS compatible with C

            self._ksp.solve(self._b, self._y)
            reason = self._ksp.getConvergedReason()
            if reason <= 0:
                raise RuntimeError(f"KSP failed in shift-invert apply: reason={reason}")

            y_arr = self._y.getArray(readonly=True).copy()
            # Project output back to velocity subspace (keep Krylov invariant)
            y_arr[self._dofs_p] = 0.0
            if not np.isfinite(y_arr).all():
                raise RuntimeError(
                    "KSP produced non-finite values; adjust sigma/nullspace."
                )
            return y_arr

        def _op_mv(x: np.ndarray) -> np.ndarray:
            if np.iscomplexobj(x):
                yr = _apply_real_vec(x.real)
                yi = _apply_real_vec(x.imag)
                out = yr + 1j * yi
            else:
                out = _apply_real_vec(x)
            # Safety: enforce subspace
            out[self._dofs_p] = 0.0
            return out

        self._linear_operator = LinearOperator(
            (self._n, self._n), matvec=_op_mv, dtype=self._lop_dtype
        )

        log_global(logger, logging.INFO, "Initialized eigensolver (ARPACK).")

    @staticmethod
    def _mu_to_lambda(mu: np.ndarray, sigma: complex) -> np.ndarray:
        return sigma + 1.0 / mu

    def solve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve EVP."""
        log_global(
            logger,
            logging.INFO,
            f"Started eigenvalue solve around {self._cfg.sigma}: "
            f"nev={self._cfg.k}, tol={self._cfg.tol}, max_it={self._cfg.maxiter}",
        )

        t0 = time.time()

        # Choose a roomy Krylov subspace if not provided
        ncv = self._cfg.ncv if self._cfg.ncv is not None else max(4 * self._cfg.k, 40)

        mu, W = eigs(
            self._linear_operator,
            k=self._cfg.k,
            which="LM",
            tol=self._cfg.tol,
            maxiter=self._cfg.maxiter,
            ncv=ncv,
        )

        elapsed = time.time() - t0
        log_global(logger, logging.INFO, f"Solve completed in {elapsed:.2f} s.")

        lam = self._mu_to_lambda(mu, self._cfg.sigma)
        idx = _sort_indices(lam, self._cfg.which_sort.to_arpack())
        lam = lam[idx]
        V = W[:, idx]

        res = _compute_residuals(self._A, self._M, lam, V)
        med_res = float(np.median(res))
        max_res = float(np.max(res))

        log_global(
            logger,
            logging.DEBUG,
            "Computed residuals: median=%.2e, max=%.2e, >1e-6=%d, >1e-4=%d.",
            med_res,
            max_res,
            int(np.sum(res > 1e-6)),
            int(np.sum(res > 1e-4)),
        )

        if (max_res > 1e-4) or (med_res > 1e-6):
            log_global(
                logger,
                logging.WARNING,
                "Poor eigenpair quality (adjust sigma and k/ncv).",
            )

        return lam, V, res
