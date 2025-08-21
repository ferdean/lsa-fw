"""LSA-FW Solver utilities."""

from __future__ import annotations

import logging
from enum import Enum, StrEnum, auto
from typing import Iterator, TypeAlias

import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

from FEM.utils import iComplexPETScVector, iPETScMatrix, iPETScVector

logger = logging.getLogger(__name__)

Scalar: TypeAlias = PETSc.ScalarType
"""Alias for the base numeric type used throughout the framework (float or complex).

Depending on how PETSc was configured, `PETSc.ScalarType` will be either
`float` (real builds) or `complex` (complex builds).
"""

_IS_COMPLEX_BUILD: bool = np.dtype(Scalar).kind == "c"


class iEpsProblemType(Enum):
    """Internal EPS problem type enumeration mapping to SLEPc.EPS.ProblemType."""

    HEP = SLEPc.EPS.ProblemType.HEP
    """Standard Hermitian eigenvalue problem. Ax = λx, with A Hermitian."""
    NHEP = SLEPc.EPS.ProblemType.NHEP
    """Standard non-Hermitian eigenvalue problem: Ax = λx, with A arbitrary."""
    GHEP = SLEPc.EPS.ProblemType.GHEP
    """Generalized Hermitian eigenvalue problem: Ax = λBx, with A, B Hermitian."""
    GNHEP = SLEPc.EPS.ProblemType.GNHEP
    """Generalized non-Hermitian eigenvalue problem: Ax = λBx, with arbitrary A, B."""
    PGNHEP = SLEPc.EPS.ProblemType.PGNHEP
    """Polynomial generalized non-Hermitian eigenvalue problem."""
    GHIEP = SLEPc.EPS.ProblemType.GHIEP
    """Generalized Hermitian indefinite eigenvalue problem."""

    def to_slepc(self) -> SLEPc.EPS.ProblemType:
        """Convert internal type to SLEPc.EPS.ProblemType."""
        return self.value

    @classmethod
    def from_slepc(cls, problem_type: SLEPc.EPS.ProblemType) -> iEpsProblemType:
        """Create internal type from a SLEPc.EPS.ProblemType."""
        try:
            return cls[problem_type.name]
        except KeyError:
            raise ValueError(f"Unsupported SLEPc EPS ProblemType: {problem_type}")

    @classmethod
    def from_string(cls, name: str) -> iEpsProblemType:
        """Create internal type from a string (case-insensitive)."""
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid problem type: {name}. Choose from {list(cls.__members__.keys())}."
            )


class PreconditionerType(StrEnum):
    """Preconditioner types available in PETSc for spectral-transform KSPs."""

    NONE = auto()
    """No preconditioning."""
    JACOBI = auto()
    """Jacobi (diagonal scaling)."""
    SOR = auto()
    """Successive over-relaxation."""
    ASM = auto()
    """Additive Schwarz Method."""
    ILU = auto()
    """Incomplete LU factorization."""
    ICC = auto()
    """Incomplete Cholesky factorization."""
    LU = auto()
    """Direct LU factorization."""
    CHOLESKY = auto()
    """Direct Cholesky factorization."""
    GAMG = auto()
    """Geometric-Algebraic Multigrid."""
    HYPRE = auto()
    """Hypre algebraic multigrid (requires Hypre support)."""
    REDUNDANT = auto()
    """Redundant coarse-grid preconditioner."""
    SHELL = auto()
    """User-defined shell preconditioner."""
    # Additional types can be added here, e.g., PILU, PYTHON, PETSC.


class KSPType(StrEnum):
    """KSP solver types."""

    CG = auto()
    """Conjugate gradient method (symmetric positive-definite)."""
    GMRES = auto()
    """Generalized minimal residual method (nonsymmetric)."""
    BICG = auto()
    """Biconjugate gradient method (nonsymmetric)."""
    BICGSTAB = auto()
    """Biconjugate gradient stabilized method."""
    RICHARDSON = auto()
    """Richardson (stationary) iteration."""
    CHEBYSHEV = auto()
    """Chebyshev semi-iterative method."""
    PREONLY = auto()
    """Use only pre-conditioner (used for direct solvers, like LU or Cholebsky)."""
    QCG = auto()
    """Quasi-minimal residual CG."""
    CGS = auto()
    """CG squared."""
    GCR = auto()
    """Generalized conjugate residual."""
    LSQR = auto()
    """Least squares QR method."""
    LGMRES = auto()
    """Loose GMRES (recycled Krylov space)."""
    FGMRES = auto()
    """Flexible GMRES (variable preconditioning)."""

    def to_petsc(self) -> str:
        """Convert internal type to a type accepted by PETSc."""
        return self.name.lower()


class iSTType(Enum):
    """Internal wrapper for the SLEPc.ST.Type spectral-transform types."""

    SHELL = SLEPc.ST.Type.SHELL
    """User-provided (shell) transform."""
    SHIFT = SLEPc.ST.Type.SHIFT
    """Simple shift: replaces A with A - sigma I, useful for targeting shifts without explicit inversion."""
    SINVERT = SLEPc.ST.Type.SINVERT
    """Shift-and-invert: uses (A - sigma I)^{-1}, improving convergence near sigma at the cost of a factorization"""
    CAYLEY = SLEPc.ST.Type.CAYLEY
    """Cayley transform: applies (A - sigma I)^{-1}(A + sigma I)."""
    PRECOND = SLEPc.ST.Type.PRECOND
    """Preconditioned shift-and-invert."""
    FILTER = SLEPc.ST.Type.FILTER
    """Polynomial filter."""

    def to_slepc(self) -> SLEPc.ST.Type:
        """Convert to the underlying SLEPc.ST.Type."""
        return self.value


class iEpsWhich(Enum):
    """Internal wrapper for SLEPc.EPS.Which enums."""

    ALL = SLEPc.EPS.Which.ALL
    LARGEST_MAGNITUDE = SLEPc.EPS.Which.LARGEST_MAGNITUDE
    SMALLEST_MAGNITUDE = SLEPc.EPS.Which.LARGEST_REAL
    LARGEST_REAL = SLEPc.EPS.Which.LARGEST_REAL
    SMALLEST_REAL = SLEPc.EPS.Which.SMALLEST_REAL
    LARGEST_IMAGINARY = SLEPc.EPS.Which.LARGEST_IMAGINARY
    SMALLEST_IMAGINARY = SLEPc.EPS.Which.SMALLEST_IMAGINARY
    TARGET_MAGNITUDE = SLEPc.EPS.Which.TARGET_MAGNITUDE
    TARGET_REAL = SLEPc.EPS.Which.TARGET_REAL
    TARGET_IMAGINARY = SLEPc.EPS.Which.TARGET_IMAGINARY
    USER = SLEPc.EPS.Which.USER

    def to_slepc(self) -> SLEPc.EPS.Which:
        """Convert to the corresponding SLEPc.EPS.Which value."""
        return self.value


class iEpsSolver:
    """Minimal wrapper around SLEPc EPS solver to provide a safe, consistent interface.

    Refer to the official PETSc documentation for more details: https://slepc.upv.es/documentation/.
    """

    def __init__(
        self,
        A: iPETScMatrix | None = None,
        M: iPETScMatrix | None = None,
        comm: PETSc.Comm = PETSc.COMM_WORLD,
    ) -> None:
        """Initialize SLEPc solver wrapper."""
        self._eps = SLEPc.EPS().create(comm=comm)
        if M is not None and A is None:
            raise ValueError(
                "Cannot set right-hand operator M without left-hand operator A."
            )
        if A is not None:
            self.set_operators(A, M)

    @property
    def raw(self) -> SLEPc.EPS:
        """Access the underlying EPS object."""
        return self._eps

    def set_operators(self, A: iPETScMatrix, M: iPETScMatrix | None = None) -> None:
        """Set the matrix operators for the generalized eigenproblem Ax + λMx = 0."""
        if M is not None:
            self._eps.setOperators(-A.raw, M.raw)
        else:
            self._eps.setOperators(-A.raw)

    def set_problem_type(self, problem_type: iEpsProblemType) -> None:
        """Set the eigenproblem type. Refer to iEpsProblemType enum."""
        self._eps.setProblemType(problem_type.to_slepc())

    def set_dimensions(
        self, number_eigenpairs: int, subspace_dimension: int | None = None
    ) -> None:
        """Set number of eigenpairs (nev) and subspace dimension (ncv)."""
        self._eps.setDimensions(
            number_eigenpairs,
            subspace_dimension if subspace_dimension is not None else PETSc.DECIDE,
        )

    def set_tolerances(self, atol: float, max_it: int) -> None:
        """Set convergence absolute tolerance and maximum iterations."""
        self._eps.setTolerances(tol=atol, max_it=max_it)

    def set_which_eigenpairs(self, which: iEpsWhich) -> None:
        """Select which eigenpairs to compute. Refer to iEpsWhich enum."""
        self._eps.setWhichEigenpairs(which.to_slepc())

    def set_target(self, sigma: float | complex) -> None:
        """Set spectral transformation shift (target)."""
        self._eps.setTarget(Scalar(sigma))

    def set_interval(self, a: float, b: float) -> None:
        """Compute eigenvalues in real interval [a, b]."""
        self._eps.setInterval(a, b)

    def set_interval_complex(self, a: float, b: float, c: float, d: float) -> None:
        """Compute eigenvalues in complex rectangle [a,b]x[c,d]."""
        self._eps.setIntervalComplex(a, b, c, d)

    def set_st_type(self, st_type: iSTType) -> None:
        """Set spectral transformation type. Refer to iSTType enum."""
        st = self._eps.getST()
        st.setType(st_type.to_slepc())

    def set_st_pc_type(self, pc_type: PreconditionerType) -> None:
        """Set the PETSc preconditioner type on the shift-and-invert ST."""
        st = self._eps.getST()
        ksp = st.getKSP()
        pc = ksp.getPC()
        pc.setType(pc_type.name.lower())

    def solve(self) -> None:
        """Run the EPS solver on configured operators and settings."""
        self._eps.solve()

    def get_num_converged(self) -> int:
        """Return number of converged eigenpairs."""
        return self._eps.getConverged()

    def get_eigenvalue(self, idx: int) -> float | complex:
        """Get the eigenvalue at index idx."""
        return self._eps.getEigenvalue(idx)

    def _get_eigenvector_real(self, idx: int) -> iComplexPETScVector:
        A, _ = self._eps.getOperators()
        vec_r = A.createVecRight()
        vec_i = A.createVecRight()
        self._eps.getEigenvector(idx, vec_r, vec_i)

        vec_i.assemble()
        if vec_i.norm() <= 1e-6:
            # Avoid unnecessary allocation of imaginary part, if vector is real
            return iComplexPETScVector(vec_r)

        return iComplexPETScVector(vec_r, vec_i)

    def _get_eigenvector_complex(self, idx: int) -> iComplexPETScVector:
        A, _ = self._eps.getOperators()
        vec = A.createVecRight()
        self._eps.getEigenvector(idx, vec)
        return iComplexPETScVector(vec)

    def get_eigenvector(self, idx: int) -> iComplexPETScVector:
        """Get the eigenvector at index idx.

        The format of the output eigenvector depends on the PETSc build:
          - In the real-scalar build, both parts (real and imaginary) are explicitly returned; so they have to be
          manipulated separately.
        - In the complex-scalar build, a single vector is used and already contains the complex data.

        In either case, the vectors are normalized (2-norm or B-norm).
        """
        if _IS_COMPLEX_BUILD:
            return self._get_eigenvector_complex(idx)
        else:
            return self._get_eigenvector_real(idx)

    def get_eigenpair(self, idx: int) -> tuple[float | complex, iComplexPETScVector]:
        """Get (eigenvalue, eigenvector) tuple at index idx."""
        return self.get_eigenvalue(idx), self.get_eigenvector(idx)

    def get_all_eigenpairs_up_to(
        self, num: int
    ) -> Iterator[tuple[float | complex, iComplexPETScVector]]:
        """Lazily yield up to `num` converged eigenpairs.

        This avoids allocating a full list if you process them one-by-one.
        """
        nconv = self.get_num_converged()
        limit = min(nconv, num)
        for i in range(limit):
            yield self.get_eigenpair(i)


class iKSP:
    """Wrapper around PETSc KSP (Krylov Space) linear solver, with a cleaner, type-safe API.

    Provides methods to configure solver type, tolerances, preconditioner,
    and to execute the solve and retrieve diagnostics.
    """

    def __init__(
        self,
        A: iPETScMatrix | None = None,
        comm: PETSc.Comm = PETSc.COMM_WORLD,
    ) -> None:
        """Initialize a KSP solver."""
        self._ksp = PETSc.KSP().create(comm=comm)
        if A is not None:
            self.set_operators(A)

    @property
    def raw(self) -> PETSc.KSP:
        """Access the underlying PETSc KSP object."""
        return self._ksp

    def set_operators(self, A: iPETScMatrix, P: iPETScMatrix | None = None) -> None:
        """Set the system matrix A and optional preconditioning matrix P.

        If P is provided, it will be used as the preconditioning operator.
        """
        if P is not None:
            self._ksp.setOperators(A.raw, P.raw)
        else:
            self._ksp.setOperators(A.raw)

    def set_type(self, ksp_type: KSPType) -> None:
        """Choose the iterative solver algorithm."""
        self._ksp.setType(ksp_type.to_petsc())

    def get_type(self) -> str:
        """Get the current PETSc KSP type string."""
        return self._ksp.getType()

    def set_tolerances(
        self,
        tol: float = 1e-12,
        max_it: int = 1000,
        rtol: float = 1e-8,
    ) -> None:
        """Set convergence criteria for the solver."""
        self._ksp.setTolerances(rtol=rtol, atol=tol, max_it=max_it)

    def set_preconditioner(self, pc_type: PreconditionerType) -> None:
        """Configure the preconditioner type."""
        pc = self._ksp.getPC()
        pc.setType(pc_type)

    def set_initial_guess_nonzero(self, flag: bool) -> None:
        """Indicate whether the initial guess vector is nonzero."""
        self._ksp.setInitialGuessNonzero(flag)

    def set_from_options(self, prefix: str | None = None) -> None:
        """Apply command-line options to this solver."""
        if prefix is not None:
            self._ksp.setOptionsPrefix(prefix)
        self._ksp.setFromOptions()

    def solve(self, b: iPETScVector, x: iPETScVector | None = None) -> None:
        """Solve the linear system Ax = b.

        If x is not provided, a new vector is created.
        """
        if x is None:
            x = b.duplicate()
        self._ksp.solve(b.raw, x.raw)

    def get_solution(self) -> iPETScVector:
        """Retrieve the solution vector from the last solve."""
        return iPETScVector(self._ksp.getSolution())

    def get_residual_norm(self) -> float:
        """Get the norm of the residual after solve."""
        return float(self._ksp.getResidualNorm())

    def get_iteration_number(self) -> int:
        """Return the number of iterations performed."""
        return self._ksp.getIterationNumber()

    def reset(self) -> None:
        """Reset internal solver state for fresh reuse."""
        self._ksp.reset()
