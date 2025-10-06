"""LSA-FW Adjoint-Based Sensitivity module.

This module provides routines to compute eigenvalue sensitivity using adjoint (left) eigenvectors.
It reuses existing FEM assemblers and solvers to compute:
 - The direct eigenpair `(sigma, v)` for a given baseflow and parameter.
 - The adjoint eigenvector `a` corresponding to that eigenpair.
 - The baseflow sensitivity `(u_mu, p_mu)` by solving the steady flow Jacobian linear system.
 - The eigenvalue sensitivity `d sigma / d mu` via the adjoint formula.

Note that, currently, the implementation is limited to `mu = Re`.
"""

import logging

import numpy as np
import ufl
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc
from mpi4py import MPI
from ufl import TestFunctions, conj, dot, grad, inner, dx, nabla_grad

from FEM.bcs import BoundaryConditions
from FEM.operators import (
    LinearizedNavierStokesAssembler,
    StationaryNavierStokesAssembler,
)
from FEM.spaces import FunctionSpaces
from FEM.utils import iComplexPETScVector, iPETScMatrix, iPETScVector, Scalar
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.linear import LinearSolver
from Solver.utils import (
    KSPType,
    PreconditionerType,
    iEpsProblemType,
    iEpsWhich,
    iSTType,
)
from lib.loggingutils import log_global

logger = logging.getLogger(__name__)

_IS_COMPLEX_BUILD: bool = np.dtype(Scalar).kind == "c"


def _hermitian(A: iPETScMatrix) -> iPETScMatrix:
    raw = A.raw.copy()
    raw.transpose()

    # Implementation note: The `iPETScMatrix` wrapper provides functionality to get the Hermitian
    # transpose of a matrix. However, this caused issues in PETSc (matrix type MATTRANSPOSEVIRTUAL),
    # so we explicitly form the transpose here.

    if _IS_COMPLEX_BUILD:
        raw.conjugate()
    return iPETScMatrix(raw)


class EigenSensitivitySolver:
    """Solver for eigenvalue sensitivity using adjoint mode (for parameter Re).

    This class computes the sensitivity of a stability eigenvalue (sigma) to the Reynolds number.
    It reuses assembled matrices across the direct and adjoint solves to avoid redundant computations.
    """

    def __init__(
        self,
        spaces: FunctionSpaces,
        bcs: BoundaryConditions,
        baseflow: dfem.Function,
        re: float,
        *,
        A: iPETScMatrix | None = None,
        M: iPETScMatrix | None = None,
        tags: dmesh.MeshTags | None = None,
        target: float | complex | None = None,
        tol_direct: float = 1e-6,
        tol_adjoint: float = 1e-3,
        tol_baseflow: float = 1e-6,
        max_it: int = 500,
        max_modes: int = 5,
        bc_energy_tol: float = 0.98,
    ) -> None:
        """Initialize the sensitivity solver for a given baseflow and Reynolds number."""
        self._spaces = spaces
        self._bcs = bcs
        self._baseflow = baseflow
        self._re = re
        self._A = A
        self._M = M
        self._tags = tags
        self._target = target
        self._tol_direct = tol_direct
        self._tol_adjoint = tol_adjoint
        self._tol_baseflow = tol_baseflow
        self._max_it = max_it
        self._max_modes = max_modes
        self._bc_energy_tol = bc_energy_tol

        # Keep assembler only for BC bookkeeping (Dirichlet DOFs) and potential future reuse.
        self._lin_assembler = LinearizedNavierStokesAssembler(
            baseflow, spaces, re, bcs=bcs, tags=tags
        )
        self._ns_assembler: StationaryNavierStokesAssembler | None = None
        self._sigma: complex | None = None  # Selected eigenvalue (direct)
        self._v_func: dfem.Function | None = None  # Direct eigenfunction (mixed space)
        self._a_func: dfem.Function | None = None  # Adjoint eigenfunction (mixed space)
        self._baseflow_sensitivity: dfem.Function | None = None

        log_global(
            logger,
            logging.INFO,
            "Initialized eigenvalue sensitivity solver for Re=%.2f",
            re,
        )

    @staticmethod
    def _fraction_energy_on_indices(
        x: PETSc.Vec | np.ndarray, idx: np.ndarray
    ) -> float:
        if idx.size == 0:
            return 0.0
        arr = x.getArray(readonly=True) if isinstance(x, PETSc.Vec) else np.asarray(x)
        total = float(np.vdot(arr, arr).real)
        if total == 0.0:
            return 0.0
        return float(np.vdot(arr[idx], arr[idx]).real) / total

    @staticmethod
    def _project_to(
        expr: ufl.Form, function_space: dfem.FunctionSpace
    ) -> dfem.Function:
        """Generic L2 projection of a UFL expression onto function_space."""
        V = function_space
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = inner(v, u) * dx
        L = inner(v, expr) * dx

        A = fem_petsc.assemble_matrix(dfem.form(a))
        A.assemble()
        b = fem_petsc.assemble_vector(dfem.form(L))

        ksp = PETSc.KSP().create(A.getComm())
        ksp.setOperators(A)
        ksp.setType("cg")
        pc = ksp.getPC()
        try:
            pc.setType("hypre")
        except Exception:
            pc.setType("jacobi")
        ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=2000)

        u_fun = dfem.Function(V)
        x = PETSc.Vec().createWithArray(u_fun.x.array, comm=A.getComm())
        ksp.solve(b, x)
        u_fun.x.array[:] = x.getArray(readonly=True)
        u_fun.x.scatter_forward()
        return u_fun

    def _collect_dirichlet_dofs(self) -> np.ndarray:
        dofs: list[int] = []
        for bc in self._lin_assembler.bcs:
            if (idx := bc.dof_indices()) is None:
                continue
            # Flatten nested structures (bc indices may be list/tuple of arrays)
            if isinstance(idx, (list, tuple)):
                for part in idx:
                    arr = np.asarray(part, dtype=np.int64).ravel()
                    if arr.size:
                        dofs.extend(int(i) for i in arr)
            else:
                arr = np.asarray(idx, dtype=np.int64).ravel()
                if arr.size:
                    dofs.extend(int(i) for i in arr)
        # Include any internally pinned pressure DOF(s) (from steady NS assembler)
        if (pinned := self._lin_assembler._dofs_p) is not None:
            arr = np.asarray(pinned, dtype=np.int64).ravel()
            if arr.size:
                dofs.extend(int(i) for i in arr)
        return (
            np.unique(np.array(dofs, dtype=np.int64))
            if dofs
            else np.empty((0,), dtype=np.int64)
        )

    def _ensure_matrices(self) -> tuple[iPETScMatrix, iPETScMatrix]:
        if self._A is None or self._M is None:
            raise RuntimeError(
                "Matrices (A, M) must be provided for sensitivity solve in complex build. "
                "Assemble them upstream (in real) and pass them here."
            )
        return self._A, self._M

    def solve_direct_mode(
        self, target: float | complex | None = None
    ) -> tuple[complex, dfem.Function]:
        """Solve the direct eigenproblem around the baseflow at Re.

        Computes the right eigenpair (sigma, v) for the linearized NS operator, filtering out spurious
        Dirichlet modes.
        """
        if target is None:
            target = self._target

        A, M = self._ensure_matrices()

        cfg = EigensolverConfig(
            num_eig=self._max_modes,
            problem_type=iEpsProblemType.GNHEP,
            atol=self._tol_direct,
            max_it=self._max_it,
        )
        es = EigenSolver(A, M, cfg, check_hermitian=False)

        if target is not None:
            es.solver.set_st_type(iSTType.SINVERT)
            es.solver.set_target(target)
            es.solver.set_st_pc_type(PreconditionerType.LU)
        else:
            es.solver.set_which_eigenpairs(iEpsWhich.LARGEST_REAL)

        log_global(
            logger,
            logging.INFO,
            "Solving direct eigenproblem (target=%s, tol=%.1e, max_it=%d, max_modes=%d).",
            target,
            self._tol_direct,
            self._max_it,
            self._max_modes,
        )

        if not (pairs := es.solve()):
            raise RuntimeError("No eigenpairs returned by the eigensolver.")

        # Filter out spurious eigenpairs caused by strong Dirichlet BCs
        constrained = self._collect_dirichlet_dofs()
        filtered: list[tuple[complex, iComplexPETScVector]] = []
        for lam, vec in pairs:
            # Build a complex view for energy localization check
            if vec.imag is None:
                v_view: PETSc.Vec | np.ndarray = vec.real.raw
            else:
                r = vec.real.raw.getArray()
                im = vec.imag.raw.getArray()
                v_view = r.astype(complex, copy=False) + 1j * im
            if (
                self._fraction_energy_on_indices(v_view, constrained)
                >= self._bc_energy_tol
            ):
                continue
            filtered.append((lam, vec))
        if not filtered:
            raise RuntimeError(
                "All computed eigenpairs were filtered as spurious (Dirichlet identity modes). "
                "Consider assembling reduced operators on free DOFs or relax 'bc_energy_tol'."
            )

        # Select the eigenpair: nearest to target if specified, otherwise largest real part
        if target is not None:
            sigma, eigvec = min(filtered, key=lambda p: abs(p[0] - target))
        else:
            sigma, eigvec = max(filtered, key=lambda p: p[0].real)
        self._sigma = sigma

        # Convert eigenvector to a dolfinx Function
        v_func = dfem.Function(self._spaces.mixed)
        if eigvec.imag is None:
            v_values = eigvec.real.raw.getArray(readonly=True)
        else:
            r = eigvec.real.raw.getArray(readonly=True)
            im = eigvec.imag.raw.getArray(readonly=True)
            v_values = r.astype(complex, copy=False) + 1j * im
        v_func.x.array[:] = v_values
        v_func.x.scatter_forward()
        self._v_func = v_func
        num_spurious = len(pairs) - len(filtered)
        log_global(
            logger,
            logging.INFO,
            "Direct eigenpair: sigma = %.4e %s %.4e j",
            sigma.real,
            "+" if sigma.imag >= 0 else "-",
            abs(sigma.imag),
        )
        if num_spurious > 0:
            log_global(
                logger,
                logging.DEBUG,
                "Filtered %d spurious mode(s) with bc_energy_tol=%.2f.",
                num_spurious,
                self._bc_energy_tol,
            )
        return sigma, v_func

    def solve_adjoint_mode(
        self, sigma: complex | None = None, v_func: dfem.Function | None = None
    ) -> dfem.Function:
        """Solve the adjoint eigenproblem (left eigenvector) for the previously found sigma, v.

        Computes the left eigenvector `a` (adjoint mode) associated with the conjugate eigenvalue sigma*.
        The eigenpair is filtered for spurious modes and normalized such that a^H B v = 1.
        """
        if sigma is None or v_func is None:
            sigma = self._sigma
            v_func = self._v_func
        if sigma is None or v_func is None:
            raise RuntimeError(
                "Direct eigenpair must be computed before adjoint solve."
            )

        A, M = self._ensure_matrices()
        A_H = _hermitian(A)
        M_H = _hermitian(M)

        constrained = self._collect_dirichlet_dofs()
        cfg = EigensolverConfig(
            num_eig=self._max_modes,
            problem_type=iEpsProblemType.GNHEP,
            atol=self._tol_adjoint,
            max_it=self._max_it,
        )
        es_adj = EigenSolver(cfg, A=A_H, M=M_H, check_hermitian=False)

        # Shift-invert around sigma* to converge to the corresponding left eigenvector.
        es_adj.solver.set_st_type(iSTType.SINVERT)
        es_adj.solver.set_st_pc_type(PreconditionerType.LU)
        es_adj.solver.set_target(sigma.conjugate())
        es_adj.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)

        log_global(
            logger,
            logging.INFO,
            "Solving adjoint eigenproblem (target sigma = %.4e%s%.4ei, tol=%.1e).",
            sigma.real,
            "-" if sigma.imag >= 0 else "+",
            abs(sigma.imag),
            self._tol_adjoint,
        )

        pairs = es_adj.solve()
        if not pairs:
            raise RuntimeError("No eigenpairs returned by the adjoint eigensolver.")

        # Filter spurious adjoint eigenpairs (Dirichlet modes)
        filtered: list[tuple[complex, iComplexPETScVector]] = []
        for lam, avec in pairs:
            if avec.imag is None:
                a_view: PETSc.Vec | np.ndarray = avec.real.raw
            else:
                r = avec.real.raw.getArray()
                im = avec.imag.raw.getArray()
                a_view = r.astype(complex, copy=False) + 1j * im
            if (
                self._fraction_energy_on_indices(a_view, constrained)
                >= self._bc_energy_tol
            ):
                continue
            filtered.append((lam, avec))
        if not filtered:
            raise RuntimeError(
                "All adjoint eigenpairs were filtered as spurious (Dirichlet identity modes). "
                "Consider assembling reduced operators on free DOFs or relax 'bc_energy_tol'."
            )

        target_star = sigma.conjugate()
        sigma_adj, a_vec = min(filtered, key=lambda p: abs(p[0] - target_star))

        # Normalize the adjoint eigenvector such that a^H B v = 1
        v_petsc = PETSc.Vec().createWithArray(v_func.x.array, comm=MPI.COMM_WORLD)
        Mv = M.raw.createVecRight()
        M.raw.mult(v_petsc, Mv)
        prod = a_vec.dot(iPETScVector(Mv))  # conjugating dot
        if prod == 0:
            raise RuntimeError("Bi-orthonormal normalization failed (a^H B v = 0).")
        a_vec.scale(1.0 / prod)

        # Convert adjoint eigenvector to Function
        a_func = dfem.Function(self._spaces.mixed)
        if a_vec.imag is None:
            a_values = a_vec.real.raw.getArray(readonly=True)
        else:
            r = a_vec.real.raw.getArray(readonly=True)
            im = a_vec.imag.raw.getArray(readonly=True)
            a_values = r.astype(complex, copy=False) + 1j * im
        a_func.x.array[:] = a_values
        a_func.x.scatter_forward()
        self._a_func = a_func

        num_spurious = len(pairs) - len(filtered)
        log_global(
            logger,
            logging.INFO,
            "Adjoint eigenpair computed (sigma* = %.4e %s %.4e j).",
            sigma_adj.real,
            "+" if sigma_adj.imag >= 0 else "-",
            abs(sigma_adj.imag),
        )
        if num_spurious > 0:
            log_global(
                logger,
                logging.INFO,
                "Filtered %d spurious adjoint mode(s) with bc_energy_tol=%.2f.",
                num_spurious,
                self._bc_energy_tol,
            )
        return a_func

    def compute_baseflow_sensitivity(self, tol: float | None = None) -> dfem.Function:
        """Solve for the baseflow sensitivity w.r.t. Re (implicit part of eigen-sensitivity)."""
        tol_lin = tol if tol is not None else self._tol_baseflow
        if self._ns_assembler is None:
            self._ns_assembler = StationaryNavierStokesAssembler(
                self._spaces, re=self._re, bcs=self._bcs, initial_guess=self._baseflow
            )
        jacobian_matrix, _ = self._ns_assembler.get_matrix_forms()

        u_base = self._baseflow.sub(0)
        v_test, _ = TestFunctions(self._spaces.mixed)
        rhs_form = dfem.form(
            -(1.0 / self._re**2) * inner(grad(u_base), grad(v_test)) * dx
        )
        rhs_vec = fem_petsc.assemble_vector(rhs_form)
        fem_petsc.apply_lifting(
            rhs_vec, [self._ns_assembler.jacobian], [list(self._ns_assembler.bcs)]
        )
        fem_petsc.set_bc(rhs_vec, list(self._ns_assembler.bcs))
        pinned = getattr(self._ns_assembler, "_dofs_p", None)
        if pinned is not None:
            for didx in np.asarray(pinned, dtype=np.int64).ravel():
                rhs_vec.setValue(int(didx), 0.0)

        rhs_vec.assemblyBegin()
        rhs_vec.assemblyEnd()
        log_global(
            logger,
            logging.INFO,
            "Solving baseflow sensitivity linear system (steady Jacobian solve).",
        )
        sens_vec = LinearSolver.solve(
            jacobian_matrix, iPETScVector(rhs_vec), ksp_type=KSPType.GMRES, tol=tol_lin
        )
        sensitivity_fn = dfem.Function(self._spaces.mixed)
        sensitivity_fn.x.array[:] = sens_vec.as_array()
        sensitivity_fn.x.scatter_forward()
        self._baseflow_sensitivity = sensitivity_fn

        return sensitivity_fn

    def evaluate_sensitivity(
        self,
        re: float | None = None,
        v_func: dfem.Function | None = None,
        a_func: dfem.Function | None = None,
        baseflow_sens: dfem.Function | None = None,
    ) -> complex:
        """Evaluate the eigenvalue sensitivity."""
        re_val = re if re is not None else self._re
        v = v_func or self._v_func
        a = a_func or self._a_func
        s = baseflow_sens or self._baseflow_sensitivity
        if v is None or a is None or s is None:
            raise RuntimeError(
                "Direct mode, adjoint mode, and baseflow sensitivity are required to evaluate d sigma/d Re."
            )
        v_u = v.sub(0)
        a_u = a.sub(0)
        u_mu = s.sub(0)

        # Explicit sensitivity term
        form_explicit = (-1.0 / re_val**2) * inner(grad(conj(a_u)), grad(v_u)) * dx
        d_sigma_exp = dfem.assemble_scalar(dfem.form(form_explicit))

        # Implicit (baseflow) sensitivity term
        form_base = (
            inner(dot(u_mu, nabla_grad(v_u)), conj(a_u)) * dx
            + inner(dot(v_u, nabla_grad(u_mu)), conj(a_u)) * dx
        )
        d_sigma_base = dfem.assemble_scalar(dfem.form(form_base))

        return d_sigma_exp + d_sigma_base

    def evaluate(self, target: float | complex | None = None) -> complex:
        """Compute the eigenvalue sensitivity for the given baseflow at Reynolds number Re."""
        self.solve_direct_mode(target=target)
        self.solve_adjoint_mode()
        self.compute_baseflow_sensitivity()
        d_sigma = self.evaluate_sensitivity()

        log_global(
            logger,
            logging.INFO,
            "Computed eigenvalue sensitivity: %.4e + %.4e * j.",
            d_sigma.real,
            d_sigma.imag,
        )

        return d_sigma

    def compute_wavemaker(
        self,
        *,
        v_func: dfem.Function | None = None,
        a_func: dfem.Function | None = None,
        degree: int = 1,
    ) -> dfem.Function:
        """Compute the structural-sensitivity 'wavemaker' field Sw(x).

        Sw(x) = ||u†(x)|| * ||u(x)|| / <u†, u>, where u, u† are the velocity parts of the direct and adjoint
        eigenfunctions. The inner product is the L2 one over the domain. See Fabre et al., AMR (2019), eqs. (15)-(16).
        """
        v_func = v_func or self._v_func
        a_func = a_func or self._a_func
        if v_func is None or a_func is None:
            raise RuntimeError("Compute direct and adjoint modes before Sw.")

        # Velocity subfields (index 0 in your mixed space)
        v_u = v_func.sub(0)
        a_u = a_func.sub(0)

        # Denominator <u†, u> (may be complex) -> use magnitude to keep Sw real and non-negative
        denom = dfem.assemble_scalar(dfem.form(inner(conj(a_u), v_u) * dx))
        denom = complex(denom)
        denom_abs = abs(denom)
        if denom_abs == 0.0:
            raise RuntimeError("Denominator <u†,u> = 0; normalization issue.")

        # Pointwise product of Hermitian norms
        Sw_expr = (
            ufl.sqrt(inner(conj(a_u), a_u)) * ufl.sqrt(inner(conj(v_u), v_u))
        ) / denom_abs

        # Project directly to the solver's pressure space
        Sw_p = self._project_to(Sw_expr, self._spaces.pressure)

        # Pack into mixed space: u = 0, p = Sw
        mixed = dfem.Function(self._spaces.mixed)
        with mixed.x.petsc_vec.localForm() as lf:
            lf.set(0.0)
        mixed.sub(1).interpolate(Sw_p)
        mixed.x.scatter_forward()
        return mixed
