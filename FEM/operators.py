"""LSA-FW FEM linearized operator assembly.

This module implements the variational forms and discrete assembly routines
for the linearized incompressible Navier-Stokes equations around a steady base flow.

It supports eigenvalue and time-dependent formulations using PETSc block matrices,
including mass, viscous, convection, shear, pressure, and divergence operators.
"""

import logging
from typing import Callable
import numpy as np

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import dolfinx.fem as dfem
from ufl import (  # type: ignore[import-untyped]
    dx,
    TrialFunction,
    TestFunction,
    TestFunctions,
    Form,
    inner,
    grad,
    div,
    dot,
    Measure,
    derivative,
)
from ufl.argument import Argument  # type: ignore[import-untyped]
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from basix.ufl import mixed_element

from .utils import iPETScMatrix, iPETScVector, iPETScNullSpace
from .spaces import FunctionSpaces
from .bcs import BoundaryConditions, apply_periodic_constraints

logger = logging.getLogger(__name__)


class VariationalForms:
    """Collector for variational forms for linearized incompressible Navier-Stokes equations.

    Note: customization of quadrature degree is not yet supported. It is considered that the defaults
    set by DOLFINx are sufficient for the current (most-common) use cases.
    """

    @staticmethod
    def mass(u: Argument, v: Argument) -> Form:
        """Mass operator."""
        return inner(u, v) * dx

    @staticmethod
    def convection(u: Argument, v: Argument, u_base: dfem.Function) -> Form:
        """Convection operator."""
        return inner(dot(u_base, grad(u)), v) * dx

    @staticmethod
    def shear(u: Argument, v: Argument, u_base: dfem.Function) -> Form:
        """Shear operator."""
        return inner(dot(u, grad(u_base)), v) * dx

    @staticmethod
    def pressure_gradient(p: Argument, v: Argument) -> Form:
        """Pressure gradient operator.

        Refer to the module documentation and to the unit tests to understand the sign convention.
        """
        return p * div(v) * dx

    @staticmethod
    def viscous(u: Argument, v: Argument, re: float) -> Form:
        """Viscous operator."""
        return (1.0 / re) * inner(grad(u), grad(v)) * dx

    @staticmethod
    def divergence(u: Argument, q: Argument) -> Form:
        """Divergence operator."""
        return inner(div(u), q) * dx

    @staticmethod
    def neumann_rhs(v: Argument, g: dfem.Function, ds: Measure) -> Form:
        """Neumann boundary condition operator."""
        return inner(v, g) * ds

    @staticmethod
    def stiffness(u: Argument, v: Argument) -> Form:
        """Stiffness (Laplace) operator. Used for the vibrating membrane benchmark."""
        return inner(grad(u), grad(v)) * dx


class LinearizedNavierStokesAssembler:
    """Finite element operator assembler for linearized Navier-Stokes equations."""

    def __init__(
        self,
        base_flow: dfem.Function,
        spaces: FunctionSpaces,
        re: float,
        bcs: BoundaryConditions | None = None,
    ):
        """Initialize the assembler."""
        if base_flow.function_space != spaces.velocity:
            raise ValueError(
                "Base flow must be defined on the same function space as the velocity."
            )

        self._base_flow = base_flow
        self._re = re
        self._dtype = PETSc.ScalarType

        self._spaces = spaces
        self._nullspace: iPETScNullSpace | None = None

        self._u, self._p = self._get_trial_functions(spaces)
        self._v, self._q = self._get_test_functions(spaces)

        (
            self._u_bcs,
            self._p_bcs,
            self._neumann_forms,
            self._robin_forms,
            self._u_maps,
            self._p_maps,
        ) = self._get_bcs(bcs)

        self._matrix_cache: dict[str, iPETScMatrix] = {}

    @staticmethod
    def _get_test_functions(
        spaces: FunctionSpaces,
    ) -> tuple[Argument, Argument]:
        return TestFunction(spaces.velocity), TestFunction(spaces.pressure)

    @staticmethod
    def _get_trial_functions(
        spaces: FunctionSpaces,
    ) -> tuple[Argument, Argument]:
        return TrialFunction(spaces.velocity), TrialFunction(spaces.pressure)

    @staticmethod
    def _get_bcs(
        bcs: BoundaryConditions | None,
    ) -> tuple[
        list[dfem.DirichletBC],
        list[dfem.DirichletBC],
        list[Form],
        list[Form],
        list[dict[int, int]],
        list[dict[int, int]],
    ]:
        if bcs is None:
            return [], [], [], [], [], []
        velocity_bcs = [bc for _, bc in bcs.velocity]
        pressure_bcs = [bc for _, bc in bcs.pressure]
        neumann_forms = [form for _, form in bcs.neumann_forms]
        robin_forms = [form for _, form in bcs.robin_forms]
        return (
            velocity_bcs,
            pressure_bcs,
            neumann_forms,
            robin_forms,
            bcs.velocity_periodic_map,
            bcs.pressure_periodic_map,
        )

    @staticmethod
    def _assemble(
        bilinear_form: Form, dirichlet_bcs: list[dfem.DirichletBC], dtype: np.dtype
    ) -> iPETScMatrix:
        ufl_form = dfem.form(bilinear_form, dtype=dtype)
        a_petsc = assemble_matrix(ufl_form, bcs=dirichlet_bcs)
        return iPETScMatrix(a_petsc)

    def _get_or_compute_matrix(
        self, key: str, assembler: Callable[[], iPETScMatrix]
    ) -> iPETScMatrix:
        if key not in self._matrix_cache:
            logger.debug(f"Computing and caching matrix: {key}")
            self._matrix_cache[key] = assembler()
        return self._matrix_cache[key]

    def _apply_periodic_constraint(
        self, obj: iPETScMatrix | iPETScVector
    ) -> iPETScMatrix | iPETScVector:
        for pmap in self._u_maps + self._p_maps:
            apply_periodic_constraints(obj, pmap)
        return obj

    def assemble_mass_matrix(self) -> iPETScMatrix:
        """Assemble mass matrix."""
        logger.info("Assembling mass matrix.")
        return self._get_or_compute_matrix(
            "mass",
            lambda: self._assemble(
                VariationalForms.mass(self._u, self._v),
                self._u_bcs,
                self._dtype,
            ),
        )

    def assemble_convection_matrix(self) -> iPETScMatrix:
        """Assemble convection matrix."""
        return self._get_or_compute_matrix(
            "convection",
            lambda: self._assemble(
                VariationalForms.convection(self._u, self._v, self._base_flow),
                self._u_bcs,
                self._dtype,
            ),
        )

    def assemble_shear_matrix(self) -> iPETScMatrix:
        """Assemble shear matrix."""
        return self._get_or_compute_matrix(
            "shear",
            lambda: self._assemble(
                VariationalForms.shear(self._u, self._v, self._base_flow),
                self._u_bcs,
                self._dtype,
            ),
        )

    def assemble_pressure_gradient_matrix(self) -> iPETScMatrix:
        """Assemble pressure gradient matrix."""
        return self._get_or_compute_matrix(
            "pressure_gradient",
            lambda: self._assemble(
                VariationalForms.pressure_gradient(self._p, self._v),
                self._p_bcs,
                self._dtype,
            ),
        )

    def assemble_viscous_matrix(self) -> iPETScMatrix:
        """Assemble viscous matrix."""
        return self._get_or_compute_matrix(
            "viscous",
            lambda: self._assemble(
                VariationalForms.viscous(self._u, self._v, self._re),
                self._u_bcs,
                self._dtype,
            ),
        )

    def assemble_divergence_matrix(self) -> iPETScMatrix:
        """Assemble divergence matrix."""
        return self._get_or_compute_matrix(
            "divergence",
            lambda: self._assemble(
                VariationalForms.divergence(self._u, self._q),
                self._u_bcs,
                self._dtype,
            ),
        )

    def assemble_robin_matrix(self) -> iPETScMatrix:
        """Assemble matrix contribution from Robin boundary conditions.

        If Robin forms are defined separately per boundary (as is typical), this assumes they are summed at runtime.
        """
        return self._get_or_compute_matrix(
            "robin",
            lambda: (
                iPETScMatrix.zeros(
                    (self._spaces.velocity_dofs[0], self._spaces.velocity_dofs[0]),
                    comm=self._base_flow.function_space.mesh.comm,
                )
                if not self._robin_forms
                else self._assemble(sum(self._robin_forms), self._u_bcs)
            ),
        )

    def assemble_linear_operator(self) -> iPETScMatrix:
        """Assemble the full linear operator. Refer to the module documentation for nomenclature details."""
        logger.info("Assembling full linear NS operator.")

        # @FIXME: This matrix addition gives problems when running in different ranks (MPI),
        # as the addition finds matrix size mismatches

        A = (
            self.assemble_shear_matrix()  # C1
            + self.assemble_convection_matrix()  # C2
            + self.assemble_viscous_matrix()  # B
            + self.assemble_robin_matrix()  # R
        )

        G = self.assemble_pressure_gradient_matrix()
        D = self.assemble_divergence_matrix()

        L = iPETScMatrix.from_nested(
            [
                [A.raw, G.raw],
                [D.raw, None],
            ]
        )

        return self._apply_periodic_constraint(L)

    def assemble_eigensystem(self) -> tuple[iPETScMatrix, iPETScMatrix]:
        """Assemble the generalized eigenvalue system (A, M).

        Refer to the module documentation for nomenclature details.
        """
        logger.info("Assembling generalized eigenvalue system (A, M).")

        n_p, _ = self._spaces.pressure_dofs

        A = self.assemble_linear_operator()
        M_v = self.assemble_mass_matrix()
        M_p = iPETScMatrix.zeros((n_p, n_p), comm=M_v.comm)

        M = iPETScMatrix.from_nested(
            [
                [M_v.raw, None],
                [None, M_p.raw],
            ]
        )
        M = self._apply_periodic_constraint(M)

        self.attach_pressure_nullspace(A)
        self.attach_pressure_nullspace(M)

        return A, M

    def assemble_neumann_rhs(self, g: dfem.Function, ds: Measure) -> iPETScVector:
        """Assemble the Neumann boundary condition RHS contribution."""
        logger.info("Assembling Neumann boundary condition RHS.")
        form = dfem.form(
            VariationalForms.neumann_rhs(self._v, g, ds), dtype=self._dtype
        )
        vec = assemble_vector(form)
        dfem.apply_lifting(vec, [form], bcs=[self._u_bcs])
        vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        for bc in self._u_bcs:
            bc.set(vec)

        return self._apply_periodic_constraint(iPETScVector(vec))

    def clear_cache(self) -> None:
        """Clear the internal matrix cache."""
        logger.info("Clearing matrix cache.")
        self._matrix_cache.clear()

    def attach_pressure_nullspace(self, mat: iPETScMatrix) -> None:
        """Attach constant-pressure nullspace to PETSc matrix.

        This method informs PETSc that the matrix has a known nullspace (constant pressure mode), allowing compatible
        Krylov solvers to handle it correctly without pinning.
        """
        logger.info("Attaching constant-pressure nullspace to matrix %s.", mat)

        if self._nullspace is None:
            n_v, _ = self._spaces.velocity_dofs
            n_p, _ = self._spaces.pressure_dofs
            arr = np.concatenate([np.zeros(n_v), np.ones(n_p)])
            vec = iPETScVector.from_array(arr, comm=mat.comm)
            self._nullspace = iPETScNullSpace.create_constant_and_vectors(vectors=[vec])

        mat.attach_nullspace(self._nullspace)


class SteadyNavierStokesAssembler:
    """Finite element operator assembler for the steady (nonlinear) incompressible Navier-Stokes flow.

    TODO: Periodic BCs not handled yet.
    """

    def __init__(
        self,
        spaces: FunctionSpaces,
        re: float,
        bcs: BoundaryConditions | None = None,
        f: dfem.Function | None = None,
    ) -> None:
        """Initialize the assembler."""
        self._spaces = spaces
        self._re = re
        self._bcs = bcs
        if f is None:
            # Default body force to 0
            self._f = dfem.Constant(
                spaces.velocity.mesh, (0.0,) * spaces.velocity.mesh.topology.dim
            )
        else:
            self._f = f

        # Mixed velocity-pressure space
        mixed_elem = mixed_element(
            [
                spaces.velocity.ufl_element(),
                spaces.pressure.ufl_element(),
            ]
        )
        self._mixed_space = dfem.functionspace(
            spaces.velocity.mesh, mixed_elem
        )  # TODO: this mixed spaces should be an output of the spaces module

        # Solution
        self._up = dfem.Function(self._mixed_space)
        self.u, self.p = self._up.split()

        # Newton increment and test functions
        self._du = TrialFunction(self._mixed_space)
        self._v, self._q = TestFunctions(self._mixed_space)

        # Boundary conditions and Neumann/Robin terms
        self._u_bcs, self._p_bcs, self._neumann_forms, self._robin_forms = (
            self._get_bcs(bcs)
        )

        # Build variational forms
        self._residual, self._jacobian = self._build_forms()

        # Caches for PETSc objects
        self._vec_cache: dict[str, iPETScVector] = {}
        self._mat_cache: dict[str, iPETScMatrix] = {}

    @staticmethod
    def _get_bcs(bcs: BoundaryConditions | None):
        if bcs is None:
            return [], [], [], []
        vel_bcs = [bc for _, bc in bcs.velocity]
        pres_bcs = [bc for _, bc in bcs.pressure]
        neumann = [form for _, form in bcs.neumann_forms]
        robin = [form for _, form in bcs.robin_forms]
        return vel_bcs, pres_bcs, neumann, robin

    @property
    def residual(self) -> dfem.Form:
        """Get residual form."""
        return self._residual

    @property
    def jacobian(self) -> dfem.Form:
        """Get jacobian."""
        return self._jacobian

    @property
    def bcs(self):
        """Return all Dirichlet BC objects (velocity + pressure)."""
        return [*self._u_bcs, *self._p_bcs]

    def _build_forms(self) -> tuple[dfem.Form, dfem.Form]:
        nu = 1.0 / self._re

        momentum_ufl = (
            inner(grad(self.u) * self.u, self._v) * dx
            + nu * inner(grad(self.u), grad(self._v)) * dx
            - inner(self.p, div(self._v)) * dx
            - inner(self._f, self._v) * dx
        )
        for form in self._neumann_forms:
            momentum_ufl += form

        continuity_ufl = inner(div(self.u), self._q) * dx

        residual_ufl = momentum_ufl + continuity_ufl

        residual = dfem.form(residual_ufl)
        jacobian = dfem.form(derivative(residual_ufl, self._up, self._du))

        return residual, jacobian

    def assemble_residual(self) -> iPETScVector:
        """Assemble and return the PETSc residual vector."""
        key = f"res_{id(self._residual)}"
        if key not in self._vec_cache:
            vec = assemble_vector(self._residual)
            spaces = dfem.extract_function_spaces([self._residual], index=0)
            block_bcs = dfem.bcs_by_block(spaces, [*self._u_bcs, *self._p_bcs])
            dfem.apply_lifting(vec, [self._jacobian], block_bcs, x0=None)
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            for bc in (*self._u_bcs, *self._p_bcs):
                bc.set(vec)
            self._vec_cache[key] = iPETScVector(vec)
        return self._vec_cache[key]

    def assemble_jacobian(self) -> iPETScMatrix:
        """Assemble and return the PETSc Jacobian matrix."""
        key = f"jac_{id(self._jacobian)}"
        if key not in self._mat_cache:
            A = assemble_matrix(self._jacobian, bcs=[*self._u_bcs, *self._p_bcs])
            A.assemble()
            self._mat_cache[key] = iPETScMatrix(A)
        return self._mat_cache[key]

    def clear_cache(self) -> None:
        """Clear cached PETSc matrices and vectors."""
        self._vec_cache.clear()
        self._mat_cache.clear()

    def solve(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-12,
        max_it: int = 25,
        linear_solver: str = "gmres",
        preconditioner: str = "ilu",
    ):
        """
        Compute the steady base flow by solving F(u,p)=0 with Newton’s method.
        Modifies self._up in place and also returns it.
        """
        # wrap the UFL forms and solution Function into a high-level problem
        problem = NonlinearProblem(
            self._residual, self._up, bcs=self.bcs, J=self._jacobian
        )

        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.rtol = rtol
        solver.atol = atol
        solver.max_it = max_it
        solver.linear_solver = linear_solver
        solver.preconditioner = preconditioner

        num_its, converged = solver.solve(self._up)
        if not converged:
            raise RuntimeError(f"Newton failed to converge in {num_its} iterations")
        return self._up
