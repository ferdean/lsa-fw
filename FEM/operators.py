"""LSA-FW FEM linearized operator assembly.

This module implements the variational forms and discrete assembly routines
for the linearized incompressible Navier-Stokes equations around a steady base flow.

It supports eigenvalue and time-dependent formulations using PETSc block matrices,
including mass, viscous, convection, shear, pressure, and divergence operators.
"""

import logging
from typing import Callable
import numpy as np

import dolfinx.fem as dfem
from ufl import dx, TrialFunction, TestFunction, Form, inner, grad, div, dot, Measure  # type: ignore[import-untyped]
from ufl.argument import Argument  # type: ignore[import-untyped]
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc

from .utils import iPETScMatrix, iPETScVector, iPETScNullSpace
from .spaces import FunctionSpaces
from .bcs import BoundaryConditions

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

        self._u_bcs, self._p_bcs, self._neumann_forms, self._robin_forms = (
            self._get_bcs(bcs)
        )

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
    ) -> tuple[list[dfem.DirichletBC], list[dfem.DirichletBC], list[Form], list[Form]]:
        if bcs is None:
            return [], [], [], []
        velocity_bcs = [bc for _, bc in bcs.velocity]
        pressure_bcs = [bc for _, bc in bcs.pressure]
        neumann_forms = [form for _, form in bcs.neumann_forms]
        robin_forms = [form for _, form in bcs.robin_forms]
        return velocity_bcs, pressure_bcs, neumann_forms, robin_forms

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

        return iPETScMatrix.from_nested(
            [
                [A.raw, G.raw],
                [D.raw, None],
            ]
        )

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

        return iPETScVector(vec)

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
            n_p, _ = self._spaces.pressure_dofs
            vec = iPETScVector.from_array(np.ones((n_p,)), comm=mat.comm)
            self._nullspace = iPETScNullSpace.create_constant_and_vectors(vectors=[vec])

        mat.attach_nullspace(self._nullspace)
