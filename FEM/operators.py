"""LSA-FW FEM linearized operator assembly.

This module implements the variational forms and discrete assembly routines
for the linearized incompressible Navier-Stokes equations around a steady base flow.

It supports eigenvalue and time-dependent formulations using PETSc block matrices,
including mass, viscous, convection, shear, pressure, and divergence operators.
"""

import logging

import dolfinx.fem as dfem
from ufl import dx, TrialFunction, TestFunction, Form, inner, grad, div, dot, Measure  # type: ignore[import-untyped]
from ufl.argument import Argument  # type: ignore[import-untyped]
from dolfinx.fem.petsc import assemble_matrix

from .utils import iPETScMatrix
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
        """Pressure gradient operator."""
        return p * div(v) * dx

    @staticmethod
    def viscous(u: Argument, v: Argument, re: float) -> Form:
        """Viscous operator."""
        return (1.0 / re) * inner(grad(u), grad(v)) * dx

    @staticmethod
    def divergence(u: Argument, v: Argument) -> Form:
        """Divergence operator."""
        return inner(div(u), v) * dx

    @staticmethod
    def neumann_rhs(v: Argument, g: dfem.Function, ds: Measure):
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

        self._u, self._p = self._get_trial_functions(spaces)
        self._v, self._q = self._get_test_functions(spaces)

        self._u_bcs, self._p_bcs, self._neumann_forms, self._robin_forms = (
            self._get_bcs(bcs)
        )

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
        return bcs.velocity, bcs.pressure, bcs.neumann_forms, bcs.robin_forms

    @staticmethod
    def _assemble(
        bilinear_form: Form, dirichlet_bcs: list[dfem.DirichletBC]
    ) -> iPETScMatrix:
        a_petsc = assemble_matrix(dfem.form(bilinear_form), bcs=dirichlet_bcs)
        a_petsc.assemble()
        return iPETScMatrix(a_petsc)

    def assemble_mass_matrix(self) -> iPETScMatrix:
        """Assemble mass matrix."""
        logger.info("Assembling mass matrix.")
        return self._assemble(VariationalForms.mass(self._u, self._v), self._u_bcs)

    def assemble_convection_matrix(self) -> iPETScMatrix:
        """Assemble convection matrix."""
        return self._assemble(
            VariationalForms.convection(self._u, self._v, self._base_flow), self._u_bcs
        )

    def assemble_shear_matrix(self) -> iPETScMatrix:
        """Assemble shear matrix."""
        return self._assemble(
            VariationalForms.shear(self._u, self._v, self._base_flow), self._u_bcs
        )

    def assemble_pressure_gradient_matrix(self) -> iPETScMatrix:
        """Assemble pressure gradient matrix."""
        return self._assemble(
            VariationalForms.pressure_gradient(self._p, self._v), self._p_bcs
        )

    def assemble_viscous_matrix(self) -> iPETScMatrix:
        """Assemble viscous matrix."""
        return self._assemble(
            VariationalForms.viscous(self._u, self._v, self._re), self._u_bcs
        )

    def assemble_divergence_matrix(self) -> iPETScMatrix:
        """Assemble divergence matrix."""
        return self._assemble(
            VariationalForms.divergence(self._u, self._q), self._u_bcs
        )

    def assemble_linear_operator(self) -> iPETScMatrix:
        """Assemble the full linear operator. Refer to the module documentation for nomenclature details."""
        logger.info("Assembling full linear NS operator.")

        A = self.assemble_viscous_matrix()
        A.axpy(1.0, self.assemble_convection_matrix())
        A.axpy(1.0, self.assemble_shear_matrix())

        G = self.assemble_pressure_gradient_matrix()
        D = self.assemble_divergence_matrix()

        return iPETScMatrix.from_nested(
            [
                [A.raw, G.raw],
                [D.raw, None],
            ]
        )

    def assemble_generalized_system(self) -> tuple[iPETScMatrix, iPETScMatrix]:
        """Assemble the generalized eigenvalue system (A, M).

        Refer to the module documentation for nomenclature details.
        """
        ...

    def assemble_neumann_rhs(self, g: dfem.Function) -> iPETScMatrix:
        """Assemble the Neumann boundary condition operator."""
        ...
