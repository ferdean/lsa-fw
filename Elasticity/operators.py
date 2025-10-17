"""LSA-FW Elasticity FEM assembler.

Assembles the undamped, small-strain, isotropic linear elasticity eigenproblem in a displacement
formulation. It uses the definition of infinitesimal strain and Hooke's law in its Lamé form.
"""

# mypy: disable-error-code="attr-defined, name-defined"

from __future__ import annotations

import logging
from dataclasses import dataclass

import ufl
from Elasticity.bcs import BoundaryConditions
import dolfinx.fem as dfem
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import MeshTags
import dolfinx.mesh as dmesh

from FEM.utils import Scalar, iPETScMatrix
from lib.loggingutils import log_rank
from FEM.utils import iMeasure

logger = logging.getLogger(__name__)


class ElasticityVariationalForms:
    """Collector for linear elasticity weak forms (isotropic, small strain)."""

    @staticmethod
    def _eps(u: ufl.Argument) -> ufl.Form:
        return ufl.sym(ufl.grad(u))  # Infinitesimal strain -- _eps(u) := 1/2 (u + u^T)

    @staticmethod
    def sigma(u: ufl.Argument, *, mu: dfem.Constant, lam: dfem.Constant) -> ufl.Form:
        d = u.ufl_domain().topological_dimension()
        return 2.0 * mu * ElasticityVariationalForms._eps(u) + lam * ufl.tr(
            ElasticityVariationalForms._eps(u)
        ) * ufl.Identity(d)

    @staticmethod
    def stiffness(
        u: ufl.Argument, v: ufl.Argument, *, mu: dfem.Constant, lam: dfem.Constant
    ) -> ufl.Form:
        return (
            ufl.inner(
                ElasticityVariationalForms.sigma(u, mu=mu, lam=lam),
                ElasticityVariationalForms._eps(v),
            )
            * ufl.dx
        )

    @staticmethod
    def mass(u: ufl.Argument, v: ufl.Argument, *, rho: dfem.Constant) -> ufl.Form:
        return rho * ufl.inner(u, v) * ufl.dx

    @staticmethod
    def body_force(v: ufl.Argument, *, b: dfem.Function) -> ufl.Form:
        # Not used in EVP
        return ufl.inner(b, v) * ufl.dx

    @staticmethod
    def traction(v: ufl.Argument, *, t: dfem.Function, ds: ufl.Measure) -> ufl.Form:
        return ufl.inner(t, v) * ds


@dataclass(frozen=True)
class MaterialProperties:
    """Define a collector for the material properties. Refer to the literature for the derivation of the values."""

    mu: dfem.Constant
    lam: dfem.Constant
    rho: dfem.Constant

    @classmethod
    def from_basic_properties(
        cls,
        mesh: dmesh.Mesh,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
    ) -> MaterialProperties:
        """Create object from basic structural properties."""
        mu_val = young_modulus / (2.0 * (1.0 + poisson_ratio))
        lam_val = (
            young_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
        )

        if mu_val <= 0 or (3 * lam_val + 2 * mu_val) <= 0:
            raise ValueError(
                "Properties infeasibility. The given values would lead to a non-symmetric nor PD system."
            )

        return cls(
            dfem.Constant(mesh, mu_val),
            dfem.Constant(mesh, lam_val),
            dfem.Constant(mesh, density),
        )


class ElasticityEigenAssembler:
    """Finite element assembler for undamped linear elasticity eigenproblems."""

    def __init__(
        self,
        space: dfem.FunctionSpace,
        *,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
        bcs: BoundaryConditions | None = None,
        tags: MeshTags | None = None,
    ) -> None:
        """Initialize."""
        self._space = space
        self._bcs = self._extract_bcs(bcs.dirichlet)
        self._material_properties = MaterialProperties.from_basic_properties(
            space.mesh, young_modulus, poisson_ratio, density
        )

        self._u = ufl.TrialFunction(self._space)
        self._v = ufl.TestFunction(self._space)

        self._ds = iMeasure.ds(self._space.mesh, tags)

        self._stiffness_form = ElasticityVariationalForms.stiffness(
            self._u,
            self._v,
            mu=self._material_properties.mu,
            lam=self._material_properties.lam,
        )
        self._mass_form = ElasticityVariationalForms.mass(
            self._u, self._v, rho=self._material_properties.rho
        )

        self._mat_cache: dict[str, iPETScMatrix] = {}

        log_rank(logger, logging.INFO, "Initialized elasticity EVP assembler.")

        # Add natural BCs (typically zero for EVP, but supported)
        if bcs.neumann is not None:
            for marker, t in bcs.neumann:
                log_rank(
                    logger,
                    logging.WARN,
                    "EVP contains non-homogeneous Neumann boundary conditions.",
                )
                self._stiffness_form -= ElasticityVariationalForms.traction(
                    self._v, t=t, ds=self._ds(marker)
                )

    @staticmethod
    def _extract_bcs(bcs: BoundaryConditions | None) -> list[dfem.DirichletBC]:
        return [bc for (_, bc) in bcs]

    def assemble_stiffness(self, *, key: str | int | None = None) -> iPETScMatrix:
        """Assemble and return the stiffness matrix."""
        key = str(key or f"k_{id(self)}")
        if key not in self._mat_cache:
            log_rank(
                logger,
                logging.INFO,
                "No cached stiffness matrix found. Assembling started.",
            )

            stiffness = assemble_matrix(
                dfem.form(self._stiffness_form, dtype=Scalar),
                bcs=list(self._bcs),
            )
            stiffness.assemble()
            self._mat_cache[key] = iPETScMatrix(stiffness)

            log_rank(
                logger,
                logging.INFO,
                "Stiffness matrix assembly successful. Matrix was cached under '%s'",
                key,
            )

        return self._mat_cache[key]

    def assemble_mass(self, *, key: str | int | None = None) -> iPETScMatrix:
        """Assemble and return the consistent mass matrix."""
        key = str(key or f"m_{id(self)}")
        if key not in self._mat_cache:
            log_rank(
                logger,
                logging.INFO,
                "No cached mass matrix found. Assembling started.",
            )

            mass = assemble_matrix(
                dfem.form(self._mass_form, dtype=Scalar), bcs=list(self._bcs)
            )
            mass.assemble()
            self._mat_cache[key] = iPETScMatrix(mass)

            log_rank(
                logger,
                logging.INFO,
                "Mass matrix assembly successful. Matrix was cached under '%s'",
                key,
            )

        return self._mat_cache[key]

    def assemble_eigensystem(self) -> tuple[iPETScMatrix, iPETScMatrix]:
        """Assemble eigensystem. Returned as (mass, stiffness)."""
        return self.assemble_mass(), self.assemble_stiffness()
