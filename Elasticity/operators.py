"""LSA-FW Elasticity FEM assembler.

Assembles the undamped, small-strain, isotropic linear elasticity eigenproblem in a displacement
formulation. It uses the definition of infinitesimal strain and Hooke's law in its Lamé form.
"""

# mypy: disable-error-code="attr-defined, name-defined"

from __future__ import annotations

import logging
from dataclasses import dataclass

import ufl
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import MeshTags
from ufl.core.expr import Expr  # type:ignore[import-untyped]
from ufl.algorithms.replace import replace  # type:ignore[import-untyped]

from Elasticity.bcs import BoundaryConditions
from FEM.utils import Scalar, iPETScMatrix, iMeasure
from lib.loggingutils import log_rank

logger = logging.getLogger(__name__)


class ElasticityVariationalForms:
    """Collector for linear elasticity weak forms (isotropic, small strain)."""

    @staticmethod
    def _eps(u: ufl.Argument) -> ufl.Form:
        return ufl.sym(ufl.grad(u))

    @staticmethod
    def sigma(u: ufl.Argument, *, mu: Expr, lam: Expr) -> ufl.Form:
        d = u.ufl_domain().topological_dimension()
        return 2.0 * mu * ElasticityVariationalForms._eps(u) + lam * ufl.tr(
            ElasticityVariationalForms._eps(u)
        ) * ufl.Identity(d)

    @staticmethod
    def stiffness(u: ufl.Argument, v: ufl.Argument, *, mu: Expr, lam: Expr) -> ufl.Form:
        return (
            ufl.inner(
                ElasticityVariationalForms.sigma(u, mu=mu, lam=lam),
                ElasticityVariationalForms._eps(v),
            )
            * ufl.dx
        )

    @staticmethod
    def mass(u: ufl.Argument, v: ufl.Argument, *, rho: Expr) -> ufl.Form:
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

    mu: dfem.Function
    lam: dfem.Function
    rho: dfem.Function
    young_modulus: dfem.Function
    poisson_ration: dfem.Function

    @staticmethod
    def _get_linear_function_space(mesh: dmesh.Mesh) -> dfem.FunctionSpace:
        return dfem.functionspace(mesh, ("DG", 0))

    @staticmethod
    def _to_function(
        mesh: dmesh.Mesh, val: float | dfem.Constant | dfem.Function
    ) -> dfem.Function:
        function_space = MaterialProperties._get_linear_function_space(mesh)
        f = dfem.Function(function_space)
        if isinstance(val, dfem.Function):
            try:
                f.interpolate(val)
            except Exception:
                f.x.array[:] = val.x.array.mean()
            return f
        if isinstance(val, dfem.Constant):
            f.x.array[:] = float(val.value)
            return f
        f.x.array[:] = float(val)
        return f

    @classmethod
    def from_basic_properties(
        cls,
        mesh: dmesh.Mesh,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
    ) -> MaterialProperties:
        """Create object from basic structural properties."""
        young_modulus_func = cls._to_function(mesh, young_modulus)
        poisson_ratio_func = cls._to_function(mesh, poisson_ratio)
        density_func = cls._to_function(mesh, density)

        mu_func = young_modulus_func / (2.0 * (1.0 + poisson_ratio_func))
        lam_func = (
            young_modulus_func
            * poisson_ratio_func
            / ((1.0 + poisson_ratio_func) * (1.0 - 2.0 * poisson_ratio_func))
        )

        return cls(
            mu_func, lam_func, density_func, young_modulus_func, poisson_ratio_func
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
        self._bcs = self._extract_bcs(bcs.dirichlet) if bcs is not None else []
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

        if bcs is not None and getattr(bcs, "neumann", None) is not None:
            for marker, t in bcs.neumann:
                log_rank(
                    logger,
                    logging.WARN,
                    "EVP contains non-homogeneous Neumann boundary conditions.",
                )
                self._stiffness_form -= ElasticityVariationalForms.traction(
                    self._v, t=t, ds=self._ds(marker)
                )

    @property
    def function_space(self) -> dfem.FunctionSpace:
        """Retrieve the function space."""
        return self._space

    @property
    def trial(self) -> ufl.Argument:
        """Retrieve trial function."""
        return self._u

    @property
    def test(self) -> ufl.Argument:
        """Retrieve test function."""
        return self._v

    @property
    def stiffness_form(self) -> ufl.Form:
        """Retrieve the stiffness (bilinear) form."""
        return self._stiffness_form

    @property
    def mass_form(self) -> ufl.Form:
        """Retrieve the mass (bilinear) form."""
        return self._mass_form

    @property
    def material_properties(self) -> MaterialProperties:
        """Retrieve material properties (exposed for differentiation)."""
        return self._material_properties

    @staticmethod
    def _extract_bcs(bcs: BoundaryConditions | None) -> list[dfem.DirichletBC]:
        return [bc for (_, bc) in bcs] if bcs is not None else []

    @staticmethod
    def _assert_param_is_valid(form: ufl.Form, param: Expr) -> None:
        if not isinstance(param, (dfem.Function, ufl.Coefficient)):
            raise TypeError(
                "`param` must be a UFL Coefficient (typically fem.Function on DG-0). "
                f"Found: {type(param)}."
            )
        form_domains = set(form.ufl_domains())
        if (
            getattr(param, "ufl_domain", None)
            and param.ufl_domain() not in form_domains
        ):
            raise ValueError(
                "`param` lives on a different UFL domain (mesh) than the form."
            )

    @staticmethod
    def _substitute_trial_test(form: ufl.Form, v_h: dfem.Function) -> ufl.Form:
        args = form.arguments()
        return replace(form, {args[0]: v_h, args[1]: v_h})

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
                logger, logging.INFO, "No cached mass matrix found. Assembling started."
            )
            mass = assemble_matrix(
                dfem.form(self._mass_form, dtype=Scalar),
                bcs=list(self._bcs),
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

    def compute_sensitivity(
        self,
        eigenfunction: dfem.Function,
        eigenvalue: float,
        param: Expr,
        dparam: Expr | None = None,
    ) -> float:
        """Compute eigenvalue sensitivity to `param`."""
        k_vv = self._substitute_trial_test(self.stiffness_form, eigenfunction)
        m_vv = self._substitute_trial_test(self.mass_form, eigenfunction)
        self._assert_param_is_valid(k_vv, param)
        self._assert_param_is_valid(m_vv, param)
        if dparam is None:
            if isinstance(param, dfem.Function):
                dparam = dfem.Function(param.function_space)
                dparam.x.array[:] = 1.0
            else:
                raise TypeError("Provide `dparam` when `param` is not a fem.Function.")

        dk = ufl.derivative(k_vv, param, dparam)
        dm = ufl.derivative(m_vv, param, dparam)

        val_k = dfem.assemble_scalar(dfem.form(dk))
        val_m = dfem.assemble_scalar(dfem.form(dm))

        return float(val_k - eigenvalue * val_m)
