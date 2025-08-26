"""LSA-FW FEM linearized operator assembly.

This module implements the variational forms and discrete assembly routines for the linearized incompressible
Navier-Stokes equations around a steady baseflow.

It supports eigenvalue and time-dependent formulations using PETSc block matrices, including mass, viscous, convection,
shear, pressure, and divergence operators.
"""

# mypy: disable-error-code="attr-defined, name-defined"

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import dolfinx.fem as dfem
import numpy as np
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
)
from dolfinx.mesh import compute_midpoints
from dolfinx.mesh import MeshTags
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (  # type: ignore[import-untyped]
    Form,
    TestFunctions,
    TrialFunctions,
    div,
    dx,
    grad,
    inner,
    dot,
    nabla_grad,
    split,
    derivative,
    ExternalOperator,
)
from ufl.argument import Argument  # type: ignore[import-untyped]

from lib.cache import CacheStore
from lib.loggingutils import log_global, log_rank

from .bcs import BoundaryConditions, apply_periodic_constraints
from .spaces import FunctionSpaces
from .utils import (
    iPETScBlockMatrix,
    iPETScMatrix,
    iPETScNullSpace,
    iMeasure,
    iPETScVector,
    Scalar,
)

logger = logging.getLogger(__name__)


def _extract_bcs(bcs: BoundaryConditions) -> tuple[
    list[dfem.DirichletBC],
    list[dfem.DirichletBC],
    list[tuple[int, dfem.Constant]],
    list[tuple[int, dfem.Constant]],
    list[tuple[int, dfem.Constant, ExternalOperator]],
    list[dict[int, int]],
    list[dict[int, int]],
]:
    velocity_bcs = [bc for _, bc in bcs.velocity]
    pressure_bcs = [bc for _, bc in bcs.pressure]
    return (
        velocity_bcs,
        pressure_bcs,
        bcs.velocity_neumann,
        bcs.pressure_neumann,
        bcs.robin_data,
        bcs.velocity_periodic_map,
        bcs.pressure_periodic_map,
    )


class BaseAssembler(ABC):
    """Abstract base class for finite element operator assemblers."""

    def __init__(
        self,
        spaces: FunctionSpaces,
        bcs: BoundaryConditions,
        *,
        tags: MeshTags | None = None,
    ) -> None:
        """Initialize base assembler."""
        (
            self._u_bcs,
            self._p_bcs,
            self._v_neumann,
            self._p_neumann,
            self._robin_data,
            self._u_maps,
            self._p_maps,
        ) = _extract_bcs(bcs)
        self._spaces = spaces
        self._ds = iMeasure.ds(spaces.mixed.mesh, tags)
        self._mat_cache: dict[str | int, iPETScMatrix] = {}
        self._vec_cache: dict[str | int, iPETScVector] = {}

    @property
    @abstractmethod
    def residual(self) -> dfem.Form:
        """Get residual form."""
        pass

    @property
    @abstractmethod
    def jacobian(self) -> dfem.Form:
        """Get jacobian form."""
        pass

    @property
    @abstractmethod
    def sol(self) -> dfem.Function:
        """Get solution function."""
        pass

    @property
    def bcs(self) -> tuple[dfem.DirichletBC, ...]:
        return (*self._u_bcs, *self._p_bcs)

    @property
    def spaces(self) -> FunctionSpaces:
        """Get the function spaces used by this assembler."""
        return self._spaces

    @abstractmethod
    def get_matrix_forms(self) -> tuple[iPETScMatrix, iPETScVector | iPETScMatrix]:
        """Assemble and return PETSc matrix and vector."""
        pass

    def clear_cache(self) -> None:
        """Clear cached PETSc matrices and vectors."""
        self._mat_cache.clear()
        self._vec_cache.clear()

    def _apply_dirichlet(self, fn: dfem.Function | PETSc.Vec) -> None:
        """Apply Dirichlet BCs in-place."""
        array = fn.x.array if isinstance(fn, dfem.Function) else fn.array
        for bc in self.bcs:
            bc.set(array)

        if isinstance(fn, dfem.Function):
            fn.x.scatter_forward()
        else:
            fn.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)


class StokesAssembler(BaseAssembler):
    """Finite element operator assembler for the steady Stokes equations."""

    def __init__(
        self,
        spaces: FunctionSpaces,
        bcs: BoundaryConditions,
        *,
        tags: MeshTags | None = None,
        f: dfem.Function | None = None,
    ) -> None:
        """Initialize Stokes assembler."""
        super().__init__(spaces, bcs, tags=tags)
        self._f = f or dfem.Constant(
            spaces.velocity.mesh, (0.0,) * spaces.velocity.mesh.topology.dim
        )
        self._g = dfem.Constant(
            spaces.velocity.mesh, (0.0,) * spaces.velocity.mesh.topology.dim
        )

        self._u, self._p = TrialFunctions(spaces.mixed)
        self._v, self._q = TestFunctions(spaces.mixed)

        self._wh = dfem.Function(spaces.mixed)
        self._apply_dirichlet(self._wh)

        # TODO: handle periodic constraints

        self._jacobian, self._residual = self._build_forms()

    @property
    def residual(self) -> dfem.Form:
        return dfem.form(self._residual, dtype=Scalar)

    @property
    def jacobian(self) -> dfem.Form:
        return dfem.form(self._jacobian, dtype=Scalar)

    @property
    def sol(self) -> dfem.Function:
        return self._wh

    def _build_forms(self) -> tuple[dfem.Form, dfem.Form]:
        bform = (
            inner(grad(self._u), grad(self._v)) * dx
            + inner(self._p, div(self._v)) * dx
            + inner(div(self._u), self._q) * dx
        )
        lform = -inner(self._f, self._v) * dx

        # The pressure term seems to have the incorrect sign. This is done deliberately, so that the resulting
        # block system is symmetric. This symmetry lets us reuse later symmetric solvers and pre-conditioners for
        # both the Stokes and Navier–Stokes operators.

        for marker, g in self._v_neumann:
            lform += inner(self._v, g) * self._ds(marker)

        for marker, g in self._p_neumann:
            lform += inner(self._q, g) * self._ds(marker)

        return bform, lform

    def get_matrix_forms(self) -> tuple[iPETScMatrix, iPETScVector]:
        """Assemble the jacobian and residual forms of the Stokes equations."""
        key_jac = f"jac_{id(self._jacobian)}"
        key_res = f"res_{id(self._residual)}"

        if key_jac not in self._mat_cache:
            A = assemble_matrix(self.jacobian, bcs=list(self.bcs))
            A.assemble()
            A_wrapper = iPETScMatrix(A)
            if not self._p_bcs:
                # Pin pressure DOF
                _, self._dofs_p = self._spaces.mixed.sub(1).collapse()
                A_wrapper.pin_dof(self._dofs_p[0])
            self._mat_cache[key_jac] = A_wrapper

        if key_res not in self._vec_cache:
            b = assemble_vector(self.residual)
            dfem.apply_lifting(b.array, [dfem.form(self._jacobian)], [list(self.bcs)])
            self._apply_dirichlet(b)
            self._vec_cache[key_res] = iPETScVector(b)

        return self._mat_cache[key_jac], self._vec_cache[key_res]


class VariationalForms:
    """Collector for variational forms for linearized incompressible Navier-Stokes equations."""

    @staticmethod
    def mass(u: Argument, v: Argument) -> Form:
        return inner(u, v) * dx

    @staticmethod
    def convection(u: Argument, v: Argument, u_base: dfem.Function) -> Form:
        return inner(dot(u_base, nabla_grad(u)), v) * dx

    @staticmethod
    def shear(u: Argument, v: Argument, u_base: dfem.Function) -> Form:
        return inner(dot(u, grad(u_base)), v) * dx

    @staticmethod
    def pressure_gradient(p: Argument, v: Argument) -> Form:
        return inner(p, div(v)) * dx

    @staticmethod
    def viscous(u: Argument, v: Argument, re: float) -> Form:
        return (1.0 / re) * inner(grad(u), grad(v)) * dx

    @staticmethod
    def divergence(u: Argument, q: Argument) -> Form:
        return inner(div(u), q) * dx

    @staticmethod
    def forcing(f: dfem.Function, v: Argument) -> Form:
        return -inner(f, v) * dx

    @staticmethod
    def stiffness(u: Argument, v: Argument) -> Form:
        return inner(grad(u), grad(v)) * dx


class StationaryNavierStokesAssembler(BaseAssembler):
    """Finite element operator assembler for stationary Navier-Stokes equations."""

    def __init__(
        self,
        spaces: FunctionSpaces,
        bcs: BoundaryConditions,
        re: float,
        *,
        tags: MeshTags | None = None,
        f: dfem.Function | None = None,
        initial_guess: dfem.Function | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(spaces, bcs, tags=tags)

        self._re = re
        self._f = f or dfem.Constant(
            spaces.velocity.mesh, (0.0,) * spaces.velocity.mesh.topology.dim
        )

        self._v, self._q = TestFunctions(spaces.mixed)
        self._wh = dfem.Function(spaces.mixed)  # Solution

        if initial_guess is not None:
            self._wh.x.array[:] = initial_guess.x.array
            self._wh.x.scatter_forward()

        self._u, self._p = split(self._wh)
        self._apply_dirichlet(self._wh)

        self._residual, self._jacobian = self._build_forms()

        log_global(
            logger,
            logging.INFO,
            "Stationary Navier Stokes assembler has been initialized.",
        )

    @property
    def residual(self) -> dfem.Form:
        return self._residual

    @property
    def jacobian(self) -> dfem.Form:
        return self._jacobian

    @property
    def sol(self) -> dfem.Function:
        return self._wh

    def _build_forms(self) -> tuple[dfem.Form, dfem.Form]:
        convection = VariationalForms.convection(self._u, self._v, self._u)
        diffusion = VariationalForms.viscous(self._u, self._v, self._re)
        pressure = VariationalForms.pressure_gradient(self._p, self._v)
        divergence = VariationalForms.divergence(self._u, self._q)
        forcing = VariationalForms.forcing(self._f, self._v)

        form = convection + diffusion + pressure + divergence + forcing

        for marker, g in self._v_neumann:
            form += inner(self._v, g) * self._ds(marker)

        for marker, g in self._p_neumann:
            form += inner(self._q, g) * self._ds(marker)

        for marker, alpha, g_expr in self._robin_data:
            form += alpha * inner(self._u - g_expr, self._v) * self._ds(marker)

        return dfem.form(form, dtype=Scalar), dfem.form(
            derivative(form, self._wh), dtype=Scalar
        )

    def get_matrix_forms(
        self, *, key_jac: str | int | None = None, key_res: str | int | None = None
    ) -> tuple[iPETScMatrix, iPETScVector]:
        """Get the matrix and vector forms for the linearized Navier-Stokes equations."""
        key_jac = key_jac or f"jac_{id(self._jacobian)}"
        key_res = key_res or f"res_{id(self._residual)}"

        if key_jac not in self._mat_cache:
            log_rank(
                logger,
                logging.INFO,
                "No cached matrix found. Assembling linearized operator.",
            )
            A = assemble_matrix(self._jacobian, bcs=list(self.bcs))
            A.assemble()
            A_wrapper = iPETScMatrix(A)
            if not self._p_bcs:
                # Pin pressure DOF
                log_rank(
                    logger,
                    logging.DEBUG,
                    "No pressure Dirichlet BCs. Pinning pressure DOF to remove nullspace.",
                )
                _, self._dofs_p = self._spaces.mixed.sub(1).collapse()
                A_wrapper.pin_dof(self._dofs_p[-1])
            self._mat_cache[key_jac] = A_wrapper

        if key_res not in self._vec_cache:
            log_rank(logger, logging.INFO, "No cached vector found. Assembling RHS.")
            b = assemble_vector(self.residual)
            dfem.apply_lifting(b.array, [dfem.form(self._jacobian)], [list(self.bcs)])
            self._apply_dirichlet(b)
            self._vec_cache[key_res] = iPETScVector(b)

        return self._mat_cache[key_jac], self._vec_cache[key_res]


class LinearizedNavierStokesAssembler(BaseAssembler):
    """FEM assembler for the linearized Navier-Stokes operator around a stationary baseflow."""

    def __init__(
        self,
        base_flow: dfem.Function,
        spaces: FunctionSpaces,
        re: float,
        bcs: BoundaryConditions,
        *,
        tags: MeshTags | None = None,
        use_sponge: bool = False,
    ) -> None:
        """Initialize the linearized Navier-Stokes assembler."""
        if base_flow.function_space != spaces.mixed:
            raise ValueError("Baseflow must be defined on the mixed function space.")

        if _has_non_homogeneous_neumann(bcs) or _has_non_homogeneous_robin(bcs):
            raise ValueError(
                "Non-homogeneous natural (flux) boundary conditions are not yet supported."
            )

        super().__init__(spaces, bcs, tags=tags)

        self._base_flow = base_flow.sub(0)  # Velocity part only
        self._re = re
        self._use_sponge = use_sponge

        self._u, self._p = TrialFunctions(spaces.mixed)
        self._v, self._q = TestFunctions(spaces.mixed)

        _, self._dofs_u = spaces.mixed.sub(0).collapse()
        _, self._dofs_p = spaces.mixed.sub(1).collapse()

        self._num_dofs = len(self._dofs_u) + len(self._dofs_p)
        self._nullspace: iPETScNullSpace | None = None

        log_global(
            logger, logging.INFO, "Initialized linearized Navier-Stokes assembler."
        )

    @property
    def residual(self) -> dfem.Form:
        raise NotImplementedError("No residual form is defined for eigenproblems.")

    @property
    def jacobian(self) -> dfem.Form:
        raise NotImplementedError("No Jacobian form is defined for eigenproblems.")

    @property
    def sol(self) -> dfem.Function:
        raise NotImplementedError("No solution function is defined for eigenproblems.")

    def assemble_linear_operator(
        self,
        *,
        key: str | int | None = None,
        cache: CacheStore | None = None,
    ) -> iPETScMatrix:
        """Assemble the linearized Navier-Stokes operator A."""
        key = str(key or f"lin_ns_{id(self)}")
        if key in self._mat_cache:
            return self._mat_cache[key]

        if cache is not None and (cached := cache.load_matrix(key)) is not None:
            self._mat_cache[key] = iPETScMatrix(cached)
            return self._mat_cache[key]

        log_rank(
            logger,
            logging.DEBUG,
            "Assembling linear operator - (%d DOFs)",
            self._num_dofs,
        )

        a = (
            VariationalForms.shear(self._u, self._v, self._base_flow)
            + VariationalForms.convection(self._u, self._v, self._base_flow)
            + VariationalForms.viscous(self._u, self._v, self._re)
            + VariationalForms.pressure_gradient(self._p, self._v)
            + VariationalForms.divergence(self._u, self._q)
        )

        if self._use_sponge:
            alpha = _get_damping_factor(self._spaces)
            a += -inner(alpha * self._u, self._v) * dx

        for marker, alpha, g_expr in self._robin_data:
            a += alpha * inner(self._u - g_expr, self._v) * self._ds(marker)

        A_raw = assemble_matrix(dfem.form(a, dtype=Scalar), bcs=list(self.bcs))
        A_raw.assemble()

        A = iPETScMatrix(A_raw)

        if not self._p_bcs:
            self.attach_pressure_nullspace(A)
            log_rank(logger, logging.DEBUG, "Attached pressure nullspace to A.")

        for pmap in (*self._u_maps, *self._p_maps):
            apply_periodic_constraints(A, pmap)

        self._mat_cache[key] = A
        if cache:
            cache.save_matrix(key, A.raw)

        return A

    def assemble_mass_matrix(
        self,
        *,
        key: str | int | None = None,
        cache: CacheStore | None = None,
    ) -> iPETScMatrix:
        """Assemble the mass matrix M."""
        key = str(key or f"mass_ns_{id(self)}")
        if key in self._mat_cache:
            return self._mat_cache[key]

        if cache is not None and (cached := cache.load_matrix(key)) is not None:
            self._mat_cache[key] = iPETScMatrix(cached)
            return self._mat_cache[key]

        log_rank(
            logger,
            logging.DEBUG,
            "Assembling mass matrix - (%d DOFs)",
            self._num_dofs,
        )

        a_mass = VariationalForms.mass(self._u, self._v)

        M_raw = assemble_matrix(dfem.form(a_mass, dtype=Scalar), bcs=list(self.bcs))
        M_raw.assemble()

        M = iPETScMatrix(M_raw)

        if not self._p_bcs:
            self.attach_pressure_nullspace(M)
            log_rank(logger, logging.DEBUG, "Attached pressure nullspace to M.")

        for pmap in (*self._u_maps, *self._p_maps):
            apply_periodic_constraints(M, pmap)

        self._mat_cache[key] = M
        if cache:
            cache.save_matrix(key, M.raw)

        return M

    def assemble_eigensystem(
        self, *, cache: CacheStore | None = None
    ) -> tuple[iPETScMatrix, iPETScMatrix]:
        """Assemble both the system operator and mass matrix (A, M)."""
        A = self.assemble_linear_operator(cache=cache)
        M = self.assemble_mass_matrix(cache=cache)
        log_rank(
            logger,
            logging.INFO,
            "Assembled eigensystem: %d pressure DOFs, %d velocity DOFs.",
            len(self._dofs_p),
            len(self._dofs_u),
        )
        return A, M

    def get_matrix_forms(self) -> tuple[iPETScMatrix, iPETScMatrix]:
        """Return assembled (A, M) for use in eigenproblems."""
        return self.assemble_eigensystem()

    def attach_pressure_nullspace(self, mat: iPETScMatrix) -> None:
        """Attach constant-pressure nullspace to the matrix."""
        if self._nullspace is None:
            arr = np.zeros(self._num_dofs, dtype=Scalar)
            arr[self._dofs_p] = 1.0
            vec = iPETScVector.from_array(arr, comm=mat.comm)
            vec.scale(1 / vec.norm)
            self._nullspace = iPETScNullSpace.from_vectors([vec])
        self._nullspace.attach_to(mat)

    def extract_subblocks(self, mat: iPETScMatrix) -> iPETScBlockMatrix:
        """Return (vv, vp, pv, pp) subblocks from mixed matrix."""
        is_v = PETSc.IS().createGeneral(self._dofs_u, comm=mat.comm)
        is_p = PETSc.IS().createGeneral(self._dofs_p, comm=mat.comm)

        return iPETScBlockMatrix(
            [
                [
                    iPETScMatrix(mat.raw.createSubMatrix(is_v, is_v)),
                    iPETScMatrix(mat.raw.createSubMatrix(is_v, is_p)),
                ],
                [
                    iPETScMatrix(mat.raw.createSubMatrix(is_p, is_v)),
                    iPETScMatrix(mat.raw.createSubMatrix(is_p, is_p)),
                ],
            ]
        )

    def test_nullspace(self, matrix: iPETScMatrix) -> None:
        """Assert correctness of nullspace (debug only)."""
        if self._nullspace is None:
            raise RuntimeError("Nullspace not initialized.")
        is_ns, residual = self._nullspace.test_matrix(matrix)
        assert is_ns, f"Nullspace test failed: residual = {residual:.3e}"


def _get_damping_factor(
    spaces: FunctionSpaces, *, frac: float = 0.20, alpha_max: float = -1.4
) -> dfem.Function:
    """Generate the damping factor to form a sponge (damping) term over the last fraction of the domain.

    This is specially useful when finding eigenfunctions living only at the scenario outlet. Those eigenmodes
    belong to the discrete approximation of the convective spectrum and carry essentially no information about
    the wake instability.
    """
    mesh = spaces.mixed.mesh

    # Determine global x-range (gathered on root)
    x_coords = mesh.geometry.x[:, 0]
    x_min = MPI.COMM_WORLD.allreduce(x_coords.min(), op=MPI.MIN)
    x_max = MPI.COMM_WORLD.allreduce(x_coords.max(), op=MPI.MAX)

    domain_length = x_max - x_min
    sponge_start = x_min + (1.0 - frac) * domain_length
    sponge_length = frac * domain_length

    # DG0 function space
    V0 = dfem.functionspace(mesh, ("DG", 0))
    alpha = dfem.Function(V0)

    # Compute cell centroids using DolfinX built-in method
    centroids = compute_midpoints(
        mesh,
        mesh.topology.dim,
        np.arange(mesh.topology.index_map(mesh.topology.dim).size_local),
    )

    # Evaluate ramp at centroids explicitly
    xi = (centroids[:, 0] - sponge_start) / sponge_length
    ramp = np.clip(xi, 0.0, 1.0)
    alpha_array = alpha_max * ramp**2

    # Assign to function
    alpha.x.array[:] = alpha_array

    return alpha


def _has_non_homogeneous_neumann(bcs: BoundaryConditions) -> bool:
    for _, g in (*bcs.velocity_neumann, *bcs.pressure_neumann):
        if isinstance(g, dfem.Constant):
            if not np.allclose(g.value, 0.0):
                return True
        else:
            return True
    return False


def _has_non_homogeneous_robin(bcs: BoundaryConditions) -> bool:
    for _, _, g in bcs.robin_data:
        if isinstance(g, dfem.Constant):
            if not np.allclose(g.value, 0.0):
                return True
        else:
            return True
    return False
