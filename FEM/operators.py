"""LSA-FW FEM linearized operator assembly.

This module implements the variational forms and discrete assembly routines
for the linearized incompressible Navier-Stokes equations around a steady base flow.

It supports eigenvalue and time-dependent formulations using PETSc block matrices,
including mass, viscous, convection, shear, pressure, and divergence operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

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
import dolfinx.fem as dfem
import numpy as np

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

logger = logging.getLogger(__name__)  # TODO: add logs to linearized assembler


def _extract_bcs(bcs: BoundaryConditions | None) -> tuple[
    list[dfem.DirichletBC],
    list[dfem.DirichletBC],
    list[tuple[int, dfem.Constant]],
    list[tuple[int, dfem.Constant]],
    list[tuple[int, dfem.Constant, ExternalOperator]],
    list[dict[int, int]],
    list[dict[int, int]],
]:
    if bcs is None:
        return [], [], [], [], [], [], []
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
        bcs: BoundaryConditions | None,
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
        self._mat_cache: dict[str, iPETScMatrix] = {}
        self._vec_cache: dict[str, iPETScVector] = {}

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
    def bcs(self) -> tuple[dfem.DirichletBC]:
        return (*self._u_bcs, *self._p_bcs)

    @property
    def spaces(self) -> FunctionSpaces:
        """Get the function spaces used by this assembler."""
        return self._spaces

    @abstractmethod
    def get_matrix_forms(self) -> tuple[iPETScMatrix, iPETScVector]:
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
        bcs: BoundaryConditions | None = None,
        f: dfem.Function | None = None,
    ) -> None:
        """Initialize Stokes assembler."""
        super().__init__(spaces, bcs)
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
            A = assemble_matrix(self.jacobian, bcs=self.bcs)
            A.assemble()
            A_wrapper = iPETScMatrix(A)
            if not self._p_bcs:
                # Pin pressure DOF
                _, self._dofs_p = self._spaces.mixed.sub(1).collapse()
                A_wrapper.pin_dof(self._dofs_p[0])
            self._mat_cache[key_jac] = A_wrapper

        if key_res not in self._vec_cache:
            b = assemble_vector(self.residual)
            dfem.apply_lifting(b.array, [dfem.form(self._jacobian)], [self.bcs])
            self._apply_dirichlet(b)
            self._vec_cache[key_res] = iPETScVector(b)

        return self._mat_cache[key_jac], self._vec_cache[key_res]


class StationaryNavierStokesAssembler(BaseAssembler):
    """Finite element operator assembler for stationary Navier-Stokes equations."""

    def __init__(
        self,
        spaces: FunctionSpaces,
        re: float,
        *,
        f: dfem.Function | None = None,
        bcs: BoundaryConditions | None = None,
        initial_guess: dfem.Function | None = None,
        tags: MeshTags | None = None,
    ):
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
        convection = inner(dot(self._u, nabla_grad(self._u)), self._v) * dx
        diffusion = (1 / self._re) * inner(grad(self._u), grad(self._v)) * dx
        pressure = inner(self._p, div(self._v)) * dx + inner(div(self._u), self._q) * dx
        forcing = -inner(self._f, self._v) * dx

        form = convection + diffusion + pressure + forcing

        # If when reading this the pressure sign seems incorrect, refer to StokesAssembler._build_forms

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
            A = assemble_matrix(self._jacobian, bcs=self.bcs)
            A.assemble()
            A_wrapper = iPETScMatrix(A)
            if not self._p_bcs:
                # Pin pressure DOF
                _, self._dofs_p = self._spaces.mixed.sub(1).collapse()
                A_wrapper.pin_dof(self._dofs_p[-1])
            self._mat_cache[key_jac] = A_wrapper

        if key_res not in self._vec_cache:
            b = assemble_vector(self.residual)
            dfem.apply_lifting(b.array, [dfem.form(self._jacobian)], [self.bcs])
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
        return -inner(p, div(v)) * dx

    @staticmethod
    def viscous(u: Argument, v: Argument, re: float) -> Form:
        return (1.0 / re) * inner(grad(u), grad(v)) * dx

    @staticmethod
    def divergence(u: Argument, q: Argument) -> Form:
        return inner(div(u), q) * dx

    @staticmethod
    def stiffness(u: Argument, v: Argument) -> Form:
        return inner(grad(u), grad(v)) * dx


class LinearizedNavierStokesAssembler:
    """FEM assembler for the linearized Navier-Stokes operator around a stationary base flow."""

    def __init__(
        self,
        base_flow: dfem.Function,
        spaces: FunctionSpaces,
        re: float,
        *,
        bcs: BoundaryConditions | None = None,
        tags: MeshTags | None = None,
        use_sponge: bool = False,
    ) -> None:
        """Initialize."""
        if base_flow.function_space != spaces.mixed:
            raise ValueError(
                "Base flow must live on the same mixed space as the solution."
            )

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
        self._cache: dict[str, iPETScMatrix] = {}

        self._use_sponge = use_sponge
        self._re = re

        self._u_base = base_flow.sub(0)

        self._nullspace: iPETScNullSpace | None = None

        self._u, self._p = TrialFunctions(self._spaces.mixed)
        self._v, self._q = TestFunctions(self._spaces.mixed)

        _, self._dofs_u = self._spaces.mixed.sub(0).collapse()
        _, self._dofs_p = self._spaces.mixed.sub(1).collapse()

        self._num_dofs = len(self._dofs_p) + len(self._dofs_u)

        log_global(
            logger, logging.INFO, "Initialized linearized Navier-Stokes assembler."
        )

    @property
    def bcs(self) -> tuple[dfem.DirichletBC]:
        return (*self._u_bcs, *self._p_bcs)

    @property
    def spaces(self) -> FunctionSpaces:
        return self._spaces

    def assemble_linear_operator(
        self,
        *,
        key: str | int | None = None,
        cache: CacheStore | None = None,
    ) -> iPETScMatrix:
        """Assemble the full linear operator."""
        key = key or f"lin_ns_{id(self)}"
        if key not in self._cache:
            if cache is not None:
                loaded = cache.load_matrix(str(key))
                if loaded is not None:
                    self._cache[key] = iPETScMatrix(loaded)
                    return self._cache[key]

            log_rank(
                logger,
                logging.DEBUG,
                "Assembling linear operator - (%d, %d) DOFs",
                self._num_dofs,
                self._num_dofs,
            )

            shear = VariationalForms.shear(self._u, self._v, self._u_base)
            convection = VariationalForms.convection(self._u, self._v, self._u_base)
            viscous = VariationalForms.viscous(self._u, self._v, self._re)
            gradient = VariationalForms.pressure_gradient(self._p, self._v)
            divergence = VariationalForms.divergence(self._u, self._q)

            full_form = shear + convection + viscous + gradient + divergence

            # Note that no neumann is handled here, as by design, only homogeneous conditions can be
            # set for eigen-analyses

            if self._use_sponge:
                alpha = _get_damping_factor(self._spaces)
                full_form += -inner(alpha * self._u, self._v) * dx

            for marker, alpha, _ in self._robin_data:
                full_form += alpha * inner(self._u, self._v) * self._ds(marker)

            mat = assemble_matrix(dfem.form(full_form, dtype=Scalar), bcs=self.bcs)
            mat.assemble()
            A = iPETScMatrix(mat)

            if not self._p_bcs:
                self.attach_pressure_nullspace(A)
                log_rank(
                    logger,
                    logging.DEBUG,
                    "Attached pressure nullspace to linear operator",
                )

            for pmap in (*self._u_maps, *self._p_maps):
                apply_periodic_constraints(A, pmap)

            self._cache[key] = A
            if cache is not None:
                cache.save_matrix(str(key), A.raw)

        return self._cache[key]

    def assemble_mass_matrix(
        self,
        *,
        key: str | int | None = None,
        cache: CacheStore | None = None,
    ) -> iPETScMatrix:
        """Assemble the mass matrix."""
        key = key or f"mass_ns_{id(self)}"
        if key not in self._cache:
            if cache is not None:
                loaded = cache.load_matrix(str(key))
                if loaded is not None:
                    self._cache[key] = iPETScMatrix(loaded)
                    return self._cache[key]

            log_rank(
                logger,
                logging.DEBUG,
                "Assembling mass matrix - (%d, %d) DOFs",
                self._num_dofs,
                self._num_dofs,
            )

            mass_form = VariationalForms.mass(self._u, self._v)

            mat = assemble_matrix(dfem.form(mass_form, dtype=Scalar), bcs=self.bcs)
            mat.assemble()
            M = iPETScMatrix(mat)

            if not self._p_bcs:
                self.attach_pressure_nullspace(M)
                log_rank(
                    logger,
                    logging.DEBUG,
                    "Attached pressure nullspace to mass matrix",
                )

            for pmap in (*self._u_maps, *self._p_maps):
                apply_periodic_constraints(M, pmap)

            self._cache[key] = M
            if cache is not None:
                cache.save_matrix(str(key), M.raw)

        return self._cache[key]

    def assemble_eigensystem(
        self, *, cache: CacheStore | None = None
    ) -> tuple[iPETScMatrix, iPETScMatrix]:
        """Return (A, M) for eigenproblem; attach constant-pressure nullspace if needed."""
        A = self.assemble_linear_operator(cache=cache)
        M = self.assemble_mass_matrix(cache=cache)
        log_rank(
            logger,
            logging.INFO,
            "Successfully assembled eigensystem; with %d pressure DOFs and %d velocity DOFs.",
            len(self._dofs_p),
            len(self._dofs_u),
        )
        log_global(
            logger,
            logging.DEBUG,
            "Use 'extract_subblocks' to obtain the matrix blocks (vv, vp, pv, pp) if needed.",
        )
        return A, M

    def attach_pressure_nullspace(self, mat: iPETScMatrix) -> None:
        """Attach constant-pressure nullspace to a mixed-space matrix.

        In incompressible flow problems the pressure is determined only up to an additive constant, so the discrete
        operator admits a nullspace corresponding to u = 0, p = cnst.

        Explicitly attaching this nullspace is mandatory whenever the assembled system  remains singular - for example,
        if there are open (Neumann) or mixed velocity boundary conditions that do not fully constrain the velocity
        field.

        However, when velocity Dirichlet conditions are applied on the entire boundary, the velocity solution is fully
        determined and the discrete divergence operator becomes surjective. In that case the constant-pressure mode
        is implicitly removed, the system is rendered non-singular, and no explicit nullspace correction is needed.
        """
        if self._nullspace is None:
            arr = np.zeros((self._num_dofs,), dtype=Scalar)
            arr[self._dofs_p] = 1.0
            vec = iPETScVector.from_array(arr, comm=mat.comm)
            vec.scale(1 / vec.norm)

            self._nullspace = iPETScNullSpace.from_vectors([vec])

        self._nullspace.attach_to(mat)

    def clear_cache(self) -> None:
        """Clear cached PETSc matrices and vectors."""
        self._cache.clear()

    def extract_subblocks(self, mat: iPETScMatrix) -> iPETScBlockMatrix:
        """Extract and return (vv, vp, pv, pp) from any mixed-space matrix."""
        if self._dofs_u is None or self._dofs_p is None:
            _, self._dofs_u = self._spaces.mixed.sub(0).collapse()
            _, self._dofs_p = self._spaces.mixed.sub(1).collapse()

        is_v = PETSc.IS().createGeneral(self._dofs_u, comm=mat.comm)
        is_p = PETSc.IS().createGeneral(self._dofs_p, comm=mat.comm)

        vv = iPETScMatrix(mat.raw.createSubMatrix(is_v, is_v))
        vp = iPETScMatrix(mat.raw.createSubMatrix(is_v, is_p))
        pv = iPETScMatrix(mat.raw.createSubMatrix(is_p, is_v))
        pp = iPETScMatrix(mat.raw.createSubMatrix(is_p, is_p))

        return iPETScBlockMatrix(
            [
                [vv, vp],
                [pv, pp],
            ]
        )

    def test_nullspace(self, matrix: iPETScMatrix) -> None:
        """Debugging usage only."""
        if self._nullspace is None:
            raise RuntimeError("No nullspace has been defined yet.")
        is_ns, residual = self._nullspace.test_matrix(matrix)
        assert is_ns, f"Nullspace is not properly created, residual: {residual:.3f}"


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
