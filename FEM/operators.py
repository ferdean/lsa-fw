"""LSA-FW FEM linearized operator assembly.

This module implements the variational forms and discrete assembly routines
for the linearized incompressible Navier-Stokes equations around a steady base flow.

It supports eigenvalue and time-dependent formulations using PETSc block matrices,
including mass, viscous, convection, shear, pressure, and divergence operators.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import dolfinx.fem as dfem
import numpy as np
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
)
from petsc4py import PETSc
from ufl import (  # type: ignore[import-untyped]
    Form,
    Measure,
    TestFunctions,
    TrialFunctions,
    derivative,
    div,
    dot,
    dx,
    grad,
    inner,
    nabla_grad,
    split,
    conj,
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
    iPETScVector,
    Scalar,
)

logger = logging.getLogger(__name__)  # TODO: add logs to linearized assembler


def _extract_bcs(bcs: BoundaryConditions | None) -> tuple[
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


class BaseAssembler(ABC):
    """Abstract base class for finite element operator assemblers."""

    def __init__(self, spaces: FunctionSpaces, bcs: BoundaryConditions | None) -> None:
        """Initialize base assembler."""
        (
            self._u_bcs,
            self._p_bcs,
            self._neumann_forms,
            self._robin_forms,
            self._uaps,
            self._paps,
        ) = _extract_bcs(bcs)
        self._spaces = spaces
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

        # TODO: check handling of neumann and robin conditions
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

    def _build_forms(self) -> tuple[Form, Form]:
        """Construct bilinear and linear UFL forms for the Stokes problem."""

        a = (
            inner(grad(self._u), grad(conj(self._v))) * dx
            + inner(self._p, div(conj(self._v))) * dx
            + inner(div(self._u), conj(self._q)) * dx
        )

        L = inner(self._f, conj(self._v)) * dx

        # Note that the pressure term is deliberately signed to yield symmetry;
        # and that test-function occurrences are conjugated for complex support

        for bc in self._neumann_forms:
            L += bc

        return a, L

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
            dfem.apply_lifting(b.array, [self.jacobian], [self.bcs])
            self._apply_dirichlet(b)
            self._vec_cache[key_res] = iPETScVector(b)

        return self._mat_cache[key_jac], self._vec_cache[key_res]


class StationaryNavierStokesAssembler(BaseAssembler):
    """Finite element operator assembler for stationary Navier-Stokes equations."""

    def __init__(
        self,
        spaces: FunctionSpaces,
        re: float,
        f: dfem.Function | None = None,
        bcs: BoundaryConditions | None = None,
        initial_guess: dfem.Function | None = None,
    ):
        """Initialize."""
        super().__init__(spaces, bcs)
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

        # TODO: handle robin conditions
        # TODO: handle periodic constraints

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
        convection = inner(dot(self._u, nabla_grad(self._u)), conj(self._v)) * dx
        diffusion = (1 / self._re) * inner(grad(self._u), grad(conj(self._v))) * dx
        pressure = (
            inner(self._p, div(conj(self._v))) * dx
            + inner(div(self._u), conj(self._q)) * dx
        )
        forcing = -inner(self._f, conj(self._v)) * dx

        form = convection + diffusion + pressure + forcing

        # If when reading this the pressure sign seems incorrect, refer to StokesAssembler._build_forms

        for bc in self._neumann_forms:
            form += bc

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
                A_wrapper.pin_dof(self._dofs_p[0])
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
        return inner(u, conj(v)) * dx

    @staticmethod
    def convection(u: Argument, v: Argument, u_base: dfem.Function) -> Form:
        return inner(dot(u_base, nabla_grad(u)), conj(v)) * dx

    @staticmethod
    def shear(u: Argument, v: Argument, u_base: dfem.Function) -> Form:
        return inner(dot(u, grad(u_base)), conj(v)) * dx

    @staticmethod
    def pressure_gradient(p: Argument, v: Argument) -> Form:
        return -inner(p, div(conj(v))) * dx

    @staticmethod
    def viscous(u: Argument, v: Argument, re: float) -> Form:
        return (1.0 / re) * inner(grad(u), grad(conj(v))) * dx

    @staticmethod
    def divergence(u: Argument, q: Argument) -> Form:
        return inner(div(u), conj(q)) * dx

    @staticmethod
    def neumann_rhs(v: Argument, g: dfem.Function, ds: Measure) -> Form:
        return inner(conj(v), g) * ds

    @staticmethod
    def stiffness(u: Argument, v: Argument) -> Form:
        return inner(grad(u), grad(conj(v))) * dx


class LinearizedNavierStokesAssembler:
    """FEM assembler for the linearized Navier-Stokes operator around a stationary base flow."""

    def __init__(
        self,
        base_flow: dfem.Function,
        spaces: FunctionSpaces,
        re: float,
        bcs: BoundaryConditions | None = None,
    ) -> None:
        """Initialize."""
        if base_flow.function_space != spaces.mixed:
            raise ValueError(
                "Base flow must live on the same mixed space as the solution."
            )

        (
            self._u_bcs,
            self._p_bcs,
            self._neumann_forms,
            self._robin_forms,
            self._uaps,
            self._paps,
        ) = _extract_bcs(bcs)

        self._u_base = base_flow.sub(0)
        self._spaces = spaces
        self._re = re
        self._cache: dict[str, iPETScMatrix] = {}
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

            for pmap in (*self._uaps, *self._paps):
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

            for pmap in (*self._uaps, *self._paps):
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
            arr = np.zeros((len(self._dofs_u + self._dofs_p),), dtype=Scalar)
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
