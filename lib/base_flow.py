"""LSA-FW Base (steady) flow computation.

This module provides functionality to compute the steady-state (base flow) solution of the incompressible Navier-Stokes
equations.
It solves for the non-dimensional velocity and pressure fields that satisfy the Navier-Stokes equations for a given
mesh, function spaces, and boundary conditions. Both Newton's method and Picard (fixed-point) iteration are supported
for handling the nonlinear convective term.

The solution obtained here (base flow) can be used in the linear stability analysis module as the base state about
which perturbations are analyzed.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from enum import StrEnum, auto

import ufl
from basix.ufl import mixed_element
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
from petsc4py import PETSc

from Meshing import Mesher

from FEM.spaces import FunctionSpaces
from FEM.bcs import BoundaryConditions
from FEM.utils import iPETScMatrix, iPETScVector


logger = logging.getLogger(__name__)


class NonlinearSolverType(StrEnum):
    """Nonlinear solver methods for the Navier-Stokes base flow."""

    NEWTON = auto()
    """Newton's method."""
    PICARD = auto()
    """Picard's iteration."""

    @classmethod
    def from_string(cls, value: str) -> NonlinearSolverType:
        """Get nonlinear solver type from a string."""
        try:
            return cls(value.lower().strip())
        except KeyError as e:
            raise ValueError(f"No solver type found for '{value}'.") from e


class LinearSolverType(StrEnum):
    """Linear solver methods for the base flow linear systems."""

    GMRES = auto()
    """Generalized Minimal Residual Method."""
    FGMRES = auto()
    """Flexible GMRES allowing variable preconditioning."""
    BICGSTAB = auto()
    """Stabilized BiConjugate Gradient."""
    TFQMR = auto()
    """Transpose-Free Quasi-Minimal Residual."""

    @classmethod
    def from_string(cls, value: str) -> LinearSolverType:
        try:
            return cls(value.lower().strip())
        except KeyError as e:
            raise ValueError(f"Invalid linear solver type '{value}'.") from e


class PreconditionerType(StrEnum):
    """Preconditioner types for the base flow linear systems."""

    HYPRE = auto()
    """Algebraic multigrid preconditioner from Hypre."""
    GAMG = auto()
    """PETSc built-in multigrid."""
    ILU = auto()
    """Incomplete LU factorization."""
    JACOBI = auto()
    """Jacobi (diagonal scaling)."""

    @classmethod
    def from_string(cls, value: str) -> PreconditionerType:
        try:
            return cls(value.lower().strip())
        except KeyError as e:
            raise ValueError(f"Invalid preconditioner type '{value}'.") from e


@dataclass
class BaseFlowSolverConfig:
    """Configuration for the base flow solver."""

    re: float
    """Reynolds number."""
    tol: float = 1e-8
    """Nonlinear residual tolerance."""
    max_iterations: int = 50
    """Maximum number of iterations."""
    nonlinear_solver_type: NonlinearSolverType = NonlinearSolverType.NEWTON
    """Nonlinear solver type (currently, only Newton or Piccard supported)."""
    linear_solver_type: LinearSolverType = LinearSolverType.GMRES
    """Linear solver type."""
    preconditioner_type: PreconditionerType = PreconditionerType.ILU
    """Preconditioner type."""


def solve_base_flow(
    mesher: Mesher,
    spaces: FunctionSpaces,
    bcs: BoundaryConditions,
    cfg: BaseFlowSolverConfig,
) -> tuple[dfem.Function, dfem.Function]:
    """Solve steady Navier-Stokes base flow via Newton or Picard.

    Note: Robin and Neumann BCs are currently not supported.
    """
    dof_break = _get_last_velocity_dof(spaces)
    ns = _get_null_space(dof_break, mesher, spaces)
    u, p, w = _initialize_functions(mesher.mesh, spaces)
    dirichlet = _transform_dirichlet_bc_to_mixed_space(mesher, bcs, w.function_space)
    ksp = _create_linear_solver(mesher, cfg)
    pressure_pin = _select_pressure_pin(spaces, bcs, mesher, dof_break)

    if cfg.nonlinear_solver_type == NonlinearSolverType.PICARD:
        u_conv = dfem.Function(spaces.velocity)
    else:
        u_conv = None

    logger.info(
        f"Starting ase flow solver: Re = {cfg.re}; method = {cfg.nonlinear_solver_type.name}."
    )

    for it in range(1, cfg.max_iterations + 1):
        if u_conv is not None:
            u_conv.x.petsc_vec.setArray(w.x.petsc_vec.getArray()[:dof_break])
            u_conv.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)

        F_form, J_form = _build_forms(cfg, w, u_conv)

        # (Re)build matrix
        A = iPETScMatrix(dfem.petsc.assemble_matrix(J_form, bcs=dirichlet))
        A.attach_nullspace(ns)
        A.pin_dof(pressure_pin)
        A.assemble()

        # (Re)build RHS *with* Dirichlet BCs enforced
        vec_b = dfem.petsc.create_vector(F_form)
        dfem.petsc.assemble_vector(vec_b, F_form)
        dfem.petsc.apply_lifting(vec_b, [J_form], [dirichlet], x0=[w.x.petsc_vec])
        dfem.petsc.set_bc(vec_b, dirichlet, x0=w.x.petsc_vec)
        b = iPETScVector(vec_b)

        res = b.norm

        logger.info(f"    [] iter {it:2d}: residual = {res:.3e}")

        if res < cfg.tol:
            logger.info(f"  [!] converged in {it} iterations.")
            break

        dx = b.duplicate()

        ksp.setOperators(A.raw)
        ksp.solve(b.raw, dx.raw)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f" [!] Linear solve failed: {ksp.getConvergedReason()}")

        logger.info(f"    [] linear iters = {ksp.getIterationNumber()}")

        w.x.petsc_vec.axpy(1.0, dx.raw)
        w.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.FORWARD
        )

    else:
        raise RuntimeError(
            f"Did not converge after {cfg.max_iterations} iterations (residual {res:.3e})."
        )

    u, p = _extract_functions_from_mixed(u, p, w, dof_break)

    return u, p


def _get_last_velocity_dof(spaces: FunctionSpaces) -> int:
    return spaces.velocity_dofs[0]


def _initialize_functions(
    mesh: dmesh.Mesh, spaces: FunctionSpaces
) -> tuple[dfem.Function, dfem.Function, dfem.Function]:
    u = dfem.Function(spaces.velocity, name="u_base")
    p = dfem.Function(spaces.pressure, name="p_base")

    mixed_elem = mixed_element(
        [spaces.velocity.ufl_element(), spaces.pressure.ufl_element()]
    )
    W = dfem.functionspace(mesh, mixed_elem)
    w = dfem.Function(W)

    w.x.petsc_vec.set(0.0)  # Set initial guess to 0
    w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)

    w.x.petsc_vec.setArray(np.r_[u.x.petsc_vec.getArray(), p.x.petsc_vec.getArray()])
    w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
    return u, p, w


def _transform_dirichlet_bc_to_mixed_space(
    mesher: Mesher,
    bcs: BoundaryConditions,
    mixed_space: dfem.FunctionSpace,
) -> list[dfem.DirichletBC]:
    velocity_space, _ = mixed_space.sub(0).collapse()
    pressure_space, _ = mixed_space.sub(1).collapse()

    facet_dim = mesher.mesh.topology.dim - 1

    mixed_space_bcs: list[dfem.DirichletBC] = []

    for marker, bc in bcs.velocity:
        facets = mesher.facet_tags.find(marker)
        dofs = dfem.locate_dofs_topological(mixed_space.sub(0), facet_dim, facets)
        g = dfem.Function(velocity_space)
        g.interpolate(bc.g)
        mixed_space_bcs.append(dfem.dirichletbc(g, dofs))

    for marker, bc in bcs.pressure:
        facets = mesher.facet_tags.find(marker)
        dofs = dfem.locate_dofs_topological(mixed_space.sub(1), facet_dim, facets)
        g = dfem.Function(pressure_space)
        g.interpolate(bc.g)
        mixed_space_bcs.append(dfem.dirichletbc(g, dofs))

    return mixed_space_bcs


def _create_linear_solver(
    mesher: Mesher, config: BaseFlowSolverConfig, rtol: float = 1e-6, atol: float = 1e-8
) -> PETSc.KSP:
    ksp = PETSc.KSP().create(mesher.mesh.comm)
    ksp.setType(config.linear_solver_type.name.lower())
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.getPC().setType(config.preconditioner_type.name.lower())
    return ksp


def _get_null_space(
    nu: int, mesher: Mesher, spaces: FunctionSpaces
) -> PETSc.MatNullSpace:
    nv_loc, nv_glob = spaces.velocity_dofs
    np_loc, np_glob = spaces.pressure_dofs
    vec = PETSc.Vec().createMPI(
        nv_loc + np_loc, nv_glob + np_glob, comm=mesher.mesh.comm
    )
    start, end = vec.getOwnershipRange()
    p0, p1 = nv_glob, nv_glob + np_glob
    owned_pressure = range(max(start, p0), min(end, p1))
    if owned_pressure:
        vec.setValues(
            list(owned_pressure),
            [1.0] * len(owned_pressure),
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
    vec.assemblyBegin()
    vec.assemblyEnd()
    norm = vec.norm()
    vec.scale(1.0 / norm if norm > 0 else 1.0)
    return PETSc.NullSpace().create(vectors=[vec], comm=mesher.mesh.comm)


def _build_forms(
    cfg: BaseFlowSolverConfig,
    w: dfem.Function,
    u_conv: dfem.Function | None,
) -> tuple[dfem.Form, ufl.Form]:
    dw = ufl.TrialFunction(w.function_space)
    vq = ufl.TestFunction(w.function_space)
    v, q = ufl.split(vq)
    u_, p_ = ufl.split(w)

    if u_conv is not None:
        conv = ufl.inner(ufl.dot(u_conv, ufl.grad(u_)), v) * ufl.dx
    else:
        conv = ufl.inner(ufl.dot(u_, ufl.grad(u_)), v) * ufl.dx

    diff = (1.0 / cfg.re) * ufl.inner(ufl.grad(u_), ufl.grad(v)) * ufl.dx
    pres = -ufl.inner(p_, ufl.div(v)) * ufl.dx
    cont = ufl.inner(q, ufl.div(u_)) * ufl.dx

    F = conv + diff + pres + cont

    return dfem.form(F), dfem.form(ufl.derivative(F, w, dw))


def _select_pressure_pin(
    spaces: FunctionSpaces, bcs: BoundaryConditions, mesher: Mesher, nu: int
) -> int:
    if bcs.pressure:
        marker, _ = bcs.pressure[0]
        facet_dim = mesher.mesh.topology.dim - 1
        facets = mesher.facet_tags.find(marker)
        p_dofs = dfem.locate_dofs_topological(spaces.pressure, facet_dim, facets)
        if p_dofs.size > 0:
            return nu + int(p_dofs[0])

    return nu  # Fallback: pin the very first pressure DOF


def _extract_functions_from_mixed(
    u: dfem.Function, p: dfem.Function, w: dfem.Function, nu: int
) -> tuple[dfem.Function, dfem.Function]:
    u.x.petsc_vec.setArray(w.x.petsc_vec.getArray()[:nu])
    p.x.petsc_vec.setArray(w.x.petsc_vec.getArray()[nu:])
    u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
    p.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
    return u, p
