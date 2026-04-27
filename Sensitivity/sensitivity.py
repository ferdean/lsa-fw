"""LSA-FW sensitivity and receptivity analysis."""

import math
from ufl import inner, dx, sqrt, TrialFunction, TestFunction
from ufl.core.expr import Expr
import dolfinx.fem as dfem
import dolfinx.fem.petsc as fem_petsc
from petsc4py import PETSc
from FEM.utils import iPETScMatrix


def _get_velocity_subspace_idx(mixed_space: dfem.FunctionSpace) -> tuple[int, int]:
    """Get the index of the velocity subspace.

    This function is used to be able to not just assume that the given functions are built
    in a velocity-pressure mixed space.
    """
    try:
        n_subspaces = mixed_space.num_sub_spaces
    except AttributeError:
        raise RuntimeError("Expected a mixed (velocity-pressure) function space.")

    if n_subspaces is None or n_subspaces < 2:
        raise RuntimeError("Function space is not mixed or has < 2 subspaces.")

    # Identify velocity and pressure subspaces by their value shape
    mesh_dim = mixed_space.mesh.topology.dim
    candidates = []
    for idx in range(n_subspaces):
        subspace = mixed_space.sub(idx)
        shp = subspace.value_shape
        candidates.append((idx, shp))

    vel_idx = None
    prs_idx = None

    for idx, shp in candidates:
        if shp == (mesh_dim,):  # vector-valued DoFs
            vel_idx = idx
        elif shp == ():  # scalar
            prs_idx = idx
    if vel_idx is None or prs_idx is None:
        raise RuntimeError(
            f"Could not identify (velocity, pressure) subspaces. "
            f"Found shapes: {candidates} with mesh_dim={mesh_dim}."
        )
    return vel_idx, prs_idx


def _project_to(
    expr: Expr,
    function_space: dfem.FunctionSpace,
    *,
    bcs: list[dfem.DirichletBC] | None = None,
) -> dfem.Function:
    """Project a UFL expression onto the given function space (generic L2 projection)."""
    bcs = bcs or []

    u = TrialFunction(function_space)
    v = TestFunction(function_space)

    a = inner(u, v) * dx
    L = inner(expr, v) * dx

    A = fem_petsc.assemble_matrix(dfem.form(a), bcs=bcs)
    A.assemble()

    b = fem_petsc.assemble_vector(dfem.form(L))
    if bcs:
        fem_petsc.apply_lifting(b, [dfem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    if bcs:
        fem_petsc.set_bc(b, bcs)

    uh = dfem.Function(function_space)

    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    pc = ksp.getPC()
    try:
        pc.setType("hypre")
    except PETSc.Error:
        pc.setType("jacobi")

    ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=2000)
    ksp.setFromOptions()

    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    return uh


def compute_receptivity(
    A: iPETScMatrix, M: iPETScMatrix, *, target_eigenvalue: complex | None = None
) -> dfem.Function:
    raise NotImplementedError("compute_receptivity is not yet implemented.")


def compute_receptivity_from_modes(
    direct_mode: dfem.Function, adjoint_mode: dfem.Function
) -> dfem.Function:
    """Compute the structural sensitivity field, given the direct and adjoint modes.

    Refer to the documentation for further details on the definition of this field (also
    known as 'wavemaker').
    """
    if direct_mode.function_space != adjoint_mode.function_space:
        raise ValueError("Direct/adjoint functions are defined on different spaces.")

    vel_subspace_idx, prs_subspace_idx = _get_velocity_subspace_idx(
        direct_mode.function_space
    )
    prs_subspace, _ = direct_mode.function_space.sub(prs_subspace_idx).collapse()

    direct_vel_mode = direct_mode.sub(vel_subspace_idx)
    adjoint_vel_mode = adjoint_mode.sub(vel_subspace_idx)

    denominator = dfem.assemble_scalar(
        dfem.form(inner(adjoint_vel_mode, direct_vel_mode) * dx)
    )
    if math.isclose((denominator_abs := abs(complex(denominator))), 0):
        raise RuntimeError(
            "Receptivity denominator is close to zero (potential zero-division errors). "
            "Hint: check for normalization issues."
        )

    receptivity_expr = (
        sqrt(inner(adjoint_vel_mode, adjoint_vel_mode))
        * sqrt(inner(direct_vel_mode, direct_vel_mode))
    ) / denominator_abs

    receptivity = _project_to(receptivity_expr, prs_subspace)

    maxval = receptivity.x.petsc_vec.norm(PETSc.NormType.NORM_INFINITY)
    if maxval > 0.0:
        receptivity.x.petsc_vec.scale(1.0 / maxval)
        receptivity.x.scatter_forward()

    receptivity_fnc = dfem.Function(direct_mode.function_space)
    with receptivity_fnc.x.petsc_vec.localForm() as lf:
        lf.set(0.0)
    receptivity_fnc.sub(prs_subspace_idx).interpolate(  # Project to scalar space
        receptivity
    )
    receptivity_fnc.x.scatter_forward()
    return receptivity_fnc
