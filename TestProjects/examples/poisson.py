"""Simple Poisson problem using DOLFINx."""

import numpy as np
import ufl  # type: ignore[import]
from mpi4py.MPI import COMM_WORLD
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from time import perf_counter_ns


def solve_poisson(
    nx: int = 10,
    ny: int = 10,
    f_value: float = 1.0,
    u_dirichlet: float = 0.0,
    petsc_options: dict | None = None,
) -> fem.Function:
    """Solve the Poisson problem.

    -Δu = f on a unit square with Dirichlet BC u = u_D on x=0.

    Args:
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        f_value: Constant source term
        u_dirichlet: Dirichlet boundary value
        petsc_options (optional): PETSc solver options

    Returns:
        The solution function uh
    """
    start_time = perf_counter_ns()

    domain = mesh.create_unit_square(COMM_WORLD, nx, ny, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, np.float64(f_value))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    u_const = fem.Constant(domain, np.array(u_dirichlet, dtype=np.float64))
    boundary_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = fem.dirichletbc(u_const, boundary_dofs, V)

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options=petsc_options or {"ksp_type": "cg"}
    )
    uh = problem.solve()

    end_time = perf_counter_ns()

    if COMM_WORLD.rank == 0:
        print(
            f"        solution found in {(end_time - start_time)/1_000_000:.4f} ms.\n"
        )

    return uh
