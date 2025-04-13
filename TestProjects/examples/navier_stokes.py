"""Solve a simplified 2D Navier–Stokes driven lid flow using DOLFINx and visualize results.

The problem simulates incompressible flow described by the Navier–Stokes equations
on a unit square domain. The top boundary (lid) moves horizontally, driving the fluid motion.
The bottom boundary remains stationary (no-slip), and side boundaries have no explicit boundary conditions,
representing open or symmetry boundaries. The fluid is initially at rest.
"""

import numpy as np
import pyvista
from time import perf_counter_ns
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from ufl import grad, inner, dx  # type: ignore[import]
from pathlib import Path
import ufl
from dolfinx.plot import vtk_mesh


def run_navier_stokes():
    """Run the Navier–Stokes driven lid solver on a unit square."""
    start_time = perf_counter_ns()

    # Mesh and function space:
    #   - 32x32 unit square mesh with triangular elements
    #   - mesh distributed across MPI ranks
    #   - function space defined with Lagrange elements (continuous piecewise polynomials) of degree 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32, mesh.CellType.triangle)
    W = fem.functionspace(domain, ("Lagrange", 2, (2,)))

    # Physical parameters
    T, dt = 2.0, 0.05  # simulation duration and timestep
    nu = 1e-2  # kinematic viscosity of fluid
    rho = 1.0  # fluid density

    # Functions
    u = fem.Function(W)  # current velocity solution (unknown)
    u_n = fem.Function(W)  # previous velocity solution (known)
    u.name, u_n.name = "u", "u_n"

    # Test function and forcing term (no external forces applied)
    v = ufl.TestFunction(W)
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))

    # Boundary conditions setup
    def _bottom_wall(x):
        return np.isclose(x[1], 0.0)

    def _top_wall(x):
        return np.isclose(x[1], 1.0)

    # Extract scalar function space from the first component of the vector space (horizontal velocity)
    # Useful to define scalar boundary conditions on a single velocity component
    V0_collapse, _ = W.sub(0).collapse()

    # No-slip condition at the bottom wall (velocity = 0)
    u_bottom = fem.Function(V0_collapse)
    u_bottom.x.array[:] = 0.0
    bc_bottom = fem.dirichletbc(
        u_bottom,
        fem.locate_dofs_geometrical((W.sub(0), V0_collapse), _bottom_wall),
        W.sub(0),
    )

    # Moving lid condition at the top wall (horizontal velocity = 10)
    u_top = fem.Function(V0_collapse)
    u_top.x.array[:] = 10.0
    bc_top = fem.dirichletbc(
        u_top,
        fem.locate_dofs_geometrical((W.sub(0), V0_collapse), _top_wall),
        W.sub(0),
    )

    bcs = [bc_bottom, bc_top]

    # Weak form residual and Jacobian
    F = (
        rho * inner((u - u_n) / dt, v) * dx
        + rho * inner(grad(u) * u, v) * dx
        + nu * inner(grad(u), grad(v)) * dx
        - inner(f, v) * dx
    )
    J = fem.form(ufl.derivative(F, u))

    # Nonlinear problem and Newton solver setup
    problem = NonlinearProblem(fem.form(F), u, bcs=bcs, J=J)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = 1e-6

    # Output to XDMF for visualization
    xdmf_path = Path("results/ns_driven_lid.xdmf")
    xdmf_path.parent.mkdir(parents=True, exist_ok=True)
    xdmf = io.XDMFFile(MPI.COMM_WORLD, xdmf_path, "w")
    xdmf.write_mesh(domain)

    # Time-stepping loop
    t = 0.0
    while t < T + 1e-8:
        solver.solve(u)
        u_n.x.array[:] = u.x.array

        geom_deg = domain.geometry.cmap.degree
        scalar_space = fem.functionspace(domain, ("Lagrange", geom_deg))
        for i in range(2):
            u_comp = fem.Function(scalar_space)
            u_comp.interpolate(u.sub(i))
            u_comp.name = f"u{i}"
            xdmf.write_function(u_comp, t)

        if MPI.COMM_WORLD.rank == 0:
            print(f"        [t = {t:.2f}] step complete")
        t += dt

    xdmf.close()
    end_time = perf_counter_ns()

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"\n        simulation completed in {(end_time - start_time) / 1e6:.4f} ms.\n"
        )

        topology, cell_types, points = vtk_mesh(W)
        grid = pyvista.UnstructuredGrid(topology, cell_types, points)
        grid.point_data["u"] = np.linalg.norm(u.x.array.reshape((-1, 2)), axis=1)

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, scalars="u", cmap="viridis")
        plotter.add_title("Driven Lid Flow - Velocity Magnitude")
        plotter.show()


if __name__ == "__main__":
    run_navier_stokes()
