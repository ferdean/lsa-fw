"""Vibrating membrane benchmark.

Solve the vibrating membrane (Dirichlet eigenvalue problem for the Laplacian on a rectangle) using the finite
element method, and validate it against the analytical solution.

This script reuses meshing, boundary condition, and solver components from LSA-FW as-is.
Because LSA-FW is tailored for Navier-Stokes, some helper functions add minimal adaptation (e.g., defining a scalar
H1 space) to apply the framework to the membrane benchmark.
"""

from pathlib import Path
from typing import Final, Sequence, cast

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import matplotlib.pyplot as plt
import numpy as np
from basix.ufl import element
from dolfinx.fem.petsc import assemble_matrix
from ufl import TestFunction, TrialFunction

from config import load_bc_config, load_facet_config, BoundaryConditionsConfig
from FEM.bcs import (
    BoundaryConditionType,
    _wrap_constant_vector,  # HACK: no private methods should be used
)
from FEM.operators import VariationalForms
from FEM.spaces import iElementFamily
from FEM.utils import iPETScMatrix
from Meshing.core import Mesher
from Meshing.plot import plot_mesh
from Meshing.utils import Shape, iCellType
from Solver.eigen import EigenSolver, EigensolverConfig, iEpsProblemType
from Solver.utils import iEpsWhich

_A: Final[float] = 2.0
_B: Final[float] = 4.0
_NUM_EIG: Final[int] = 15
_MESH_DIVS: Final[tuple[int, int]] = (32, 32)
_CONFIG_DIR: Final[Path] = Path(__file__).parent / "vibrating_membrane"

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def get_mesher(nxy: tuple[int, int], *, show_plot: bool = False) -> Mesher:
    """Generate a rectangular mesh with LSA-FW Mesher and apply boundary tags."""
    mesher = Mesher(
        shape=Shape.BOX,
        n=nxy,
        cell_type=iCellType.TRIANGLE,
        domain=((0.0, 0.0), (_A, _B)),
    )
    mesh = mesher.generate()
    tags = load_facet_config(_CONFIG_DIR / "mesh_tags.toml")
    mesher.mark_boundary_facets(tags)
    if show_plot:
        plot_mesh(mesh, tags=mesher.facet_tags)
    return mesher


def get_function_space(mesh: dmesh.Mesh) -> dfem.FunctionSpace:
    """Create the vector function space for the membrane benchmark.

    Wraps the degree-1 Lagrange element definition from LSA-FW FEM. The framework's original spaces target Navier-Stokes
    variables, so this helper constructs the appropriate space for the membrane.
    """
    basix_el = element(
        family=iElementFamily.LAGRANGE.to_dolfinx(),
        cell=iCellType.TRIANGLE.to_basix(),
        degree=2,
        shape=(),
    )
    return dfem.functionspace(mesh, basix_el)


def define_velocity_bcs(
    mesh: dmesh.Mesh,
    V: dfem.FunctionSpace,
    tags: dmesh.MeshTags,
    configs: Sequence[BoundaryConditionsConfig],
) -> list[dfem.DirichletBC]:
    """Assemble homogeneous Dirichlet velocity BCs on V for the vibrating membrane.

    Uses LSA-FW FEM bcs definitions.
    """
    dim = mesh.topology.dim
    bcs: list[dfem.DirichletBC] = []
    for cfg in configs:
        marker = cfg.marker
        facets = tags.find(marker)
        if (  # TODO: check this in main code
            cfg.type == BoundaryConditionType.DIRICHLET_VELOCITY.value
        ):
            fn = dfem.Function(V)
            interp = (
                cfg.value if callable(cfg.value) else _wrap_constant_vector(cfg.value)
            )
            fn.interpolate(interp)

            dofs = dfem.locate_dofs_topological(V, dim - 1, facets)
            bcs.append(dfem.dirichletbc(fn, dofs))

    return bcs


def assemble_eigensystem(
    V: dfem.FunctionSpace,
    bcs: list[dfem.DirichletBC],
) -> tuple[iPETScMatrix, iPETScMatrix]:
    """Assemble the mass and stiffness matrices for the membrane eigenproblem.

    Leverages VariationalForms.mass and VariationalForms.stiffness from LSA-FW FEM.
    Mocks the original LSA-FW FEM assembly process, which is designed for Navier-Stokes problems.
    Dirichlet BCs are applied to enforce zero displacements on the boundary.
    """
    u = TrialFunction(V)
    v = TestFunction(V)

    def _assemble(form):
        mat = assemble_matrix(dfem.form(form), bcs=bcs)
        return iPETScMatrix(mat)

    M = _assemble(VariationalForms.mass(u, v))
    A = _assemble(VariationalForms.stiffness(u, v))

    M.assemble()
    A.assemble()

    return M, A


def get_analytic_eigenvalues(num: int, a: float = _A, b: float = _B) -> list[float]:
    """Compute the first num analytic eigenvalues for a rectangular membrane.

    Computes lambda_mn = pi^2*(m^2/a^2 + n^2/b^2) for m,n >= 1. Refer to the documentation for further details.
    """
    vals: list[float] = []
    limit = int(np.ceil(np.sqrt(num) * 1.5))
    for m in range(1, limit + 1):
        for n in range(1, limit + 1):
            vals.append((np.pi**2) * (m**2 / a**2 + n**2 / b**2))
    vals.sort()
    return vals[:num]


def _run_case(nxy: tuple[int, int]) -> tuple[list[float], list[float]]:
    mesher = get_mesher(nxy)
    V = get_function_space(mesher.mesh)
    bc_configs = load_bc_config(_CONFIG_DIR / "bcs.toml")

    # mesher.facet_tags is Optional in the Mesher API; we mark and require it for this benchmark.
    assert (
        mesher.facet_tags is not None
    ), "Facet tags must be defined (mark_boundary_facets must have run)."
    tags: dmesh.MeshTags = cast(dmesh.MeshTags, mesher.facet_tags)

    bcs = define_velocity_bcs(mesher.mesh, V, tags, bc_configs)

    M, A = assemble_eigensystem(V, bcs)

    cfg = EigensolverConfig(
        problem_type=iEpsProblemType.GHEP,
        num_eig=_NUM_EIG + 4,  # Extra to filter out spurious modes
        atol=1e-8,
        max_it=1000,
    )
    solver = EigenSolver(cfg, A, M)
    solver.solver.set_which_eigenpairs(iEpsWhich.SMALLEST_REAL)
    numerical = solver.solve()

    # Note: we are seeing spurious lambda \approx 1 modes because applying homogeneous Dirichlet BCs replaces
    # boundary‐dof rows/cols with identity, which analytically gives 1·x = lambda·1·x -> lambda=1 for each pinned dof.
    # These are not physical membrane modes and are thus dropped
    filtered = [(ev, vec) for ev, vec in numerical if abs(ev.real - 1.0) > cfg.atol]

    numerical_vals = np.array(
        [float(ev.real) for ev, _ in filtered][:_NUM_EIG]
    ).tolist()
    analytic_vals = get_analytic_eigenvalues(_NUM_EIG)

    return numerical_vals, analytic_vals


def run_default() -> None:
    """Execute the FEM benchmark and compare numerical vs analytic eigenvalues."""

    numerical, analytic = _run_case(_MESH_DIVS)

    print("\nBenchmark - Vibrating membrane:\n")
    print(f"\t2D domain: [0.0, {_A}] x [0.0, {_B}], mesh divisions: {_MESH_DIVS}")
    print(f"\tNumber of eigenvalues computed: {_NUM_EIG}")
    print(f"\tAnalytical eigenvalues: lambda_mn = pi^2*(m^2/{_A}^2 + n^2/{_B}^2)\n")

    print(f"\t{'mode':>4}   {'lambda_num':>12}   {'lambda_ana':>12}   {'rel_err':>10}")
    for i, (num_val, ana_val) in enumerate(zip(numerical, analytic), start=1):
        rel_err = abs(num_val - ana_val) / ana_val
        print(f"\t{i:4d}   {num_val:12.6e}   {ana_val:12.6e}   {rel_err:10.2e}")
    print(
        "\n\t[!] Average relative error:"
        f" {np.mean([abs(num_val - ana_val) / ana_val for num_val, ana_val in zip(numerical, analytic)]):.2e}\n"
    )


def run_convergence_analysis(resolutions: list[tuple[int, int]] | None = None) -> None:
    """Run the vibrating membrane benchmark across mesh resolutions and plot convergence.

    This routine validates the correctness of the FEM implementation (mesh, space, assembly, solver) by checking how
    the numerical eigenvalue converges to the known analytical solution as the mesh is refined.

    The expected convergence rate for quadratic (P2) elements solving the Laplacian eigenvalue problem is 4 (i.e., the
    error decreases as h^4). Any deviation from this indicates issues in the implementation (e.g., incorrect assembly,
    boundary condition mishandling, or solver configuration).
    """
    resolutions = resolutions or [(8, 8), (16, 16), (32, 32), (64, 64), (96, 96)]

    hs: list[float] = []
    rel_errors: list[float] = []

    print("\nBenchmark - Vibrating membrane (convergence analysis):\n")
    print("\tQuadratic elements (P2, Lagrange)")
    print(f"\tAnalytical eigenvalues: lambda_mn = pi^2*(m^2/{_A}^2 + n^2/{_B}^2)\n")

    for nxy in resolutions:
        print(f"\t[>] Running mesh {nxy}...")

        numerical, analytic = _run_case(nxy)
        rel_err = float(
            np.mean(
                [
                    abs(num_val - ana_val) / ana_val
                    for num_val, ana_val in zip(numerical, analytic)
                ]
            )
        )
        print(f"\t        rel_err = {rel_err:.2e}")

        hs.append(1 / max(nxy))
        rel_errors.append(rel_err)

    # Keep original lists as lists; create arrays for vectorized math to avoid
    # reassigning a different type to list-typed variables (mypy clean).
    hs_arr = np.array(hs, dtype=float)
    rel_errors_arr = np.array(rel_errors, dtype=float)

    # Generate reference line: e_ref = C * h^4
    h_ref = np.array([hs_arr[0], hs_arr[-1]], dtype=float)
    e_ref = rel_errors_arr[-2] * (h_ref / hs_arr[-2]) ** 4

    plt.figure(figsize=(6, 4))
    plt.loglog(hs_arr, rel_errors_arr, marker="o", linestyle="-", color="k")
    plt.loglog(h_ref, e_ref, linestyle="--", color="r", label="$h^4$")

    plt.xlabel(r"Mesh size (-)")
    plt.ylabel(r"Relative error (-)")
    plt.grid(True, which="both", ls="--")
    plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        Path(__file__).parent
        / "vibrating_membrane"
        / "vibrating_membrane_convergence.png",
        dpi=300,
    )

    print(
        "\n\tConvergence plot saved as 'vibrating_membrane/vibrating_membrane_convergence.png'."
    )


if __name__ == "__main__":
    run_default()
    run_convergence_analysis([(8, 8), (16, 16), (32, 32), (64, 64), (96, 96)])
