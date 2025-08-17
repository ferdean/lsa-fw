"""Unit tests for Solver.linear (LinearSolver)."""

from pathlib import Path

import numpy as np
import pytest

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
from mpi4py import MPI

from config import BoundaryConditionsConfig
from FEM.bcs import (
    BoundaryConditions,
    BoundaryConditionType,
    define_bcs,
)
from FEM.spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from FEM.operators import StokesAssembler
from Meshing.core import Mesher
from Meshing.utils import Shape, iCellType
from Solver.linear import LinearSolver


@pytest.fixture(scope="module")
def test_mesher() -> Mesher:
    """Define a small 2D test mesher on the unit square."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(12, 12), cell_type=iCellType.TRIANGLE)
    _ = mesher.generate()
    mesher.mark_boundary_facets(lambda x: 1)  # Mark all boundary facets as '1'
    return mesher


@pytest.fixture(scope="module")
def test_mesh(test_mesher: Mesher) -> dmesh.Mesh:
    """Provide the mesh."""
    return test_mesher.mesh


@pytest.fixture(scope="module")
def test_spaces(test_mesh: dmesh.Mesh) -> FunctionSpaces:
    """Taylor-Hood spaces."""
    return define_spaces(test_mesh, type=FunctionSpaceType.TAYLOR_HOOD)


@pytest.fixture(scope="module")
def test_bcs(test_mesher: Mesher, test_spaces: FunctionSpaces) -> BoundaryConditions:
    """Homogeneous Dirichlet velocity on all boundaries; leave pressure unconstrained (assembler pins one DOF)."""
    configs = [
        BoundaryConditionsConfig(
            marker=1,
            type=BoundaryConditionType.DIRICHLET_VELOCITY,
            value=(0.0, 0.0),
        )
    ]
    return define_bcs(test_mesher, test_spaces, configs)


@pytest.fixture
def stokes_assembler_rhs0(
    test_spaces: FunctionSpaces, test_bcs: BoundaryConditions
) -> StokesAssembler:
    """Stokes assembler with zero body force (RHS == 0)."""
    return StokesAssembler(spaces=test_spaces, bcs=test_bcs)


@pytest.fixture
def stokes_assembler_rhs(
    test_spaces: FunctionSpaces, test_bcs: BoundaryConditions
) -> StokesAssembler:
    """Stokes assembler with nonzero body force to guarantee a nontrivial RHS."""
    f = dfem.Constant(test_spaces.velocity.mesh, (1.0, 0.0))
    return StokesAssembler(spaces=test_spaces, bcs=test_bcs, f=f)


class TestLinearSolverSciPy:
    """SciPy direct factorization path."""

    @pytest.mark.skipif(
        MPI.COMM_WORLD.size > 1, reason="SciPy direct runs only on rank 0 in serial."
    )
    def test_direct_scipy_solve_smoke(
        self, stokes_assembler_rhs: StokesAssembler
    ) -> None:
        """Solve with SciPy LU; solution should be finite and nonzero for a nontrivial RHS."""
        solver = LinearSolver(stokes_assembler_rhs)
        sol = solver.direct_scipy_solve(show_plot=False, key="scipy_rhs")
        arr = sol.x.array.copy()

        assert np.all(np.isfinite(arr))
        assert np.linalg.norm(arr) > 0.0, "Expected nonzero solution for nontrivial RHS"

    @pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Serial-only check.")
    def test_direct_scipy_caches_factor(
        self, stokes_assembler_rhs: StokesAssembler
    ) -> None:
        """Second call with the same key should reuse assembled A, b and the LU factor."""
        solver = LinearSolver(stokes_assembler_rhs)
        _ = solver.direct_scipy_solve(show_plot=False, key="cache")
        # Call again with same key (should hit _ab_cache and _lu_cache internally)
        sol2 = solver.direct_scipy_solve(show_plot=False, key="cache")
        assert sol2 is stokes_assembler_rhs.sol  # Same Function object is updated


class TestLinearSolverPETSc:
    """PETSc KSP path (GMRES default)."""

    def test_gmres_converges_and_records_residuals(
        self, stokes_assembler_rhs: StokesAssembler
    ) -> None:
        """GMRES should run, record a residual history, and last residual <= first."""
        solver = LinearSolver(stokes_assembler_rhs)
        _ = solver.gmres_solve(
            tol=1e-12,
            rtol=1e-8,
            max_it=200,
            restart=30,
            show_plot=False,
            key="gmres1",
            enable_monitor=True,
        )

        hist = solver.get_residual_history(key="gmres1")
        assert isinstance(hist, list)
        assert len(hist) > 0, "Residual history should have entries"
        assert hist[-1] <= hist[0] + 1e-30  # Non-increasing in practice

    def test_plot_residuals_writes_png(
        self, tmp_path: Path, stokes_assembler_rhs: StokesAssembler
    ) -> None:
        """After a GMRES run with monitoring, plot_residuals should emit a PNG."""
        solver = LinearSolver(stokes_assembler_rhs)
        _ = solver.gmres_solve(
            tol=1e-10, rtol=1e-8, max_it=150, key="gmres_plot", enable_monitor=True
        )

        out = tmp_path / "ksp_residuals.png"
        solver.plot_residuals(key="gmres_plot", output_path=out, title="title")

        assert out.exists()
        assert out.stat().st_size > 0

    def test_cache_key_separates_histories(
        self, stokes_assembler_rhs: StokesAssembler
    ) -> None:
        """Different keys should maintain independent residual histories."""
        solver = LinearSolver(stokes_assembler_rhs)

        _ = solver.gmres_solve(
            tol=1e-10, rtol=1e-8, max_it=60, key="K1", enable_monitor=True
        )
        hist1 = solver.get_residual_history(key="K1")
        assert len(hist1) > 0

        _ = solver.gmres_solve(
            tol=1e-10, rtol=1e-8, max_it=40, key="K2", enable_monitor=True
        )
        hist2 = solver.get_residual_history(key="K2")
        assert len(hist2) > 0

        # Ensure last-run retrieval returns the latest key by default
        last = solver.get_residual_history()
        assert last == hist2

        # Histories must be distinct objects and (likely) different lengths
        assert hist1 is not hist2
        assert len(hist1) != len(hist2) or not np.allclose(hist1, hist2)

    def test_cg_runs_without_crashing(
        self, stokes_assembler_rhs0: StokesAssembler
    ) -> None:
        """CG on a saddle-point system is not mathematically appropriate, but code path should not crash."""
        solver = LinearSolver(stokes_assembler_rhs0)
        sol = solver.cg_solve(
            tol=1e-8,
            rtol=1e-8,
            max_it=5,
            show_plot=False,
            key="cg_smoke",
            enable_monitor=True,
        )
        # Just assert we got a Function back and the vector is finite
        assert sol is stokes_assembler_rhs0.sol
        assert np.all(np.isfinite(sol.x.array))
