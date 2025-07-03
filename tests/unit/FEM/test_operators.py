"""Tests for FEM.operators module."""

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import numpy as np
import pytest

from FEM.bcs import (
    BoundaryCondition,
    BoundaryConditions,
    BoundaryConditionType,
    define_bcs,
)
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from Meshing import Mesher, Shape, iCellType


@pytest.fixture(scope="module")
def test_mesher() -> Mesher:
    """Define test mesher."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(16, 16), cell_type=iCellType.TRIANGLE)
    _ = mesher.generate()
    mesher.mark_boundary_facets(lambda x: 1)  # Mark all boundary facets with marker '1'
    return mesher


@pytest.fixture(scope="module")
def test_mesh(test_mesher: Mesher) -> dmesh.Mesh:
    """Define test mesh."""
    return test_mesher.mesh


@pytest.fixture(scope="module")
def test_spaces(test_mesh: dmesh.Mesh) -> FunctionSpaces:
    """Define test function spaces."""
    return define_spaces(test_mesh, type=FunctionSpaceType.TAYLOR_HOOD)


@pytest.fixture(scope="module")
def test_bcs(test_mesher: Mesher, test_spaces: FunctionSpaces) -> None:
    """Test Dirichlet velocity boundary condition."""
    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.DIRICHLET_VELOCITY,
            value=(1.0, 0.0),
        )
    ]
    return define_bcs(test_mesher, test_spaces, configs)


@pytest.fixture(scope="module")
def zero_base_flow(test_spaces: FunctionSpaces) -> dfem.Function:
    """Define a zero velocity field on the velocity space."""
    u_base = dfem.Function(test_spaces.mixed)
    u_base.x.array[:] = 0.0
    return u_base


@pytest.fixture
def test_assembler(
    zero_base_flow: dfem.Function,
    test_spaces: FunctionSpaces,
    test_bcs: BoundaryConditions,
) -> LinearizedNavierStokesAssembler:
    """Define an assembler for the linearized system with zero base flow."""
    return LinearizedNavierStokesAssembler(
        base_flow=zero_base_flow, spaces=test_spaces, re=1.0, bcs=test_bcs
    )


class TestLinearizedAssembler:
    """Tests for the perturbed flow assembler."""

    def test_linear_operator_shape(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Test that the assembled linear operator shapes match the expected dimensions."""
        n_u, _ = test_assembler.spaces.velocity_dofs
        n_p, _ = test_assembler.spaces.pressure_dofs

        L = test_assembler.assemble_linear_operator()

        assert L.shape == (n_u + n_p, n_u + n_p)

    def test_mass_shape(self, test_assembler: LinearizedNavierStokesAssembler) -> None:
        """Test that the assembled linear operator shapes match the expected dimensions."""
        n_u, _ = test_assembler.spaces.velocity_dofs
        n_p, _ = test_assembler.spaces.pressure_dofs

        M = test_assembler.assemble_mass_matrix()

        assert M.shape == (n_u + n_p, n_u + n_p)

    def test_mass_matrix_positive_definite(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Verify that the mass matrix is symmetric positive definite."""
        M = test_assembler.assemble_mass_matrix()

        assert M.is_numerically_symmetric()
        for _ in range(10):
            x = M.create_vector_right()
            x.set_random()
            assert x.dot(M @ x) > 0
            del x  # Garbage-collect vector to ensure randomization

    def test_mass_subblocks(self, test_assembler: LinearizedNavierStokesAssembler):
        """Check mass subblocks."""
        M = test_assembler.assemble_mass_matrix()
        blocks = test_assembler.extract_subblocks(M)

        n_u, _ = test_assembler.spaces.velocity_dofs

        Mv = blocks[0, 0]

        assert Mv.shape == (n_u, n_u)
        assert Mv.is_numerically_symmetric()
        for _ in range(10):
            y = Mv.create_vector_right()
            y.set_random()
            assert y.dot(Mv @ y) > 0

        # All other blocks should be zero
        assert blocks[0, 1].norm < 1e-10  # vp
        assert blocks[1, 0].norm < 1e-10  # pv
        assert blocks[1, 1].norm < 1e-10  # pp

    def test_linear_operator_subblocks(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Check that the assembled operator and its true sub-blocks are non-empty where expected."""
        L = test_assembler.assemble_linear_operator()
        # The full operator must have non-zero entries
        assert L.norm > 0

        blocks = test_assembler.extract_subblocks(L)
        # Velocity–velocity block must be non-zero
        assert blocks[0, 0].norm > 0
        # Gradient and divergence blocks should both be non-zero
        assert blocks[1, 0].norm > 0
        assert blocks[0, 1].norm > 0
        # Pressure–pressure block should be zero
        assert blocks[1, 1].norm < 1e-10

    def test_gradient_divergence_adjoints(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Test that the gradient and divergence matrices are adjoint to each other.

        Note that, since the pressure is defined with a negative sign, the adjoint condition is actually D = - G^T.
        """
        L = test_assembler.assemble_linear_operator()
        subblocks = test_assembler.extract_subblocks(L)

        G = subblocks[0, 1]
        D = subblocks[1, 0]

        D.axpy(1.0, G.T)  # D + G^T
        assert D.norm < 1e-10

    def test_pressure_nullspace(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Test that the attached nullspace contains exactly the constant-pressure mode."""
        A, _ = test_assembler.assemble_eigensystem()
        ns = A.get_nullspace()
        assert ns is not None

        # Should be exactly one null vector (the constant-pressure mode)
        assert ns.dimension == 1
        basis = ns.basis
        assert len(basis) == 1

        # Build a 'pure' pressure vector: ones on p-dofs, zero elsewhere
        mixed = test_assembler.spaces.mixed
        _, dofs_p = mixed.sub(1).collapse()
        x = A.create_vector_right()
        arr = np.zeros((x.size,), dtype=float)
        arr[dofs_p] = 1.0
        x.set_array(arr)
        x.ghost_update()

        # Test that the basis vector itself is in the nullspace
        ok_b, norm_b = ns.test_vector(A, basis[0])
        assert ok_b, f"Basis vector failed nullspace test (residual norm={norm_b:.2e})"

        # Test that the constructed pressure-only vector is in the nullspace
        ok_x, norm_x = ns.test_vector(A, x)
        assert (
            ok_x
        ), f"Pressure vector failed nullspace test (residual norm={norm_x:.2e})"

        # Test the whole nullspace against the matrix
        ok_all, norm_all = ns.test_matrix(A)
        assert (
            ok_all
        ), f"NullSpace.test_matrix failed (max residual norm={norm_all:.2e})"

        # Remove the nullspace component from x and check it goes to zero
        ns.remove(x)
        x.ghost_update()
        assert x.norm < 1e-12, f"NullSpace.remove left nonzero norm={x.norm:.2e}"

    def test_cache(self, test_assembler: LinearizedNavierStokesAssembler) -> None:
        """Verify that specifying the same key reuses the cache, and different keys give new objects."""
        A1 = test_assembler.assemble_linear_operator(key="A")
        A2 = test_assembler.assemble_linear_operator(key="A")
        assert A1 is A2, "Same key should return the cached matrix"

        A3 = test_assembler.assemble_linear_operator(key="B")
        assert A3 is not A1, "Different key should force a rebuild"

        # After clear_cache, even same key must rebuild
        test_assembler.clear_cache()
        A4 = test_assembler.assemble_linear_operator(key="A")
        assert A4 is not A1, "clear_cache should invalidate previous entries"

        # Repeat for mass matrix
        M1 = test_assembler.assemble_mass_matrix(key="M")
        M2 = test_assembler.assemble_mass_matrix(key="M")
        assert M1 is M2, "Same key should return the cached mass matrix"

        M3 = test_assembler.assemble_mass_matrix(key="N")
        assert M3 is not M1, "Different key should force rebuild of mass matrix"

        test_assembler.clear_cache()
        M4 = test_assembler.assemble_mass_matrix(key="M")
        assert M4 is not M1, "clear_cache should also invalidate mass matrix entries"
