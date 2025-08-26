"""Tests for FEM.operators module."""

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import numpy as np
import pytest

from config import BoundaryConditionsConfig
from FEM.bcs import (
    BoundaryConditions,
    BoundaryConditionType,
    define_bcs,
)
from FEM.operators import (
    LinearizedNavierStokesAssembler,
    StationaryNavierStokesAssembler,
)
from FEM.spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from Meshing.core import Mesher
from Meshing.utils import Shape, iCellType


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
def test_bcs(test_mesher: Mesher, test_spaces: FunctionSpaces) -> BoundaryConditions:
    """Test Dirichlet velocity boundary condition."""
    configs = [
        BoundaryConditionsConfig(
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
    """Define an assembler for the linearized system with zero baseflow."""
    return LinearizedNavierStokesAssembler(
        base_flow=zero_base_flow, spaces=test_spaces, re=1.0, bcs=test_bcs
    )


@pytest.fixture
def test_stationary_assembler(
    test_spaces: FunctionSpaces, test_bcs: BoundaryConditions
) -> StationaryNavierStokesAssembler:
    return StationaryNavierStokesAssembler(
        spaces=test_spaces,
        re=1.0,
        bcs=test_bcs,
    )


class TestLinearizedAssembler:
    """Tests for the perturbation flow assembler."""

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

    def test_mass_subblocks(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Check mass subblocks."""
        M = test_assembler.assemble_mass_matrix()
        blocks = test_assembler.extract_subblocks(M)

        n_u, _ = test_assembler.spaces.velocity_dofs

        # Name blocks and assert non-None for type narrowing
        vv = blocks[0, 0]  # velocity–velocity
        vp = blocks[0, 1]  # velocity–pressure
        pv = blocks[1, 0]  # pressure–velocity
        pp = blocks[1, 1]  # pressure–pressure

        assert vv is not None
        assert vp is not None
        assert pv is not None
        assert pp is not None

        # VV block properties
        assert vv.shape == (n_u, n_u)
        assert vv.is_numerically_symmetric()
        for _ in range(10):
            y = vv.create_vector_right()
            y.set_random()
            assert y.dot(vv @ y) > 0

        # All other blocks should be (numerically) zero
        assert vp.norm < 1e-10
        assert pv.norm < 1e-10
        assert pp.norm < 1e-10

    def test_linear_operator_subblocks(
        self, test_assembler: LinearizedNavierStokesAssembler
    ) -> None:
        """Check that the assembled operator and its true sub-blocks are non-empty where expected."""
        L = test_assembler.assemble_linear_operator()
        # The full operator must have non-zero entries
        assert L.norm > 0

        blocks = test_assembler.extract_subblocks(L)

        # Name blocks and assert non-None for type narrowing
        vv = blocks[0, 0]  # velocity–velocity
        vp = blocks[0, 1]  # velocity–pressure (gradient)
        pv = blocks[1, 0]  # pressure–velocity (divergence)
        pp = blocks[1, 1]  # pressure–pressure

        assert vv is not None
        assert vp is not None
        assert pv is not None
        assert pp is not None

        # Expected non-zeros
        assert vv.norm > 0
        assert pv.norm > 0
        assert vp.norm > 0
        # PP should be zero
        assert pp.norm < 1e-10

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

        D.axpy(-1.0, G.T)
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

    def test_non_homogeneous_natural_bcs(
        self, test_spaces: FunctionSpaces, zero_base_flow: dfem.Function
    ) -> None:
        """Test that non-homogeneous Neumann or Robin BCs raise an error."""
        # Neumann BC with non-zero value
        neumann_bcs = BoundaryConditions(
            velocity=[],
            pressure=[],
            velocity_neumann=[
                (1, dfem.Constant(test_spaces.velocity.mesh, (1.0, 0.0)))
            ],
            pressure_neumann=[],
            robin_data=[],
            velocity_periodic_map=[],
            pressure_periodic_map=[],
        )
        with pytest.raises(ValueError):
            LinearizedNavierStokesAssembler(
                base_flow=zero_base_flow, spaces=test_spaces, re=1.0, bcs=neumann_bcs
            )

        # Robin BC with non-zero target value
        robin_target = dfem.Constant(test_spaces.velocity.mesh, (0.1, 0.0))
        robin_bcs = BoundaryConditions(
            velocity=[],
            pressure=[],
            velocity_neumann=[],
            pressure_neumann=[],
            robin_data=[
                (1, dfem.Constant(test_spaces.velocity.mesh, 1.0), robin_target)
            ],
            velocity_periodic_map=[],
            pressure_periodic_map=[],
        )
        with pytest.raises(ValueError):
            LinearizedNavierStokesAssembler(
                base_flow=zero_base_flow, spaces=test_spaces, re=1.0, bcs=robin_bcs
            )


class TestStationaryAssembler:
    """Tests for the stationary flow assembler."""

    def test_jacobian_residual_shape(
        self, test_stationary_assembler: StationaryNavierStokesAssembler
    ) -> None:
        """Test that the assembled jacobian and residual match the expected dimensions."""
        A, b = test_stationary_assembler.get_matrix_forms()
        n = A.shape[0]
        assert A.shape == (n, n)
        assert b.size == n

    def test_jacobian_positive_definite(
        self, test_stationary_assembler: StationaryNavierStokesAssembler
    ) -> None:
        """Test that the jacobian is PD."""
        A, _ = test_stationary_assembler.get_matrix_forms()
        for _ in range(10):
            x = A.create_vector_right()
            x.set_random()
            val = x.dot(A @ x)
            assert val > 0 or np.isclose(val, 0)

    def test_nonzero_rhs(
        self, test_spaces: FunctionSpaces, test_bcs: BoundaryConditions
    ) -> None:
        """Verify that the RHS changes if the forcing term is nonzero."""
        zero_rhs_assembler = StationaryNavierStokesAssembler(
            spaces=test_spaces,
            re=1.0,
            bcs=test_bcs,
            f=dfem.Constant(test_spaces.velocity.mesh, (0.0, 0.0)),
        )
        _, b0 = zero_rhs_assembler.get_matrix_forms()

        nonzero_rhs_assembler = StationaryNavierStokesAssembler(
            spaces=test_spaces,
            re=1.0,
            bcs=test_bcs,
            f=dfem.Constant(test_spaces.velocity.mesh, (1.0, 0.0)),
        )
        _, b1 = nonzero_rhs_assembler.get_matrix_forms()

        assert not np.allclose(b0.raw, b1.raw)

    def test_cache(
        self, test_stationary_assembler: StationaryNavierStokesAssembler
    ) -> None:
        """Verify that specifying the same key reuses the cache, and different keys give new objects."""
        A1, b1 = test_stationary_assembler.get_matrix_forms(key_jac="A", key_res="b")
        A2, b2 = test_stationary_assembler.get_matrix_forms(key_jac="A", key_res="b")
        assert A1 is A2
        assert b1 is b2

        A3, b3 = test_stationary_assembler.get_matrix_forms(key_jac="C", key_res="d")
        assert A3 is not A1
        assert b3 is not b1

        test_stationary_assembler.clear_cache()
        A4, b4 = test_stationary_assembler.get_matrix_forms(key_jac="A", key_res="b")
        assert A4 is not A1
        assert b4 is not b1
