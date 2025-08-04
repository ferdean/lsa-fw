"""Unit tests for FEM.bcs module."""

import dolfinx.fem as dfem
import numpy as np
import pytest
import ufl

from FEM.bcs import (
    BoundaryCondition,
    BoundaryConditionType,
    apply_periodic_constraints,
    compute_periodic_dof_pairs,
    define_bcs,
)
from FEM.spaces import FunctionSpaces, FunctionSpaceType, define_spaces
from FEM.utils import iPETScMatrix
from Meshing.core import Mesher
from Meshing.utils import Shape
from Meshing.utils import iCellType


@pytest.fixture(scope="module")
def mesher_with_tags() -> Mesher:
    """Create a unit square mesh with facet markers for testing.

    Note: This fixture assumes that the mesh file is valid and that the meshing module has been fully tested.
    Although this creates a dependency on external state and violates strict test isolation, it is a deliberate
    and common trade-off in staged test suites (e.g., testing FEM only after validating Meshing).
    """
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(4, 4), cell_type=iCellType.TRIANGLE)
    _ = mesher.generate()

    def _marker_fn(x: np.ndarray) -> int:
        if np.isclose(x[0], 0.0):
            return 1
        elif np.isclose(x[0], 1.0):
            return 2
        return 99

    mesher.mark_boundary_facets(_marker_fn)
    return mesher


@pytest.fixture
def spaces(mesher_with_tags: Mesher) -> FunctionSpaces:
    """Define function spaces for the Navier-Stokes problem."""
    return define_spaces(mesher_with_tags.mesh, FunctionSpaceType.TAYLOR_HOOD)


def test_dirichlet_velocity_bc(
    mesher_with_tags: Mesher, spaces: FunctionSpaces
) -> None:
    """Test Dirichlet velocity boundary condition."""
    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.DIRICHLET_VELOCITY,
            value=(1.0, 0.0),
        )
    ]
    bcs = define_bcs(mesher_with_tags, spaces, configs)

    assert len(bcs.velocity) == 1
    marker, bc = bcs.velocity[0]
    assert marker == 1
    assert isinstance(bc, dfem.DirichletBC)
    assert bcs.pressure == []
    assert bcs.pressure_neumann == []
    assert bcs.velocity_neumann == []
    assert bcs.robin_data == []
    assert bcs.pressure_periodic_map == []
    assert bcs.velocity_periodic_map == []


def test_dirichlet_pressure_bc(
    mesher_with_tags: Mesher, spaces: FunctionSpaces
) -> None:
    """Test Dirichlet pressure boundary condition."""
    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.DIRICHLET_PRESSURE,
            value=1.0,
        )
    ]
    bcs = define_bcs(mesher_with_tags, spaces, configs)

    assert len(bcs.pressure) == 1
    marker, bc = bcs.pressure[0]
    assert marker == 1
    assert isinstance(bc, dfem.DirichletBC)
    assert bcs.velocity == []
    assert bcs.pressure_neumann == []
    assert bcs.velocity_neumann == []
    assert bcs.robin_data == []
    assert bcs.pressure_periodic_map == []
    assert bcs.velocity_periodic_map == []


def test_callable_dirichlet_bc(
    mesher_with_tags: Mesher, spaces: FunctionSpaces
) -> None:
    """Test Dirichlet BC with a callable value."""

    def _velocity_field(x: np.ndarray) -> np.ndarray:
        return np.vstack((np.sin(x[0]), np.cos(x[1])))

    config = BoundaryCondition(
        marker=1,
        type=BoundaryConditionType.DIRICHLET_VELOCITY,
        value=_velocity_field,
    )

    bcs = define_bcs(mesher_with_tags, spaces, [config])

    assert len(bcs.velocity) == 1
    _, bc = bcs.velocity[0]
    assert isinstance(bc, dfem.DirichletBC)


def test_neumann_bc(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test Neumann boundary condition."""
    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.NEUMANN_VELOCITY,
            value=(0.0, -1.0),
        )
    ]
    bcs = define_bcs(mesher_with_tags, spaces, configs)

    assert len(bcs.velocity_neumann) == 1
    marker, form = bcs.velocity_neumann[0]
    assert marker == 1
    assert isinstance(form, dfem.Constant)
    assert bcs.velocity == []
    assert bcs.pressure == []
    assert bcs.pressure_neumann == []
    assert bcs.robin_data == []
    assert bcs.pressure_periodic_map == []
    assert bcs.velocity_periodic_map == []


def test_robin_bc(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test Robin boundary condition."""
    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.ROBIN,
            value=(0.0, 0.0),
            robin_alpha=0.1,
        )
    ]
    bcs = define_bcs(mesher_with_tags, spaces, configs)

    assert len(bcs.robin_data) == 1
    marker, alpha, g_expr = bcs.robin_data[0]
    assert marker == 1
    assert isinstance(alpha, dfem.Constant)
    assert isinstance(g_expr, (ufl.ExternalOperator, dfem.Constant))
    assert bcs.velocity == []
    assert bcs.pressure == []
    assert bcs.velocity_neumann == []
    assert bcs.pressure_neumann == []
    assert bcs.velocity_periodic_map == []
    assert bcs.pressure_periodic_map == []


def test_mixed_bcs(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test combination of Dirichlet, Neumann, and Robin boundary conditions."""
    configs = [
        BoundaryCondition(
            marker=1, type=BoundaryConditionType.DIRICHLET_VELOCITY, value=(1.0, 0.0)
        ),
        BoundaryCondition(
            marker=1, type=BoundaryConditionType.NEUMANN_VELOCITY, value=(0.0, -1.0)
        ),
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.ROBIN,
            value=(0.0, 0.0),
            robin_alpha=0.1,
        ),
        BoundaryCondition(
            marker=1, type=BoundaryConditionType.DIRICHLET_PRESSURE, value=1.0
        ),
    ]
    bcs = define_bcs(mesher_with_tags, spaces, configs)

    assert len(bcs.velocity) == 1
    assert len(bcs.pressure) == 1
    assert len(bcs.velocity_neumann) == 1
    assert len(bcs.robin_data) == 1

    for _, bc in bcs.velocity + bcs.pressure:
        assert isinstance(bc, dfem.DirichletBC)

    _, g_vec = bcs.velocity_neumann[0]
    assert isinstance(g_vec, (dfem.Constant, ufl.ExternalOperator))

    _, alpha, g_expr = bcs.robin_data[0]
    assert isinstance(alpha, dfem.Constant)
    assert isinstance(g_expr, (dfem.Constant, ufl.ExternalOperator))

    assert bcs.pressure_neumann == []
    assert bcs.velocity_periodic_map == []
    assert bcs.pressure_periodic_map == []


def test_periodic_pairs(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test that pairs of facets are grouped together for periodic boundary conditions."""
    pairs = compute_periodic_dof_pairs(
        spaces.velocity, mesher_with_tags, from_marker=1, to_marker=2
    )
    assert pairs

    coords = spaces.velocity.tabulate_dof_coordinates()[
        :, : mesher_with_tags.mesh.geometry.dim
    ]

    for td, sd in pairs.items():
        x_to, y_to = coords[td]
        x_fr, y_fr = coords[sd]

        assert pytest.approx(x_to - x_fr, rel=1e-6) == 1.0
        assert pytest.approx(y_to, rel=1e-6) == y_fr


def test_apply_periodic() -> None:
    """Test applying periodic boundary conditions."""
    #    [[ 1,  2,  3,  4],
    #     [ 5,  6,  7,  8],
    #     [ 9, 10, 11, 12],
    #     [13, 14, 15, 16]]

    original = np.arange(16).reshape(4, 4) + 1
    M = iPETScMatrix.from_matrix(original)
    periodic_map = {3: 0}

    apply_periodic_constraints(M, periodic_map)

    # Expected result:
    # - row_0 ← row_0 + row_3 = [1 + 13, 2 + 14, 3 + 15, 4 + 16] = [14, 16, 18, 20]
    # - col_0 ← col_0 + col_3 = [14 + 20, 5 + 8, 9 + 12, 13 + 16] = [34, 13, 21, 29]
    # - row_3 and col_3 are zeroed out
    # - M[3, 3] = 1

    M_expected = np.array(
        [
            [34, 16, 18, 0],
            [13, 6, 7, 0],
            [21, 10, 11, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(M.as_array(), M_expected)
