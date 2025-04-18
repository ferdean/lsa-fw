"""Unit tests for FEM.bcs module."""

import pytest
import dolfinx.fem as dfem
import ufl

from Meshing import Mesher, Shape
from Meshing.utils import iCellType
from FEM.bcs import (
    define_bcs,
    BoundaryCondition,
    BoundaryConditionType,
)
from FEM.spaces import define_spaces, FunctionSpaceType, FunctionSpaces


@pytest.fixture(scope="module")
def mesher_with_tags() -> Mesher:
    """Create a unit square mesh with facet markers for testing.

    Note: This fixture assumes that the mesh file is valid and that the meshing module has been fully tested.
    Although this creates a dependency on external state and violates strict test isolation, it is a deliberate
    and common trade-off in staged test suites (e.g., testing FEM only after validating Meshing).
    """
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(4, 4), cell_type=iCellType.TRIANGLE)

    _ = mesher.generate()

    mesher.mark_boundary_facets(lambda x: 1)  # Mark all boundary facets with marker '1'
    return mesher


@pytest.fixture
def spaces(mesher_with_tags: Mesher) -> FunctionSpaces:
    """Define function spaces for the Navier-Stokes problem."""
    return define_spaces(mesher_with_tags.mesh, FunctionSpaceType.TAYLOR_HOOD)


def test_dirichlet_velocity_bc(
    mesher_with_tags: Mesher, spaces: FunctionSpaces
) -> None:
    """Test Dirichlet velocity boundary condition."""
    V = spaces.velocity
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.DIRICHLET_VELOCITY,
            value=(1.0, 0.0),
        )
    ]
    bcs = define_bcs(
        mesher_with_tags.mesh, spaces, mesher_with_tags.facet_tags, u, v, configs
    )

    assert len(bcs.velocity) == 1
    assert isinstance(bcs.velocity[0], dfem.DirichletBC)
    assert bcs.pressure == []
    assert bcs.neumann_forms == []
    assert bcs.robin_forms == []


def test_dirichlet_pressure_bc(
    mesher_with_tags: Mesher, spaces: FunctionSpaces
) -> None:
    """Test Dirichlet pressure boundary condition."""
    V = spaces.velocity
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.DIRICHLET_PRESSURE,
            value=1.0,
        )
    ]
    bcs = define_bcs(
        mesher_with_tags.mesh, spaces, mesher_with_tags.facet_tags, u, v, configs
    )

    assert len(bcs.pressure) == 1
    assert isinstance(bcs.pressure[0], dfem.DirichletBC)
    assert bcs.velocity == []
    assert bcs.neumann_forms == []
    assert bcs.robin_forms == []


def test_neumann_bc(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test Neumann boundary condition."""
    V = spaces.velocity
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.NEUMANN,
            value=(0.0, -1.0),
        )
    ]
    bcs = define_bcs(
        mesher_with_tags.mesh, spaces, mesher_with_tags.facet_tags, u, v, configs
    )

    assert len(bcs.neumann_forms) == 1
    assert isinstance(bcs.neumann_forms[0], ufl.Form)
    assert bcs.velocity == []
    assert bcs.pressure == []
    assert bcs.robin_forms == []


def test_robin_bc(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test Robin boundary condition."""
    V = spaces.velocity
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    configs = [
        BoundaryCondition(
            marker=1,
            type=BoundaryConditionType.ROBIN,
            value=(0.0, 0.0),
            robin_alpha=0.1,
        )
    ]
    bcs = define_bcs(
        mesher_with_tags.mesh, spaces, mesher_with_tags.facet_tags, u, v, configs
    )

    assert len(bcs.robin_forms) == 2
    assert all(isinstance(f, ufl.Form) for f in bcs.robin_forms)
    assert bcs.velocity == []
    assert bcs.pressure == []
    assert bcs.neumann_forms == []


def test_mixed_bcs(mesher_with_tags: Mesher, spaces: FunctionSpaces) -> None:
    """Test combination of Dirichlet, Neumann, and Robin boundary conditions."""
    V = spaces.velocity
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    configs = [
        BoundaryCondition(
            marker=1, type=BoundaryConditionType.DIRICHLET_VELOCITY, value=(1.0, 0.0)
        ),
        BoundaryCondition(
            marker=1, type=BoundaryConditionType.NEUMANN, value=(0.0, -1.0)
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
    bcs = define_bcs(
        mesher_with_tags.mesh, spaces, mesher_with_tags.facet_tags, u, v, configs
    )

    assert len(bcs.velocity) == 1
    assert len(bcs.pressure) == 1
    assert len(bcs.neumann_forms) == 1
    assert len(bcs.robin_forms) == 2
    assert all(isinstance(b, dfem.DirichletBC) for b in bcs.velocity + bcs.pressure)
    assert isinstance(bcs.neumann_forms[0], ufl.Form)
    assert all(isinstance(f, ufl.Form) for f in bcs.robin_forms)
