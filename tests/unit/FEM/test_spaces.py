"""Unit tests for FEM.spaces module."""

import pytest
from dolfinx.fem import FunctionSpace
from dolfinx.mesh import Mesh

from FEM.spaces import (
    FunctionSpaceType,
    define_spaces,
)
from Meshing import Mesher, Shape, iCellType


@pytest.fixture(scope="module")
def test_mesh() -> Mesh:
    """Load a pre-generated mesh from disk for FEM tests.

    Note: This fixture assumes that the mesh file is valid and that the meshing module has been fully tested.
    Although this creates a dependency on external state and violates strict test isolation, it is a deliberate
    and common trade-off in staged test suites (e.g., testing FEM only after validating Meshing).
    """
    mesher = Mesher(
        shape=Shape.BOX,
        n=(12, 12, 12),
        cell_type=iCellType.HEXAHEDRON,
        domain=((0, 0, 0), (10, 5, 10)),
    )
    return mesher.generate()


@pytest.mark.parametrize(
    "space_type",
    [
        FunctionSpaceType.TAYLOR_HOOD,
        FunctionSpaceType.MINI,
        FunctionSpaceType.SIMPLE,
    ],
)
def test_define_spaces_valid_types(test_mesh: Mesh, space_type: FunctionSpaceType):
    """Test the define_spaces function with valid space types."""
    spaces = define_spaces(test_mesh, type=space_type)

    assert isinstance(spaces.velocity, FunctionSpace)
    assert isinstance(spaces.pressure, FunctionSpace)
    assert isinstance(spaces.mixed, FunctionSpace)
    assert spaces.velocity.mesh is test_mesh
    assert spaces.pressure.mesh is test_mesh


def test_taylor_hood_elements(test_mesh: Mesh):
    """Test the elements of the Taylor-Hood function spaces."""
    spaces = define_spaces(test_mesh, type=FunctionSpaceType.TAYLOR_HOOD)

    vel_elem = spaces.velocity.ufl_element()
    pre_elem = spaces.pressure.ufl_element()

    assert vel_elem.sobolev_space.name == "H1"
    assert pre_elem.sobolev_space.name == "H1"

    assert vel_elem.degree == 2
    assert pre_elem.degree == 1

    assert vel_elem.cell == pre_elem.cell

    assert (
        vel_elem.family_name == "P"
    )  # note that 'P' is the canonical short name used in basix for Lagrange
    assert pre_elem.family_name == "P"

    assert vel_elem.reference_value_shape == (test_mesh.geometry.dim,)
    assert pre_elem.reference_value_shape == ()


def test_simple_elements(test_mesh: Mesh) -> None:
    """Test the elements of the SIMPLE function spaces."""
    spaces = define_spaces(test_mesh, type=FunctionSpaceType.SIMPLE)

    vel_elem = spaces.velocity.ufl_element()
    pre_elem = spaces.pressure.ufl_element()

    assert vel_elem.sobolev_space.name == "H1"
    assert pre_elem.sobolev_space.name == "H1"

    assert vel_elem.degree == 1
    assert pre_elem.degree == 1

    assert vel_elem.cell == pre_elem.cell

    assert vel_elem.family_name == "P"
    assert pre_elem.family_name == "P"

    assert vel_elem.reference_value_shape == (test_mesh.geometry.dim,)
    assert pre_elem.reference_value_shape == ()


def test_mini_elements(test_mesh: Mesh) -> None:
    """Test the elements of the MINI function space."""
    spaces = define_spaces(test_mesh, type=FunctionSpaceType.MINI)

    vel_elem = spaces.velocity.ufl_element()
    pre_elem = spaces.pressure.ufl_element()

    assert vel_elem.sobolev_space.name == "H1"
    assert pre_elem.sobolev_space.name == "H1"

    assert vel_elem.family_name == "custom"  # from enriched element
    assert pre_elem.family_name == "P"

    assert vel_elem.degree == 3  # from bubble degree
    assert pre_elem.degree == 1

    assert vel_elem.cell == pre_elem.cell

    assert vel_elem.reference_value_shape == (test_mesh.geometry.dim,)
    assert pre_elem.reference_value_shape == ()


def test_dg_raises_not_implemented(test_mesh: Mesh):
    """Test that the define_spaces function raises NotImplementedError for DG spaces."""
    with pytest.raises(NotImplementedError) as err:
        define_spaces(test_mesh, type=FunctionSpaceType.DG)
    assert "not yet supported." in str(err)
