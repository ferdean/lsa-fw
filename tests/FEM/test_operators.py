"""Tests for FEM.operators module."""

import pytest

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from Meshing import Mesher, Shape, iCellType

from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import FunctionSpaces, define_spaces, FunctionSpaceType


@pytest.fixture(scope="module")
def test_mesh() -> dmesh.Mesh:
    """Define test mesh."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(4, 4), cell_type=iCellType.TRIANGLE)
    return mesher.generate()


@pytest.fixture(scope="module")
def test_spaces(test_mesh: dmesh.Mesh) -> FunctionSpaces:
    """Define test function spaces."""
    return define_spaces(test_mesh, type=FunctionSpaceType.TAYLOR_HOOD)


@pytest.fixture(scope="module")
def zero_base_flow(test_spaces: FunctionSpaces) -> dfem.Function:
    """Define a zero velocity field on the velocity space."""
    u_base = dfem.Function(test_spaces.velocity)
    u_base.x.array[:] = 0.0
    return u_base


@pytest.fixture
def test_assembler(
    zero_base_flow: dfem.Function, test_spaces: FunctionSpaces
) -> LinearizedNavierStokesAssembler:
    """Define an assembler for the linearized system with zero base flow."""
    return LinearizedNavierStokesAssembler(
        base_flow=zero_base_flow, spaces=test_spaces, re=1.0
    )


def test_shapes(
    test_assembler: LinearizedNavierStokesAssembler, test_spaces: FunctionSpaces
) -> None:
    """Test that all assembled operator shapes match the expected dimensions."""
    u_dofs = (
        test_spaces.velocity.dofmap.index_map.size_global
        * test_spaces.velocity.dofmap.index_map_bs
    )
    p_dofs = (
        test_spaces.pressure.dofmap.index_map.size_global
        * test_spaces.pressure.dofmap.index_map_bs
    )

    viscous = test_assembler.assemble_viscous_matrix()
    convection = test_assembler.assemble_convection_matrix()
    shear = test_assembler.assemble_shear_matrix()
    grad = test_assembler.assemble_pressure_gradient_matrix()
    div = test_assembler.assemble_divergence_matrix()
    mass = test_assembler.assemble_mass_matrix()

    assert viscous.shape == (u_dofs, u_dofs)
    assert convection.shape == (u_dofs, u_dofs)
    assert shear.shape == (u_dofs, u_dofs)
    assert grad.shape == (u_dofs, p_dofs)
    assert div.shape == (p_dofs, u_dofs)
    assert mass.shape == (u_dofs, u_dofs)


def test_operator_nonzero_entries(test_assembler: LinearizedNavierStokesAssembler):
    """Test that all assembled operators have non-zero entries."""
    assert test_assembler.assemble_mass_matrix().nonzero_entries > 0
    assert test_assembler.assemble_viscous_matrix().nonzero_entries > 0
    assert (
        test_assembler.assemble_convection_matrix().nonzero_entries >= 0  # u_base = 0
    )
    assert test_assembler.assemble_shear_matrix().nonzero_entries >= 0  # u_base = 0
    assert test_assembler.assemble_pressure_gradient_matrix().nonzero_entries > 0
    assert test_assembler.assemble_divergence_matrix().nonzero_entries > 0


def test_matrix_symmetry_stokes(
    test_assembler: LinearizedNavierStokesAssembler,
) -> None:
    """Test that the assembled viscous matrix is (numerically) symmetric."""
    A = test_assembler.assemble_viscous_matrix()
    assert A.is_numerically_symmetric()


def test_mass_matrix_positive_definite(
    test_assembler: LinearizedNavierStokesAssembler,
) -> None:
    """Verify that the mass matrix is symmetric positive definite."""
    M = test_assembler.assemble_mass_matrix()

    x = (
        M.raw.createVecRight()
    )  # TODO: Maybe it makes sense to also define a wrapper for vectors
    x.setRandom()

    y = M.raw.createVecLeft()
    M.raw.mult(x, y)

    dot = x.dot(y)

    assert M.is_numerically_symmetric()
    assert dot > 0


def test_gradient_divergence_adjoints(
    test_assembler: LinearizedNavierStokesAssembler,
) -> None:
    """Test that the gradient and divergence matrices are adjoint to each other."""
    G = test_assembler.assemble_pressure_gradient_matrix()
    D = test_assembler.assemble_divergence_matrix()

    D.axpy(-1.0, G.T)  # D - G^T
    assert D.norm < 1e-10
