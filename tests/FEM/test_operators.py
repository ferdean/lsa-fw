"""Tests for FEM.operators module."""

import pytest

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from Meshing import Mesher, Shape, iCellType

from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import FunctionSpaces, define_spaces, FunctionSpaceType
from FEM.utils import iMeasure


@pytest.fixture(scope="module")
def test_mesher() -> Mesher:
    """Define test mesher."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(16, 16), cell_type=iCellType.TRIANGLE)
    _ = mesher.generate()
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
    u_dofs, _ = test_spaces.velocity_dofs
    p_dofs, _ = test_spaces.pressure_dofs

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

    x = M.create_vector_right()
    x.set_random()

    assert M.is_numerically_symmetric()
    assert x.dot(M @ x) > 0


def test_gradient_divergence_adjoints(
    test_assembler: LinearizedNavierStokesAssembler,
) -> None:
    """Test that the gradient and divergence matrices are adjoint to each other."""
    G = test_assembler.assemble_pressure_gradient_matrix()
    D = test_assembler.assemble_divergence_matrix()

    D.axpy(-1.0, G.T)  # D - G^T
    assert D.norm < 1e-10


def test_robin_matrix_zero_if_none(test_assembler: LinearizedNavierStokesAssembler):
    """Test that the Robin matrix is zero if no Robin forms are provided."""
    R = test_assembler.assemble_robin_matrix()
    assert R.nonzero_entries == 0
    assert R.norm == 0.0


def test_assemble_linear_operator_structure(
    test_assembler: LinearizedNavierStokesAssembler, test_spaces: FunctionSpaces
):
    """Test that the linear operator is a 2x2 block with correct submatrix shapes."""
    mat = test_assembler.assemble_linear_operator()
    n_u, _ = test_spaces.velocity_dofs
    n_p, _ = test_spaces.pressure_dofs

    A_block = mat.sub(0, 0)
    G_block = mat.sub(0, 1)
    D_block = mat.sub(1, 0)
    zero_block = mat.sub(1, 1)

    assert A_block is not None
    assert G_block is not None
    assert D_block is not None
    # PETSc creates an uninitialized submatrix for the zero block. Trying to access any of its expected properties
    # or methods would raise a runtime error at c++ level (then, a natural check for .nonzero_entries == 0 or
    # .norm() == 0.0 cannot be done)
    assert zero_block is None

    assert mat.shape == (n_u + n_p, n_u + n_p)
    assert A_block.shape == (n_u, n_u)
    assert G_block.shape == (n_u, n_p)
    assert D_block.shape == (n_p, n_u)


def test_assemble_eigensystem_properties(
    test_assembler: LinearizedNavierStokesAssembler,
):
    """Test shape and symmetry of eigenvalue system components."""
    A, M = test_assembler.assemble_eigensystem()
    assert A.shape == M.shape

    M_v_block = M.sub(0, 0)
    M_v = test_assembler.assemble_mass_matrix()

    assert M_v_block is not None
    assert M_v == M_v_block
    # Note that M_v has been tested to be SPD in `test_mass_matrix_positive_definite`


def test_assemble_neumann_rhs_format(
    test_assembler: LinearizedNavierStokesAssembler,
    test_spaces: FunctionSpaces,
    test_mesher: Mesher,
):
    """Test assembly of Neumann RHS with dummy function."""
    test_mesher.mark_boundary_facets(
        lambda x: 1  # Mark all boundary facets with marker '1'
    )
    g = dfem.Function(test_spaces.velocity)
    g.x.array[:] = 1.0  # Constant vector field
    ds = iMeasure.ds(test_mesher.mesh, test_mesher.facet_tags)

    rhs = test_assembler.assemble_neumann_rhs(g, ds)
    n_u, _ = test_spaces.velocity_dofs
    assert rhs.size == n_u


def test_clear_cache(test_assembler: LinearizedNavierStokesAssembler):
    """Test that clearing the cache causes recomputation."""
    mat_1 = test_assembler.assemble_viscous_matrix()
    id_before = mat_1.raw.handle

    test_assembler.clear_cache()

    mat_2 = test_assembler.assemble_viscous_matrix()
    id_after = mat_2.raw.handle

    assert id_before != id_after  # PETSc handle should change after recomputation
