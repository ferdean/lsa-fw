"""Tests for FEM.utils module."""

import pytest

from ufl import TrialFunction, TestFunction, inner, dx
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.fem import form

from Meshing import Mesher, Shape
from Meshing.utils import iCellType
from FEM.spaces import define_spaces
from FEM.utils import iPETScMatrix


@pytest.fixture
def test_matrix() -> iPETScMatrix:
    """Define the matrix under test.

    Note: This fixture assumes that the mesh file is valid and that the meshing module has been fully tested.
    Although this creates a dependency on external state and violates strict test isolation, it is a deliberate
    and common trade-off in staged test suites (e.g., testing FEM only after validating Meshing).
    """
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(4, 4), cell_type=iCellType.TRIANGLE)
    mesh = mesher.generate()

    spaces = define_spaces(mesh)
    u = TrialFunction(spaces.velocity)
    v = TestFunction(spaces.velocity)
    a = inner(u, v) * dx

    a_petsc = assemble_matrix(form(a), bcs=[])
    a_petsc.assemble()

    return iPETScMatrix(a_petsc)


def test_shape_and_nnz(test_matrix: iPETScMatrix) -> None:
    """Check that the matrix has correct shape and non-zero entries."""
    assert isinstance(test_matrix, iPETScMatrix)
    assert test_matrix.shape[0] == test_matrix.shape[1]
    assert test_matrix.nonzero_entries > 0


def test_matrix_symmetry(test_matrix: iPETScMatrix) -> None:
    """Check whether the matrix reports symmetry (may be False)."""
    assert isinstance(test_matrix.is_symmetric, bool)


def test_str_and_print(test_matrix: iPETScMatrix) -> None:
    """Ensure the string representation and `.print()` method work correctly."""
    s = str(test_matrix)
    assert "iPETScMatrix" in s
    test_matrix.print()  # Should not raise


def test_raw_access(test_matrix: iPETScMatrix) -> None:
    """Verify that raw access returns the underlying PETSc matrix."""
    raw = test_matrix.raw
    assert isinstance(raw, PETSc.Mat)
    assert raw.getSize() == test_matrix.shape


def test_scaling_shifting(test_matrix: iPETScMatrix) -> None:
    """Test matrix scaling and shifting of diagonal entries."""
    row, col = 0, 0
    before = test_matrix.get_value(row, col)

    test_matrix.shift(3.0)
    shifted = test_matrix.get_value(row, col)
    assert pytest.approx(shifted - before, abs=1e-10) == 3.0

    test_matrix.scale(0.5)
    scaled = test_matrix.get_value(row, col)
    assert pytest.approx(scaled, abs=1e-10) == 0.5 * shifted


def test_zero_all_entries(test_matrix: iPETScMatrix) -> None:
    """Test that zeroing the matrix sets all stored values to zero."""
    test_matrix.zero_all_entries()
    rows, _ = test_matrix.shape

    # Check a representative set of rows for zero values
    for row in range(min(rows, 100)):
        _, values = test_matrix.get_row(row)
        for val in values:
            assert pytest.approx(val, abs=1e-12) == 0.0


def test_get_row_and_value(test_matrix: iPETScMatrix) -> None:
    """Check that row extraction and value retrieval are consistent."""
    row = 0
    cols, values = test_matrix.get_row(row)

    assert isinstance(cols, list)
    assert isinstance(values, list)
    assert len(cols) == len(values)

    for col, val in zip(cols, values):
        retrieved = test_matrix.get_value(row, col)
        assert pytest.approx(retrieved, abs=1e-10) == val


def test_type(test_matrix: iPETScMatrix) -> None:
    """Verify that the matrix type string is correctly reported."""
    mat_type = test_matrix.type
    assert isinstance(mat_type, str)
    assert mat_type in {
        "aij",
        "mpiaij",
        "seqaij",
        "baij",
        "sbaij",
        "matfree",
    }  # PETSc matrix types
