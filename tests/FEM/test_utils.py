"""Tests for FEM.utils module."""

import pytest
from pathlib import Path

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


def test_matrix_export(tmp_path: Path) -> None:
    """Test that iPETScMatrix.export() writes a binary matrix file."""
    A = PETSc.Mat().createAIJ([4, 4], comm=PETSc.COMM_SELF)
    A.setUp()
    A.setValue(1, 1, 3.14)
    A.assemble()

    wrapped = iPETScMatrix(A)
    filepath = tmp_path / "exported_matrix.bin"
    wrapped.export(str(filepath))

    assert filepath.exists()
    assert filepath.stat().st_size > 0


def test_matrix_addition(test_matrix: iPETScMatrix) -> None:
    """Test that matrix addition creates a new matrix with correct summed values."""
    A = test_matrix
    B = test_matrix  # Since immutable, reusing is fine
    C = A + B

    row, col = 0, 0
    expected = A.get_value(row, col) + B.get_value(row, col)
    actual = C.get_value(row, col)
    assert pytest.approx(actual, abs=1e-10) == expected


def test_matrix_invalid_addition(test_matrix: iPETScMatrix) -> None:
    """Ensure that adding matrices with incompatible shapes raises ValueError."""
    shape = test_matrix.shape
    wrong_mat = PETSc.Mat().createAIJ(
        [shape[0] + 1, shape[1] + 1], comm=PETSc.COMM_SELF
    )
    wrong_mat.setUp()
    wrong_mat.assemble()
    wrong = iPETScMatrix(wrong_mat)

    with pytest.raises(ValueError, match="Incompatible matrix shapes"):
        _ = test_matrix + wrong


def test_axpy_operation(test_matrix: iPETScMatrix) -> None:
    """Test that AXPY modifies the target matrix as expected."""
    A = iPETScMatrix(test_matrix.raw.duplicate(copy=True))
    B = test_matrix
    alpha = 2.0

    row, col = 0, 0
    before = A.get_value(row, col)
    contribution = alpha * B.get_value(row, col)

    A.axpy(alpha, B)

    after = A.get_value(row, col)
    assert pytest.approx(after, abs=1e-10) == before + contribution


def test_from_nested(test_matrix: iPETScMatrix) -> None:
    """Test creating a MatNest from two compatible matrices."""
    A = test_matrix.raw
    zero = A.duplicate()
    zero.zeroEntries()
    zero.assemble()

    nested = iPETScMatrix.from_nested(
        [
            [A, zero],
            [zero, None],
        ]
    )

    assert isinstance(nested, iPETScMatrix)
    assert nested.shape == (2 * A.getSize()[0], 2 * A.getSize()[1])
