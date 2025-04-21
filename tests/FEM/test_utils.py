"""Tests for FEM.utils module."""

import pytest
from pathlib import Path
import numpy as np

from petsc4py import PETSc

from FEM.utils import iPETScMatrix, iPETScVector


class TestVector:
    """Test suite for the iPETScVector wrapper class."""

    def test_zeros_and_size(self) -> None:
        """Test creation of zero vector and verify size and contents."""
        vec = iPETScVector.zeros(5)
        assert vec.size == 5
        assert np.allclose(vec.as_array(), 0.0)

    def test_from_array_and_as_array(self) -> None:
        """Test vector creation from NumPy array and conversion back."""
        arr = np.array([1.0, 2.0, 3.0])
        vec = iPETScVector.from_array(arr)
        np.testing.assert_array_equal(vec.as_array(), arr)

    def test_addition_and_radd(self) -> None:
        """Test vector addition and right-hand addition."""
        v1 = iPETScVector.from_array(np.array([1.0, 2.0]))
        v2 = iPETScVector.from_array(np.array([3.0, 4.0]))
        result1 = v1 + v2
        result2 = v2 + v1
        np.testing.assert_array_equal(result1.as_array(), [4.0, 6.0])
        np.testing.assert_array_equal(result2.as_array(), [4.0, 6.0])

    def test_subtraction(self) -> None:
        """Test vector subtraction."""
        v1 = iPETScVector.from_array(np.array([5.0, 6.0]))
        v2 = iPETScVector.from_array(np.array([2.0, 1.0]))
        diff = v1 - v2
        np.testing.assert_array_equal(diff.as_array(), [3.0, 5.0])

    def test_scalar_multiplication_and_dot(self) -> None:
        """Test scalar multiplication and dot product."""
        v = iPETScVector.from_array(np.array([2.0, -1.0]))
        scaled = v * 3.0
        assert isinstance(scaled, iPETScVector)
        np.testing.assert_allclose(scaled.as_array(), [6.0, -3.0])

        dot = v * iPETScVector.from_array(np.array([4.0, 5.0]))
        assert pytest.approx(dot, abs=1e-10) == 3

    def test_rmul(self) -> None:
        """Test right scalar multiplication."""
        v = iPETScVector.from_array(np.array([2.0, 3.0]))
        result = 2.0 * v
        np.testing.assert_allclose(result.as_array(), [4.0, 6.0])

    def test_outer_product(self) -> None:
        """Test vector outer product resulting in a matrix."""
        a = iPETScVector.from_array(np.array([1.0, 2.0]))
        b = iPETScVector.from_array(np.array([3.0, 4.0]))
        mat = a @ b
        assert isinstance(mat, iPETScMatrix)
        assert mat.shape == (2, 2)
        assert pytest.approx(mat[0, 0]) == 3.0
        assert pytest.approx(mat[1, 1]) == 8.0

    def test_get_and_set_item(self) -> None:
        """Test vector indexing access and modification."""
        v = iPETScVector.zeros(3)
        v[1] = 5.0
        assert pytest.approx(v[1]) == 5.0

    def test_copy_and_scale(self) -> None:
        """Test copying and scaling of a vector."""
        v1 = iPETScVector.from_array(np.array([1.0, 2.0]))
        v2 = v1.copy()
        v2.scale(2.0)
        np.testing.assert_allclose(v2.as_array(), [2.0, 4.0])
        np.testing.assert_allclose(
            v1.as_array(), [1.0, 2.0]
        )  # Ensure original unchanged

    def test_zero_entries_and_random(self) -> None:
        """Test zeroing and random assignment of entries."""
        v = iPETScVector.zeros(3)
        v.set_random()
        assert not np.allclose(v.as_array(), 0.0)
        v.zero_all_entries()
        assert np.allclose(v.as_array(), 0.0)

    def test_get_set_value_methods(self) -> None:
        """Test get_value and set_value methods."""
        v = iPETScVector.zeros(2)
        v.set_value(0, 42)
        assert pytest.approx(v.get_value(0), abs=1e-10) == 42

    def test_norm(self) -> None:
        """Test norm."""
        v = iPETScVector.from_array(np.array([3.0, 4.0]))
        assert pytest.approx(v.norm, abs=1e-10) == 5.0

    def test_export_vector(self, tmp_path: Path) -> None:
        """Test exporting a vector to binary file."""
        v = iPETScVector.from_array(np.array([1.0, 2.0]))
        filepath = tmp_path / "vector_export.bin"
        v.export(filepath)
        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_print(self) -> None:
        """Test the print method of vector."""
        v = iPETScVector.zeros(2)
        v.print()  # Should not raise


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
        "nested",
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


class TestMatrix:
    """Test suite for the iPETScMatrix wrapper class."""

    def test_from_nested(self) -> None:
        """Test building a nested matrix from blocks, and converting it to aij."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 1.0)
        A.setValue(1, 1, 2.0)
        A.assemble()

        Z = A.copy()
        Z.zeroEntries()
        Z.assemble()

        nested = iPETScMatrix.from_nested(
            [
                [A, Z],
                [Z, None],
            ]
        )
        aij = nested.to_aij()  # Convert to aij matrix for better indexing

        assert isinstance(nested, iPETScMatrix)
        assert nested.shape == (4, 4)
        assert pytest.approx(aij[0, 0]) == 1.0
        assert pytest.approx(aij[1, 1]) == 2.0
        assert aij[2, 2] == 0.0

    def test_addition(self) -> None:
        """Test matrix addition and reflected addition."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 1.0)
        A.assemble()

        mat1 = iPETScMatrix(A)
        mat2 = iPETScMatrix(A.copy())

        mat_sum = mat1 + mat2

        assert pytest.approx(mat_sum[0, 0], abs=1e-10) == 2.0
        assert pytest.approx((mat2 + mat1)[0, 0], abs=1e-10) == 2.0

    def test_matmul(self) -> None:
        """Test matrix @ vector and matrix @ matrix operations."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 2.0)
        A.setValue(1, 1, 3.0)
        A.assemble()

        mat = iPETScMatrix(A)
        vec = iPETScVector.from_array(np.array([1.0, 2.0]))

        result_vec = mat @ vec
        mat2 = mat @ mat

        np.testing.assert_allclose(result_vec.as_array(), [2.0, 6.0])
        assert isinstance(mat2, iPETScMatrix)
        assert pytest.approx(mat2[0, 0], abs=1e-10) == 4.0

    def test_rmatmul_vector_matrix(self) -> None:
        """Test vector @ matrix operation."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 2.0)
        A.setValue(1, 1, 3.0)
        A.assemble()

        mat = iPETScMatrix(A)
        vec = iPETScVector.from_array(np.array([3.0, 4.0]))

        result = vec @ mat
        np.testing.assert_allclose(result.as_array(), [6.0, 12.0])

    def test_indexing_get_set(self) -> None:
        """Test matrix[i, j] get and set functionality."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.assemble()

        mat = iPETScMatrix(A)
        mat[1, 0] = 3.0

        assert pytest.approx(mat[1, 0], abs=1e-10) == 3.0

    def test_properties_transpose_and_shape(self) -> None:
        """Test properties: shape, transpose, nonzero_entries, norm, type."""
        A = PETSc.Mat().createAIJ([2, 3], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 1, 1.0)
        A.setValue(1, 2, 2.0)
        A.assemble()

        mat = iPETScMatrix(A)

        assert mat.shape == (2, 3)
        assert mat.T.shape == (3, 2)
        assert mat.nonzero_entries == 2

        assert isinstance(mat.norm, float)
        assert isinstance(mat.type, str)

    def test_symmetry_checks(self) -> None:
        """Test is_symmetric and is_numerically_symmetric methods."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 1.0)
        A.setValue(0, 1, 2.0)
        A.setValue(1, 0, 2.0)
        A.setValue(1, 1, 1.0)
        A.assemble()

        mat = iPETScMatrix(A)

        assert mat.is_symmetric
        assert mat.is_numerically_symmetric(tol=1e-12)

    def test_scale_and_shift(self) -> None:
        """Test scale and shift operations on matrix."""
        A = PETSc.Mat().createAIJ([1, 1], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 1.0)
        A.assemble()

        mat = iPETScMatrix(A)

        mat.shift(2.0)
        assert pytest.approx(mat[0, 0], abs=1e-10) == 3.0

        mat.scale(0.5)
        assert pytest.approx(mat[0, 0], abs=1e-10) == 1.5

    def test_zero_all_entries(self) -> None:
        """Test that zero_all_entries() clears the matrix."""
        A = PETSc.Mat().createAIJ([1, 1], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 5.0)
        A.assemble()

        mat = iPETScMatrix(A)
        mat.zero_all_entries()

        assert pytest.approx(mat[0, 0], abs=1e-10) == 0.0

    def test_get_value_and_get_row(self) -> None:
        """Test get_value and get_row consistency."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(1, 0, 2.0)
        A.assemble()

        mat = iPETScMatrix(A)

        cols, values = mat.get_row(1)
        for col, val in zip(cols, values):
            assert pytest.approx(mat.get_value(1, col), abs=1e-10) == val

    def test_axpy(self) -> None:
        """Test AXPY operation on matrices."""
        A = PETSc.Mat().createAIJ([1, 1], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 0, 1.0)
        A.assemble()

        mat = iPETScMatrix(A)
        mat.axpy(2.0, mat)

        assert pytest.approx(mat[0, 0], abs=1e-10) == 3.0

    def test_create_vector_right_left(self) -> None:
        """Test creation of left and right-hand vectors from matrix."""
        A = PETSc.Mat().createAIJ([3, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.assemble()

        mat = iPETScMatrix(A)

        v_r = mat.create_vector_right()
        v_l = mat.create_vector_left()

        assert v_r.size == 2
        assert v_l.size == 3

    def test_export_matrix(self, tmp_path: Path) -> None:
        """Test exporting matrix to binary format."""
        A = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_SELF)
        A.setUp()
        A.setValue(0, 1, 4.2)
        A.assemble()

        mat = iPETScMatrix(A)
        path = tmp_path / "matrix_output.bin"
        mat.export(path)

        assert path.exists()
        assert path.stat().st_size > 0
