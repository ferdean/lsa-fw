"""Tests for FEM.utils module."""

# mypy: disable-error-code="attr-defined, name-defined"


from pathlib import Path

import numpy as np
import pytest
from petsc4py import PETSc
from scipy import sparse

from FEM.utils import iComplexPETScVector, iPETScMatrix, iPETScNullSpace, iPETScVector

_complex_build = np.issubdtype(PETSc.ScalarType, np.complexfloating)

skip_complex = pytest.mark.skipif(
    not _complex_build, reason="Complex-valued PETSc build required"
)

skip_real = pytest.mark.skipif(
    _complex_build, reason="Real-valued PETSc build required"
)


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

    @skip_complex
    def test_scale_complex(self) -> None:
        """Scale a vector by a complex factor."""
        arr = np.array([1 + 1j, 2 - 2j])
        vec = iPETScVector.from_array(arr)

        factor = 0.5 + 0.25j
        vec.scale(factor)

        result = vec.as_array()
        expected = arr * factor

        np.testing.assert_allclose(result, expected, atol=1e-12)

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

    @skip_complex
    def test_set_value_complex(self) -> None:
        """Test set_value and get_value for complex entries."""
        vec = iPETScVector.zeros(3)

        vec.set_value(0, 1 + 2j)
        vec.set_value(2, -3 - 4j)

        assert pytest.approx(vec.get_value(0), abs=1e-12) == 1 + 2j
        assert pytest.approx(vec.get_value(1), abs=1e-12) == 0.0 + 0.0j
        assert pytest.approx(vec.get_value(2), abs=1e-12) == -3 - 4j

    def test_norm(self) -> None:
        """Test norm."""
        v = iPETScVector.from_array(np.array([3.0, 4.0]))
        assert pytest.approx(v.norm, abs=1e-10) == 5.0

    @skip_complex
    def test_norm_complex(self) -> None:
        """Norm of a complex vector."""
        arr = np.array([3 + 4j, -1 + 2j, 0 - 1j], dtype=PETSc.ScalarType)
        vec = iPETScVector.from_array(arr)
        # PETSc returns a real 2-norm even for complex vectors
        norm = vec.norm
        expected = np.linalg.norm(arr)
        assert pytest.approx(norm, abs=1e-12) == expected

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


class TestMatrix:
    """Test suite for the iPETScMatrix wrapper class."""

    def test_create_aij(self) -> None:
        """Test creating an aij sparse matrix."""
        shape = (3, 4)
        mat = iPETScMatrix.create_aij(shape, comm=PETSc.COMM_WORLD, nnz=2)

        assert isinstance(mat, iPETScMatrix)
        assert mat.shape == shape
        assert mat.nonzero_entries == 0
        assert "aij" in mat.type.lower()

    def test_from_nested(self) -> None:
        """Test building a nested matrix from blocks, and converting it to aij."""
        A = iPETScMatrix.create_aij((2, 2))
        A[0, 0] = 1.0
        A[1, 1] = 2.0
        A.assemble()

        Z = A.duplicate()
        Z.zero_all_entries()
        Z.assemble()

        nested = iPETScMatrix.from_nested(
            [
                [A.raw, Z.raw],
                [Z.raw, None],
            ]
        )
        aij = nested.to_aij()  # Convert to aij matrix for better indexing

        assert isinstance(nested, iPETScMatrix)
        assert nested.shape == (4, 4)
        assert pytest.approx(aij[0, 0]) == 1.0
        assert pytest.approx(aij[1, 1]) == 2.0
        assert aij[2, 2] == 0.0

    def test_zeros(self) -> None:
        """Test building a zero-matrix."""
        shape = (4, 5)
        mat = iPETScMatrix.zeros(shape)

        assert isinstance(mat, iPETScMatrix)
        assert mat.shape == shape
        assert mat.nonzero_entries == 0

    def test_from_matrix_self(self) -> None:
        """Test creating a matrix from another matrix."""
        A = iPETScMatrix.create_aij((2, 2), comm=PETSc.COMM_WORLD, nnz=1)
        B = iPETScMatrix.from_matrix(A)
        assert B is A

    def test_from_matrix_wraps_petsc_mat(self) -> None:
        """Test that passing a raw PETSc.Mat wraps a iPETScMatrix."""
        raw = PETSc.Mat().createAIJ((3, 3), comm=PETSc.COMM_WORLD)
        raw.setValue(0, 0, 7.0)
        raw.assemble()
        M = iPETScMatrix.from_matrix(raw)

        assert isinstance(M, iPETScMatrix)
        assert M.shape == (3, 3)
        assert pytest.approx(M[0, 0]) == 7.0
        assert "aij" in M.type.lower()

    def test_from_matrix_from_array(self) -> None:
        """Test creating a matrix from a numpy array."""
        arr = np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, -3.5, 0.0],
            ]
        )
        M = iPETScMatrix.from_matrix(arr, comm=PETSc.COMM_WORLD)

        assert isinstance(M, iPETScMatrix)
        assert M.shape == arr.shape
        assert pytest.approx(M[0, 0]) == arr[0, 0]
        assert pytest.approx(M[0, 2]) == arr[0, 2]
        assert pytest.approx(M[1, 1]) == arr[1, 1]

    def test_from_matrix_from_scipy_sparse(self) -> None:
        """SciPy sparse matrices should convert to AIJ with same sparsity and values."""
        data = np.array([10, 20, 30])
        rows = np.array([0, 1, 2])
        cols = np.array([2, 0, 1])
        sp = sparse.coo_matrix((data, (rows, cols)), shape=(4, 4))
        M = iPETScMatrix.from_matrix(sp, comm=PETSc.COMM_WORLD)
        M.assemble()

        assert isinstance(M, iPETScMatrix)
        assert M.shape == (4, 4)
        for i, j, v in zip(rows, cols, data):
            assert pytest.approx(M[i, j]) == v
        assert M.nonzero_entries == len(data)

    def test_from_matrix_invalid_input(self) -> None:
        """Passing an unsupported type should raise a TypeError."""
        with pytest.raises(TypeError):
            _ = iPETScMatrix.from_matrix("not a matrix", comm=PETSc.COMM_WORLD)

    def test_addition(self) -> None:
        """Test matrix addition and reflected addition."""
        A = iPETScMatrix.create_aij((2, 2))
        A[0, 0] = 1.0
        A.assemble()

        B = A.duplicate(copy=True)

        C = A + B

        assert pytest.approx(C[0, 0], abs=1e-10) == 2.0
        assert pytest.approx((B + A)[0, 0], abs=1e-10) == 2.0

    def test_matmul(self) -> None:
        """Test matrix @ vector and matrix @ matrix operations."""
        A = iPETScMatrix.from_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
        vec = iPETScVector.from_array(np.array([1.0, 2.0]))

        result_vec = A @ vec
        mat2 = A @ A

        np.testing.assert_allclose(result_vec.as_array(), [2.0, 6.0])
        assert isinstance(mat2, iPETScMatrix)
        assert pytest.approx(mat2[0, 0], abs=1e-10) == 4.0

    def test_rmatmul_vector_matrix(self) -> None:
        """Test vector @ matrix operation."""
        A = iPETScMatrix.from_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
        vec = iPETScVector.from_array(np.array([3.0, 4.0]))

        result = vec @ A
        np.testing.assert_allclose(result.as_array(), [6.0, 12.0])

    def test_indexing_get_set(self) -> None:
        """Test matrix[i, j] get and set functionality."""
        A = iPETScMatrix.create_aij((2, 2))
        A[1, 0] = 3.0
        A.assemble()

        assert pytest.approx(A[1, 0], abs=1e-10) == 3.0

    def test_properties_transpose_and_shape(self) -> None:
        """Test properties: shape, transpose, nonzero_entries, norm, type."""
        A = iPETScMatrix.from_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]))

        transpose = A.T.to_aij()  # Convert just to support indexing

        assert A.shape == (2, 3)
        assert A.T.shape == (3, 2)
        assert transpose[2, 1] == 2.0
        assert transpose[1, 0] == 1.0

    @skip_complex
    def test_hermitian(self) -> None:
        """Test Hermitian transpose for complex matrices."""
        A = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0 + 0.0j, 4.0 - 2.0j],
                    [3.0 + 5.0j, 2.0 + 0.0j],
                ]
            )
        )

        H = A.H.to_aij()  # Convert just to support indexing

        for i in range(2):
            for j in range(2):
                expected = A[j, i].conjugate()
                assert pytest.approx(H[i, j], abs=1e-12) == expected

    def test_real_hermitian(self) -> None:
        """Test Hermitian transpose for real matrices."""
        A = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, 4.0],
                    [3.0, 2.0],
                ]
            )
        )

        assert A.T.to_aij() == A.H.to_aij()

    def test_symmetry_checks(self) -> None:
        """Test is_symmetric and is_numerically_symmetric methods."""
        A = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, 2.0],
                    [2.0, 2.0],
                ]
            )
        )

        assert A.is_symmetric
        assert A.is_numerically_symmetric(tol=1e-12)

    def test_scale_and_shift(self) -> None:
        """Test scale and shift operations on matrix."""
        A = iPETScMatrix.create_aij((1, 1))
        A[0, 0] = 1.0
        A.assemble()

        A.shift(2.0)
        assert pytest.approx(A[0, 0], abs=1e-10) == 3.0

        A.scale(0.5)
        assert pytest.approx(A[0, 0], abs=1e-10) == 1.5

    @skip_complex
    def test_scale_with_complex_factor(self) -> None:
        """Scale by a purely imaginary factor."""
        A = iPETScMatrix.create_aij((1, 1))
        A[0, 0] = 2.0 + 0.0j
        A.assemble()

        factor = 0.0 + 1.0j
        A.scale(factor)

        expected = (2.0 + 0.0j) * factor
        assert pytest.approx(A[0, 0], abs=1e-12) == expected

    def test_zero_all_entries(self) -> None:
        """Test that zero_all_entries() clears the matrix."""
        A = iPETScMatrix.create_aij((1, 1))
        A[0, 0] = 5.0
        A.assemble()

        A.zero_all_entries()
        assert pytest.approx(A[0, 0], abs=1e-10) == 0.0

    def test_get_value_and_get_row(self) -> None:
        """Test get_value and get_row consistency."""
        A = iPETScMatrix.create_aij((2, 2))
        A[1, 0] = 2.0
        A.assemble()

        cols, values = A.get_row(1)
        for col, val in zip(cols, values):
            assert pytest.approx(A.get_value(1, col), abs=1e-10) == val

    def test_axpy(self) -> None:
        """Test AXPY operation on matrices."""
        A = iPETScMatrix.create_aij((1, 1))
        A[0, 0] = 1.0
        A.assemble()

        A.axpy(2.0, A)
        assert pytest.approx(A[0, 0], abs=1e-10) == 3.0

    @skip_complex
    def test_axpy_with_complex_alpha(self) -> None:
        """Test complex AXPY operation on matrices."""
        A = iPETScMatrix.create_aij((1, 1))
        A[0, 0] = 1.0 + 0.0j
        A.assemble()

        alpha = 2.0 + 3.0j
        A.axpy(alpha, A)

        expected = (alpha + 1.0) * (1.0 + 0.0j)
        assert pytest.approx(A[0, 0], abs=1e-12) == expected

    def test_create_vector_right_left(self) -> None:
        """Test creation of left and right-hand vectors from matrix."""
        A = iPETScMatrix.create_aij((3, 2))
        A.assemble()

        v_r = A.create_vector_right()
        v_l = A.create_vector_left()

        assert v_r.size == 2
        assert v_l.size == 3

    def test_pin_dof(self) -> None:
        """Test pinning a DOF by zeroing its row and column and setting the diagonal."""
        A = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )

        A.pin_dof(0)

        assert pytest.approx(A[0, 0], abs=1e-10) == 1.0
        assert pytest.approx(A[0, 1], abs=1e-10) == 0.0
        assert pytest.approx(A[1, 0], abs=1e-10) == 0.0
        assert pytest.approx(A[1, 1], abs=1e-10) == 4.0

    def test_attach_nullspace(self) -> None:
        """Test attaching a constant nullspace to a matrix."""
        A = iPETScMatrix.create_aij((3, 3))
        A.assemble()

        ns_vec = iPETScVector.create_seq(3)
        for i in range(3):
            ns_vec[i] = 1.0
        ns_vec.assemble()

        ns = iPETScNullSpace.create_constant_and_vectors(A.comm, [ns_vec])
        A.attach_nullspace(ns)

        ns_retrieved = A.get_nullspace()
        assert ns_retrieved is not None
        assert ns_retrieved.test_matrix(A)

    def test_export_matrix(self, tmp_path: Path) -> None:
        """Test exporting matrix to binary format."""
        A = iPETScMatrix.create_aij((2, 2))
        A[0, 1] = 4.2
        A.assemble()

        filepath = tmp_path / "matrix_output.bin"
        A.export(filepath)

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_get_row_simple(self):
        """Test get row."""
        M = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, 0.0, 2.0],
                    [0.0, 3.0, 0.0],
                ]
            )
        )

        cols0, vals0 = M.get_row(0)
        assert cols0 == [0, 1, 2]
        assert pytest.approx(vals0) == [1.0, 0.0, 2.0]

        cols1, vals1 = M.get_row(1)
        assert cols1 == [0, 1, 2]
        assert pytest.approx(vals1) == [0.0, 3.0, 0.0]

    def test_get_row_sparse(self):
        """Test get row with sparse matrices."""
        M = iPETScMatrix.zeros((3, 3))
        M[0, 1] = 42.0
        M.assemble()

        cols0, vals0 = M.get_row(0)
        assert cols0 == [1]
        assert pytest.approx(vals0) == [42]

        cols1, vals1 = M.get_row(1)
        assert cols1 == []
        assert pytest.approx(vals1) == []

    def test_get_column_simple(self):
        """Test get column."""
        M = iPETScMatrix.from_matrix(
            np.array(
                [
                    [5.0, 0.0],
                    [0.0, 7.0],
                    [8.0, 9.0],
                ]
            )
        )

        rows0, vals0 = M.get_column(0)
        assert rows0 == [0, 1, 2]
        assert pytest.approx(vals0) == [5.0, 0.0, 8.0]

        rows1, vals1 = M.get_column(1)
        assert rows1 == [0, 1, 2]
        assert pytest.approx(vals1) == [0.0, 7.0, 9.0]

    def test_get_column_sparse(self):
        """Test get column."""
        M = iPETScMatrix.zeros((2, 3))
        M[0, 0] = 1.0
        M.assemble()

        rows0, vals0 = M.get_column(0)
        assert rows0 == [0]
        assert pytest.approx(vals0) == [1.0]

    def test_add_value_accumulates(self):
        """Test add value."""
        M = iPETScMatrix.zeros((3, 3))
        M.add_value(1, 2, 4.2)
        M.add_value(1, 2, 1.3)
        M.assemble()

        v = M[1, 2]
        assert pytest.approx(v, abs=1e-12) == 5.5

        cols, vals = M.get_row(1)
        assert cols == [2]
        assert pytest.approx(vals) == [5.5]


class TestNullSpace:
    """Tests for the iPETScNullSpace wrapper."""

    def _build_tridiag_matrix(self) -> iPETScMatrix:
        """Build a simple 3x3 tridiagonal matrix whose nullspace is span{[1,1,1]}."""
        return iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-1.0, 2.0, -1.0],
                    [0.0, -1.0, 1.0],
                ],
                dtype=float,
            )
        )

    def test_from_vectors_and_test(self) -> None:
        """Construct nullspace from the constant basis vector and verify test()."""
        A = self._build_tridiag_matrix()
        v1 = iPETScVector.from_array(np.array([1.0, 1.0, 1.0]))
        ns = iPETScNullSpace.from_vectors([v1])
        assert ns.test_matrix(A)

    def test_create_constant_and_test(self) -> None:
        """Test that create_constant() yields the all-ones nullspace for this A."""
        A = self._build_tridiag_matrix()
        ns = iPETScNullSpace.create_constant(comm=A.comm)
        assert ns.has_constant()
        assert ns.test_matrix(A)

    def test_remove_constant(self) -> None:
        """Test that removing a constant nullspace subtracts the mean from the vector."""
        v = iPETScVector.from_array(np.array([2.0, 3.0, 4.0]))
        ns = iPETScNullSpace.create_constant(comm=v.comm)
        ns.remove(v)
        arr = v.as_array()
        # after removal, should be [-1, 0, 1]
        np.testing.assert_allclose(arr, np.array([-1.0, 0.0, 1.0]), atol=1e-12)

    def test_constant_and_vectors(self) -> None:
        """Test that create_constant_and_vectors should include both constant and extra modes."""
        A = self._build_tridiag_matrix()
        v2 = iPETScVector.from_array(np.array([2.0, 2.0, 2.0]))
        ns = iPETScNullSpace.create_constant_and_vectors(A.comm, [v2])

        assert ns.has_constant()
        assert ns.test_matrix(A)


class TestComplexVector:
    """Tests for the iComplexPETScVector wrapper."""

    @skip_real
    def test_initialization_real(self) -> None:
        """Test initialization with real build."""
        vec = iPETScVector.zeros(3)
        real_vec = iComplexPETScVector(vec)
        imag_vec = iComplexPETScVector(real=vec, imag=vec)

        assert real_vec.imag is None
        assert imag_vec.imag is not None

    @skip_complex
    def test_initialization_complex(self) -> None:
        """Test initialization with complex build (imag is never set)."""
        vec = iPETScVector.zeros(3)
        real_vec = iComplexPETScVector(vec)
        imag_vec = iComplexPETScVector(real=vec, imag=vec)

        assert real_vec.imag is None
        assert imag_vec.imag is None

    @skip_real
    def test_lazy_imag_allocation_real(self) -> None:
        """Test lazy imaginary allocation with real build."""
        real_vec = iPETScVector.zeros(2)
        comp_vec = iComplexPETScVector(real_vec)
        comp_vec[0] = 1.0
        comp_vec[1] = -2.5
        comp_vec.assemble()

        # After setting only real values, imaginary part stays None
        assert comp_vec.imag is None

        comp_vec[1] = 3.0 + 4.0j  # Triggers allocation of the imaginary vector
        assert comp_vec.imag is not None
        assert comp_vec[1] == 3.0 + 4.0j
        assert isinstance(comp_vec[0], complex)
        assert comp_vec[0] == 1.0 + 0j

    @skip_complex
    def test_lazy_imag_allocation_complex(self) -> None: ...

    def test_is_complex_property(self) -> None:
        """Test .is_complex both in real and complex PETSc builds."""
        real_vec = iComplexPETScVector.from_array(
            np.array([1.0, 0.0, 3.0]), comm=PETSc.COMM_WORLD
        )
        # If PETSc itself is built with complex support, is_complex must be True even for real matrices
        if np.dtype(PETSc.ScalarType).kind == "c":
            assert real_vec.is_complex
        else:
            assert not real_vec.is_complex

        c_vec = iComplexPETScVector.from_array(
            np.array([1.0 + 0j, 0.0 + 2j, 3.0 + 0j]), comm=PETSc.COMM_WORLD
        )
        assert c_vec.is_complex

    def test_add(self) -> None:
        """Test addition."""
        v1 = iComplexPETScVector.from_array(np.array([1 + 2j, 3 + 4j]))
        v2 = iComplexPETScVector.from_array(np.array([5 + 6j, 7 + 8j]))

        v_sum = v1 + v2

        assert v_sum[0] == pytest.approx(6 + 8j)
        assert v_sum[1] == pytest.approx(10 + 12j)

    def test_sub(self) -> None:
        """Test subtraction."""
        v1 = iComplexPETScVector.from_array(np.array([1 + 2j, 3 + 4j]))
        v2 = iComplexPETScVector.from_array(np.array([5 + 6j, 7 + 8j]))

        v_diff = v2 - v1

        assert v_diff[0] == pytest.approx(4 + 4j)
        assert v_diff[1] == pytest.approx(4 + 4j)

    @skip_real
    def test_add_sub_lazy_alloc(self) -> None:
        """Test that adding two purely real vectors yields imag=None (lazy behavior)."""
        v1 = iComplexPETScVector.from_array(np.array([1, 3]))
        v2 = iComplexPETScVector.from_array(np.array([5, 7]))

        v_sum = v1 + v2

        assert v_sum.imag is None
        assert v_sum[0] == pytest.approx(6)
        assert v_sum[1] == pytest.approx(10)

    @skip_real
    def test_scalar_mul_and_scale_real(self) -> None:
        """Test scalar mul and in-place scale with real build and lazy imag allocation."""
        comp = iComplexPETScVector.from_array(np.array([2.0, 0.5]))

        # Real scalar multiply
        v2 = comp * 2.0
        assert isinstance(v2[0], (int, float, complex))
        assert v2.imag is None
        assert v2[0] == pytest.approx(4.0)
        assert v2[1] == pytest.approx(1.0)

        # Complex scalar multiply: triggers imag allocation
        v3 = comp * (1 + 1j)
        assert v3.imag is not None
        assert v3[0] == pytest.approx(2.0 + 2.0j)
        assert v3[1] == pytest.approx(0.5 + 0.5j)

        # In-place scale
        comp.scale(1 + 1j)
        assert comp.imag is not None
        assert comp[0] == pytest.approx(2.0 + 2.0j)
        assert comp[1] == pytest.approx(0.5 + 0.5j)

    @skip_complex
    def test_scalar_mul_and_scale_complex(self) -> None:
        """Test scalar mul and scale with complex build (imag never allocated)."""
        comp = iComplexPETScVector.from_array(np.array([2.0, 0.5]))
        v2 = comp * 2.0
        assert comp.imag is None
        assert v2[0] == pytest.approx(4.0)
        assert v2[1] == pytest.approx(1.0)

        v3 = comp * (1 + 1j)
        assert comp.imag is None
        assert v3[0] == pytest.approx(2.0 + 2.0j)
        assert v3[1] == pytest.approx(0.5 + 0.5j)

        comp.scale(1 + 1j)
        assert comp.imag is None
        assert comp[0] == pytest.approx(2.0 + 2.0j)
        assert comp[1] == pytest.approx(0.5 + 0.5j)

    def test_matmul_mat_vec(self) -> None:
        """Test iPETScMatrix @ iComplexPETScVector."""
        mat = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )
        vec = iComplexPETScVector.from_array(np.array([1 + 1j, 2 + 2j]))

        res = mat @ vec

        # Expected: [1*(1+1j)+2*(2+2j), 3*(1+1j)+4*(2+2j)]
        exp0 = (1 + 1j) + 2 * (2 + 2j)
        exp1 = 3 * (1 + 1j) + 4 * (2 + 2j)

        assert isinstance(res, iComplexPETScVector)
        assert res.is_complex
        assert res[0] == pytest.approx(exp0)
        assert res[1] == pytest.approx(exp1)

    def test_matmul_vec_mat(self) -> None:
        """Test iComplexPETScVector @ iPETScMatrix."""
        mat = iPETScMatrix.from_matrix(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )
        vec = iComplexPETScVector.from_array(np.array([1 + 1j, 2 + 2j]))

        res = vec @ mat

        # expected: [(1+1j)*1 + (2+2j)*3, (1+1j)*2 + (2+2j)*4 ]
        exp0 = (1 + 1j) * 1 + (2 + 2j) * 3
        exp1 = (1 + 1j) * 2 + (2 + 2j) * 4

        assert isinstance(res, iComplexPETScVector)
        assert res.is_complex
        assert res[0] == pytest.approx(exp0)
        assert res[1] == pytest.approx(exp1)

    @skip_real
    def test_norm_and_dot_real(self) -> None:
        """Test norm() and dot() in real build."""
        v = iComplexPETScVector.from_array(np.array([3 + 4j, 0]), comm=PETSc.COMM_WORLD)
        w = iComplexPETScVector.from_array(np.array([3 - 4j, 0]), comm=PETSc.COMM_WORLD)

        assert v.norm() == pytest.approx(5.0)
        assert w.norm() == pytest.approx(5.0)

        # v·v = 25
        dvv = v.dot(v)
        assert dvv.real == pytest.approx(25.0)
        assert dvv.imag == pytest.approx(0.0)

        # w·w = 25
        dww = w.dot(w)
        assert dww.real == pytest.approx(25.0)
        assert dww.imag == pytest.approx(0.0)

        # v·w = (3 - 4j)*(3 - 4j) = -7 - 24j
        dvw = v.dot(w)
        assert dvw.real == pytest.approx(-7.0)
        assert dvw.imag == pytest.approx(-24.0)

        # w·v = (3 + 4j)*(3 + 4j) = -7 + 24j
        dwv = w.dot(v)
        assert dwv.real == pytest.approx(-7.0)
        assert dwv.imag == pytest.approx(24.0)

    @skip_complex
    def test_norm_and_dot_complex(self) -> None: ...

    def test_copy_and_equality(self) -> None:
        """Test copy() and equality operators using from_array."""
        data = np.array([1.0, 2 + 3j, -4.5])
        vec = iComplexPETScVector.from_array(data, comm=PETSc.COMM_WORLD)

        vec_copy = vec.copy()
        assert vec_copy == vec

        # mutating original breaks equality
        vec[0] = 9.0
        assert vec_copy != vec
