"""LSA-FW FEM utilities."""

from __future__ import annotations

import logging

from enum import Enum, auto
from typing import overload, TypeAlias
from pathlib import Path
from copy import deepcopy
from scipy import sparse
import numpy as np


from basix import ElementFamily as DolfinxElementFamily
from dolfinx.mesh import Mesh, MeshTags
from ufl import Measure  # type: ignore[import-untyped]
from petsc4py import PETSc

logger = logging.getLogger(__name__)

Scalar: TypeAlias = PETSc.ScalarType
"""Alias for the base numeric type used throughout the framework (float or complex).

Depending on how PETSc was configured, `PETSc.ScalarType` will be either
`float` (real builds) or `complex` (complex builds).
"""

_IS_COMPLEX_BUILD: bool = np.dtype(Scalar).kind == "c"


class iElementFamily(Enum):
    """Internal element family identifiers.

    Provides an abstraction over basix.ElementFamily with IDE-friendly names,
    and conversion to/from Dolfinx.
    """

    LAGRANGE = auto()
    """Continuous Lagrange element (typically P1, P2, etc.)."""
    P = auto()
    """Alias for Lagrange (polynomial basis)."""
    BUBBLE = auto()
    """Interior bubble function (zero on element boundaries)."""
    RT = auto()
    """Raviart-Thomas element (H(div)-conforming)."""
    BDM = auto()
    """Brezzi-Douglas-Marini element (H(div)-conforming, higher order)."""
    CR = auto()
    """Crouzeix-Raviart element (nonconforming, continuous at edge midpoints)."""
    DPC = auto()
    """Discontinuous piecewise constant (or higher) element on quadrilateral/hexahedral meshes."""
    N1E = auto()
    """Nédélec 1st kind (H(curl)-conforming)."""
    N2E = auto()
    """Nédélec 2nd kind (H(curl)-conforming, higher order)."""
    HHJ = auto()
    """Hellan-Herrman-Johnson element (for symmetric tensors in plate bending)."""
    REGGE = auto()
    """Regge element (used for elasticity with symmetric gradients)."""
    SERENDIPITY = auto()
    """Reduced basis for tensor-product cells (fewer DOFs than full Q space)."""
    HERMITE = auto()
    """Hermite element (includes derivative DOFs, e.g., for $C^1$ continuity)."""
    ISO = auto()
    """Isoparametric element (geometry interpolated using same basis)."""

    def to_dolfinx(self) -> DolfinxElementFamily:
        """Convert to Dolfinx (Basix) ElementFamily enum."""
        return _MAP_TO_DOLFINX[self]

    @classmethod
    def from_dolfinx(cls, family: DolfinxElementFamily) -> iElementFamily:
        """Convert from Dolfinx ElementFamily to internal enum."""
        return _MAP_FROM_DOLFINX[family]

    @classmethod
    def from_string(cls, name: str) -> iElementFamily:
        """Create from string (case-insensitive)."""
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown element family '{name}'. Valid options: {list(cls.__members__.keys())}"
            )


class iMeasure:
    """Helper for constructing tagged facet measures."""

    @staticmethod
    def ds(mesh: Mesh, tags: MeshTags, name: str = "ds") -> Measure:
        return Measure(name, domain=mesh, subdomain_data=tags)

    @staticmethod
    def dS(mesh: Mesh, tags: MeshTags, name: str = "dS") -> Measure:
        return Measure(name, domain=mesh, subdomain_data=tags)


class iPETScMatrix:
    """Minimal wrapper around PETSc matrix to provide a consistent interface.

    Note that this is not a complete wrapper and does not implement all methods or properties of a PETSc matrix.
    This is intended to be a light-weight typed solution to improve the interface with PETSc.

    Refer to the official PETSc documentation for more details: https://petsc.org/release/docs/manualpages/Mat/.
    """

    def __init__(self, mat: PETSc.Mat) -> None:
        """Initialize PETSc matrix wrapper."""
        try:
            mat.assemble()
        except Exception:
            logger.warning("PETSc matrix could not be assembled upon initialization.")
        self._mat = mat

    @classmethod
    def from_nested(
        cls, blocks: list[list[PETSc.Mat | None]], comm: PETSc.Comm = PETSc.COMM_WORLD
    ) -> iPETScMatrix:
        """Create a nested PETSc matrix from a list of blocks and wrap it as an iPETScMatrix.

        The input is a list of lists representing a block matrix structure.
        Each inner list corresponds to a block row, and its elements correspond to block columns.
        PETSc.Mat instances are used where sub-blocks exist, and `None` is used for empty blocks.

        Example:

            nested = iPETScMatrix.from_nested([
                [A, G],
                [D, None],
            ])

        This builds the nested matrix:
            [ A   G  ]
            [ D   0  ]
        """
        mat = PETSc.Mat().createNest(blocks, comm=comm)
        mat.assemble()
        return cls(mat)

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        comm: PETSc.Comm = PETSc.COMM_WORLD,
        nnz: int | None = None,
    ) -> iPETScMatrix:
        """Initialize a zero matrix of given size."""
        mat = PETSc.Mat().createAIJ(shape, nnz=nnz, comm=comm)
        mat.assemble()
        return cls(mat)

    @classmethod
    def create_aij(
        cls,
        shape: tuple[int],
        comm: PETSc.Comm = PETSc.COMM_SELF,
        nnz: int | list[int] | None = None,
    ) -> iPETScMatrix:
        """Create an AIJ (sparse) matrix of the given global shape and wrap it.

        This is roughly equivalent to:

            A = PETSc.Mat().createAIJ(shape, nnz=nnz, comm=comm)
            A.setUp()
            return iPETScMatrix(A)

        If you a different preallocation pattern (e.g. per-row) is needed,
        pass in a sequence of nnz counts instead of a single int.
        """
        mat = PETSc.Mat().createAIJ(shape, nnz=nnz, comm=comm)
        mat.setUp()
        return cls(mat)

    @classmethod
    def from_matrix(
        cls,
        matrix: PETSc.Mat | iPETScMatrix | np.ndarray | sparse.spmatrix,
        comm: PETSc.Comm = PETSc.COMM_WORLD,
    ) -> iPETScMatrix:
        """Construct a matrix from various matrix-like objects.

        - If given an iPETScMatrix, returns it directly.
        - If given a PETSc.Mat, wraps and returns it.
        - If given a NumPy ndarray, creates a dense PETSc matrix.
        - If given a SciPy sparse matrix, creates an AIJ PETSc matrix with identical sparsity.
        """
        if isinstance(matrix, cls):
            return matrix

        if isinstance(matrix, PETSc.Mat):
            return cls(matrix)

        if isinstance(matrix, np.ndarray):
            mat = PETSc.Mat().createDense(matrix.shape, array=matrix, comm=comm)
            mat.assemble()
            return cls(mat)

        if sparse.isspmatrix(matrix):
            csr = matrix.tocsr()
            row_nnz = (csr.indptr[1:] - csr.indptr[:-1]).tolist()
            mat = PETSc.Mat().createAIJ(
                csr.shape,
                nnz=row_nnz,
                comm=comm,
            )
            coo = csr.tocoo()
            for i, j, v in zip(coo.row, coo.col, coo.data):
                mat.setValue(i, j, v, addv=PETSc.InsertMode.INSERT_VALUES)
            mat.assemble()
            return cls(mat)

        raise TypeError(
            f"Cannot construct iPETScMatrix from object of type {type(matrix)}"
        )

    def __str__(self) -> str:
        return f"iPETScMatrix(shape={self.shape}, nnz={self.nonzero_entries})"

    def __add__(self, other: object) -> iPETScMatrix:
        """Perform matrix addition."""
        if not isinstance(other, iPETScMatrix):
            raise NotImplementedError(f"Cannot add iPETScMatrix with {type(other)}")
        if self.shape != other.shape:
            raise ValueError(
                f"Incompatible matrix shapes: {self.shape} vs {other.shape}"
            )

        result = self._mat.duplicate(copy=True)
        result.axpy(Scalar(1.0), other.raw)
        return iPETScMatrix(result)

    def __radd__(self, other: object) -> iPETScMatrix:
        """Perform matrix addition from the right."""
        return self.__add__(other)

    @overload
    def __matmul__(self, other: iPETScVector) -> iPETScVector: ...

    @overload
    def __matmul__(self, other: iPETScMatrix) -> iPETScMatrix: ...

    def __matmul__(
        self, other: iPETScVector | iPETScMatrix
    ) -> iPETScVector | iPETScMatrix:
        """Perform matrix-vector or matrix-matrix multiplication."""
        match other:
            case iPETScVector():
                if self.shape[1] != other.size:
                    raise ValueError(
                        f"Incompatible matrix-vector shapes: {self.shape[1]} vs {other.size}"
                    )
                product = self._mat.createVecRight()
                self._mat.mult(other.raw, product)
                return iPETScVector(product)

            case iPETScMatrix():
                if self.shape[1] != other.shape[0]:
                    raise ValueError(
                        f"Incompatible matrix-matrix shapes: {self.shape} vs {other.shape}"
                    )
                try:
                    result = self._mat.matMult(other.raw)
                except PETSc.Error as e:
                    raise NotImplementedError(
                        f"Matrix multiplication not supported for type '{self.type}': {e}"
                    )
                return iPETScMatrix(result)

            case _:
                return NotImplemented

    def __rmatmul__(self, other: object) -> iPETScVector:
        """Perform vector-matrix multiplication."""
        if not isinstance(other, iPETScVector):
            raise NotImplementedError(
                f"Cannot multiply iPETScMatrix with {type(other)}"
            )
        if self.shape[0] != other.size:
            raise ValueError(
                f"Incompatible matrix and vector sizes: {self.shape[0]} vs {other.size}"
            )
        result = self._mat.createVecLeft()
        self._mat.multTranspose(other.raw, result)
        return iPETScVector(result)

    def __getitem__(self, indices: tuple[int, int]) -> Scalar:
        """Get a value via direct indexation (i.e., matrix[i, j])."""
        if self.type.lower() == "nest":
            raise NotImplementedError(
                "Direct indexing is not supported for nested matrices. "
                "Use `raw.getNestSubMat()` to access individual blocks."
            )
        return self.get_value(indices[0], indices[1])

    def __setitem__(
        self, indices: tuple[int, int], value: int | float | complex
    ) -> None:
        """Set a value via direct indexation (i.e., matrix[i, j] = value)."""
        if self.type.lower() == "nest":
            raise NotImplementedError(
                "Direct assignment is not supported for nested matrices. "
                "Use sub-matrix access or flatten the structure manually."
            )
        self._mat.setValue(
            indices[0], indices[1], Scalar(value), addv=PETSc.InsertMode.INSERT_VALUES
        )

    def __eq__(self, other: object) -> bool:
        """Matrix equality."""
        if not isinstance(other, iPETScMatrix):
            return NotImplemented

        if self.shape != other.shape:
            return False

        diff = self.raw.copy()
        diff.axpy(
            Scalar(-1.0),
            other.raw,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )
        return diff.norm() < 1e-12

    @property
    def raw(self) -> PETSc.Mat:
        """Return the underlying PETSc matrix."""
        return self._mat

    @property
    def comm(self) -> PETSc.Comm:
        """Return the PETSc communicator associated with the matrix."""
        return self._mat.comm

    @property
    def T(self) -> iPETScMatrix:
        """Return the transpose of the matrix."""
        return iPETScMatrix(PETSc.Mat().createTranspose(self._mat))

    @property
    def H(self) -> iPETScMatrix:
        """Return the Hermitian transpose of the matrix."""
        return iPETScMatrix(PETSc.Mat().createHermitianTranspose(self._mat))

    @property
    def shape(self) -> tuple[int, int]:
        return self._mat.getSize()

    @property
    def nonzero_entries(self) -> int:
        """Return the number of non-zero entries in the matrix."""
        if "dense" in self.type:
            logger.warning(
                "The current matrix is dense, so all the entries are memory-allocated. "
                "Then, the value returned by this property might not be useful."
            )
        return self._mat.getInfo()["nz_used"]

    @property
    def norm(self) -> Scalar:
        """Return the Frobenius norm of the matrix."""
        return self._mat.norm()

    @property
    def type(self) -> str:
        """Return the PETSc matrix type (e.g., 'aij', 'baij')."""
        return self._mat.getType()

    @property
    def is_symmetric(self) -> bool:
        """Check whether the matrix is symmetric."""
        return self._mat.isSymmetric()

    def is_numerically_symmetric(self, tol: float = 1e-6) -> bool:
        """Check whether the matrix is numerically symmetric.

        This is more robust than `is_symmetric`, which may fail due to rounding errors,
        reordering, or boundary condition insertions.
        """
        diff = self.raw.copy()
        diff.axpy(
            Scalar(-1.0),
            self.T.raw,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )
        return diff.norm() < tol

    def is_hermitian(self) -> bool:
        """Check whether the matrix is Hermitian.

        For real matrices, it should return the same as `is_numerically_symetric`.
        """
        return self._mat.isHermitian()

    def is_numerically_hermitian(self, tol: float = 1e-4) -> bool:
        """Check whether the matrix is numerically Hermitian.

        This is more robust than `is_hermitian`, which may fail due to rounding errors,
        reordering, or boundary condition insertions.
        """
        diff = self.raw.copy()
        diff.axpy(
            Scalar(-1.0),
            self.H.to_aij().raw,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )
        return diff.norm() < tol

    def assemble(self) -> None:
        """(Re)assemble matrix after any insert."""
        self._mat.assemble()

    def print(self) -> None:
        """Print the matrix."""
        self._mat.view()

    def scale(self, alpha: int | float | complex) -> None:
        """Scale the matrix by a constant factor."""
        self._mat.scale(Scalar(alpha))

    def shift(self, alpha: int | float | complex) -> None:
        """Shift the diagonal of the matrix by a constant."""
        self._mat.shift(Scalar(alpha))

    def sub(self, row: int, col: int) -> iPETScMatrix | None:
        """Return the (row, col)-th submatrix from a nested matrix."""
        if self.type.lower() != "nest":
            raise NotImplementedError(
                "Submatrix access is only available for nested matrices."
            )

        sub_mat = self._mat.getNestSubMatrix(row, col)
        if sub_mat.handle == 0:
            return None
        return iPETScMatrix(sub_mat)

    def zero_all_entries(self) -> None:
        """Zero all entries in the matrix."""
        self._mat.zeroEntries()

    def get_value(self, row: int, col: int) -> Scalar:
        """Get the value at position (row, col)."""
        return self._mat.getValue(row, col)

    def get_row(self, row: int) -> tuple[list[int], list[Scalar]]:
        """Get column indices and values for a specific row."""
        cols, values = self._mat.getRow(row)
        return cols.tolist(), values.tolist()

    def axpy(self, alpha: int | float | complex, other: object) -> None:
        """Perform an AXPY operation: this = alpha * other + this."""
        if not isinstance(other, iPETScMatrix):
            raise NotImplementedError(f"Cannot add iPETScMatrix with {type(other)}")
        if self.shape != other.shape:
            raise ValueError(
                f"Incompatible matrix shapes: {self.shape} vs {other.shape}"
            )
        self._mat.axpy(
            Scalar(alpha),
            other.raw,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )

    def create_vector_right(self) -> iPETScVector:
        """Create a vector for right-hand side operations."""
        return iPETScVector(self._mat.createVecRight())

    def create_vector_left(self) -> iPETScVector:
        """Create a vector for left-hand side operations."""
        return iPETScVector(self._mat.createVecLeft())

    def duplicate(self, copy: bool = False) -> iPETScMatrix:
        """Return a new matrix with the same layout (and optionally, values)."""
        new_mat = self._mat.duplicate(copy=copy)
        return iPETScMatrix(new_mat)

    def to_aij(self) -> iPETScMatrix:
        """Convert a nested (MatNest) matrix to a flat AIJ matrix."""
        if self.type.lower() not in ("nest", "hermitiantranspose", "transpose"):
            raise NotImplementedError(
                "Only MatNest matrices can be flattened with `to_aij()`."
            )

        aij = self._mat.convert("aij")
        aij.assemble()
        return iPETScMatrix(aij)

    def as_array(self) -> np.ndarray:
        """Return the vector as a NumPy array."""
        return self._mat.getArray().copy()

    def zero_row_columns(
        self, rows: list[int], diag: int | float | complex = 0.0
    ) -> None:
        """Zero row and column and place a given value in the diagonal."""
        self._mat.zeroRowsColumns(rows, diag=Scalar(diag))

    def pin_dof(self, index: int) -> None:
        """Pin a single DOF by zeroing its row and column and placing a 1 on the diagonal.

        This is commonly used to eliminate nullspaces and ensure compatibility with solvers that do not handle
        them, e.g., fixing pressure at a point in incompressible flow problems.
        """
        self._mat.zeroRowsColumns([index], diag=Scalar(1.0))

    def attach_nullspace(self, nullspace: iPETScNullSpace) -> None:
        """Attach an existing PETSc nullspace object to this matrix."""
        self._mat.setNearNullSpace(nullspace.raw)
        self._mat.setNullSpace(nullspace.raw)

    def get_nullspace(self) -> iPETScNullSpace | None:
        """Retrieve attached nullspace."""
        try:
            return iPETScNullSpace(self.raw.getNullSpace())
        except Exception:
            return None

    def export(self, path: Path) -> None:
        """Export the matrix to a binary file."""
        viewer = PETSc.Viewer().createBinary(
            str(path), mode=PETSc.Viewer.Mode.WRITE, comm=self.comm
        )
        viewer(self.raw)
        viewer.destroy()


class iPETScVector:
    """Minimal wrapper around PETSc vector to provide a consistent interface.

    Note that this is not a complete wrapper and does not implement all methods or properties of a PETSc vector.
    This is intended to be a light-weight typed solution to improve the interface with PETSc.
    Refer to the official PETSc documentation for more details: https://petsc.org/release/docs/manualpages/Vec/.
    """

    def __init__(self, vec: PETSc.Vec) -> None:
        """Initialize PETSc vector wrapper."""
        try:
            vec.assemble()
        except Exception:
            logger.warning("PETSc vector could not be assembled upon initialization.")
        self._vec = vec

    @classmethod
    def zeros(cls, size: int, comm: PETSc.Comm = PETSc.COMM_WORLD) -> iPETScVector:
        """Initialize a zero vector of given size."""
        vec = PETSc.Vec().create(comm=comm)
        vec.setSizes(size)
        vec.setFromOptions()
        vec.set(Scalar(0.0))
        return cls(vec)

    @classmethod
    def from_array(
        cls, array: np.ndarray, comm: PETSc.Comm = PETSc.COMM_WORLD
    ) -> iPETScVector:
        """Create a vector from a NumPy array."""
        vec = PETSc.Vec().createWithArray(deepcopy(array), comm=comm)
        vec.setFromOptions()
        return cls(vec)

    @classmethod
    def create_seq(cls, size: int, comm: PETSc.Comm = PETSc.COMM_SELF) -> iPETScVector:
        """Create a sequential (non-parallel) vector of the given size."""
        vec = PETSc.Vec().createSeq(size, comm=comm)
        vec.setFromOptions()
        return cls(vec)

    def __add__(self, other: object) -> iPETScVector:
        """Perform vector addition."""
        if not isinstance(other, iPETScVector):
            raise NotImplementedError(f"Cannot add iPETScVector with {type(other)}")
        if self.size != other.size:
            raise ValueError(f"Incompatible vector sizes: {self.size} vs {other.size}")
        if self.comm != other.comm:
            raise ValueError(
                f"Incompatible vector communicators: {self.comm} vs {other.comm}"
            )
        result = self.copy()
        result._vec.axpy(Scalar(1.0), other.raw)
        return result

    def __radd__(self, other: object) -> iPETScVector:
        """Perform vector addition from the right."""
        return self.__add__(other)

    def __sub__(self, other: object) -> iPETScVector:
        """Perform vector subtraction."""
        if not isinstance(other, iPETScVector):
            raise NotImplementedError(
                f"Cannot subtract iPETScVector with {type(other)}"
            )
        if self.size != other.size:
            raise ValueError(f"Incompatible vector sizes: {self.size} vs {other.size}")
        if self.comm != other.comm:
            raise ValueError(
                f"Incompatible vector communicators: {self.comm} vs {other.comm}"
            )
        result = self.copy()
        result._vec.axpy(Scalar(-1.0), other.raw)
        return result

    @overload
    def __mul__(self, other: int | float | complex) -> iPETScVector: ...

    @overload
    def __mul__(self, other: iPETScVector) -> float | complex: ...

    def __mul__(
        self, other: int | float | complex | iPETScVector
    ) -> iPETScVector | float | complex:
        """Perform scalar-vector multiplication or inner product."""
        if isinstance(other, iPETScVector):
            return self.dot(other)
        result = self.copy()
        result.scale(other)
        return result

    def __rmul__(self, alpha: int | float | complex) -> iPETScVector:
        """Perform vector-scalar multiplication."""
        return self.__mul__(alpha)

    def __matmul__(self, other: iPETScVector) -> iPETScMatrix:
        """Vector outer product."""
        if not isinstance(other, iPETScVector):
            return NotImplemented

        logger.warning(
            "Vector outer product creates a dense matrix; if you need a sparse result, "
            "override __matmul__ for sparse vectors."
        )

        A = PETSc.Mat().createDense([self.size, other.size], comm=self.comm)
        A.setUp()
        x = self.raw
        y = other.raw
        for i in range(self.size):
            xi = x.getValue(i)
            for j in range(other.size):
                yj = y.getValue(j)
                A.setValue(i, j, xi * yj)
        A.assemble()
        return iPETScMatrix(A)

    def __getitem__(self, index: int) -> int | float | complex:
        """Get a value via direct indexation (i.e., vector[i])."""
        return self._vec.getValue(index)

    def __setitem__(self, index: int, value: int | float | complex) -> None:
        """Set a value via direct indexation (i.e., vector[i] = value)."""
        self._vec.setValue(index, Scalar(value), addv=PETSc.InsertMode.INSERT_VALUES)
        self._vec.assemble()

    def __eq__(self, other: object) -> bool:
        """Vector equality."""
        if not isinstance(other, iPETScVector):
            return NotImplemented

        if self.size != other.size:
            return False

        diff = self.raw.duplicate()
        diff.axpy(Scalar(-1.0), self.raw, other.raw)  # diff = other - self
        return diff.norm() < 1e-12

    @property
    def raw(self) -> PETSc.Vec:
        """Return the underlying PETSc vector."""
        return self._vec

    @property
    def comm(self) -> PETSc.Comm:
        """Return the PETSc communicator associated with the vector."""
        return self._vec.getComm()

    @property
    def size(self) -> int:
        """Return the size of the vector."""
        return self._vec.getSize()

    @property
    def norm(self) -> Scalar:
        """Return the 2-norm of the vector."""
        return self._vec.norm()

    def copy(self) -> iPETScVector:
        """Create a copy of the vector."""
        return iPETScVector(self._vec.copy())

    def duplicate(self) -> iPETScVector:
        """Return a new vector of the same size (optionally copying values)."""
        new_vec = self._vec.duplicate()
        return iPETScVector(new_vec)

    def assemble(self) -> iPETScVector:
        """(Re)assemble vector after any change."""
        self._vec.assemble()

    def scale(self, alpha: int | float | complex) -> None:
        """Scale the vector by a constant factor."""
        self._vec.scale(alpha)

    def zero_all_entries(self) -> None:
        """Zero all entries in the vector."""
        self._vec.set(Scalar(0.0))

    def get_value(self, i: int) -> Scalar:
        """Get the value at index i."""
        return self._vec.getValue(i)

    def set_value(self, i: int, value: int | float | complex) -> None:
        """Set the value at index i."""
        self._vec.setValue(i, Scalar(value))

    def as_array(self) -> np.ndarray:
        """Return the vector as a NumPy array."""
        return self._vec.getArray().copy()

    def ghost_update(
        self,
        addv: PETSc.InsertMode = PETSc.InsertMode.INSERT_VALUES,
        mode: PETSc.ScatterMode = PETSc.ScatterMode.FORWARD,
    ) -> None:
        """Synchronize ghost (parallel) entries."""
        self._vec.ghostUpdate(addv=addv, mode=mode)

    def axpy(self, alpha: int | float | complex, other: iPETScVector) -> None:
        """Perform an AXPY operation: this = alpha * other + this."""
        self._vec.axpy(Scalar(alpha), other._vec)

    def set_array(self, array: np.ndarray) -> None:
        """Directly replace the underlying array (and rebuild ghost entries)."""
        self._vec.setArray(array)

    def set_random(self, rng: PETSc.Random | None = None) -> None:
        """Set the vector to random values."""
        if rng is None:
            rng = PETSc.Random().create(comm=self._vec.comm)
            rng.setFromOptions()
        self._vec.setRandom(rng)

    def dot(self, other: iPETScVector) -> int | float | complex:
        """Vector inner product."""
        return self._vec.dot(other.raw)

    def print(self) -> None:
        self._vec.view()

    def export(self, path: Path) -> None:
        """Export the vector to a binary file."""
        viewer = PETSc.Viewer().createBinary(
            str(path), mode=PETSc.Viewer.Mode.WRITE, comm=self._vec.comm
        )
        self._vec.view(viewer)


class iComplexPETScVector:
    """Wrapper for PETSc vectors supporting an optional imaginary part.

    Note that if the PETSc build is real and the user never uses the imaginary part (never sets a complex value or
    scales by a complex number), then self._imag stays None (saving memory). The imaginary vector is allocated on
    the first assignment or operation that produces a non-zero imaginary component.
    """

    def __init__(
        self,
        real: PETSc.Vec | iPETScVector,
        imag: PETSc.Vec | iPETScVector | None = None,
    ):
        """Create complex vector."""
        self._real = real if isinstance(real, iPETScVector) else iPETScVector(real)
        if _IS_COMPLEX_BUILD:
            # In a complex PETSc build, ignore any provided imaginary part
            self._imag = None
            if imag is not None:
                logger.warning(
                    "PETSc is built in a complex-floating point configuration. "
                    "Please use `iPETScVector`."
                )
        else:
            # In a real PETSc build, accept or lazy-allocate imag
            self._imag = (
                None
                if imag is None
                else imag if isinstance(imag, iPETScVector) else iPETScVector(imag)
            )

    @classmethod
    def from_array(
        cls, data: np.ndarray | sparse.spmatrix, comm: PETSc.Comm = PETSc.COMM_SELF
    ) -> iComplexPETScVector:
        """Construct an iComplexPETScVector from
        - a 1-D numpy array or
        - a SciPy sparse column-vector (shape (n,1) or (1,n)).
        """
        if isinstance(data, np.ndarray):
            arr = data
        elif sparse.isspmatrix(data):
            mat = data.tocsr()
            r, c = mat.shape
            if c == 1:
                arr = mat.toarray().ravel()
            elif r == 1:
                arr = mat.toarray().ravel()
            else:
                raise ValueError(f"Cannot treat shape {mat.shape} as vector")
        else:
            raise TypeError(f"Unsupported type {type(data)}")

        n = arr.size
        real = iPETScVector.zeros(n, comm=comm)
        imag = None
        if not _IS_COMPLEX_BUILD and np.iscomplexobj(arr):
            imag = iPETScVector.zeros(n, comm=comm)

        # Only set non-zeros
        # For a dense array we still scan all entries, but sparse input
        # came through as dense only where needed
        nz = np.nonzero(arr)[0]
        for i in nz:
            real[i] = arr[i].real
            if imag is not None:
                imag[i] = arr[i].imag

        real.assemble()
        if imag is not None:
            imag.assemble()

        if _IS_COMPLEX_BUILD:
            return cls(real)
        else:
            return cls(real, imag)

    @property
    def real(self) -> iPETScVector:
        """Get the real part of this vector."""
        return self._real

    @property
    def imag(self) -> iPETScVector | None:
        """Get the imaginary part (iPETScVector) of this vector, or None if unused."""
        return self._imag

    @property
    def is_complex(self) -> bool:
        """Return True if this vector carries any imaginary component."""
        return _IS_COMPLEX_BUILD or (self._imag is not None)

    def get_value(self, index: int) -> float | complex:
        """Get the value at the given index."""
        if self._imag is None:
            return self._real.get_value(index)
        return self._real.get_value(index) + 1j * self._imag.get_value(index)

    def set_value(self, index: int, value: float | complex) -> None:
        """Set the value at the given index. Accepts float (real) or complex."""
        if _IS_COMPLEX_BUILD:
            # PETSc will handle both real and complex values natively.
            self._real.set_value(index, value)
            return

        # Real‐only PETSc: we need to unpack complex ourselves.
        if isinstance(value, complex):
            a, b = float(value.real), float(value.imag)
            self._real.set_value(index, a)
            if b != 0.0 or self._imag is not None:
                if self._imag is None:
                    self._imag = self._real.duplicate()
                self._imag.set_value(index, b)
        else:
            # Pure real assignment
            self._real.set_value(index, value)
            if self._imag is not None:
                # Ensure any existing imag part is zeroed out here.
                self._imag.set_value(index, 0.0)

    def __getitem__(self, index: int) -> float | complex:
        """Indexing (vec[idx])."""
        return self.get_value(index)

    def __setitem__(self, index: int, value: float | complex) -> None:
        """Index assignment (vec[idx] = value)."""
        self.set_value(index, value)

    def __add__(self, other: iComplexPETScVector | iPETScVector) -> iComplexPETScVector:
        """Vector addition."""
        if isinstance(other, iPETScVector):
            return self.__add__(iComplexPETScVector(other))

        if not isinstance(other, iComplexPETScVector):
            return NotImplemented

        if _IS_COMPLEX_BUILD:
            return iComplexPETScVector(self._real + other.real)

        # Both purely real
        if self._imag is None and other.imag is None:
            return iComplexPETScVector(self._real + other.real)

        # At least one has imag
        new_real = self._real + other.real
        zero_self = self._imag or (self._real * 0.0)
        zero_other = other.imag or (other.real * 0.0)
        new_imag = zero_self + zero_other
        return iComplexPETScVector(new_real, new_imag)

    def __sub__(self, other: iComplexPETScVector | iPETScVector) -> iComplexPETScVector:
        """Vector subtraction."""
        if isinstance(other, iPETScVector):
            return self.__sub__(iComplexPETScVector(other))

        if not isinstance(other, iComplexPETScVector):
            return NotImplemented

        if _IS_COMPLEX_BUILD:
            return iComplexPETScVector(self._real - other.real)

        if self._imag is None and other.imag is None:
            return iComplexPETScVector(self._real - other.real)

        new_real = self._real - other.real
        zero_self = self._imag or (self._real * 0.0)
        zero_other = other.imag or (other.real * 0.0)
        new_imag = zero_self - zero_other
        return iComplexPETScVector(new_real, new_imag)

    def __mul__(self, scalar: int | float | complex) -> iComplexPETScVector:
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented

        if _IS_COMPLEX_BUILD:
            return iComplexPETScVector(self._real * scalar)

        # Real PETSc build: implement (a + ib)*(x + iy)
        if isinstance(scalar, complex):
            a, b = scalar.real, scalar.imag
            if self._imag is None:
                # x-only vector
                new_real = self._real * a
                new_imag = (self._real * b) if b != 0.0 else None
                return iComplexPETScVector(new_real, new_imag)
            # Both parts exist
            new_real = (self._real * a) - (self._imag * b)
            new_imag = (self._real * b) + (self._imag * a)
            return iComplexPETScVector(new_real, new_imag)

        # Real‐only scalar
        alpha = float(scalar)
        new_real = self._real * alpha
        new_imag = (self._imag * alpha) if self._imag is not None else None
        return iComplexPETScVector(new_real, new_imag)

    def __rmul__(self, scalar: float | complex) -> iComplexPETScVector:
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    @overload
    def __matmul__(self, other: iPETScMatrix) -> iComplexPETScVector: ...

    @overload
    def __matmul__(self, other: iPETScVector) -> iPETScMatrix: ...

    @overload
    def __matmul__(self, other: iComplexPETScVector) -> iPETScMatrix: ...

    def __matmul__(
        self, other: iPETScMatrix | iPETScVector | iComplexPETScVector
    ) -> iComplexPETScVector | iPETScMatrix:
        """Perform vector cross product and vector-matrix multiplication."""
        if isinstance(other, (iPETScVector, iComplexPETScVector)):
            raise NotImplementedError(
                "Vector-vector cross product not implemented yet."
            )

        if isinstance(other, iPETScMatrix):
            real_out = self.real @ other

            if self.imag is None:
                return iComplexPETScVector(real_out)

            imag_out = self.imag @ other
            return iComplexPETScVector(real_out, imag_out)

        raise NotImplementedError(f"Cannot multiply complex vector by {type(other)}")

    @overload
    def __rmatmul__(self, other: iPETScMatrix) -> iComplexPETScVector: ...

    @overload
    def __rmatmul__(self, other: iPETScVector) -> iPETScMatrix: ...

    @overload
    def __rmatmul__(self, other: iComplexPETScVector) -> iPETScMatrix: ...

    def __rmatmul__(
        self, other: iPETScMatrix | iPETScVector | iComplexPETScVector
    ) -> iComplexPETScVector | iPETScMatrix:
        """Perform vector cross product and matrix - vector multiplication."""
        if isinstance(other, (iPETScVector, iComplexPETScVector)):
            return NotImplemented  # Let the left operand handle it

        if isinstance(other, iPETScMatrix):
            real_out = other @ self.real

            if self.imag is None:
                return iComplexPETScVector(real_out)

            imag_out = other @ self.imag
            return iComplexPETScVector(real_out, imag_out)

        raise NotImplementedError(f"Cannot multiply {type(other)} by complex vector")

    def __eq__(self, other: object) -> bool:
        """Equality comparison (within small numerical tolerance)."""
        if not isinstance(other, iComplexPETScVector):
            return False
        diff = self - other
        return diff.norm() < 1e-12

    def assemble(self) -> None:
        """Assemble both underlying PETSc vectors."""
        self._real.assemble()
        if self._imag is not None:
            self._imag.assemble()

    def norm(self) -> float:
        """Compute the Euclidean (2-)norm of the complex vector."""
        if _IS_COMPLEX_BUILD:
            return self._real.norm

        if self._imag is None:
            return self._real.norm
        real_n = self._real.norm
        imag_n = self._imag.norm
        return float(np.sqrt(real_n**2 + imag_n**2))

    def dot(self, other: iComplexPETScVector | iPETScVector) -> complex:
        """Hermitian-inner-product with another complex vector.

        Note that, by convention, the method conjugates the first vector (self).
        """
        if isinstance(other, iPETScVector):
            return self.dot(iComplexPETScVector(other))

        if not isinstance(other, iComplexPETScVector):
            raise TypeError("Dot product requires a iComplexPETScVector.")

        if _IS_COMPLEX_BUILD:
            return self._real.dot(other.real)

        a, b = self._real, self._imag or (self._real * 0.0)
        c, d = other.real, other.imag or (other.real * 0.0)
        real_part = a.dot(c) + b.dot(d)
        imag_part = a.dot(d) - b.dot(c)
        return real_part + 1j * imag_part

    def scale(self, scalar: float | complex) -> None:
        """In-place scale of this vector by a scalar (float or complex)."""
        if _IS_COMPLEX_BUILD:
            self._real.scale(scalar)
            return

        if isinstance(scalar, complex):
            a, b = scalar.real, scalar.imag
            if self._imag is None:
                if b == 0.0:
                    self._real.scale(a)
                else:
                    orig = self._real.copy()
                    self._real.scale(a)
                    self._imag = orig
                    self._imag.scale(b)
            else:
                xr, xi = self._real.copy(), self._imag.copy()
                self._real = (xr * a) - (xi * b)
                self._imag = (xr * b) + (xi * a)
        else:
            alpha = float(scalar)
            self._real.scale(alpha)
            if self._imag is not None:
                self._imag.scale(alpha)

    def copy(self) -> iComplexPETScVector:
        """Deep copy of this ComplexPETScVector."""
        if _IS_COMPLEX_BUILD or self._imag is None:
            return iComplexPETScVector(self._real.copy())
        return iComplexPETScVector(self._real.copy(), self._imag.copy())


class iPETScNullSpace:
    """Minimal wrapper around PETSc nullspace to provide a consistent interface.

    A nullspace (or kernel) of a matrix A is the set of all vectors x such that A*x = 0.
    In many PDE and linear algebra applications, correctly identifying and removing nullspace components
    is crucial for solver stability (e.g., incompressible flow nullspaces).

    This class provides a minimal typed interface to create, attach, test, and remove nullspaces.
    """

    def __init__(self, raw: PETSc.NullSpace) -> None:
        """Create nullspace from PETSc object."""
        self._raw = raw
        self._comm = raw.getComm()

    def __repr__(self) -> str:
        info = "constant" if self.has_constant() else "vector-based"
        return f"<iPETScNullSpace {info}, comm={self._comm}>"

    @property
    def raw(self) -> PETSc.NullSpace:
        """Return the underlying PETSc NullSpace object."""
        return self._raw

    @property
    def comm(self) -> PETSc.Comm:
        """MPI communicator for the null space."""
        return self._comm

    @classmethod
    def from_vectors(cls, vectors: list[iPETScVector]) -> iPETScNullSpace:
        """Create a NullSpace from a list of iPETScVector basis vectors (no constant)."""
        if not vectors:
            raise ValueError("Cannot create NullSpace from empty vector list")
        # Ensure all vectors share the same communicator
        comm = vectors[0].raw.comm
        for v in vectors:
            if v.raw.comm != comm:
                raise ValueError("All vectors must have the same communicator")
        raw_vecs = [v.raw for v in vectors]
        ns = PETSc.NullSpace().create(constant=False, vectors=raw_vecs, comm=comm)
        return cls(ns)

    @classmethod
    def create_constant(cls, comm: PETSc.Comm = PETSc.COMM_WORLD) -> iPETScNullSpace:
        """Create a NullSpace containing only the constant vector."""
        ns = PETSc.NullSpace().create(constant=True, vectors=(), comm=comm)
        return cls(ns)

    @classmethod
    def create_constant_and_vectors(
        cls, vectors: list[iPETScVector], comm: PETSc.Comm | None = PETSc.COMM_WORLD
    ) -> iPETScNullSpace:
        """Create a NullSpace with the constant vector and additional basis vectors."""
        if not vectors:
            return cls.create_constant(comm=comm)
        base_comm = comm
        if comm is None:
            base_comm = vectors[0].raw.comm
        raw_vecs = [v.raw for v in vectors]
        ns = PETSc.NullSpace().create(constant=True, vectors=raw_vecs, comm=base_comm)
        return cls(ns)

    def has_constant(self) -> bool:
        """Return whether this nullspace contains the constant vector."""
        return self.raw.hasConstant()

    def test(self, matrix: iPETScMatrix) -> bool:
        """Test if this nullspace is valid for the given matrix.

        Returns True if A*x = 0 for all nullspace vectors.
        """
        try:
            return self._raw.test(matrix.raw)
        except Exception as e:
            logger.warning(f"NullSpace.test failed: {e}")
            return False

    def remove(self, vector: iPETScVector) -> None:
        """Remove all components of the null space from the given vector in-place."""
        try:
            self._raw.remove(vector.raw)
        except Exception as e:
            logger.warning(f"NullSpace.test failed: {e}")

    def attach_to(self, mat: iPETScMatrix) -> None:
        """Attach this nullspace to an iPETScMatrix (sets both NullSpace and NearNullSpace)."""
        mat.raw.setNullSpace(self.raw)
        mat.raw.setNearNullSpace(self.raw)


_MAP_TO_DOLFINX: dict[iElementFamily, DolfinxElementFamily] = {
    iElementFamily.LAGRANGE: DolfinxElementFamily.P,
    iElementFamily.P: DolfinxElementFamily.P,
    iElementFamily.BUBBLE: DolfinxElementFamily.bubble,
    iElementFamily.RT: DolfinxElementFamily.RT,
    iElementFamily.BDM: DolfinxElementFamily.BDM,
    iElementFamily.CR: DolfinxElementFamily.CR,
    iElementFamily.DPC: DolfinxElementFamily.DPC,
    iElementFamily.N1E: DolfinxElementFamily.N1E,
    iElementFamily.N2E: DolfinxElementFamily.N2E,
    iElementFamily.HHJ: DolfinxElementFamily.HHJ,
    iElementFamily.REGGE: DolfinxElementFamily.Regge,
    iElementFamily.SERENDIPITY: DolfinxElementFamily.serendipity,
    iElementFamily.HERMITE: DolfinxElementFamily.Hermite,
    iElementFamily.ISO: DolfinxElementFamily.iso,
}

_MAP_FROM_DOLFINX: dict[DolfinxElementFamily, iElementFamily] = {
    v: k for k, v in _MAP_TO_DOLFINX.items()
}
