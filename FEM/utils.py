"""LSA-FW FEM utilities."""

from __future__ import annotations

from enum import Enum, auto
from typing import overload
from pathlib import Path
import numpy as np

from basix import ElementFamily as DolfinxElementFamily
from dolfinx.mesh import Mesh, MeshTags
from ufl import Measure  # type: ignore[import-untyped]
from petsc4py import PETSc


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
        result.axpy(1.0, other.raw)
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
                raise NotImplementedError(
                    f"Matrix cannot be multiplied with object of type {type(other)}"
                )

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

    def __getitem__(self, indices: tuple[int, int]) -> float:
        """Get a value via direct indexation (i.e., matrix[i, j])."""
        if self.type.lower() == "nest":
            raise NotImplementedError(
                "Direct indexing is not supported for nested matrices. "
                "Use `raw.getNestSubMat()` to access individual blocks."
            )
        return self.get_value(indices[0], indices[1])

    def __setitem__(self, indices: tuple[int, int], value: int | float) -> None:
        """Set a value via direct indexation (i.e., matrix[i, j] = value)."""
        if self.type.lower() == "nest":
            raise NotImplementedError(
                "Direct assignment is not supported for nested matrices. "
                "Use sub-matrix access or flatten the structure manually."
            )
        self._mat.setValue(
            indices[0], indices[1], float(value), addv=PETSc.InsertMode.INSERT_VALUES
        )
        self._mat.assemble()

    def __eq__(self, other: object) -> bool:
        """Matrix equality."""
        if not isinstance(other, iPETScMatrix):
            return NotImplemented

        if self.shape != other.shape:
            return False

        diff = self.raw.copy()
        diff.axpy(-1.0, other.raw, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
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
        transposed = PETSc.Mat().createTranspose(self._mat)
        transposed.assemble()
        return iPETScMatrix(transposed)

    @property
    def shape(self) -> tuple[int, int]:
        return self._mat.getSize()

    @property
    def nonzero_entries(self) -> int:
        """Return the number of non-zero entries in the matrix."""
        return self._mat.getInfo()["nz_used"]

    @property
    def norm(self) -> float:
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

    def is_numerically_symmetric(self, tol: float = 1e-10) -> bool:
        """Check whether the matrix is numerically symmetric.

        This is more robust than `is_symmetric`, which may fail due to rounding errors,
        reordering, or boundary condition insertions.
        """
        diff = self.raw.copy()
        diff.axpy(
            -1.0, self.T.raw, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN
        )
        return diff.norm() < tol

    def print(self) -> None:
        """Print the matrix."""
        self._mat.view()

    def scale(self, alpha: int | float) -> None:
        """Scale the matrix by a constant factor."""
        self._mat.scale(float(alpha))

    def shift(self, alpha: int | float) -> None:
        """Shift the diagonal of the matrix by a constant."""
        self._mat.shift(float(alpha))

    def sub(self, row: int, col: int) -> iPETScMatrix | None:
        """Return the (row, col)-th submatrix from a nested matrix."""
        if self.type.lower() != "nest":
            raise NotImplementedError(
                "Submatrix access is only available for nested matrices."
            )

        sub_mat = self._mat.getNestSubMatrix(row, col)
        if sub_mat is None:
            return None
        return iPETScMatrix(sub_mat)

    def zero_all_entries(self) -> None:
        """Zero all entries in the matrix."""
        self._mat.zeroEntries()

    def get_value(self, row: int, col: int) -> float:
        """Get the value at position (row, col)."""
        return self._mat.getValue(row, col)

    def get_row(self, row: int) -> tuple[list[int], list[float]]:
        """Get column indices and values for a specific row."""
        cols, values = self._mat.getRow(row)
        return cols.tolist(), values.tolist()

    def axpy(self, alpha: int | float, other: object) -> None:
        """Perform an AXPY operation: this = alpha * other + this."""
        if not isinstance(other, iPETScMatrix):
            raise NotImplementedError(f"Cannot add iPETScMatrix with {type(other)}")
        if self.shape != other.shape:
            raise ValueError(
                f"Incompatible matrix shapes: {self.shape} vs {other.shape}"
            )
        self._mat.axpy(
            float(alpha),
            other.raw,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )

    def create_vector_right(self) -> iPETScVector:
        """Create a vector for right-hand side operations."""
        return iPETScVector(self._mat.createVecRight())

    def create_vector_left(self) -> iPETScVector:
        """Create a vector for left-hand side operations."""
        return iPETScVector(self._mat.createVecLeft())

    def to_aij(self) -> iPETScMatrix:
        """Convert a nested (MatNest) matrix to a flat AIJ matrix."""
        if self.type.lower() != "nest":
            raise NotImplementedError(
                "Only MatNest matrices can be flattened with `to_aij()`."
            )

        aij = self._mat.convert("aij")
        aij.assemble()
        return iPETScMatrix(aij)

    def pin_dof(self, index: int) -> None:
        """Pin a single DOF by zeroing its row and column and placing a 1 on the diagonal.

        This is commonly used to eliminate nullspaces and ensure compatibility with solvers that do not handle
        them, e.g., fixing pressure at a point in incompressible flow problems.
        """
        self._mat.zeroRowsColumns([index], diag=1.0)

    def attach_nullspace(self, nullspace: PETSc.MatNullSpace) -> None:
        """Attach an existing PETSc nullspace object to this matrix."""
        self._mat.setNullSpace(nullspace)

    def get_nullspace(self) -> PETSc.MatNullSpace | None:
        """Retrieve attached nullspace."""
        try:
            return self.raw.getNullSpace()
        except Exception:
            return None

    def export(self, path: Path) -> None:
        """Export the matrix to a binary file."""
        viewer = PETSc.Viewer().createBinary(
            str(path), mode=PETSc.Viewer.Mode.WRITE, comm=self._mat.comm
        )
        self._mat.view(viewer)


class iPETScVector:
    """Minimal wrapper around PETSc vector to provide a consistent interface.

    Note that this is not a complete wrapper and does not implement all methods or properties of a PETSc vector.
    This is intended to be a light-weight typed solution to improve the interface with PETSc.
    Refer to the official PETSc documentation for more details: https://petsc.org/release/docs/manualpages/Vec/.
    """

    def __init__(self, vec: PETSc.Vec) -> None:
        """Initialize PETSc vector wrapper."""
        self._vec = vec

    @classmethod
    def zeros(cls, size: int, comm: PETSc.Comm = PETSc.COMM_WORLD) -> iPETScVector:
        """Initialize a zero vector of given size."""
        vec = PETSc.Vec().create(comm=comm)
        vec.setSizes(size)
        vec.setFromOptions()
        vec.set(0.0)
        return cls(vec)

    @classmethod
    def from_array(
        cls, array: np.ndarray, comm: PETSc.Comm = PETSc.COMM_WORLD
    ) -> iPETScVector:
        """Create a vector from a NumPy array."""
        vec = PETSc.Vec().createWithArray(array, comm=comm)
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
        result._vec.axpy(1.0, other.raw)
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
        result._vec.axpy(-1.0, other.raw)
        return result

    @overload
    def __mul__(self, other: int | float) -> iPETScVector: ...

    @overload
    def __mul__(self, other: iPETScVector) -> float: ...

    def __mul__(self, other: int | float | iPETScVector) -> iPETScVector | float:
        """Perform scalar-vector multiplication."""
        if isinstance(other, (int, float)):
            result = self.copy()
            result.scale(other)
            return result

        if isinstance(other, iPETScVector):
            return self.dot(other)

    def __rmul__(self, alpha: int | float) -> iPETScVector:
        """Perform vector-scalar multiplication"""
        return self.__mul__(alpha)

    def __matmul__(self, other: iPETScVector) -> iPETScMatrix:
        """Vector outer product."""
        if not isinstance(other, iPETScVector):
            return NotImplemented

        A = PETSc.Mat().createDense([self.size, other.size], comm=self.comm)
        A.setUp()

        x = self.raw
        y = other.raw

        # Outer product: A[i,j] = x[i] * y[j]
        for i in range(self.size):
            xi = x.getValue(i)
            for j in range(other.size):
                yj = y.getValue(j)
                A.setValue(i, j, xi * yj)

        A.assemble()
        return iPETScMatrix(A)

    def __getitem__(self, index: int) -> float:
        """Get a value via direct indexation (i.e., vector[i])."""
        return self._vec.getValue(index)

    def __setitem__(self, index: int, value: int | float) -> None:
        """Set a value via direct indexation (i.e., vector[i] = value)."""
        self._vec.setValue(index, float(value), addv=PETSc.InsertMode.INSERT_VALUES)
        self._vec.assemble()

    def __eq__(self, other: object) -> bool:
        """Vector equality."""
        if not isinstance(other, iPETScVector):
            return NotImplemented

        if self.size != other.size:
            return False

        diff = self.raw.duplicate()
        diff.waxpy(-1.0, self.raw, other.raw)  # diff = other - self
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
    def norm(self) -> float:
        """Return the 2-norm of the vector."""
        return self._vec.norm()

    def copy(self) -> iPETScVector:
        """Create a copy of the vector."""
        return iPETScVector(self._vec.copy())

    def scale(self, alpha: float) -> None:
        """Scale the vector by a constant factor."""
        self._vec.scale(alpha)

    def zero_all_entries(self) -> None:
        """Zero all entries in the vector."""
        self._vec.set(0.0)

    def get_value(self, i: int) -> float:
        """Get the value at index i."""
        return self._vec.getValue(i)

    def set_value(self, i: int, value: float) -> None:
        """Set the value at index i."""
        self._vec.setValue(i, value)

    def as_array(self) -> np.ndarray:
        """Return the vector as a NumPy array."""
        return self._vec.getArray().copy()

    def set_random(self, rng: PETSc.Random | None = None) -> None:
        """Set the vector to random values."""
        if rng is None:
            rng = PETSc.Random().create(comm=self._vec.comm)
            rng.setFromOptions()
        self._vec.setRandom(rng)

    def dot(self, other: iPETScVector) -> float:
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
