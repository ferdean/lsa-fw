"""LSA-FW FEM utilities."""

from enum import Enum, auto

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
    def from_dolfinx(cls, family: DolfinxElementFamily) -> "iElementFamily":
        """Convert from Dolfinx ElementFamily to internal enum."""
        return _MAP_FROM_DOLFINX[family]

    @classmethod
    def from_string(cls, name: str) -> "iElementFamily":
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
    This is intended to be a light-weight typed solution to improve developing experience.
    """

    def __init__(self, mat: PETSc.Mat) -> None:
        """Initialize PETSc matrix wrapper."""
        self._mat = mat

    def __str__(self) -> str:
        return f"iPETScMatrix(shape={self.shape}, nnz={self.nonzero_entries})"

    def print(self) -> None:
        """Print the matrix."""
        self._mat.view()

    @property
    def raw(self) -> PETSc.Mat:
        """Return the underlying PETSc matrix."""
        return self._mat

    @property
    def shape(self) -> tuple[int, int]:
        return self._mat.getSize()

    @property
    def nonzero_entries(self) -> int:
        """Return the number of non-zero entries in the matrix."""
        return self._mat.getInfo()["nz_used"]

    @property
    def type(self) -> str:
        """Return the PETSc matrix type (e.g., 'aij', 'baij')."""
        return self._mat.getType()

    @property
    def is_symmetric(self) -> bool:
        """Check whether the matrix is symmetric."""
        return self._mat.isSymmetric()

    def scale(self, alpha: float) -> None:
        """Scale the matrix by a constant factor."""
        self._mat.scale(alpha)

    def shift(self, alpha: float) -> None:
        """Shift the diagonal of the matrix by a constant."""
        self._mat.shift(alpha)

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
