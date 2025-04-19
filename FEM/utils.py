"""LSA-FW FEM utilities."""

from enum import Enum, auto

from basix import ElementFamily as DolfinxElementFamily
from dolfinx.mesh import Mesh, MeshTags
from ufl import Measure  # type: ignore[import-untyped]


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
