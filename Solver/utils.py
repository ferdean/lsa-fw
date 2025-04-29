"""LSA-FW Solver utilities."""

from __future__ import annotations

from enum import Enum
from slepc4py import SLEPc


class iEPSProblemType(Enum):
    """Internal EPS problem type enumeration mapping to SLEPc.EPS.ProblemType."""

    HEP = SLEPc.EPS.ProblemType.HEP
    """Standard Hermitian eigenvalue problem. Ax = λx, with A Hermitian."""
    NHEP = SLEPc.EPS.ProblemType.NHEP
    """Standard non-Hermitian eigenvalue problem: Ax = λx, with A arbitrary."""
    GHEP = SLEPc.EPS.ProblemType.GHEP
    """Generalized Hermitian eigenvalue problem: Ax = λBx, with A, B Hermitian."""
    GNHEP = SLEPc.EPS.ProblemType.GNHEP
    """Generalized non-Hermitian eigenvalue problem: Ax = λBx, with arbitrary A, B."""
    PGNHEP = SLEPc.EPS.ProblemType.PGNHEP
    """Polynomial generalized non-Hermitian eigenvalue problem."""
    GHIEP = SLEPc.EPS.ProblemType.GHIEP
    """Generalized Hermitian indefinite eigenvalue problem."""

    def to_slepc(self) -> SLEPc.EPS.ProblemType:
        """Convert internal type to SLEPc.EPS.ProblemType."""
        return self.value

    @classmethod
    def from_slepc(cls, problem_type: SLEPc.EPS.ProblemType) -> "iEPSProblemType":
        """Create internal type from a SLEPc.EPS.ProblemType."""
        try:
            return cls[problem_type.name]
        except KeyError:
            raise ValueError(f"Unsupported SLEPc EPS ProblemType: {problem_type}")

    @classmethod
    def from_string(cls, name: str) -> iEPSProblemType:
        """Create internal type from a string (case-insensitive)."""
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid problem type: {name}. Choose from {list(cls.__members__.keys())}."
            )
