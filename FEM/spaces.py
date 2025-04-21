"""LSA-FW FEM function spaces implementation.

This module defines and groups function spaces used in the discretization of the incompressible Navier-Stokes equations,
such as those for velocity and pressure fields.
"""

from dataclasses import dataclass
from functools import cached_property
from enum import Enum, auto
from dolfinx.fem import functionspace, FunctionSpace
import dolfinx.mesh as dmesh
from basix.ufl import element, enriched_element
import logging
from typing import assert_never

from Meshing.utils import iCellType

from .utils import iElementFamily

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FunctionSpaces:
    """Container for function spaces used in the Navier-Stokes solver."""

    velocity: FunctionSpace
    """Velocity function space."""
    pressure: FunctionSpace
    """Pressure function space."""

    @cached_property
    def quad_degree(self, offset: int = 1) -> int:
        """Get quadrature degree."""
        v_deg = self.velocity.ufl_element().degree
        p_deg = self.pressure.ufl_element().degree
        return max(v_deg, p_deg) + offset

    @cached_property
    def velocity_dofs(self) -> tuple[int, int]:
        """Return (local, global) number of velocity DOFs."""
        return self._dof_count(self.velocity)

    @cached_property
    def pressure_dofs(self) -> tuple[int, int]:
        """Return (local, global) number of pressure DOFs."""
        return self._dof_count(self.pressure)

    @staticmethod
    def _dof_count(space: FunctionSpace) -> tuple[int, int]:
        local = space.dofmap.index_map.size_local * space.dofmap.bs
        global_ = space.dofmap.index_map.size_global * space.dofmap.bs
        return local, global_


class FunctionSpaceType(Enum):
    """Supported function space types for incompressible Navier-Stokes problems.

    Each space type corresponds to a canonical element pair commonly used in the literature and numerical practice.
    The polynomial degree of velocity and pressure elements is fixed and not configurable, to ensure well-defined
    stability properties (e.g., inf-sup condition). If custom degrees are needed, use the element constructor directly.
    """

    TAYLOR_HOOD = auto()
    """Taylor-Hood element pair.

    Uses continuous quadratic (P2) elements for velocity and continuous linear (P1) elements for pressure.
    """
    MINI = auto()
    """MINI element pair.

    Uses continuous linear (P1) elements enriched with bubble functions for velocity, and continuous linear (P1)
    elements for pressure.
    """
    SIMPLE = auto()
    """Equal-order element pair.

    Uses continuous linear (P1) elements for both velocity and pressure.
    Note: not inf-sup stable; typically requires stabilization techniques.
    """
    DG = auto()
    """Discontinuous Galerkin (DG) element pair.

    Employs discontinuous function spaces for velocity and pressure.
    Not currently supported in this framework.
    """


def define_spaces(
    mesh: dmesh.Mesh, type: FunctionSpaceType = FunctionSpaceType.TAYLOR_HOOD
) -> FunctionSpaces:
    """Define function spaces for the Navier-Stokes problem.

    Args:
        mesh: The mesh on which to define the function spaces.
        type (optional): The type of function space to create. Defaults to Taylor-Hood.
    """
    cell = iCellType.from_dolfinx(mesh.topology.cell_type).to_basix()
    match type:
        case FunctionSpaceType.TAYLOR_HOOD:
            p2 = element(
                family=iElementFamily.LAGRANGE.to_dolfinx(),
                cell=cell,
                degree=2,
                shape=(mesh.geometry.dim,),
            )
            p1 = element(
                family=iElementFamily.LAGRANGE.to_dolfinx(), cell=cell, degree=1
            )

            velocity = functionspace(mesh, p2)
            pressure = functionspace(mesh, p1)

        case FunctionSpaceType.MINI:
            p1_v = element(
                family=iElementFamily.LAGRANGE.to_dolfinx(),
                cell=cell,
                degree=1,
                shape=(mesh.geometry.dim,),
            )
            p1 = element(
                family=iElementFamily.LAGRANGE.to_dolfinx(), cell=cell, degree=1
            )
            bubble = element(
                family=iElementFamily.BUBBLE.to_dolfinx(),
                cell=cell,
                degree=3,  # bubble element is usually of higher order
                shape=(mesh.geometry.dim,),
            )
            enriched = enriched_element([p1_v, bubble])

            velocity = functionspace(mesh, enriched)
            pressure = functionspace(mesh, p1)

        case FunctionSpaceType.SIMPLE:
            logger.warning(
                "Using equal-order P1-P1 function spaces. This may lead to instability."
            )

            p1_v = element(
                family=iElementFamily.LAGRANGE.to_dolfinx(),
                cell=cell,
                degree=1,
                shape=(mesh.geometry.dim,),
            )
            p1_p = element(
                family=iElementFamily.LAGRANGE.to_dolfinx(), cell=cell, degree=1
            )

            velocity = functionspace(mesh, p1_v)
            pressure = functionspace(mesh, p1_p)

        case FunctionSpaceType.DG:
            raise NotImplementedError(
                "Discontinuous Galerkin (DG) function spaces are not yet supported."
            )

        case _:
            assert_never(type)

    return FunctionSpaces(velocity=velocity, pressure=pressure)
