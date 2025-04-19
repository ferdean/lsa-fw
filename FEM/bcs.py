"""LSA-FW FEM boundary conditions.

Defines Dirichlet, Neumann, and Robin conditions for the incompressible Navier-Stokes equations.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Sequence, assert_never

import numpy as np
import ufl  # type: ignore[import-untyped]
import dolfinx.fem as dfem
import dolfinx.mesh as dmesh

from .spaces import FunctionSpaces
from .utils import iMeasure

__all__ = ["BoundaryConditionType", "BoundaryConditions", "define_bcs"]


class BoundaryConditionType(Enum):
    """Supported boundary condition types."""

    DIRICHLET_VELOCITY = auto()
    """Strong Dirichlet BC for velocity."""
    DIRICHLET_PRESSURE = auto()
    """Strong Dirichlet BC for pressure."""
    NEUMANN = auto()
    """Weak Neumann BC."""
    ROBIN = auto()
    """Weak Robin BC."""


@dataclass(frozen=True)
class BoundaryCondition:
    """Configuration object for a single boundary condition."""

    marker: int
    """Facet marker for the boundary condition."""
    type: BoundaryConditionType
    """Type of boundary condition."""
    value: float | tuple[float, ...] | Callable[[np.ndarray], np.ndarray]
    """Value of the boundary condition.

    May be:
      - a scalar (e.g. for pressure)
      - a tuple (e.g. constant vector velocity)
      - a callable returning a NumPy array evaluated at spatial coordinates.
    """
    robin_alpha: float | None = None
    """Robin coefficient (only used for Robin BCs)."""


@dataclass
class BoundaryConditions:
    """Container for all boundary condition components."""

    velocity: list[dfem.DirichletBC]
    """Strong velocity Dirichlet BCs to be applied to the system."""
    pressure: list[dfem.DirichletBC]
    """Strong pressure Dirichlet BCs to be applied to the system."""
    neumann_forms: list[ufl.Form]
    """Neumann boundary contributions to the weak form."""
    robin_forms: list[ufl.Form]
    """Robin boundary contributions to the weak form."""


def define_bcs(
    mesh: dmesh.Mesh,
    spaces: FunctionSpaces,
    tags: dmesh.MeshTags,
    u_trial: ufl.Coefficient,
    v_test: ufl.TestFunction,
    configs: Sequence[BoundaryCondition],
) -> BoundaryConditions:
    """
    Define and construct all boundary conditions.

    Args:
        mesh: The simulation mesh.
        spaces: Function space container.
        tags: Boundary facet tags.
        u_trial: Trial function used in variational form.
        v_test: Test function.
        configs: List of boundary condition configurations.
    """
    dim = mesh.topology.dim
    geom_dim = mesh.geometry.dim
    ds = iMeasure.ds(mesh, tags)

    velocity_bcs: list[dfem.DirichletBC] = []
    pressure_bcs: list[dfem.DirichletBC] = []
    neumann_forms: list[ufl.Form] = []
    robin_forms: list[ufl.Form] = []

    for cfg in configs:
        marker = cfg.marker
        facets = tags.find(marker)

        match cfg.type:
            case BoundaryConditionType.DIRICHLET_VELOCITY:
                V = spaces.velocity
                fn = dfem.Function(V)

                interpolator = (
                    cfg.value
                    if callable(cfg.value)
                    else _wrap_constant_vector(cfg.value)
                )
                fn.interpolate(interpolator)

                dofs = dfem.locate_dofs_topological(V, dim - 1, facets)
                velocity_bcs.append(dfem.dirichletbc(fn, dofs))

            case BoundaryConditionType.DIRICHLET_PRESSURE:
                Q = spaces.pressure
                fn = dfem.Function(Q)

                interpolator = (
                    cfg.value
                    if callable(cfg.value)
                    else _wrap_constant_scalar(cfg.value)
                )
                fn.interpolate(interpolator)

                dofs = dfem.locate_dofs_topological(Q, dim - 1, facets)
                pressure_bcs.append(dfem.dirichletbc(fn, dofs))

            case BoundaryConditionType.NEUMANN:
                g = (
                    cfg.value
                    if callable(cfg.value)
                    else lambda _: np.full(geom_dim, cfg.value)
                )
                g_expr = ufl.as_vector(g(ufl.SpatialCoordinate(mesh)))
                neumann_forms.append(ufl.dot(g_expr, v_test) * ds(marker))

            case BoundaryConditionType.ROBIN:
                if cfg.robin_alpha is None:
                    raise ValueError("Robin alpha must be set for Robin BCs.")
                g = (
                    cfg.value
                    if callable(cfg.value)
                    else lambda _: np.full(geom_dim, cfg.value)
                )
                g_expr = ufl.as_vector(g(ufl.SpatialCoordinate(mesh)))

                robin_forms.append(
                    cfg.robin_alpha * ufl.dot(u_trial, v_test) * ds(marker)
                )
                robin_forms.append(
                    cfg.robin_alpha * ufl.dot(g_expr, v_test) * ds(marker)
                )

            case _:
                assert_never(cfg.type)

    return BoundaryConditions(
        velocity=velocity_bcs,
        pressure=pressure_bcs,
        neumann_forms=neumann_forms,
        robin_forms=robin_forms,
    )


def _wrap_constant_vector(
    value: float | tuple[float, ...],
) -> Callable[[np.ndarray], np.ndarray]:
    array = np.atleast_1d(value).astype(float)  # Ensure 1D float array
    vector = array[:, np.newaxis]  # Shape (dim, 1)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        return np.tile(vector, (1, x.shape[1]))  # Shape (dim, n_points)

    return _interpolator


def _wrap_constant_scalar(value: float) -> Callable[[np.ndarray], np.ndarray]:

    def _interpolator(x: np.ndarray) -> np.ndarray:
        return np.full((1, x.shape[1]), float(value))

    return _interpolator


def _wrap_ufl_vector(
    value: float | tuple[float, ...] | Callable,
) -> ufl.core.expr.Expr:
    if callable(value):
        return ufl.as_vector(value(ufl.SpatialCoordinate(None)))  # type: ignore[arg-type]
    array = np.atleast_1d(value)
    return ufl.as_vector(array)
