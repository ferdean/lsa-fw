"""LSA-FW FEM boundary conditions.

Defines Dirichlet, Neumann, and Robin conditions for the incompressible Navier-Stokes equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Callable, Sequence, assert_never

import dolfinx.fem as dfem
import numpy as np
from petsc4py import PETSc
import ufl  # type: ignore[import-untyped]

from config import BoundaryConditionsConfig
from Meshing import Mesher

from .spaces import FunctionSpaces
from .utils import iPETScMatrix, iPETScVector, Scalar


class BoundaryConditionType(StrEnum):
    """Supported boundary condition types."""

    DIRICHLET_VELOCITY = auto()
    """Strong Dirichlet BC for velocity."""
    DIRICHLET_PRESSURE = auto()
    """Strong Dirichlet BC for pressure."""
    NEUMANN_VELOCITY = auto()
    """Weak Neumann BC for velocity (traction)."""
    NEUMANN_PRESSURE = auto()
    """Weak Neumann BC for pressure (flux)."""
    PERIODIC = auto()
    """Periodic BC."""
    ROBIN = auto()
    """Weak Robin BC."""

    @classmethod
    def from_string(cls, value: str) -> BoundaryConditionType:
        """Get boundary condition type from a string."""
        try:
            return cls(value.lower().strip().replace(" ", "_"))
        except KeyError:
            raise ValueError(f"No type found for {value}.")


@dataclass(frozen=True)
class BoundaryCondition:
    """Configuration object for a single boundary condition."""

    marker: int
    """Facet marker for the boundary condition."""
    type: BoundaryConditionType
    """Type of boundary condition."""
    value: (
        float | tuple[int, ...] | tuple[float, ...] | Callable[[np.ndarray], np.ndarray]
    )
    """Value of the boundary condition.

    May be:
      - a scalar (e.g. for pressure)
      - a tuple (e.g. constant vector velocity)
      - a tuple of 2 ints (for periodic BC)
      - a callable returning a NumPy array evaluated at spatial coordinates.
    """
    robin_alpha: float | None = None
    """Robin coefficient (only used for Robin BCs)."""

    @classmethod
    def from_config(cls, config: BoundaryConditionsConfig) -> BoundaryCondition:
        """Get boundary condition from config."""
        return cls(
            marker=config.marker,
            type=BoundaryConditionType.from_string(config.type),
            value=config.value,
            robin_alpha=config.robin_alpha,
        )


@dataclass
class BoundaryConditions:
    """Container for all boundary condition components."""

    velocity: list[tuple[int, dfem.DirichletBC]]
    """Strong velocity Dirichlet BCs to be applied to the system."""
    pressure: list[tuple[int, dfem.DirichletBC]]
    """Strong pressure Dirichlet BCs to be applied to the system."""
    velocity_neumann: list[tuple[int, dfem.Constant]]
    """Velocity Neumann boundary contributions to the weak form."""
    pressure_neumann: list[tuple[int, dfem.Constant]]
    """Pressure Neumann boundary contributions to the weak form."""
    robin_data: list[tuple[int, dfem.Constant, ufl.ExternalOperator]]
    """Robin boundary contributions to the weak form."""
    velocity_periodic_map: list[dict[int, int]]
    """Periodic maps for velocity BC."""
    pressure_periodic_map: list[dict[int, int]]
    """Periodic maps for pressure BC."""


def define_bcs(
    mesher: Mesher, spaces: FunctionSpaces, configs: Sequence[BoundaryCondition]
) -> BoundaryConditions:
    """Define and construct all boundary conditions."""
    mesh = mesher.mesh
    tags = mesher.facet_tags
    dim = mesh.topology.dim

    vel_subspace = spaces.mixed.sub(0)
    pres_subspace = spaces.mixed.sub(1)

    velocity_bcs: list[int, dfem.DirichletBC] = []
    pressure_bcs: list[int, dfem.DirichletBC] = []
    velocity_neumann: list[int, dfem.Constant] = []
    pressure_neumann: list[int, dfem.Constant] = []
    robin_data: list[tuple[int, dfem.Constant, ufl.ExternalOperator]] = []
    velocity_periodic_map: list[dict[int, int]] = []
    pressure_periodic_map: list[dict[int, int]] = []

    for cfg in configs:
        marker = cfg.marker
        facets = tags.find(marker)

        match cfg.type:
            case BoundaryConditionType.DIRICHLET_VELOCITY:
                fn = dfem.Function(spaces.velocity)

                interpolator = (
                    cfg.value
                    if callable(cfg.value)
                    else _wrap_constant_vector(cfg.value)
                )
                fn.interpolate(interpolator)

                dofs = dfem.locate_dofs_topological(
                    (vel_subspace, spaces.velocity), dim - 1, facets
                )
                velocity_bcs.append((marker, dfem.dirichletbc(fn, dofs, vel_subspace)))

            case BoundaryConditionType.DIRICHLET_PRESSURE:
                fn = dfem.Function(spaces.pressure)

                interpolator = (
                    cfg.value
                    if callable(cfg.value)
                    else _wrap_constant_vector(cfg.value)
                )
                fn.interpolate(interpolator)

                dofs = dfem.locate_dofs_topological(
                    (pres_subspace, spaces.pressure), dim - 1, facets
                )
                pressure_bcs.append((marker, dfem.dirichletbc(fn, dofs, pres_subspace)))

            case BoundaryConditionType.NEUMANN_VELOCITY:
                if callable(cfg.value):
                    g_vec = ufl.as_vector(cfg.value(ufl.SpatialCoordinate(mesh)))
                else:
                    g_vec = dfem.Constant(mesh, cfg.value)
                velocity_neumann.append((marker, g_vec))

            case BoundaryConditionType.NEUMANN_PRESSURE:
                if callable(cfg.value):
                    h_expr = ufl.as_scalar(cfg.value(ufl.SpatialCoordinate(mesh)))
                else:
                    h_expr = dfem.Constant(mesh, float(cfg.value))
                pressure_neumann.append((marker, h_expr))

            case BoundaryConditionType.ROBIN:
                if cfg.robin_alpha is None:
                    raise ValueError("robin_alpha must be provided for Robin BC")
                alpha = dfem.Constant(mesh, Scalar(cfg.robin_alpha))

                if callable(cfg.value):
                    g_expr = ufl.as_vector(cfg.value(ufl.SpatialCoordinate(mesh)))
                else:
                    g_expr = dfem.Constant(mesh, cfg.value)

                robin_data.append((cfg.marker, alpha, g_expr))

            case BoundaryConditionType.PERIODIC:
                from_marker, to_marker = cfg.value
                v_map = compute_periodic_dof_pairs(
                    spaces.velocity, mesher, from_marker, to_marker
                )
                p_map = compute_periodic_dof_pairs(
                    spaces.pressure, mesher, from_marker, to_marker
                )
                velocity_periodic_map.append(v_map)
                pressure_periodic_map.append(p_map)

            case _:
                assert_never(cfg.type)

    return BoundaryConditions(
        velocity=velocity_bcs,
        pressure=pressure_bcs,
        velocity_neumann=velocity_neumann,
        pressure_neumann=pressure_neumann,
        robin_data=robin_data,
        velocity_periodic_map=velocity_periodic_map,
        pressure_periodic_map=pressure_periodic_map,
    )


def compute_periodic_dof_pairs(
    V: dfem.FunctionSpace,
    mesher: Mesher,
    from_marker: int,
    to_marker: int,
    facet_dim: int | None = None,
    tolerance: float = 1e-8,
) -> dict[int, int]:
    """Identify periodic DOF pairs between 'from' and 'to' tagged boundary facets.

    Args:
        V: Function space on the mesh.
        facet_tags: MeshTags marking boundary facets (dim = mesh.topology.dim - 1).
        from_marker: integer tag value for the source boundary facets.
        to_marker: integer tag value for the target boundary facets.
        facet_dim: topological dimension of facets (defaults to mesh.topology.dim - 1).
        tolerance: maximum distance for matching DOF coordinates.

    Returns:
        A dict mapping each target DOF index to its corresponding source DOF index.
    """
    mesh = mesher.mesh
    facet_dim = facet_dim or mesh.topology.dim - 1
    coords = V.tabulate_dof_coordinates()[:, : mesh.geometry.dim]

    # Extract the facet indices for each marker
    facets = mesher.facet_tags.indices
    markers = mesher.facet_tags.values
    facets_from = facets[markers == from_marker]
    facets_to = facets[markers == to_marker]

    # Find all DOFs on each facets
    from_dofs = dfem.locate_dofs_topological(V, facet_dim, facets_from)
    to_dofs = dfem.locate_dofs_topological(V, facet_dim, facets_to)
    if from_dofs.size == 0 or to_dofs.size == 0:
        raise ValueError(
            f"No DOFs found on facets for markers {from_marker} or {to_marker}"
        )

    # Compute translation vector = centroid(to) - centroid(from)
    from_coords = coords[from_dofs]
    to_coords = coords[to_dofs]
    translation = to_coords.mean(axis=0) - from_coords.mean(axis=0)

    # Compute pairs
    pairs: dict[int, int] = {}
    for td in to_dofs:
        shifted = from_coords + translation
        dists = np.linalg.norm(shifted - coords[td], axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > tolerance:
            raise ValueError(
                f"Could not match target DOF {td!r}: min distance {dists[idx]:.3g} "
                f"exceeds tolerance {tolerance}"
            )
        pairs[int(td)] = int(from_dofs[idx])

    return pairs


def apply_periodic_constraints(
    obj: iPETScMatrix | iPETScVector, periodic_map: dict[int, int]
) -> None:
    """Merge periodic contributions and zero out 'to' DOFs on a PETSc matrix in-place.

    For matrices:
        1. Pulls the original row and column entries of each 'to' DOF.
        2. Adds them into the corresponding 'from' row and column.
        3. Zeros out the 'to' row and column, then places a 1 on its diagonal.
        4. Calls assemble() to finalize PETSc's internal state.

    For vectors:
        1. For each (to, from) pair, adds vec[to] into vec[from].
        2. Zeros out all vec[to] entries.
        3. Calls assemble() to finalize PETSc's internal state.
    """
    if isinstance(obj, iPETScMatrix):
        # TODO: this dynamic new‐nonzero allocation is only a temporary hack to let the post‐assembly periodic
        # merge run without preallocation errors. Once we fold periodic BCs into assemble_matrix() (refer to
        # FEM.operators), these options can be removed.
        obj.raw.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)
        obj.raw.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        for to_dof, from_dof in periodic_map.items():
            cols, row_vals = obj.get_row(to_dof)  # row
            for c, v in zip(cols, row_vals):
                obj.add_value(from_dof, c, v)
            obj.assemble()

            rows, col_vals = obj.get_column(to_dof)  # column
            for r, v in zip(rows, col_vals):
                obj.add_value(r, from_dof, v)
            obj.assemble()

        # Zero-out and pin each 'to' DOF
        obj.zero_row_columns(list(periodic_map.keys()), diag=1.0)
        obj.assemble()

    elif isinstance(obj, iPETScVector):
        for to_dof, from_dof in periodic_map.items():
            v_to = obj.get_value(to_dof)
            v_fr = obj.get_value(from_dof)
            obj.set_value(from_dof, v_fr + v_to)

        for to_dof in periodic_map:
            obj.set_value(to_dof, 0.0)
        obj.assemble()

    else:
        raise TypeError(
            f"Unsupported object type: {type(obj)}. Expected iPETScMatrix or iPETScVector."
        )


def _wrap_constant_vector(
    value: float | tuple[float, ...],
) -> Callable[[np.ndarray], np.ndarray]:
    array = np.atleast_1d(value).astype(float)  # Ensure 1D float array
    vector = array[:, np.newaxis]  # Shape (dim, 1)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        return np.tile(vector, (1, x.shape[1]))  # Shape (dim, n_points)

    return _interpolator
