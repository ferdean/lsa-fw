"""LSA-FW FEM boundary conditions.

Defines Dirichlet, Neumann, and Robin conditions for the incompressible Navier-Stokes equations.
Support for periodic boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Callable, Sequence, Any, Tuple, cast

import dolfinx.fem as dfem
import dolfinx.mesh as dmesh
import numpy as np
from petsc4py import PETSc
import ufl  # type: ignore[import-untyped]

from config import BoundaryConditionsConfig
from Meshing.core import Mesher

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
    SYMMETRY = auto()
    """Symmetric BC (no penetration, free-slip)."""

    # Elasticity ---
    DIRICHLET_DISPLACEMENT = auto()
    """Strong Dirichlet BC for displacement."""

    @classmethod
    def from_string(cls, value: str) -> BoundaryConditionType:
        """Get boundary condition type from a string."""
        try:
            return cls(value.lower().strip().replace(" ", "_"))
        except (KeyError, ValueError) as e:
            raise ValueError(f"No type found for {value}.") from e


@dataclass
class BoundaryConditions:
    """Container for all boundary conditions of a domain."""

    velocity: list[tuple[int, dfem.DirichletBC]]
    """Strong velocity Dirichlet BCs to be applied to the system."""
    pressure: list[tuple[int, dfem.DirichletBC]]
    """Strong pressure Dirichlet BCs to be applied to the system."""
    velocity_neumann: list[tuple[int, ufl.ExternalOperator | dfem.Constant]]
    """Velocity Neumann boundary contributions to the weak form."""
    pressure_neumann: list[tuple[int, ufl.ExternalOperator | dfem.Constant]]
    """Pressure Neumann boundary contributions to the weak form."""
    robin_data: list[tuple[int, dfem.Constant, ufl.ExternalOperator | dfem.Constant]]
    """Robin boundary contributions to the weak form."""
    velocity_periodic_map: list[dict[int, int]]
    """Periodic maps for velocity BC."""
    pressure_periodic_map: list[dict[int, int]]
    """Periodic maps for pressure BC."""


def define_bcs(
    mesher: Mesher, spaces: FunctionSpaces, configs: Sequence[BoundaryConditionsConfig]
) -> BoundaryConditions:
    """Define and construct all boundary conditions."""
    mesh = mesher.mesh
    tags = mesher.facet_tags
    dim = mesh.topology.dim

    if tags is None:
        raise ValueError("Mesh boundaries are not properly tagged.")

    vel_subspace = spaces.mixed.sub(0)
    pres_subspace = spaces.mixed.sub(1)

    velocity_bcs: list[tuple[int, dfem.DirichletBC]] = []
    pressure_bcs: list[tuple[int, dfem.DirichletBC]] = []
    velocity_neumann: list[tuple[int, ufl.ExternalOperator | dfem.Constant]] = []
    pressure_neumann: list[tuple[int, ufl.ExternalOperator | dfem.Constant]] = []
    robin_data: list[
        tuple[int, dfem.Constant, ufl.ExternalOperator | dfem.Constant]
    ] = []
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
                    else _wrap_constant_scalar(_to_scalar_value(cfg.value))
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
                    g_vec = _to_vector_constant(mesh, cfg.value)
                velocity_neumann.append((marker, g_vec))

            case BoundaryConditionType.NEUMANN_PRESSURE:
                if callable(cfg.value):
                    h_expr = ufl.as_scalar(cfg.value(ufl.SpatialCoordinate(mesh)))
                else:
                    h_expr = _to_scalar_constant(mesh, cfg.value)
                pressure_neumann.append((marker, h_expr))

            case BoundaryConditionType.ROBIN:
                if cfg.robin_alpha is None:
                    raise ValueError("robin_alpha must be provided for Robin BC")
                alpha = dfem.Constant(mesh, Scalar(cfg.robin_alpha))
                if callable(cfg.value):
                    g_expr = ufl.as_vector(cfg.value(ufl.SpatialCoordinate(mesh)))
                else:
                    g_expr = _to_vector_constant(mesh, cfg.value)
                robin_data.append((cfg.marker, alpha, g_expr))

            case BoundaryConditionType.PERIODIC:
                # Expect a pair of integer facet markers
                if not (isinstance(cfg.value, tuple) and len(cfg.value) == 2):
                    raise TypeError(
                        "PERIODIC.value must be a tuple[int, int] of (from_marker, to_marker)"
                    )
                from_marker_raw, to_marker_raw = cast(Tuple[Any, Any], cfg.value)
                if not isinstance(from_marker_raw, (int, np.integer)) or not isinstance(
                    to_marker_raw, (int, np.integer)
                ):
                    raise TypeError("PERIODIC markers must be integers.")
                from_marker_i = int(from_marker_raw)
                to_marker_i = int(to_marker_raw)
                v_map = compute_periodic_dof_pairs(
                    spaces.velocity, mesher, from_marker_i, to_marker_i
                )
                p_map = compute_periodic_dof_pairs(
                    spaces.pressure, mesher, from_marker_i, to_marker_i
                )
                velocity_periodic_map.append(v_map)
                pressure_periodic_map.append(p_map)

            case BoundaryConditionType.SYMMETRY:
                # TODO: Right now, the `comp` parameter is hard-coded to '1'. This parameter should be configurable
                # per boundary via TOML file.
                bc = _dirichlet_component_zero(vel_subspace, comp=1, facets=facets)
                velocity_bcs.append((marker, bc))

            case _:
                raise AssertionError(f"Unhandled boundary condition type: {cfg.type!r}")

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
    function_space: dfem.FunctionSpace,
    mesher: Mesher,
    from_marker: int,
    to_marker: int,
    facet_dim: int | None = None,
    tolerance: float = 1e-8,
) -> dict[int, int]:
    """Identify periodic DOF pairs between 'from' and 'to' tagged boundary facets.

    Returns a dict mapping each target DOF index to its corresponding source DOF index.
    """
    mesh = mesher.mesh
    facet_dim = facet_dim or mesh.topology.dim - 1
    coords = function_space.tabulate_dof_coordinates()[:, : mesh.geometry.dim]

    tags = mesher.facet_tags
    if tags is None:
        raise ValueError("Mesh boundaries are not properly tagged.")

    # Extract the facet indices for each marker
    facets = tags.indices
    markers = tags.values
    facets_from = facets[markers == from_marker]
    facets_to = facets[markers == to_marker]

    # Find all DOFs on each facets
    from_dofs = dfem.locate_dofs_topological(function_space, facet_dim, facets_from)
    to_dofs = dfem.locate_dofs_topological(function_space, facet_dim, facets_to)
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
    """Apply a periodic map to a matrix or vector in-place.

    For matrices:
        1. Pulls the original row and column entries of each 'to' DOF.
        2. Adds them into the corresponding 'from' row and column.
        3. Zeros out the 'to' row and column, then places a 1 on its diagonal.
        4. Calls assemble() to finalize PETSc's internal state.

    For vectors:
        1. For each (to, from) pair, adds vec[to] into vec[from].
        2. Zeros out all vec[to] entries.
        3. Calls assemble() to finalize PETSc's internal state.

    Examples of usage are shown in `doc.models.fem-bcs` and `tests/unit/FEM/test_bcs.py`.
    """
    if isinstance(obj, iPETScMatrix):
        # Temporary: relax PETSc preallocation complaints for post-assembly edits
        obj.raw.setOption(PETSc.Mat.Option.NEW_NONZERO_LOCATIONS, True)  # type: ignore[attr-defined]
        obj.raw.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)  # type: ignore[attr-defined]

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


def _dirichlet_component_zero(
    subspace: dfem.FunctionSpace, comp: int, facets: np.ndarray
) -> dfem.DirichletBC:
    comp_subspace = subspace.sub(comp)  # Component subspace (scalar)
    fdim = subspace.mesh.topology.dim - 1

    dofs_pair = dfem.locate_dofs_topological((comp_subspace, subspace), fdim, facets)
    dofs_sub = dofs_pair[0] if isinstance(dofs_pair, (list, tuple)) else dofs_pair

    zero = dfem.Constant(subspace.mesh, Scalar(0.0))
    return dfem.dirichletbc(zero, dofs_sub, comp_subspace)


def _wrap_constant_vector(
    value: float | tuple[float, ...],
) -> Callable[[np.ndarray], np.ndarray]:
    array = np.atleast_1d(value).astype(float)
    vector = array[:, np.newaxis]  # Shape (dim, 1)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        return np.tile(vector, (1, x.shape[1]))  # Shape (dim, n_points)

    return _interpolator


def _wrap_constant_scalar(value: float) -> Callable[[np.ndarray], np.ndarray]:
    v = float(value)

    def _interpolator(x: np.ndarray) -> np.ndarray:
        return np.full((1, x.shape[1]), v)

    return _interpolator


def _to_vector_constant(
    mesh: dmesh.Mesh, value: float | tuple[float, ...] | tuple[int, ...]
) -> dfem.Constant:
    dim = mesh.geometry.dim
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, dim)
    if arr.size != dim:
        raise ValueError(
            f"Vector value must have length {dim}, got shape {tuple(arr.shape)}"
        )
    return dfem.Constant(mesh, arr)


def _to_scalar_constant(mesh: dmesh.Mesh, value: Any) -> dfem.Constant:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return dfem.Constant(mesh, Scalar(value))
    raise TypeError(
        f"Scalar value expected for pressure Neumann/Dirichlet, got {type(value)}"
    )


def _to_scalar_value(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(
        f"Scalar value expected for pressure Dirichlet/Neumann, got {type(value)}"
    )
