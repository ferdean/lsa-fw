"""LSA-FW configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import tomllib


def read_toml(path: Path) -> dict[str, Any]:
    """Read TOML file."""
    if not path.exists():
        raise FileNotFoundError(f"TOML config file not found at '{path}'")
    with path.open("rb") as p:
        return tomllib.load(p)


@dataclass(frozen=True)
class BoundaryConditionsConfig:
    """Configured boundary conditions."""

    marker: int
    """Facet marker for the boundary condition."""
    type: str
    """Type of boundary condition."""
    value: float | tuple[float, ...] | tuple[int, int]
    """Boundary condition value.

    Note that callable boundary conditions are still not supported via configuration file.
    """
    robin_alpha: float | None = None
    """Robin coefficient."""


def load_bc_config(path: Path) -> Sequence[BoundaryConditionsConfig]:
    """Load configured boundary conditions.

    The file must contain an array of tables called `[[BC]]`, with each entry specifying:
    - `marker`: Mesh tag for boundary facets.
    - `type`: Type of boundary condition (e.g. "dirichlet_velocity", "neumann").
    - `value`: Scalar or vector value to impose.
    - `robin_alpha`: Robin coefficient (required if `type` is "robin").

    Example:
        [[BC]]
        marker = 1
        type = "dirichlet_velocity"
        value = [0.0, 0.0]

        [[BC]]
        marker = 2
        type = "dirichlet_pressure"
        value = 1.5
    """
    cfg = read_toml(path)
    bcs: Sequence[BoundaryConditionsConfig] = []
    for bc in cfg.get("BC"):
        raw_value = bc.get("value", 0.0)
        # for periodic we expect two ints
        if bc.get("type").lower().strip() == "periodic":
            if (
                isinstance(raw_value, list)
                and len(raw_value) == 2
                and all(isinstance(v, int) for v in raw_value)
            ):
                value = (raw_value[0], raw_value[1])
            else:
                raise TypeError("Periodic BC value must be two integer markers.")

        elif isinstance(raw_value, list):
            value = tuple(float(v) for v in raw_value)
        elif isinstance(raw_value, (int, float)):
            value = float(raw_value)
        else:
            raise TypeError(f"Unsupported value type: {type(raw_value)}")

        bcs.append(
            BoundaryConditionsConfig(
                marker=bc.get("marker"),
                type=bc.get("type"),
                value=value,
                robin_alpha=bc.get("robin_alpha"),
            )
        )
    return bcs


@dataclass(frozen=True)
class CylinderFlowGeometryConfig:
    """Geometrical configuration of a cylinder-flow problem."""

    dim: int
    """Problem dimensions (2 or 3)."""
    cylinder_radius: float
    """Radius of the cylinder."""
    cylinder_center: tuple[float, ...]
    """Coordinates of the cylinder center, as (x, y, [z])."""
    x_range: tuple[float, ...]
    """X-range of the domain relative to the global frame."""
    y_range: tuple[float, ...]
    """Y-range of the domain relative to the global frame."""
    resolution: float
    """Base mesh element size."""
    resolution_around_cylinder: float
    """Mesh element size around the cylinder."""
    influence_radius: float
    """Radius around the cylinder to apply local refinement."""
    z_range: tuple[float, float] | None = None
    """Optional y-range of the domain relative to the global frame."""


def load_cylinder_flow_config(path: Path) -> CylinderFlowGeometryConfig:
    """Load cylinder flow geometrical configuration."""
    raw = read_toml(path)
    raw["cylinder_center"] = tuple(raw["cylinder_center"])
    raw["x_range"] = tuple(raw["x_range"])
    raw["y_range"] = tuple(raw["y_range"])
    if "z_range" in raw:
        raw["z_range"] = tuple(raw["z_range"])
    return CylinderFlowGeometryConfig(**raw)


@dataclass(frozen=True)
class StepFlowGeometryConfig:
    """Geometrical configuration of a backward-step flow problem."""

    dim: int
    """Problem dimensions (2 or 3)."""
    inlet_length: float
    """Length of the inlet."""
    step_height: float
    """Height of the step."""
    outlet_length: float
    """Length of the outlet."""
    channel_height: float
    """Height of the channel."""
    resolution: float
    """Base mesh element size."""
    width: float | None = None
    """Width of the channel (required only for 3D)."""
    refinement_factor: float | None = None
    """Refinement factor in the step area."""


def load_step_flow_config(path: Path) -> StepFlowGeometryConfig:
    """Load step flow geometrical configuration."""
    cfg = read_toml(path)
    return StepFlowGeometryConfig(**cfg)


@dataclass(frozen=True)
class FacetCondition:
    """Condition on a single axis (x, y, or z) to match a facet."""

    axis: str
    """Coordinate axis to evaluate (x, y, or z)."""
    equals: float | None = None
    """Match if axis value is approximately equal to this value."""
    less_than: float | None = None
    """Match if axis value is strictly less than this value."""
    greater_than: float | None = None
    """Match if axis value is strictly greater than this value."""


@dataclass(frozen=True)
class FacetRule:
    """Tagging rule for mesh boundary facets."""

    marker: int
    """Tag to assign to facets that match this rule."""
    when: FacetCondition | None = None
    """Condition under which to apply this rule (optional if `otherwise` is set)."""
    otherwise: bool = False
    """Set to True to make this rule the fallback if no others match."""


def load_facet_config(path: Path) -> Callable[[np.ndarray], int]:
    """Load facet marker rules from a TOML file and return a callable marker function.

    This function reads a list of facet rules from a TOML file. Each rule specifies how to tag facets by evaluating the
    coordinates of their midpoints. The rules can match on `x`, `y`, or `z` coordinates, using one of:

    - `equals`: match if coordinate is approximately equal (with tolerance)
    - `less_than`: match if coordinate is strictly less than a value
    - `greater_than`: match if coordinate is strictly greater than a value
    - `otherwise = true`: fallback tag if no other rule matched

    The rules are evaluated in order.

    Example:
        [[FaceTag]]
        marker = 1
        when = { axis = "x", equals = 0.0 }

        [[FaceTag]]
        marker = 2
        when = { axis = "x", equals = 1.0 }

        [[FaceTag]]
        marker = 99
        otherwise = true
    """
    cfg = read_toml(path)
    rules = cfg.get("FaceTag", [])

    def _make_rule(rule) -> Callable[[np.ndarray], bool]:
        if rule.get("otherwise", False):
            return lambda _: True

        axis_index = {"x": 0, "y": 1, "z": 2}[rule["when"]["axis"]]
        val = rule["when"].get("equals")
        lt = rule["when"].get("less_than")
        gt = rule["when"].get("greater_than")

        def _cond(x: np.ndarray) -> bool:
            coord = x[axis_index]
            if val is not None and np.isclose(coord, val):
                return True
            if lt is not None and coord < lt:
                return True
            if gt is not None and coord > gt:
                return True
            return False

        return _cond

    conditions = [_make_rule(rule) for rule in rules]
    markers = [rule["marker"] for rule in rules]

    def marker_fn(x: np.ndarray) -> int:
        for cond, marker in zip(conditions, markers):
            if cond(x):
                return marker
        raise RuntimeError("No matching rule and no 'otherwise' fallback defined.")

    return marker_fn
