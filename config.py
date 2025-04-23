"""LSA-FW configuration."""

import tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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
    value: float | tuple[float, ...]
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
        raw_value = bc.get("value")
        if isinstance(raw_value, list):
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
