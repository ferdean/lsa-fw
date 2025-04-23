"""Tests for config."""

from pathlib import Path

import config


def _create_tmp_toml(path: Path, content: str) -> None:
    path.write_text(content.strip())


def test_load_bc_config(tmp_path: Path):
    """Test loading the boundary conditions configuration."""
    toml = """\
[[BC]]
marker = 1
type = "dirichlet_velocity"
value = [0.0, 1.0]

[[BC]]
marker = 2
type = "dirichlet_pressure"
value = 2.5

[[BC]]
marker = 3
type = "robin"
value = [1.0, 0.0]
robin_alpha = 5.0
"""
    config_path = tmp_path / "bcs.toml"
    _create_tmp_toml(config_path, toml)

    assert config.load_bc_config(config_path) == [
        config.BoundaryConditionsConfig(
            marker=1, type="dirichlet_velocity", value=(0.0, 1.0)
        ),
        config.BoundaryConditionsConfig(marker=2, type="dirichlet_pressure", value=2.5),
        config.BoundaryConditionsConfig(
            marker=3, type="robin", value=(1.0, 0.0), robin_alpha=5.0
        ),
    ]
