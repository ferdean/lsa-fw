"""Tests for config."""

from pathlib import Path

import numpy as np

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

[[BC]]
marker = 4
type = "periodic"
value = [4, 5]

[[BC]]
marker = 5
type = "periodic"
value = [4, 5]
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
        config.BoundaryConditionsConfig(marker=4, type="periodic", value=(4, 5)),
        config.BoundaryConditionsConfig(marker=5, type="periodic", value=(4, 5)),
    ]


def test_load_cylinder_flow_config(tmp_path: Path):
    """Test loading cylinder flow geometry configuration."""
    toml = """\
dim = 2
x_range = [0.0, 2.2]
y_range = [0.0, 0.41]
cylinder_radius = 0.05
cylinder_center = [0.2, 0.2]
resolution = 0.05
resolution_around_cylinder = 0.01
influence_radius = 0.15
"""
    config_path = tmp_path / "cylinder.toml"
    _create_tmp_toml(config_path, toml)

    assert config.load_cylinder_flow_config(
        config_path
    ) == config.CylinderFlowGeometryConfig(
        dim=2,
        cylinder_radius=0.05,
        cylinder_center=(0.2, 0.2),
        x_range=(0.0, 2.2),
        y_range=(0.0, 0.41),
        resolution=0.05,
        resolution_around_cylinder=0.01,
        influence_radius=0.15,
        z_range=None,
    )


def test_load_step_flow_config(tmp_path: Path):
    """Test loading step flow geometry configuration."""
    toml = """\
dim = 3
inlet_length = 1.0
step_height = 0.2
outlet_length = 4.0
channel_height = 1.0
resolution = 0.05
width = 0.8
"""
    config_path = tmp_path / "step.toml"
    _create_tmp_toml(config_path, toml)

    assert config.load_step_flow_config(config_path) == config.StepFlowGeometryConfig(
        dim=3,
        inlet_length=1.0,
        step_height=0.2,
        outlet_length=4.0,
        channel_height=1.0,
        resolution=0.05,
        width=0.8,
    )


def test_load_facet_config(tmp_path: Path):
    """Test marker function generated from TOML facet config."""
    toml = """\
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
    config_path = tmp_path / "step.toml"
    _create_tmp_toml(config_path, toml)

    fn = config.load_facet_config(config_path)

    assert fn(np.array([0.0, 0.5, 0.0])) == 1
    assert fn(np.array([1.0, 0.5, 0.0])) == 2
    assert fn(np.array([0.5, 0.5, 0.0])) == 99
