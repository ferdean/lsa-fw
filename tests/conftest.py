"""Unit-test framework configuration."""

from pathlib import Path

import pytest


@pytest.fixture
def temp_msh_file(tmp_path: Path) -> Path:
    """Create a temporary GMSH .msh file with a valid triangle mesh and physical groups."""
    msh = tmp_path / "mesh.msh"
    msh.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "domain"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 1 1 0
$EndNodes
$Elements
2
1 2 2 1 1 1 2 3
2 2 2 1 1 2 4 3
$EndElements
"""
    )
    return msh
