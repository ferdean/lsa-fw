"""Unit tests for Meshing.core module."""

import pytest
from Meshing import Shape, Mesher, iCellType, Format
from pathlib import Path


def test_cell_type_enum_mapping():
    """Ensure iCellType maps correctly to dolfinx CellType."""
    assert iCellType.TRIANGLE.to_dolfinx().name == "triangle"
    assert iCellType.HEXAHEDRON.to_dolfinx().value == iCellType.HEXAHEDRON.value


@pytest.mark.parametrize(
    "shape, n, cell_type, expected_topo_dim",
    [
        (Shape.UNIT_INTERVAL, (8,), iCellType.INTERVAL, 1),
        (Shape.UNIT_SQUARE, (4, 4), iCellType.TRIANGLE, 2),
        (Shape.UNIT_CUBE, (2, 2, 2), iCellType.HEXAHEDRON, 3),
        (Shape.BOX, (3, 3), iCellType.QUADRILATERAL, 2),
        (Shape.BOX, (2, 2, 2), iCellType.HEXAHEDRON, 3),
    ],
)
def test_generate_mesh(shape, n, cell_type, expected_topo_dim):
    """Test mesh generation for different shape/cell configurations."""
    mesher = Mesher(shape=shape, n=n, cell_type=cell_type)
    mesh = mesher.generate()

    assert mesh.topology.dim == expected_topo_dim
    assert mesh.topology.index_map(expected_topo_dim).size_local > 0
    assert mesh.geometry.x.shape[0] > 0


def test_box_custom_domain():
    """Test mesh generation with a custom domain."""
    mesher = Mesher(
        shape=Shape.BOX,
        n=(3, 3),
        cell_type=iCellType.QUADRILATERAL,
        domain=((2.0, 5.0), (4.0, 6.0)),
    )
    mesh = mesher.generate()
    x_coords = mesh.geometry.x[:, 0]
    assert x_coords.min() >= 2.0
    assert x_coords.max() <= 4.0


def test_invalid_dimension_raises():
    """Ensure error is raised when using more than 3 dimensions."""
    with pytest.raises(ValueError, match="between 1 and 3"):
        Mesher(shape=Shape.UNIT_CUBE, n=(1, 2, 3, 4))


def test_missing_custom_file_raises():
    """Ensure custom shapes require a file path."""
    with pytest.raises(ValueError, match="requires a custom file"):
        Mesher(shape=Shape.CUSTOM_XDMF)


@pytest.mark.parametrize("export_format", list(Format))
def test_export_formats(tmp_path: Path, export_format: Format):
    """Test mesh export to all supported formats."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(2, 2), cell_type=iCellType.TRIANGLE)
    mesher.generate()

    ext_map = {
        Format.XDMF: "xdmf",
        Format.VTK: "vtk",
        Format.GMSH: "msh",
    }
    path = tmp_path / f"mesh.{ext_map[export_format]}"
    mesher.export(path, export_format)

    assert path.exists(), f"{export_format} export failed"


def test_from_file_msh(temp_msh_file: Path):
    """Test importing a mesh from a MSH file."""
    mesher = Mesher.from_file(temp_msh_file, Shape.CUSTOM_MSH)
    mesh = mesher.mesh

    assert mesh.topology.dim == 2
    assert mesh.geometry.x.shape[0] == 4


# TODO: Implement this test when XDMF test data is available
# def test_from_file_xdmf(temp_xdmf_file: Path):
#     mesher = Mesher.from_file(temp_xdmf_file, Shape.CUSTOM_XDMF, gdim=2)
#     mesh = mesher.mesh
#     assert mesh.topology.dim == 2
#     assert mesh.geometry.x.shape[0] > 0
