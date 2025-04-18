"""Unit tests for Meshing.core module."""

import pytest
from pathlib import Path
from Meshing import Shape, Mesher, iCellType, Format


def test_cell_type_enum_mapping():
    """Ensure iCellType maps correctly to dolfinx CellType. Quick smoke test."""
    assert iCellType.TRIANGLE.to_dolfinx().name == "triangle"
    assert iCellType.HEXAHEDRON.to_dolfinx().value == iCellType.HEXAHEDRON.value


@pytest.mark.parametrize(
    "shape, n, cell_type, expected_topo_dim, expected_cell_name",
    [
        (Shape.UNIT_INTERVAL, (8,), iCellType.INTERVAL, 1, "interval"),
        (Shape.UNIT_SQUARE, (4, 4), iCellType.TRIANGLE, 2, "triangle"),
        (Shape.UNIT_CUBE, (2, 2, 2), iCellType.HEXAHEDRON, 3, "hexahedron"),
        (Shape.BOX, (3, 3), iCellType.QUADRILATERAL, 2, "quadrilateral"),
        (Shape.BOX, (2, 2, 2), iCellType.HEXAHEDRON, 3, "hexahedron"),
    ],
)
def test_generate_mesh(shape, n, cell_type, expected_topo_dim, expected_cell_name):
    """Test mesh generation for different shape/cell configurations."""
    mesher = Mesher(shape=shape, n=n, cell_type=cell_type)
    mesh = mesher.generate()

    assert mesh.topology.dim == expected_topo_dim
    assert mesh.topology.cell_name() == expected_cell_name
    assert mesh.topology.index_map(expected_topo_dim).size_local > 0
    assert mesh.geometry.x.shape[0] > 0


def test_custom_domain_2d():
    """Test mesh generation with a custom domain in 2D."""
    domain = ((2.0, 4.0), (5.0, 6.0))  # (xmin, ymin), (xmax, ymax)
    mesher = Mesher(
        shape=Shape.BOX,
        n=(3, 3),
        cell_type=iCellType.QUADRILATERAL,
        domain=domain,
    )
    mesh = mesher.generate()
    coords = mesh.geometry.x

    for dim, (min_expected, max_expected) in enumerate(zip(*domain)):
        actual_min = coords[:, dim].min()
        actual_max = coords[:, dim].max()
        assert actual_min == pytest.approx(min_expected)
        assert actual_max == pytest.approx(max_expected)


def test_custom_domain_3d():
    """Test mesh generation with a custom domain in 3D."""
    domain = ((2.0, 5.0, 0.0), (4.0, 6.0, 3.0))
    mesher = Mesher(
        shape=Shape.BOX,
        n=(3, 3, 9),
        cell_type=iCellType.TETRAHEDRON,
        domain=domain,  # (xmin, ymin, zmin), (xmax, ymax, zmax)
    )
    mesh = mesher.generate()
    coords = mesh.geometry.x

    for dim, (min_expected, max_expected) in enumerate(zip(*domain)):
        actual_min = coords[:, dim].min()
        actual_max = coords[:, dim].max()
        assert actual_min == pytest.approx(min_expected)
        assert actual_max == pytest.approx(max_expected)


def test_invalid_dimension():
    """Ensure error is raised when using more than 3 dimensions."""
    with pytest.raises(ValueError, match="between 1 and 3"):
        Mesher(shape=Shape.UNIT_CUBE, n=(1, 2, 3, 4))


def test_missing_custom_files():
    """Ensure custom shapes require a file path."""
    with pytest.raises(ValueError, match="requires a custom file"):
        Mesher(shape=Shape.CUSTOM_XDMF)


@pytest.mark.parametrize("export_format", {Format.XDMF})
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

    assert path.exists()


def test_zero_cells_raises():
    """Test that passing zero cells raises an error."""
    with pytest.raises(ValueError):
        Mesher(shape=Shape.UNIT_SQUARE, n=(0, 0), cell_type=iCellType.TRIANGLE)


def test_invalid_custom_file_path():
    """Test that non-existent custom mesh file raises error."""
    invalid_path = Path("non_existent_file.msh")
    with pytest.raises(FileNotFoundError):
        Mesher.from_file(invalid_path, Shape.CUSTOM_MSH)
