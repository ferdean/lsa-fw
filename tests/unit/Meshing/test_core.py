"""Unit tests for Meshing.core module."""

from pathlib import Path

import dolfinx
import dolfinx.mesh as dmesh
import numpy as np
import pytest
from mpi4py import MPI

from config import CylinderFlowGeometryConfig, StepFlowGeometryConfig
from Meshing import Format, Geometry, Mesher, Shape, iCellType


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
def test_generate_mesh(
    shape: Shape,
    n: tuple[int, ...],
    cell_type: iCellType,
    expected_topo_dim: int,
    expected_cell_name: str,
) -> None:
    """Test mesh generation for different shape/cell configurations."""
    mesher = Mesher(shape=shape, n=n, cell_type=cell_type)
    mesh = mesher.generate()

    assert mesh.topology.dim == expected_topo_dim
    assert mesh.topology.cell_name() == expected_cell_name
    assert mesh.topology.index_map(expected_topo_dim).size_local > 0
    assert mesh.geometry.x.shape[0] > 0


def test_custom_domain_2d() -> None:
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


def test_custom_domain_3d() -> None:
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


def test_invalid_dimension() -> None:
    """Ensure error is raised when using more than 3 dimensions."""
    with pytest.raises(ValueError, match="between 1 and 3"):
        Mesher(shape=Shape.UNIT_CUBE, n=(1, 2, 3, 4))


def test_missing_custom_files() -> None:
    """Ensure custom shapes require a file path."""
    with pytest.raises(ValueError, match="requires a custom file"):
        Mesher(shape=Shape.CUSTOM_XDMF)


@pytest.mark.parametrize("export_format", {Format.XDMF})
def test_export_formats(tmp_path: Path, export_format: Format) -> None:
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


def test_zero_cells_raises() -> None:
    """Test that passing zero cells raises an error."""
    with pytest.raises(ValueError):
        Mesher(shape=Shape.UNIT_SQUARE, n=(0, 0), cell_type=iCellType.TRIANGLE)


def test_invalid_custom_file_path() -> None:
    """Test that non-existent custom mesh file raises error."""
    invalid_path = Path("non_existent_file.msh")
    with pytest.raises(FileNotFoundError):
        Mesher.from_file(invalid_path, Shape.CUSTOM_MSH)


def test_mark_boundary_facets_assigns_tags() -> None:
    """Test mark_boundary_facets correctly assigns integer facet tags."""
    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(4, 4), cell_type=iCellType.TRIANGLE)
    mesh = mesher.generate()

    # Tag: left=1, right=2, rest=99
    def _marker_fn(x: np.ndarray) -> int:
        if np.isclose(x[0], 0.0):
            return 1
        elif np.isclose(x[0], 1.0):
            return 2
        return 99

    mesher.mark_boundary_facets(_marker_fn)
    tags = mesher.facet_tags

    assert tags is not None
    assert tags.dim == mesh.topology.dim - 1
    assert tags.values.size > 0
    assert set(tags.values).issubset({1, 2, 99})


@pytest.mark.skipif(
    dolfinx.default_scalar_type == np.complex128,
    reason="Read/write XDMF meshes currently not supported in complex config.",
)
def test_facet_tags_export_import(tmp_path: Path) -> None:
    """Test that facet tags are exported to and reloaded from XDMF."""
    path = tmp_path / "mesh.xdmf"

    mesher = Mesher(shape=Shape.UNIT_SQUARE, n=(2, 2), cell_type=iCellType.TRIANGLE)
    mesher.generate()
    mesher.mark_boundary_facets(lambda x: 1 if x[0] < 1e-12 else 2)
    mesher.export(path, Format.XDMF)

    # Re-import
    mesher2 = Mesher.from_file(path, Shape.CUSTOM_XDMF)
    tags = mesher2.facet_tags

    assert tags is not None
    assert set(tags.values).issubset({1, 2})


def test_mesher_from_mesh() -> None:
    """Test Mesher.from_mesh correctly wraps a given dolfinx Mesh and tags."""
    mesh = dmesh.create_unit_interval(
        MPI.COMM_WORLD, 4, np.float64  # Create a simple 1D mesh
    )
    # Create dummy facet tags on endpoints
    facet_dim = mesh.topology.dim - 1
    entities = np.array([0, 1], dtype=np.int64)
    values = np.array([10, 20], dtype=np.int32)
    facet_tags = dmesh.meshtags(mesh, facet_dim, entities, values)

    mesher = Mesher.from_mesh(mesh, facet_tags=facet_tags, cell_tags=None)

    assert mesher.mesh is mesh
    assert mesher.facet_tags is facet_tags
    assert mesher.cell_tags is None
    assert mesher._shape == Shape.PREDEFINED
    assert mesher._gdim == mesh.topology.dim
    coords = mesh.geometry.x
    expected_domain = (
        tuple(coords.min(axis=0).tolist()),
        tuple(coords.max(axis=0).tolist()),
    )
    assert mesher._domain == expected_domain


def test_from_geometry_step_flow_2d():
    """Test Mesher.from_geometry for the 2D step flow geometry."""
    cfg = StepFlowGeometryConfig(
        dim=2,
        inlet_length=1.0,
        step_height=0.5,
        outlet_length=2.0,
        channel_height=1.0,
        resolution=0.2,
    )
    mesher = Mesher.from_geometry(Geometry.STEP_FLOW, cfg)
    mesh = mesher.mesh
    domain = mesher.domain
    expected_domain = (
        (-cfg.inlet_length, -cfg.step_height),
        (cfg.outlet_length, cfg.channel_height - cfg.step_height),
    )

    assert mesh.topology.dim == 2
    assert mesher._shape == Shape.PREDEFINED
    assert domain[0][0] == pytest.approx(expected_domain[0][0])
    assert domain[0][1] == pytest.approx(expected_domain[0][1])
    assert domain[1][0] == pytest.approx(expected_domain[1][0])
    assert domain[1][1] == pytest.approx(expected_domain[1][1])
    assert mesher.facet_tags is None
    assert mesher.cell_tags is None


def test_from_geometry_cylinder_flow_2d():
    """Test Mesher.from_geometry for the 2D cylinder flow geometry."""
    cfg = CylinderFlowGeometryConfig(
        dim=2,
        length=2.0,
        height=1.0,
        cylinder_radius=0.25,
        cylinder_center=(0.5, 0.5),
        resolution=0.2,
        resolution_around_cylinder=0.1,
        influence_radius=0.5,
    )
    mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cfg)
    mesh = mesher.mesh
    domain = mesher.domain
    expected_domain = ((0.0, 0.0), (cfg.length, cfg.height))

    assert mesh.topology.dim == 2
    assert mesher._shape == Shape.PREDEFINED
    assert domain[0][0] == pytest.approx(expected_domain[0][0])
    assert domain[0][1] == pytest.approx(expected_domain[0][1])
    assert domain[1][0] == pytest.approx(expected_domain[1][0])
    assert domain[1][1] == pytest.approx(expected_domain[1][1])
    assert mesher.facet_tags is None
    assert mesher.cell_tags is None
