"""LSA-FW Meshing pre-defined geometries.

This module provides geometry generators for standard CFD benchmark cases.
These generators build fully configurable domains, such as cylinder flow and step flow.
"""

import gmsh  # type: ignore[import-untyped]
from mpi4py import MPI
from dolfinx.io import gmshio
import dolfinx.mesh as dmesh
from typing import Callable

from config import CylinderFlowGeometryConfig, StepFlowGeometryConfig

from .utils import Geometry, iCellType

GeometryConfig = CylinderFlowGeometryConfig | StepFlowGeometryConfig


def cylinder_flow(
    cfg: CylinderFlowGeometryConfig,
    comm: MPI.Intracomm,
    _: iCellType = iCellType.TRIANGLE,  # Currently unused, placeholder for future extension
) -> dmesh.Mesh:
    """Generate a mesh for cylinder flow in a rectangular channel.

    Performs automatic local refinement around the cylinder boundary.

    Supports both 2D and 3D cases, depending on the dimensionality of `cylinder_center`.

    Args:
        length: Length of the channel.
        height: Height of the channel (2D/3D).
        cylinder_radius: Radius of the cylinder.
        cylinder_center: Center position of the cylinder (x, y) or (x, y, z).
        width: Channel width (required for 3D).
        resolution: Characteristic mesh resolution.
        resolution_cylinder: Desired resolution near the cylinder.
        influence_radius: Radius of influence for refinement around the cylinder.
        _: Cell type (currently unused).
        comm: MPI communicator.
    """
    if cfg.dim == 2:
        geo = _initialize_model("cylinder2d")
        xc, yc = cfg.cylinder_center
        z = 0.0

        rect = [
            geo.addPoint(0, 0, z, cfg.resolution),
            geo.addPoint(cfg.length, 0, z, cfg.resolution),
            geo.addPoint(cfg.length, cfg.height, z, cfg.resolution),
            geo.addPoint(0, cfg.height, z, cfg.resolution),
        ]
        loop_rect = geo.addCurveLoop(
            [geo.addLine(rect[i], rect[(i + 1) % 4]) for i in range(4)]
        )

        center = geo.addPoint(xc, yc, z, cfg.resolution)
        pts = [
            geo.addPoint(xc, yc + cfg.cylinder_radius, z, cfg.resolution),
            geo.addPoint(xc + cfg.cylinder_radius, yc, z, cfg.resolution),
            geo.addPoint(xc, yc - cfg.cylinder_radius, z, cfg.resolution),
            geo.addPoint(xc - cfg.cylinder_radius, yc, z, cfg.resolution),
        ]
        arcs = [
            geo.addCircleArc(pts[0], center, pts[1]),
            geo.addCircleArc(pts[1], center, pts[2]),
            geo.addCircleArc(pts[2], center, pts[3]),
            geo.addCircleArc(pts[3], center, pts[0]),
        ]
        loop_cyl = geo.addCurveLoop(arcs)
        surface = geo.addPlaneSurface([loop_rect, loop_cyl])
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Fluid")
        geo.synchronize()

        _define_refinement_edges(
            arcs,
            cfg.resolution,
            cfg.resolution_around_cylinder,
            cfg.cylinder_radius,
            cfg.influence_radius,
        )
        return _finalize_and_extract(comm, gdim=cfg.dim)

    elif cfg.dim == 3:
        if cfg.width is None:
            raise ValueError("Width must be provided for 3D cylinder flow.")

        xc, yc, zc = cfg.cylinder_center
        if not (0 < xc < cfg.length and 0 < yc < cfg.height):
            raise ValueError("Cylinder center must be inside the channel.")
        if cfg.cylinder_radius * 2 >= min(cfg.height, cfg.width):
            raise ValueError("Cylinder radius too large for given height/width.")

        occ = _initialize_model("cylinder3d", use_occ=True)
        box = occ.addBox(0, 0, 0, cfg.length, cfg.height, cfg.width)
        cyl = occ.addCylinder(xc, yc, 0, 0, 0, cfg.width, cfg.cylinder_radius)
        fluid = occ.cut([(3, box)], [(3, cyl)], removeObject=True, removeTool=False)
        occ.synchronize()

        if not fluid[0]:
            raise RuntimeError("Boolean cut failed.")

        gmsh.model.addPhysicalGroup(
            3, [tag for dim, tag in fluid[0] if dim == 3], tag=1
        )
        gmsh.model.setPhysicalName(3, 1, "Fluid")

        # Extract cylinder surfaces
        surf = gmsh.model.getBoundary([(3, cyl)], oriented=False, recursive=False)
        lateral_faces = [
            tag
            for dim, tag in surf
            if dim == 2
            and _is_lateral_surface(tag, xc, yc, zc, cfg.cylinder_radius, cfg.width)
        ]

        if not lateral_faces:
            raise RuntimeError("No lateral cylinder surfaces found for refinement.")

        _define_refinement_faces(
            lateral_faces,
            cfg.resolution,
            cfg.resolution_around_cylinder,
            cfg.influence_radius,
        )

        return _finalize_and_extract(comm, gdim=cfg.dim, use_occ=True)

    raise ValueError("Only 2D or 3D supported.")


def step_flow(
    cfg: StepFlowGeometryConfig,
    comm: MPI.Intracomm,
    _: iCellType = iCellType.TRIANGLE,  # Currently unused, placeholder for future extension
) -> dmesh.Mesh:
    """Generate a backward-facing step flow domain.

    Args:
        inlet_length: Length before the step.
        step_height: Height of the vertical step.
        outlet_length: Length after the step.
        channel_height: Full height of the outlet channel.
        width: Channel width (extrusion in 3D). If None, a 2D domain is created.
        resolution: Characteristic mesh element size.
        _: Cell type (currently unused).
        comm: MPI communicator.
    """
    if cfg.dim == 2:
        geo = _initialize_model("step2d")
        pts = [
            geo.addPoint(0, 0, 0, cfg.resolution),
            geo.addPoint(cfg.inlet_length, 0, 0, cfg.resolution),
            geo.addPoint(cfg.inlet_length, cfg.step_height, 0, cfg.resolution),
            geo.addPoint(
                cfg.inlet_length + cfg.outlet_length, cfg.step_height, 0, cfg.resolution
            ),
            geo.addPoint(
                cfg.inlet_length + cfg.outlet_length,
                cfg.channel_height,
                0,
                cfg.resolution,
            ),
            geo.addPoint(0, cfg.channel_height, 0, cfg.resolution),
        ]
        lines = [geo.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        lines.append(geo.addLine(pts[-1], pts[0]))
        loop = geo.addCurveLoop(lines)
        surface = geo.addPlaneSurface([loop])
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        gmsh.model.setPhysicalName(2, 1, "Fluid")
        geo.synchronize()
        return _finalize_and_extract(comm, gdim=2)

    elif cfg.dim == 3:
        if cfg.width is None:
            raise ValueError("Width must be provided for 3D step flow.")
        occ = _initialize_model("step3d", use_occ=True)
        base_pts = [
            occ.addPoint(0, 0, 0, cfg.resolution),
            occ.addPoint(cfg.inlet_length, 0, 0, cfg.resolution),
            occ.addPoint(cfg.inlet_length, cfg.step_height, 0, cfg.resolution),
            occ.addPoint(
                cfg.inlet_length + cfg.outlet_length, cfg.step_height, 0, cfg.resolution
            ),
            occ.addPoint(
                cfg.inlet_length + cfg.outlet_length,
                cfg.channel_height,
                0,
                cfg.resolution,
            ),
            occ.addPoint(0, cfg.channel_height, 0, cfg.resolution),
        ]
        wire = [
            occ.addLine(base_pts[i], base_pts[i + 1]) for i in range(len(base_pts) - 1)
        ]
        wire.append(occ.addLine(base_pts[-1], base_pts[0]))
        loop = occ.addCurveLoop(wire)
        surface = occ.addPlaneSurface([loop])
        vol = occ.extrude([(2, surface)], 0, 0, cfg.width)
        occ.synchronize()

        top_tags = [tag for dim, tag in vol if dim == 3]
        gmsh.model.addPhysicalGroup(3, top_tags, tag=1)
        gmsh.model.setPhysicalName(3, 1, "Fluid")
        return _finalize_and_extract(comm, gdim=3, use_occ=True)

    raise ValueError("Only 2D or 3D supported.")


def _initialize_model(name: str, use_occ: bool = False):
    gmsh.initialize()
    gmsh.model.add(name)
    return gmsh.model.occ if use_occ else gmsh.model.geo


def _finalize_and_extract(
    comm: MPI.Comm, gdim: int = 2, use_occ: bool = False
) -> dmesh.Mesh:
    if use_occ:
        gmsh.model.occ.synchronize()
    else:
        gmsh.model.geo.synchronize()
    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.generate(gdim)
    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim)
    gmsh.finalize()
    return mesh


def _is_lateral_surface(
    tag: int, xc: float, yc: float, zc: float, r: float, h: float, tol: float = 1e-3
) -> bool:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
    dx = abs((xmin + xmax) / 2 - xc)
    dy = abs((ymin + ymax) / 2 - yc)
    dz = abs((zmin + zmax) / 2 - (zc + h / 2))
    return dx < r + tol and dy < r + tol and dz < h / 2 + tol


def _define_refinement_edges(
    edge_tags: list[int], max_size: float, min_size: float, r: float, delta: float
):
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", edge_tags)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", min_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", max_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r + delta)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)


def _define_refinement_faces(
    face_tags: list[int], max_size: float, min_size: float, delta: float
):
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", face_tags)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", min_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", max_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", delta)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)


_GEOMETRY_MAP: dict[Geometry, Callable[..., dmesh.Mesh]] = {
    Geometry.CYLINDER_FLOW: cylinder_flow,
    Geometry.STEP_FLOW: step_flow,
}


def get_geometry(
    geometry: Geometry, config: GeometryConfig, comm: MPI.Intracomm
) -> dmesh.Mesh:
    """Dispatch and generate a pre-defined CFD benchmark geometry mesh.

    Args:
        name: refer to Geometry enum.
        config: Geometry-specific configuration.
        comm: MPI communicator.
    """
    return _GEOMETRY_MAP[geometry](config, comm)
