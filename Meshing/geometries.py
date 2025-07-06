"""LSA-FW Meshing pre-defined geometries.

This module provides geometry generators for standard CFD benchmark cases.
These generators build fully configurable domains, such as cylinder flow and step flow.
"""

from typing import Callable

import numpy as np
import dolfinx.mesh as dmesh
import gmsh  # type: ignore[import-untyped]
from dolfinx.io import gmshio
from mpi4py import MPI

from config import CylinderFlowGeometryConfig, StepFlowGeometryConfig
from .utils import Geometry, iCellType

GeometryConfig = CylinderFlowGeometryConfig | StepFlowGeometryConfig


def cylinder_flow(
    cfg: CylinderFlowGeometryConfig,
    comm: MPI.Intracomm,
    _: iCellType = iCellType.TRIANGLE,
) -> dmesh.Mesh:
    """Generate a mesh for cylinder flow in a rectangular channel.

    Performs automatic local refinement around the cylinder boundary and wake. Supports both 2D and 3D cases.
    """
    xmin, xmax = cfg.x_range
    ymin, ymax = cfg.y_range

    if cfg.dim == 2:
        geo = _initialize_model("cylinder2d")
        xc, yc = cfg.cylinder_center

        # Set mesh options for accurate circle representation
        n_circle_points = 32
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumCirclePoints", n_circle_points)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        # Create rectangle (channel)
        pts = [
            geo.addPoint(x, y, 0.0, cfg.resolution)
            for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
        ]
        lines = [geo.addLine(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]
        rect_loop = geo.addCurveLoop(lines)

        # Create cylinder using sufficient points for the circle
        circ_pts = []
        for i in range(n_circle_points):
            angle = 2 * np.pi * i / n_circle_points
            x = xc + cfg.cylinder_radius * np.cos(angle)
            y = yc + cfg.cylinder_radius * np.sin(angle)
            circ_pts.append(geo.addPoint(x, y, 0.0, cfg.resolution_around_cylinder))
        arcs = [
            geo.addCircleArc(
                circ_pts[i],  # Start point
                geo.addPoint(xc, yc, 0.0, cfg.resolution_around_cylinder),  # Center
                circ_pts[(i + 1) % n_circle_points],  # End point
            )
            for i in range(n_circle_points)
        ]
        circ_loop = geo.addCurveLoop(arcs)

        # Create surface with hole for the cylinder
        surf = geo.addPlaneSurface([rect_loop, circ_loop])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surf], 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid")

        # Refinement around the cylinder
        fid_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fid_dist, "EdgesList", arcs)
        fid_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(fid_thr, "InField", fid_dist)
        gmsh.model.mesh.field.setNumber(
            fid_thr, "SizeMin", cfg.resolution_around_cylinder
        )
        gmsh.model.mesh.field.setNumber(fid_thr, "SizeMax", cfg.resolution)
        gmsh.model.mesh.field.setNumber(fid_thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(fid_thr, "DistMax", 2 * cfg.influence_radius)

        # Wake-refinement box
        wx0 = xc + cfg.cylinder_radius
        wx1 = min(xmax, wx0 + cfg.influence_length)
        wy0, wy1 = yc - 1.5 * cfg.cylinder_radius, yc + 1.5 * cfg.cylinder_radius
        fid_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(
            fid_box, "VIn", 2 * cfg.resolution_around_cylinder
        )
        gmsh.model.mesh.field.setNumber(fid_box, "VOut", cfg.resolution)
        gmsh.model.mesh.field.setNumber(fid_box, "XMin", wx0)
        gmsh.model.mesh.field.setNumber(fid_box, "XMax", wx1)
        gmsh.model.mesh.field.setNumber(fid_box, "YMin", wy0)
        gmsh.model.mesh.field.setNumber(fid_box, "YMax", wy1)

        # Combine all refinements
        fid_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fid_min, "FieldsList", [fid_thr, fid_box])
        gmsh.model.mesh.field.setAsBackgroundMesh(fid_min)

        # Optimize mesh
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 25)

        return _finalize_and_extract(comm, gdim=2)

    if cfg.dim == 3:
        if cfg.z_range is None:
            raise ValueError("z_range must be provided for 3D cylinder flow.")
        xmin, xmax = cfg.x_range
        ymin, ymax = cfg.y_range
        zmin, zmax = cfg.z_range
        xc, yc, zc = cfg.cylinder_center

        occ = _initialize_model("cylinder3d", use_occ=True)
        box = occ.addBox(xmin, ymin, zmin, xmax - xmin, ymax - ymin, zmax - zmin)
        cyl = occ.addCylinder(xc, yc, zmin, 0, 0, zmax - zmin, cfg.cylinder_radius)
        fluid = occ.cut([(3, box)], [(3, cyl)], removeObject=True, removeTool=False)
        occ.synchronize()
        if not fluid[0]:
            raise RuntimeError("Boolean cut failed.")

        tags3d = [t for d, t in fluid[0] if d == 3]
        gmsh.model.addPhysicalGroup(3, tags3d, 1)
        gmsh.model.setPhysicalName(3, 1, "Fluid")

        # Cylinder-surface refinement
        surf = gmsh.model.getBoundary([(3, cyl)], oriented=False, recursive=False)
        lateral = [
            t
            for d, t in surf
            if d == 2
            and _is_lateral_surface(t, xc, yc, zc, cfg.cylinder_radius, zmax - zmin)
        ]
        fid_dist = gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(fid_dist, "FacesList", lateral)
        fid_thr = gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(fid_thr, "InField", fid_dist)
        gmsh.model.mesh.field.setNumber(
            fid_thr, "SizeMin", cfg.resolution_around_cylinder
        )
        gmsh.model.mesh.field.setNumber(fid_thr, "SizeMax", cfg.resolution)
        gmsh.model.mesh.field.setNumber(fid_thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(fid_thr, "DistMax", cfg.influence_radius)

        # Wake-refinement volume
        wx0 = xc
        wx1 = min(xmax, wx0 + cfg.influence_length)
        wy0, wy1 = yc - 1.1 * cfg.cylinder_radius, yc + 1.1 * cfg.cylinder_radius
        fid_box = gmsh.model.mesh.field.add("Box", 3)
        gmsh.model.mesh.field.setNumber(fid_box, "VIn", cfg.resolution_around_cylinder)
        gmsh.model.mesh.field.setNumber(fid_box, "VOut", cfg.resolution * 0.9)
        gmsh.model.mesh.field.setNumber(fid_box, "XMin", wx0)
        gmsh.model.mesh.field.setNumber(fid_box, "XMax", wx1)
        gmsh.model.mesh.field.setNumber(fid_box, "YMin", wy0)
        gmsh.model.mesh.field.setNumber(fid_box, "YMax", wy1)
        gmsh.model.mesh.field.setNumber(fid_box, "ZMin", zmin)
        gmsh.model.mesh.field.setNumber(fid_box, "ZMax", zmax)

        # Combine and optimize
        fid_min = gmsh.model.mesh.field.add("Min", 5)
        gmsh.model.mesh.field.setNumbers(fid_min, "FieldsList", [fid_thr, fid_box])
        gmsh.model.mesh.field.setAsBackgroundMesh(fid_min)

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        return _finalize_and_extract(comm, gdim=3, use_occ=True)

    raise ValueError("Only 2D or 3D supported.")


def step_flow(
    cfg: StepFlowGeometryConfig,
    comm: MPI.Intracomm,
    _: iCellType = iCellType.TRIANGLE,
) -> dmesh.Mesh:
    if cfg.dim == 2:
        geo = _initialize_model("step2d")
        pts = [
            geo.addPoint(0, 0, 0, cfg.resolution),
            geo.addPoint(-cfg.inlet_length, 0, 0, cfg.resolution),
            geo.addPoint(
                -cfg.inlet_length,
                cfg.channel_height - cfg.step_height,
                0,
                cfg.resolution,
            ),
            geo.addPoint(
                cfg.outlet_length,
                cfg.channel_height - cfg.step_height,
                0,
                cfg.resolution,
            ),
            geo.addPoint(cfg.outlet_length, -cfg.step_height, 0, cfg.resolution),
            geo.addPoint(0, -cfg.step_height, 0, cfg.resolution),
        ]
        lines = [geo.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        lines.append(geo.addLine(pts[-1], pts[0]))
        loop = geo.addCurveLoop(lines)
        surf = geo.addPlaneSurface([loop])
        gmsh.model.addPhysicalGroup(2, [surf], 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid")
        geo.synchronize()
        _add_step_refinement(cfg)
        return _finalize_and_extract(comm, gdim=2)

    if cfg.dim == 3:
        if cfg.width is None:
            raise ValueError("Width must be provided for 3D step flow.")
        occ = _initialize_model("step3d", use_occ=True)
        pts = [
            occ.addPoint(0, 0, 0, cfg.resolution),
            occ.addPoint(-cfg.inlet_length, 0, 0, cfg.resolution),
            occ.addPoint(
                -cfg.inlet_length,
                cfg.channel_height - cfg.step_height,
                0,
                cfg.resolution,
            ),
            occ.addPoint(
                cfg.outlet_length,
                cfg.channel_height - cfg.step_height,
                0,
                cfg.resolution,
            ),
            occ.addPoint(cfg.outlet_length, -cfg.step_height, 0, cfg.resolution),
            occ.addPoint(0, -cfg.step_height, 0, cfg.resolution),
        ]
        lines = [occ.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        lines.append(occ.addLine(pts[-1], pts[0]))
        loop = occ.addCurveLoop(lines)
        surf = occ.addPlaneSurface([loop])
        vol = occ.extrude([(2, surf)], 0, 0, cfg.width)
        occ.synchronize()
        _add_step_refinement(cfg)
        top = [t for d, t in vol if d == 3]
        gmsh.model.addPhysicalGroup(3, top, 1)
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
    return (
        abs((xmin + xmax) * 0.5 - xc) < r + tol
        and abs((ymin + ymax) * 0.5 - yc) < r + tol
        and abs((zmin + zmax) * 0.5 - (zc + h * 0.5)) < h * 0.5 + tol
    )


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


def _add_step_refinement(cfg: StepFlowGeometryConfig):
    if cfg.refinement_factor is None:
        return
    h_in = cfg.resolution * cfg.refinement_factor
    fid = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(fid, "VIn", h_in)
    gmsh.model.mesh.field.setNumber(fid, "VOut", cfg.resolution)
    gmsh.model.mesh.field.setNumber(fid, "XMin", 0.0)
    gmsh.model.mesh.field.setNumber(fid, "XMax", cfg.outlet_length / 2)
    gmsh.model.mesh.field.setNumber(fid, "YMin", -cfg.step_height)
    gmsh.model.mesh.field.setNumber(fid, "YMax", 0.0)
    if cfg.dim == 3:
        gmsh.model.mesh.field.setNumber(fid, "ZMin", 0.0)
        gmsh.model.mesh.field.setNumber(fid, "ZMax", cfg.width)
    gmsh.model.mesh.field.setAsBackgroundMesh(fid)


_GEOMETRY_MAP: dict[Geometry, Callable[..., dmesh.Mesh]] = {
    Geometry.CYLINDER_FLOW: cylinder_flow,
    Geometry.STEP_FLOW: step_flow,
}


def get_geometry(
    geometry: Geometry, config: GeometryConfig, comm: MPI.Intracomm
) -> dmesh.Mesh:
    """Dispatch and generate a pre-defined CFD benchmark geometry mesh.

    Args:
        geometry: refer to Geometry enum.
        config: Geometry-specific configuration.
        comm: MPI communicator.
    """
    return _GEOMETRY_MAP[geometry](config, comm)
