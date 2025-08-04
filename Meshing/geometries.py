"""LSA-FW Meshing pre-defined geometries.

This module provides geometry generators for standard CFD benchmark cases.
These generators build fully configurable domains, such as cylinder flow and step flow.
"""

from typing import Callable

from dolfinx.io import gmshio
import dolfinx.mesh as dmesh
import gmsh  # type: ignore[import-untyped]
from mpi4py import MPI
import numpy as np

from config import CylinderFlowGeometryConfig, StepFlowGeometryConfig

from .utils import Geometry

GeometryConfig = CylinderFlowGeometryConfig | StepFlowGeometryConfig


def get_geometry(
    geometry: Geometry, config: GeometryConfig, comm: MPI.Intracomm
) -> dmesh.Mesh:
    """Dispatch and generate a pre-defined CFD benchmark geometry mesh."""
    return _GEOMETRY_MAP[geometry](config, comm)


def _cylinder_flow(
    cfg: CylinderFlowGeometryConfig,
    comm: MPI.Intracomm,
) -> dmesh.Mesh:
    """Generate a mesh for cylinder flow in a rectangular channel.

    Performs automatic local refinement around the cylinder boundary.
    """
    xmin, xmax = cfg.x_range
    ymin, ymax = cfg.y_range

    if cfg.dim == 2:
        geo = _initialize_model("cylinder2d")
        xc, yc = cfg.cylinder_center

        # Mesh options for smooth circle
        n_circle_points = 32
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumCirclePoints", n_circle_points)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        # Channel rectangle
        pts = [
            geo.addPoint(x, y, 0.0, cfg.resolution)
            for x, y in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
        ]
        lines = [geo.addLine(a, b) for a, b in zip(pts, pts[1:] + pts[:1])]
        rect_loop = geo.addCurveLoop(lines)

        # Circle arc points
        circ_pts = []
        for i in range(n_circle_points):
            angle = 2 * np.pi * i / n_circle_points
            x = xc + cfg.cylinder_radius * np.cos(angle)
            y = yc + cfg.cylinder_radius * np.sin(angle)
            circ_pts.append(geo.addPoint(x, y, 0.0, cfg.resolution_around_cylinder))
        arcs = [
            geo.addCircleArc(
                circ_pts[i],
                geo.addPoint(xc, yc, 0.0, cfg.resolution_around_cylinder),
                circ_pts[(i + 1) % n_circle_points],
            )
            for i in range(n_circle_points)
        ]
        circ_loop = geo.addCurveLoop(arcs)

        # Surface with hole
        surf = geo.addPlaneSurface([rect_loop, circ_loop])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2, [surf], 1)
        gmsh.model.setPhysicalName(2, 1, "Fluid")

        # Local refinement near cylinder only
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

        # Apply refinement field
        gmsh.model.mesh.field.setAsBackgroundMesh(fid_thr)

        # Mesh optimization
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

        # Combine and optimize
        fid_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(fid_min, "FieldsList", [fid_thr])
        gmsh.model.mesh.field.setAsBackgroundMesh(fid_min)

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        return _finalize_and_extract(comm, gdim=3, use_occ=True)

    raise ValueError("Only 2D or 3D supported.")


def _step_flow(
    cfg: StepFlowGeometryConfig,
    comm: MPI.Intracomm,
) -> dmesh.Mesh:
    """Generate a mesh for the backward-facing step problem."""
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
    Geometry.CYLINDER_FLOW: _cylinder_flow,
    Geometry.STEP_FLOW: _step_flow,
}
