"""Cylinder flow: parameter analysis (channel length L)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Any

import tomllib
import numpy as np

from FEM.bcs import define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.utils import Scalar
from Meshing.core import Mesher
from Meshing.plot import PlotMode, plot_mesh
from Meshing.utils import Geometry
from Solver.baseflow import BaseFlowSolver, export_function
from config import load_bc_config, load_cylinder_flow_config, load_facet_config
from lib.loggingutils import setup_logging

logger = logging.getLogger(__name__)

__example_name__ = "cylinder_flow"
__show_plots__ = False  # batch-friendly

_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_BASE_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_RE: Final[float] = 40.0

_SAVE_DIR.mkdir(parents=True, exist_ok=True)

setup_logging(verbose=True)


def _dump_geometry_toml(d: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f'dim = {int(d["dim"])}')
    lines.append(f'cylinder_radius = {float(d["cylinder_radius"])}')
    cx, cy = d["cylinder_center"]
    lines.append(f"cylinder_center = [{float(cx)}, {float(cy)}]")
    x0, x1 = d["x_range"]
    y0, y1 = d["y_range"]
    lines.append(f"x_range = [{float(x0)}, {float(x1)}]")
    lines.append(f"y_range = [{float(y0)}, {float(y1)}]")
    lines.append(f'resolution = {float(d["resolution"])}')
    lines.append(
        f'resolution_around_cylinder = {float(d["resolution_around_cylinder"])}'
    )
    lines.append(f'influence_radius = {float(d["influence_radius"])}')
    return "\n".join(lines) + "\n"


def _dump_facets_toml(face_tags: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for tag in face_tags:
        parts.append("[[FaceTag]]")
        parts.append(f"marker = {int(tag['marker'])}")
        if "when" in tag and tag["when"] is not None:
            axis = tag["when"]["axis"]
            equals = tag["when"]["equals"]
            parts.append(f'when = {{ axis = "{axis}", equals = {float(equals)} }}')
        if tag.get("otherwise", False):
            parts.append("otherwise = true")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mutate_geometry_xmax(geom: dict[str, Any], L: float) -> None:
    xr = list(geom["x_range"])
    xr[1] = float(L)
    geom["x_range"] = xr


def _mutate_facets_outlet_equals(face_tags: list[dict[str, Any]], L: float) -> None:
    found = False
    for tag in face_tags:
        if int(tag.get("marker", -1)) == 2 and "when" in tag and tag["when"]:
            when = tag["when"]
            if when.get("axis") == "x":
                when["equals"] = float(L)
                found = True
                break
    if not found:
        raise ValueError("Outlet FaceTag (marker=2, axis='x') not found in facets.toml")


def _run_for_length(L: int, cfg_dir: Path) -> None:
    save_dir = _SAVE_DIR / f"{__example_name__}_L{L}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Start parametric analysis case: L=%s.", L)

    cylinder_cfg = load_cylinder_flow_config(cfg_dir / "geometry.toml")
    facet_cfg = load_facet_config(cfg_dir / "facets.toml")
    bcs_cfg = load_bc_config(_BASE_CFG_DIR / "bcs.toml")
    bcs_pert_cfg = load_bc_config(_BASE_CFG_DIR / "bcs_perturbation.toml")

    # Meshing ---
    mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
    mesher.mark_boundary_facets(facet_cfg)
    if __show_plots__:
        plot_mesh(mesher.mesh, mode=PlotMode.INTERACTIVE, tags=mesher.facet_tags)

    mesh_dir = save_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesher.export(mesh_dir / "mesh.xdmf")

    # Spaces and BCs ---
    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
    bcs = define_bcs(mesher, spaces, bcs_cfg)
    bcs_perturbation = define_bcs(mesher, spaces, bcs_pert_cfg)

    # Baseflow ---
    bf_solver = BaseFlowSolver(
        spaces, bcs=bcs, re=_RE, tags=mesher.facet_tags, use_sponge=False
    )
    baseflow = bf_solver.solve(ramp=True, steps=2, show_plot=False, plot_scale=0.0)
    export_function(baseflow, save_dir / "baseflow")

    # Linearized operators and export ---
    assembler = LinearizedNavierStokesAssembler(
        baseflow, spaces, _RE, bcs=bcs_perturbation, tags=mesher.facet_tags
    )
    A, M = assembler.assemble_eigensystem()

    mat_dir = save_dir / "matrices" / "wo_pressure"
    mat_dir.mkdir(parents=True, exist_ok=True)
    A.export(mat_dir / "A.mtx")
    M.export(mat_dir / "M.mtx")


def main() -> None:
    """Main script entry point."""
    if Scalar is not np.float64:
        raise RuntimeError("This script requires a real (float64) PETSc/SLEPc build.")

    geom0 = _read_toml(_BASE_CFG_DIR / "geometry.toml")
    facets0 = _read_toml(_BASE_CFG_DIR / "facets.toml")

    for L in range(40, 220, 10):
        geom = {**geom0}
        facets = {**facets0}

        _mutate_geometry_xmax(geom, L)
        face_tags = facets.get("FaceTag")
        if not isinstance(face_tags, list):
            raise ValueError("facets.toml must contain an array 'FaceTag'")
        face_tags = [dict(tag) for tag in face_tags]
        _mutate_facets_outlet_equals(face_tags, L)

        cfg_dir = _SAVE_DIR / f"_cfg_L{L}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        _write_text(cfg_dir / "geometry.toml", _dump_geometry_toml(geom))
        _write_text(cfg_dir / "facets.toml", _dump_facets_toml(face_tags))

        _run_for_length(L, cfg_dir)


if __name__ == "__main__":
    main()
