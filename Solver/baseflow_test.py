"""Solve a driven cavity flow using the BaseFlowSolver."""

from __future__ import annotations

from pathlib import Path

from Meshing import Mesher, Geometry
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import BoundaryCondition, define_bcs
from FEM.operators import StokesAssembler

from config import load_facet_config, load_bc_config, load_cylinder_flow_config

from Solver.linear import LinearSolver
from Solver.plot import plot_streamlines

_CFG_DIR = Path("config_files/2D")


def run_stokes(re: float = 1.0, n: tuple[int, int] = (12, 12)) -> None:
    """Compute the steady cavity flow and write the result to XDMF."""
    cfg = load_cylinder_flow_config(_CFG_DIR / "cylinder" / "cylinder_flow.toml")
    mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, config=cfg)

    marker_fn = load_facet_config(_CFG_DIR / "cylinder" / "mesh_tags.toml")
    mesher.mark_boundary_facets(marker_fn)

    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

    bc_cfgs = [
        BoundaryCondition.from_config(cfg)
        for cfg in load_bc_config(_CFG_DIR / "cylinder" / "bcs.toml")
    ]
    bcs = define_bcs(mesher, spaces, bc_cfgs)

    assembler = StokesAssembler(spaces=spaces, re=re, bcs=bcs)
    solver = LinearSolver(assembler)

    wh = solver.direct_lu_solve(show_plot=False)
    plot_streamlines(wh, n_points=20, factor=0.005)


if __name__ == "__main__":
    run_stokes()
