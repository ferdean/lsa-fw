"""Solve a driven cavity flow using the BaseFlowSolver."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

from Meshing import Mesher, Geometry
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import BoundaryCondition, define_bcs

from Solver.baseflow import BaseFlowSolver

from config import load_bc_config, load_facet_config, load_cylinder_flow_config
from lib.loggingutils import setup_logging

logger = logging.getLogger(__name__)
setup_logging(verbose=True)

_CFG_DIR: typing.Final[Path] = Path("config_files/2D/cylinder")
_RE: typing.Final[float] = 80.0

# Generate mesh
cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "cylinder_flow.toml")
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
marker_fn = load_facet_config(_CFG_DIR / "mesh_tags.toml")
mesher.mark_boundary_facets(marker_fn)

# Define function spaces (Taylor-Hood)
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# Load boundary conditions
bc_cfgs = [
    BoundaryCondition.from_config(cfg) for cfg in load_bc_config(_CFG_DIR / "bcs.toml")
]
bcs = define_bcs(mesher, spaces, bc_cfgs)

# Compute baseflow
baseflow_solver = BaseFlowSolver(spaces, bcs=bcs)
baseflow = baseflow_solver.solve(
    _RE,
    ramp=True,
    steps=5,
    damping_factor=1.0,  # Undamped Newton solve
    show_plot=True,
)
