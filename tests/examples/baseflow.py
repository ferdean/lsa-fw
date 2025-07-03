"""Solve the baseflow of a typical flow-over-a-cylinder problem."""

from __future__ import annotations

import logging
import typing
from pathlib import Path

from petsc4py import PETSc

from Meshing import Mesher, Shape
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import BoundaryCondition, define_bcs
from FEM.plot import plot_mixed_function
from Solver.baseflow import BaseFlowSolver, export_baseflow, load_baseflow
from config import load_bc_config
from lib.loggingutils import setup_logging

logger = logging.getLogger(__name__)

setup_logging(verbose=True)

_TEST_I_O: bool = False

_CFG_DIR: typing.Final[Path] = Path("config_files") / "2D" / "cylinder"
_RE: typing.Final[float] = 100.0

# Load mesh from disk
mesher = Mesher.from_file(
    Path("out") / "mesh" / "2D" / "cylinder_fixed.xdmf", shape=Shape.CUSTOM_XDMF
)

# Define function spaces
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
    steps=1,
    damping_factor=1.0,  # Undamped Newton solve
    show_plot=False,
)

# Check export/import
if _TEST_I_O:
    PETSc.Options().setValue("viewer_binary_skip_info", "")

    path = Path("out") / "baseflow" / "2D" / f"cylinder_re_{_RE:.2f}"
    export_baseflow(baseflow, path, linear_velocity_ok=False)

    del baseflow  # Ensure data comes only from disk

    baseflow = load_baseflow(path, spaces)

plot_mixed_function(baseflow, scale=0.02)
