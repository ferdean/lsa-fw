"""Cylinder flow: assemble (A, M), solve leading eigenpairs, and save results.

Note: this script targets a single Re for clarity.
"""

import logging
from pathlib import Path
from typing import Final

import numpy as np

from FEM.bcs import define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.plot import plot_mixed_function
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.utils import Scalar
from Meshing.core import Mesher
from Meshing.plot import PlotMode, plot_mesh
from Meshing.utils import Geometry
from Solver.baseflow import BaseFlowSolver, export_function
from config import (
    load_bc_config,
    load_cylinder_flow_config,
    load_facet_config,
)
from lib.loggingutils import setup_logging

logger = logging.getLogger(__name__)

# Configuration ---

if Scalar is not np.float64:
    raise RuntimeError("This script requires a real (float64) PETSc/SLEPc build.")


__example_name = "nominal"
__show_plots = False

SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
RE: Final[float] = 60.0

# Setup ---

SAVE_DIR.mkdir(parents=True, exist_ok=True)
setup_logging(verbose=True, output_path=SAVE_DIR / "logs")

cylinder_cfg = load_cylinder_flow_config(CFG_DIR / "geometry.toml")
facet_cfg = load_facet_config(CFG_DIR / "facets.toml")
bcs_cfg = load_bc_config(CFG_DIR / "bcs.toml")
bcs_pert_cfg = load_bc_config(CFG_DIR / "bcs_perturbation.toml")

# Mesh
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
mesher.mark_boundary_facets(facet_cfg)
if __show_plots:
    plot_mesh(mesher.mesh, tags=mesher.facet_tags)

# Taylor–Hood spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# BCs
bcs = define_bcs(mesher, spaces, bcs_cfg)
bcs_perturbation = define_bcs(mesher, spaces, bcs_pert_cfg)

# Baseflow ---

bf_solver = BaseFlowSolver(spaces, bcs=bcs, tags=mesher.facet_tags)
baseflow = bf_solver.solve(re=RE, ramp=True, steps=2, show_plot=True, plot_scale=0.0)

if __show_plots:
    plot_mixed_function(
        baseflow,
        mode=PlotMode.STATIC,
        output_path=Path(__file__).parent / "fig.pdf",
        domain=((-5, 0), (20, 5)),
    )

export_function(baseflow, SAVE_DIR / f"{__example_name}" / "baseflow")

# Linearized operators ---

assembler = LinearizedNavierStokesAssembler(
    baseflow, spaces, RE, bcs=bcs_perturbation, tags=mesher.facet_tags, use_sponge=False
)
A, M = assembler.assemble_eigensystem()

A.export(SAVE_DIR / "matrices" / f"{__example_name}" / "A.mtx")
M.export(SAVE_DIR / "matrices" / f"{__example_name}" / "M.mtx")
