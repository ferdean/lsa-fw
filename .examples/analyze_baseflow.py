"""Cylinder flow: analyze baseflow (wake)."""

import logging
import csv
from pathlib import Path
from typing import Final

import numpy as np

from FEM.bcs import define_bcs
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.utils import Scalar
from Meshing.core import Mesher
from Meshing.utils import Geometry
from Solver.baseflow import BaseFlowSolver, compute_drag, compute_recirculation_length
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


SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
CYL_MARKER: Final[int] = 5

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

# Taylor–Hood spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# BCs
bcs = define_bcs(mesher, spaces, bcs_cfg)
bcs_perturbation = define_bcs(mesher, spaces, bcs_pert_cfg)

# Recirculation length and drag ---

results_path = SAVE_DIR / "recirculation_lengths.csv"
with results_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["re", "recirculation_length", "drag_x"])

    for re in range(2, 64, 2):
        bf_solver = BaseFlowSolver(spaces, bcs=bcs, tags=mesher.facet_tags)
        baseflow = bf_solver.solve(re=re, ramp=True, show_plot=False, steps=2)

        try:
            recirculation_length = compute_recirculation_length(baseflow)
        except RuntimeError:
            recirculation_length = 0.0

        drag_x = compute_drag(
            baseflow, re=re, facet_tags=mesher.facet_tags, cylinder_marker=CYL_MARKER
        )

        logger.info(
            "Re = %.1f | L_recirc = %.4f | Fx = %.6f", re, recirculation_length, drag_x
        )

        writer.writerow([re, recirculation_length, drag_x])
