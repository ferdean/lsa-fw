"""Cylinder flow: assemble EVP matrices (A, M) for multiple Reynolds numbers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np

from FEM.bcs import define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
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
from lib.loggingutils import setup_logging, log_global


logger = logging.getLogger(__name__)

__show_plots__ = False

# Configuration -----
if Scalar is not np.float64:
    raise RuntimeError("This script requires a real (float64) PETSc/SLEPc build.")

_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR.mkdir(parents=True, exist_ok=True)

_REYNOLDS: Final[tuple[int, ...]] = tuple(range(40, 91, 5))

# Load config files (once)
cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
facet_cfg = load_facet_config(_CFG_DIR / "facets.toml")
bcs_cfg = load_bc_config(_CFG_DIR / "bcs.toml")
bcs_pert_cfg = load_bc_config(_CFG_DIR / "bcs_perturbation.toml")

# Create mesh (once) and facet tags
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
mesher.mark_boundary_facets(facet_cfg)

if __show_plots__:
    plot_mesh(mesher.mesh, mode=PlotMode.INTERACTIVE, tags=mesher.facet_tags)

# Taylor–Hood spaces (once)
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# Boundary conditions (once)
bcs = define_bcs(mesher, spaces, bcs_cfg)
bcs_perturbation = define_bcs(mesher, spaces, bcs_pert_cfg)

# Loop over Reynolds numbers
for re in _REYNOLDS:
    # Per-case directory structure
    case_dir = _SAVE_DIR / f"reynolds_{re:.1f}"
    (case_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)

    setup_logging(verbose=True, output_path=case_dir / "logs")

    # Export mesh for this case to keep folder self-contained
    mesher.export(case_dir / "mesh.xdmf")

    logger.info("Starting case Re = %.1f", re)

    # Solve baseflow -----
    bf_solver = BaseFlowSolver(spaces, bcs=bcs, re=re, tags=mesher.facet_tags)
    baseflow = bf_solver.solve(
        ramp=True, steps=2, show_plot=__show_plots__, plot_scale=0.0
    )
    export_function(baseflow, case_dir / "baseflow")

    # Assemble operators -----
    assembler = LinearizedNavierStokesAssembler(
        baseflow, spaces, re, bcs=bcs_perturbation, tags=mesher.facet_tags
    )
    A, M = assembler.assemble_eigensystem()

    A.export(case_dir / "matrices" / "A.mtx")
    M.export(case_dir / "matrices" / "M.mtx")

    log_global(
        logger,
        logging.DEBUG,
        "[Re=%.1f] Assembled linearized operator matrix: shape=%s, nnz=%d, norm=%.3e",
        re,
        A.shape,
        A.nonzero_entries,
        A.norm,
    )
    log_global(
        logger,
        logging.DEBUG,
        "[Re=%.1f] Assembled mass matrix: shape=%s, nnz=%d, norm=%.3e",
        re,
        M.shape,
        M.nonzero_entries,
        M.norm,
    )

    logger.info("Finished case Re = %.1f", re)
