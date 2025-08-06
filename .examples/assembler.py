"""Generate the linear operator and mass matrices for a simple cylinder flow problem.

Several byproducts are also obtained for further post-process:
- Case mesh
- Baseflow solutions
- Recirculation length

Iterates over Re from 2 to 60 in steps of 2.
"""

import csv
import logging
from pathlib import Path
from typing import Final

import numpy as np

from config import load_bc_config, load_cylinder_flow_config, load_facet_config
from FEM.bcs import define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.plot import plot_mixed_function
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.utils import Scalar
from lib.loggingutils import setup_logging
from Meshing.core import Mesher
from Meshing.geometries import Geometry
from Meshing.plot import plot_mesh
from Solver.baseflow import (
    BaseFlowSolver,
    compute_recirculation_length,
    export_baseflow,
    load_baseflow,
)

logger = logging.getLogger(__name__)

if Scalar is not np.float64:
    raise RuntimeError("This script is only compatible with PETSc/SLEPc real builds.")

__show_plots = True  # Toggle plots

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_CASE_DIR: Final[Path] = Path("cases") / "cylinder"
_WAKE_DIR: Final[Path] = _CASE_DIR / "baseflow" / "wake.csv"


_CASE_DIR.mkdir(parents=True, exist_ok=True)
setup_logging(verbose=True, output_path=_CASE_DIR / "logs")

cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
facet_cfg = load_facet_config(_CFG_DIR / "facets.toml")
bcs_cfg = load_bc_config(_CFG_DIR / "bcs.toml")
bcs_perturbation_cfg = load_bc_config(_CFG_DIR / "bcs_perturbation.toml")


# Get (or create) mesh
try:
    mesher = Mesher.from_file(_CASE_DIR / "mesh" / "mesh.xdmf", gdim=2)
except (ValueError, RuntimeError):
    mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
    mesher.mark_boundary_facets(facet_cfg)
    mesher.export(_CASE_DIR / "mesh" / "mesh.xdmf")

if __show_plots:
    plot_mesh(mesher.mesh, tags=mesher.facet_tags)

# Define function spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# Define boundary conditions
bcs = define_bcs(mesher, spaces, bcs_cfg)
bcs_perturbation = define_bcs(mesher, spaces, bcs_perturbation_cfg)

for re in range(20, 60):
    # Solve baseflow
    try:
        baseflow = load_baseflow(_CASE_DIR / "baseflow" / f"re_{int(re)}", spaces)
        if __show_plots:
            plot_mixed_function(baseflow, scale=0)

    except Exception:
        bf_solver = BaseFlowSolver(spaces, bcs=bcs, tags=mesher.facet_tags)
        baseflow = bf_solver.solve(
            re,
            ramp=True,
            tol=1e-12,
            steps=2,
            damping_factor=1.0,  # Undamped Newton
            show_plot=__show_plots,
        )
        export_baseflow(baseflow, _CASE_DIR / "baseflow" / f"re_{int(re)}")

    # Compute recirculation length
    try:
        l_x = compute_recirculation_length(baseflow)
    except RuntimeError:
        l_x = cylinder_cfg.cylinder_radius

    # Append wake data
    write_header = not _WAKE_DIR.exists()
    with open(_WAKE_DIR, "a", newline="") as cf:
        writer = csv.writer(cf)
        if write_header:
            writer.writerow(["re", "l_x"])
        writer.writerow([re, l_x])

    # Assemble and export linear stability matrices
    assembler = LinearizedNavierStokesAssembler(
        baseflow,
        spaces,
        re,
        bcs=bcs_perturbation,
        tags=mesher.facet_tags,
        use_sponge=False,
    )
    A, M = assembler.assemble_eigensystem()

    mat_dir = _CASE_DIR / "matrices" / f"re_{int(re)}"
    mat_dir.mkdir(parents=True, exist_ok=True)
    A.export(mat_dir / "A.mtx")
    M.export(mat_dir / "M.mtx")
