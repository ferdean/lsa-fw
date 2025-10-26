"""Cylinder flow: compute adjoint EV and sensitivity measures on pre-assembled (A, M)."""

import logging
from pathlib import Path
from typing import Final

from FEM.bcs import define_bcs
from FEM.plot import plot_mixed_function
from FEM.spaces import FunctionSpaceType, define_spaces
from FEM.utils import iPETScMatrix
from Meshing.core import Mesher
from Meshing.plot import PlotMode
from Meshing.utils import Geometry
from Sensitivity import EigenSensitivitySolver
from Solver.baseflow import export_function, load_function

from config import load_bc_config, load_cylinder_flow_config, load_facet_config
from lib.loggingutils import setup_logging

__show_plots__ = True

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"

_REYNOLDS: Final[float] = 60.0
_TARGET: Final[complex] = 0.05 + 0.74j


logger = logging.getLogger(__name__)
setup_logging(verbose=True)

# Build mesh/spaces
cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
bcs_cfg = load_bc_config(_CFG_DIR / "bcs.toml")
facet_cfg = load_facet_config(_CFG_DIR / "facets.toml")

mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
mesher.mark_boundary_facets(facet_cfg)

spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
bcs = define_bcs(mesher, spaces, bcs_cfg)

case_dir = _SAVE_DIR / f"reynolds_{_REYNOLDS:.1f}"
eig_dir = _SAVE_DIR / "eigs"
mat_dir = case_dir / "matrices"
A_path = mat_dir / "A.mtx"
M_path = mat_dir / "M.mtx"
bf_path = case_dir / "baseflow"

if not A_path.exists() or not M_path.exists():
    raise ValueError(f"Missing matrices in '{mat_dir}'")
if not bf_path.exists():
    raise ValueError(f"Missing baseflow in '{bf_path}'")


# Load matrices and baseflow
logger.info("Loading matrices from '%s'", mat_dir)
A = iPETScMatrix.from_path(A_path)
A.assemble()
M = iPETScMatrix.from_path(M_path)
M.assemble()

logger.info("Loading baseflwo from '%s'", bf_path)
baseflow = load_function(bf_path, spaces)
if __show_plots__:
    plot_mixed_function(baseflow, PlotMode.INTERACTIVE, scale=0.0)

# Sensitivity ops
sense_solver = EigenSensitivitySolver(
    spaces, bcs, baseflow, _REYNOLDS, tags=mesher.facet_tags, target=_TARGET, A=A, M=M
)

sigma, direct_mode = sense_solver.solve_direct_mode(_TARGET)
adjoint_mode = sense_solver.solve_adjoint_mode(sigma, direct_mode)

export_function(direct_mode, case_dir / "direct_mode", name="direct_mode")
export_function(adjoint_mode, case_dir / "adjoint_mode", name="adjoint_mode")

if __show_plots__:
    plot_mixed_function(direct_mode, PlotMode.INTERACTIVE, title="Direct", scale=0.0)
    plot_mixed_function(adjoint_mode, PlotMode.INTERACTIVE, title="Adjoint", scale=0.0)

    for ext in ("pdf", "png"):
        plot_mixed_function(
            adjoint_mode,
            PlotMode.STATIC,
            output_path=eig_dir / f"adjoint_eig_real.{ext}",
            domain=((-5, 0), (20, 5)),
            streamline_density=0.0,
        )
        plot_mixed_function(
            adjoint_mode,
            PlotMode.STATIC,
            output_path=eig_dir / f"adjoint_eig_real.{ext}",
            domain=((-5, 0), (20, 5)),
            streamline_density=0.0,
        )
