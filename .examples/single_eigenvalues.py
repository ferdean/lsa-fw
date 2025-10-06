"""Cylinder flow: solve EVP on pre-assembled (A, M) and write sigma.

This script loads matrices previously assembled by the main EVP assembly script, solves for the eigenpair closest to a
given shift-invert target, and writes the dominant eigenvalue to disk.

Note:
- The matrices are not included in the repo to keep it lightweight. Adjust MAT_DIR to point to the folder containing
`A.mtx` and `M.mtx`.
"""

from pathlib import Path
import logging
from typing import Final

from FEM.plot import plot_mixed_function
from Meshing.plot import PlotMode
from Meshing.utils import Geometry
from config import load_cylinder_flow_config
import dolfinx.fem as dfem

from FEM.spaces import FunctionSpaceType, define_spaces
from FEM.utils import iPETScMatrix
from Meshing.core import Mesher
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import PreconditionerType, iSTType
from lib.loggingutils import setup_logging

__show_plots__ = True
__example_name__ = "cylinder_flow"

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_MAT_DIR: Final[Path] = _SAVE_DIR / f"{__example_name__}" / "matrices"
_TARGET: Final[complex] = -0.05
_NUM_EIG: Final[int] = 5
_EIG_INDEX: Final[int] = 0
_ATOL: Final[float] = 1e-3

logger = logging.getLogger(__name__)
setup_logging(verbose=True)

A_path = _MAT_DIR / "A.mtx"
M_path = _MAT_DIR / "M.mtx"

if not A_path.exists() or not M_path.exists():
    raise FileNotFoundError(f"Missing A.mtx or M.mtx in {_MAT_DIR}")
logger.info("Loading matrices from %s", _MAT_DIR)

A = iPETScMatrix.from_path(A_path)
M = iPETScMatrix.from_path(M_path)
A.assemble()
M.assemble()

logger.info("A: shape=%s, nnz=%d, norm=%.3e.", A.shape, A.nonzero_entries, A.norm)
logger.info("M: shape=%s, nnz=%d, norm=%.3e.", M.shape, M.nonzero_entries, M.norm)

# Solve EVP ------
cfg = EigensolverConfig(num_eig=_NUM_EIG, atol=_ATOL)
es = EigenSolver(A, M, cfg=cfg, check_hermitian=False)

es.solver.set_st_type(iSTType.SINVERT)
es.solver.set_target(_TARGET)
es.solver.set_st_pc_type(PreconditionerType.LU)

nconv = es.solver.solve()

sigma = es.solver.get_eigenvalue(_EIG_INDEX)

out_path = _MAT_DIR / f"sigma_eig{_EIG_INDEX}.txt"
out_path.write_text(f"{sigma.real} {sigma.imag}\n", encoding="utf-8")
logger.info("Wrote sigma to %s", out_path)

if __show_plots__:
    cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
    mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

    eigenvector = es.solver.get_eigenvector(_EIG_INDEX)

    eigenfunction = dfem.Function(spaces.mixed)
    eigenfunction.x.array[:] = eigenvector.real.as_array()

    plot_mixed_function(eigenfunction, PlotMode.INTERACTIVE)
