"""Cylinder flow: solve EVP on pre-assembled (A, M) and write sigma.

This script loads matrices previously assembled by the main EVP assembly script, solves for the eigenpair closest to a
given shift-invert target, and writes the dominant eigenvalue to disk.

Note:
- The matrices are not included in the repo to keep it lightweight. Adjust MAT_DIR to point to the folder containing
`A.mtx` and `M.mtx`.
- This script is just a single-eig version of `eigenvalues.py`, with additional post-process actions (like enhanced
export and plotting).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import dolfinx.fem as dfem

from FEM.plot import plot_mixed_function
from FEM.spaces import FunctionSpaceType, define_spaces
from FEM.utils import iPETScMatrix
from Meshing.core import Mesher
from Meshing.plot import PlotMode
from Meshing.utils import Geometry
from Solver.baseflow import export_function
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import PreconditionerType, iEpsWhich, iSTType

from config import load_cylinder_flow_config
from lib.loggingutils import setup_logging

__show_plots__ = False
__export__ = False

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_NUM_EIG: Final[int] = 5
_EIG_INDEX: Final[int] = 0
_ATOL: Final[float] = 1e-3

_REYNOLDS: Final[float] = 47.0
_TARGET: Final[complex] = 0.7316769290210628j

_OUT_DIR: Final[Path] = _SAVE_DIR / f"reynolds_{_REYNOLDS:.1f}" / "adjoint_mode"

logger = logging.getLogger(__name__)
setup_logging(verbose=True)

_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _hermitian(A: iPETScMatrix) -> iPETScMatrix:
    raw = A.raw.copy()
    raw.transpose()
    raw.conjugate()
    return iPETScMatrix(raw)


# Build mesh/spaces (needed to export eigenfunction)
cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

case_dir = _SAVE_DIR / f"reynolds_{_REYNOLDS:.1f}"
mat_dir = case_dir / "matrices"
A_path = mat_dir / "A.mtx"
M_path = mat_dir / "M.mtx"

if not A_path.exists() or not M_path.exists():
    raise ValueError(f"Missing matrices in '{mat_dir}'")

logger.info("Loading matrices from '%s'", mat_dir)

# Load matrices
A = iPETScMatrix.from_path(A_path)
A.assemble()
M = iPETScMatrix.from_path(M_path)
M.assemble()

logger.info("A: shape=%s, nnz=%d, norm=%.3e", A.shape, A.nonzero_entries, A.norm)
logger.info("M: shape=%s, nnz=%d, norm=%.3e", M.shape, M.nonzero_entries, M.norm)

# Solve EVP
cfg = EigensolverConfig(num_eig=_NUM_EIG, atol=_ATOL)
es = EigenSolver(A, M, cfg=cfg, check_hermitian=False)

es.solver.set_st_type(iSTType.SINVERT)
es.solver.set_target(_TARGET)
es.solver.set_st_pc_type(PreconditionerType.LU)
es.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)

es.solver.solve()

epais = es.solver.get_all_eigenpairs_up_to(es.solver.get_num_converged())
evals, _ = zip(*epais)
print(evals)

# Post-process (eigvec)
# Note for self: the iPETScComplexVector wrapper, when used in a complex PETSc build, works exactly equal to
# iPETScVector, and it collects both real and imaginary part in the `real` vector, and sets `imag` to None.
# This causes inconsistencies, like the fact that to access the imaginary part, one has to access the real part:
# >>> imag_vec = complex_vec.real.as_array().imag
# This class made sense in real PETSc builds, as it is a nice work-around when one cannot work with complex numbers,
# but its use must be reassessed for complex builds.

# Optionally: export eigenvectors and eigenvalues
if __export__:
    sigma = es.solver.get_eigenvalue(_EIG_INDEX)
    out_sigma = _OUT_DIR / f"sigma_eig_trans_{_EIG_INDEX}.txt"
    out_sigma.write_text(f"{sigma.real} {sigma.imag}\n", encoding="utf-8")

    logger.info("Wrote sigma to '%s'", out_sigma)

    eigvec = es.solver.get_eigenvector(_EIG_INDEX)
    adjoint_mode = dfem.Function(spaces.mixed)
    adjoint_mode.x.array[:] = eigvec.real.as_array().real
    export_function(adjoint_mode, _OUT_DIR, name="adjoint_mode")

    logger.info("Exported eigenfunctions (real/imag) to XDMF in '%s'", case_dir)

# Optionally: static plots (real part)
if __show_plots__:
    for ext in ("pdf", "png"):
        plot_mixed_function(
            adjoint_mode,
            PlotMode.STATIC,
            output_path=_OUT_DIR / f"adjoint_mode.{ext}",
            domain=((-3, 0), (15, 3)),
            streamline_density=0.0,
            normalize=True,
        )
