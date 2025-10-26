"""Cylinder flow: solve EVP on pre-assembled (A, M) and write sigma + eigenvector/eigenfunction."""

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
from Solver.utils import PreconditionerType, iSTType

from config import load_cylinder_flow_config
from lib.loggingutils import setup_logging

__show_plots__ = True  # plotting optional

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_NUM_EIG: Final[int] = 5
_EIG_INDEX: Final[int] = 0
_ATOL: Final[float] = 1e-3

_REYNOLDS: Final[float] = 60.0
_TARGET: Final[complex] = 0.05 + 0.74j

_OUT_DIR: Final[Path] = _SAVE_DIR / f"reynolds_{_REYNOLDS:.1f}" / "eigs"

logger = logging.getLogger(__name__)
setup_logging(verbose=True)
_OUT_DIR.mkdir(parents=True, exist_ok=True)

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

es.solver.solve()

# Post-process (eigval)
sigma = es.solver.get_eigenvalue(_EIG_INDEX)

out_sigma = _OUT_DIR / f"sigma_eig{_EIG_INDEX}.txt"
out_sigma.write_text(f"{sigma.real} {sigma.imag}\n", encoding="utf-8")
logger.info("Wrote sigma to '%s'", out_sigma)

# Post-process (eigvec)
# Note for self: the iPETScComplexVector wrapper, when used in a complex PETSc build, works exactly equal to
# iPETScVector, and it collects both real and imaginary part in the `real` vector, and sets `imag` to None.
# This causes inconsistencies, like the fact that to access the imaginary part, one has to access the real part:
# >>> imag_vec = complex_vec.real.as_array().imag
# This class made sense in real PETSc builds, as it is a nice work-around when one cannot work with complex numbers,
# but its use must be reassessed for complex builds.
eigvec = es.solver.get_eigenvector(_EIG_INDEX)

eigfun_real = dfem.Function(spaces.mixed)
eigfun_real.x.array[:] = eigvec.real.as_array().real

eigfun_imag = dfem.Function(spaces.mixed)
eigfun_imag.x.array[:] = eigvec.real.as_array().imag

export_function(eigfun_real, _OUT_DIR, name="eigfun_real")
export_function(eigfun_imag, _OUT_DIR, name="eigfun_imag")

logger.info("Exported eigenfunctions (real/imag) to XDMF in '%s'", case_dir)

# Optionally: static plots (real part)
if __show_plots__:
    for ext in ("pdf", "png"):
        plot_mixed_function(
            eigfun_real,
            PlotMode.STATIC,
            output_path=_OUT_DIR / f"eig_real.{ext}",
            domain=((-5, 0), (20, 5)),
            streamline_density=0.0,
        )
        plot_mixed_function(
            eigfun_imag,
            PlotMode.STATIC,
            output_path=_OUT_DIR / f"eig_imag.{ext}",
            domain=((-5, 0), (20, 5)),
            streamline_density=0.0,
        )
