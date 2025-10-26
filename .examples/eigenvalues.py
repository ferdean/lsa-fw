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

__show_plots__ = False

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_NUM_EIG: Final[int] = 5
_EIG_INDEX: Final[int] = 0
_ATOL: Final[float] = 1e-3

_REYNOLDS: Final[tuple[float, ...]] = tuple(range(40, 91, 5))
_TARGETS: Final[tuple[complex, ...]] = (  # Obtained from DOI:10.1115/1.4042737
    (-0.03 + 0.7197388769374216j),
    0.7316769290210628j,
    (0.018 + 0.7379601143282424j),
    (0.03 + 0.742986662573986j),
    (0.05 + 0.744243299635422j),
    (0.061 + 0.7461282552275759j),
    (0.072 + 0.7461282552275759j),
    (0.085 + 0.744557458900781j),
    (0.09 + 0.742986662573986j),
    (0.1 + 0.7398450699203962j),
    (0.115 + 0.7351326809400116j),
)

logger = logging.getLogger(__name__)
setup_logging(verbose=True)


if __show_plots__:
    cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
    mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)


for re, target in zip(_REYNOLDS, _TARGETS):
    case_dir = _SAVE_DIR / f"reynolds_{re:.1f}"
    mat_dir = case_dir / "matrices"
    A_path = mat_dir / "A.mtx"
    M_path = mat_dir / "M.mtx"

    if not A_path.exists() or not M_path.exists():
        logger.warning("Skipping Re = %.1f: missing matrices in '%s'", re, mat_dir)
        continue

    logger.info("[Re=%.1f] Loading matrices from '%s'", re, mat_dir)

    # Load matrices
    A = iPETScMatrix.from_path(A_path)
    A.assemble()
    M = iPETScMatrix.from_path(M_path)
    M.assemble()

    logger.info(
        "[Re=%.1f] A: shape=%s, nnz=%d, norm=%.3e",
        re,
        A.shape,
        A.nonzero_entries,
        A.norm,
    )
    logger.info(
        "[Re=%.1f] M: shape=%s, nnz=%d, norm=%.3e",
        re,
        M.shape,
        M.nonzero_entries,
        M.norm,
    )

    # Solve EVP
    cfg = EigensolverConfig(num_eig=_NUM_EIG, atol=_ATOL)
    es = EigenSolver(A, M, cfg=cfg, check_hermitian=False)

    es.solver.set_st_type(iSTType.SINVERT)
    es.solver.set_target(target)
    es.solver.set_st_pc_type(PreconditionerType.LU)

    es.solver.solve()

    sigma = es.solver.get_eigenvalue(_EIG_INDEX)

    out_path = case_dir / f"sigma_eig{_EIG_INDEX}.txt"
    out_path.write_text(f"{sigma.real} {sigma.imag}\n", encoding="utf-8")
    logger.info("[Re=%.1f] Wrote sigma to '%s'", re, out_path)

    # Optional: plot eigenfunction (real part)
    if __show_plots__:
        eigenvector = es.solver.get_eigenvector(_EIG_INDEX)
        eigenfunction = dfem.Function(spaces.mixed)
        eigenfunction.x.array[:] = eigenvector.real.as_array()
        plot_mixed_function(eigenfunction, PlotMode.INTERACTIVE)

logger.info("All cases processed.")
