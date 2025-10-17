"""Steel plate eigenmodes: solve, save figures, and report errors vs NAFEMS."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Final

import numpy as np
from dolfinx import fem as dfem

from Elasticity.bcs import AxisNormalBc, define_bcs
from Elasticity.operators import ElasticityEigenAssembler
from Elasticity.plot import (
    plot_displacement,
    save_displacement,
)  # save_displacement supports xy_view/elev/azim
from Elasticity.spaces import define_space
from Elasticity.utils import process_modes
from Meshing.core import Mesher
from Meshing.utils import Shape, iCellType
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import PreconditionerType, iEpsProblemType, iSTType
from config import load_facet_config
from lib.loggingutils import setup_logging

_L: Final[float] = 10.0
_H: Final[float] = 1.0
_E: Final[float] = 200e9
_NU: Final[float] = 0.300
_RHO: Final[float] = 8000.0
_EXPECTED_FREQS: Final[tuple[float, ...]] = (
    45.897,
    109.440,
    109.440,
    167.890,
    193.590,
    206.190,
    206.190,
)

logger = logging.getLogger(__name__)
setup_logging(verbose=True)

facet_cfg = load_facet_config(Path(__file__).parent / "config" / "facets.toml")

mesher = Mesher(
    shape=Shape.BOX,
    n=(32, 32, 6),
    cell_type=iCellType.HEXAHEDRON,
    domain=((0.0, 0.0, 0.0), (_L, _L, _H)),
)
mesh = mesher.generate()
mesher.mark_boundary_facets(facet_cfg)

function_space = define_space(mesh)
bcs = define_bcs(
    mesher,
    function_space,
    axis_normal=(
        AxisNormalBc(tags=(1, 2), axis=2, value=0.0),
        AxisNormalBc(tags=(3, 4), axis=2, value=0.0),
    ),
)

assembler = ElasticityEigenAssembler(
    function_space, young_modulus=_E, poisson_ratio=_NU, density=_RHO, bcs=bcs
)
M, K = assembler.assemble_eigensystem()

# Eigensolve ---
cfg = EigensolverConfig(num_eig=25)
es = EigenSolver(K, M, cfg=cfg, check_hermitian=False)
sol = es.solver
sol.set_problem_type(iEpsProblemType.GHEP)
sol.set_target(0.0)
sol.set_st_type(iSTType.SINVERT)
sol.set_st_pc_type(PreconditionerType.CHOLESKY)
sol.set_dimensions(number_eigenpairs=24)

pairs = es.solve()

# Postprocessing ---
modes = process_modes(pairs, K, M, function_space, skip_below_hz=2e-1)
expected = np.array(_EXPECTED_FREQS, dtype=float)
computed = np.array([md.fn for md in modes], dtype=float)
k = min(len(expected), len(computed))
ix_c = np.argsort(computed)[:k]
ix_e = np.argsort(expected)[:k]

abs_err = computed[ix_c] - expected[ix_e]
rel_err = abs_err / expected[ix_e]

logger.info("Eigenfrequency comparison (first %d modes)", k)
for rank, (i_c, i_e) in enumerate(zip(ix_c, ix_e), 1):
    logger.info(
        "#%02d  f_comp=%8.3f Hz | f_ref=%8.3f Hz | Δ=%+8.3f Hz | rel_err=%+6.3f%%",
        rank,
        computed[i_c],
        expected[i_e],
        abs_err[rank - 1],
        100.0 * rel_err[rank - 1],
    )
logger.info(
    "Summary: MAE=%.3f Hz | MRE=%.3f%% | Max |rel_err|=%.3f%%",
    float(np.mean(np.abs(abs_err))),
    float(100.0 * np.mean(np.abs(rel_err))),
    float(100.0 * np.max(np.abs(rel_err))),
)

fig_dir = Path(__file__).parent / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)

for idx, md in enumerate(modes, 1):
    logger.info(
        "#%02d  f=%.3f Hz | ω=%.3e rad/s | mass_norm=%s | ω²(RQ)=%.3e",
        idx,
        md.fn,
        md.wn,
        "OK" if md.mass_chk else "FAIL",
        md.rq_omega2,
    )

    # Save static figure (matplotlib). Use xy_view for higher modes for clarity.
    azi = -90 if md.fn >= 180.0 else -50
    ele = 90 if md.fn >= 180.0 else 18
    save_displacement(
        md.function,
        fig_dir / f"eig_{idx:02d}.png",
        scale=500.0,
        cmap="viridis",
        part="abs",
        show_mesh=False,
        title=f"f = {md.fn:.2f} Hz",
        elev=ele,
        azim=azi,
    )
