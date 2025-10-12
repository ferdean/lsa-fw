"""Compute natural frequencies and eigenmodes of an steel plate."""

import logging
from typing import Final

from Elasticity.operators import ElasticityEigenAssembler
from Elasticity.plot import DisplacementPlotConfig, plot_displacement
from Elasticity.spaces import define_space
from Elasticity.utils import process_modes
from Meshing.core import Mesher
from Meshing.utils import Shape, iCellType
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import PreconditionerType, iEpsProblemType, iSTType

from lib.loggingutils import setup_logging

logger = logging.getLogger(__name__)

_E = 2.10e11
_NU = 0.225
_RHO = 7800.0

_EXPECTED_FREQS: Final[tuple[float, ...]] = (
    230.2814112586767,
    332.8661238317352,
    389.7818906781604,
    587.2834023871483,
    1001.8836175631717,
    1071.3541228161757,
    1170.0464071573176,
    1265.0270282091594,
    1752.1685190432913,
    1933.2904724408518,
    2170.141046646646,
    2550.1373576559085,
    3385.788831012662,
)

setup_logging(verbose=True)

mesher = Mesher(
    shape=Shape.BOX,
    n=(24, 24, 3),
    cell_type=iCellType.TETRAHEDRON,
    domain=((0.0, 0.0, 0.0), (0.3, 0.3, 0.006)),
)
mesh = mesher.generate()

function_space = define_space(mesh)

assembler = ElasticityEigenAssembler(
    function_space, young_modulus=_E, poisson_ratio=_NU, density=_RHO
)

M, K = assembler.assemble_eigensystem()

# assert M.is_numerically_symmetric(tol=1e-6)
# assert K.is_numerically_symmetric(tol=5e-5)

cfg = EigensolverConfig(num_eig=25)

es = EigenSolver(K, M, cfg=cfg, check_hermitian=False)

sol = es.solver
sol.set_problem_type(iEpsProblemType.GHEP)
sol.set_target(0.0)
sol.set_st_type(iSTType.SINVERT)
sol.set_st_pc_type(PreconditionerType.CHOLESKY)
sol.set_dimensions(number_eigenpairs=20)

pairs = es.solve()


modes = process_modes(pairs, K, M, function_space, skip_below_hz=1e-1)

for idx, md in enumerate(modes[:12], 1):
    logger.info(
        "#%02d  f = %.3f Hz | wn = %.3e rad/s | mass_norm_check = %s | ω²(RQ) = %.3e",
        idx,
        md.fn,
        md.wn,
        "OK" if md.mass_chk else "FAIL",
        md.rq_omega2,
    )
    plot_cfg = DisplacementPlotConfig(
        title=f"f = {md.fn:.2f} Hz", show_edges=False, scale=0.05
    )
    plot_displacement(md.function, cfg=plot_cfg)
