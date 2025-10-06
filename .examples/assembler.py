"""Cylinder flow: assemble EVP matrices (A, M)."""

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

__example_name__ = "cylinder_flow"
__show_plots__ = True
__enable_tests__ = True

# Configuration -----
if Scalar is not np.float64:
    raise RuntimeError("This script requires a real (float64) PETSc/SLEPc build.")

_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_RE: Final[float] = 40.0
_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Setup -----
setup_logging(verbose=True, output_path=_SAVE_DIR / "logs")

# Load config files
cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry.toml")
facet_cfg = load_facet_config(_CFG_DIR / "facets.toml")
bcs_cfg = load_bc_config(_CFG_DIR / "bcs.toml")
bcs_pert_cfg = load_bc_config(_CFG_DIR / "bcs_perturbation.toml")

# Create benchmark mesh
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
mesher.mark_boundary_facets(facet_cfg)

if __show_plots__:
    plot_mesh(mesher.mesh, mode=PlotMode.INTERACTIVE, tags=mesher.facet_tags)

mesher.export(_SAVE_DIR / f"{__example_name__}" / "mesh.xdmf")

# Taylor–Hood spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# Boundary conditions
bcs = define_bcs(mesher, spaces, bcs_cfg)
bcs_perturbation = define_bcs(mesher, spaces, bcs_pert_cfg)


# Solve baseflow -----
bf_solver = BaseFlowSolver(
    spaces, bcs=bcs, re=_RE, tags=mesher.facet_tags, use_sponge=False
)
baseflow = bf_solver.solve(
    ramp=True,
    steps=2,
    show_plot=__show_plots__,
    plot_scale=0.0,
)
export_function(baseflow, _SAVE_DIR / f"{__example_name__}" / "baseflow")

# Assemble operators -----
assembler = LinearizedNavierStokesAssembler(
    baseflow, spaces, _RE, bcs=bcs_perturbation, tags=mesher.facet_tags
)
A, M = assembler.assemble_eigensystem()

A.export(_SAVE_DIR / f"{__example_name__}" / "matrices" / "A.mtx")
M.export(_SAVE_DIR / f"{__example_name__}" / "matrices" / "M.mtx")

log_global(
    logger,
    logging.DEBUG,
    "Assembled linearized operator matrix: shape=%s, nnz=%d, norm=%.3e",
    A.shape,
    A.nonzero_entries,
    A.norm,
)
log_global(
    logger,
    logging.DEBUG,
    "Assembled mass matrix: shape=%s, nnz=%d, norm=%.3e",
    M.shape,
    M.nonzero_entries,
    M.norm,
)

# Tests -----
if __enable_tests__:
    _TOL: Final[float] = 1e-10
    n_u, _ = spaces.velocity_dofs
    n_p, _ = spaces.pressure_dofs

    A_blocks = assembler.extract_subblocks(A)
    M_blocks = assembler.extract_subblocks(M)

    A_vv = A_blocks[0, 0]  # Velocity–velocity
    A_vp = A_blocks[0, 1]  # Velocity–pressure
    A_pv = A_blocks[1, 0]  # Pressure–velocity
    A_pp = A_blocks[1, 1]  # Pressure–pressure

    M_vv = M_blocks[0, 0]
    M_vp = M_blocks[0, 1]
    M_pv = M_blocks[1, 0]
    M_pp = M_blocks[1, 1]

    _, dofs_p = spaces.mixed.sub(1).collapse()
    right_vec = A.create_vector_right()
    arr = np.zeros((right_vec.size,), dtype=float)
    arr[dofs_p] = 1.0
    right_vec.set_array(arr)
    right_vec.ghost_update()

    # Check shape and non-triviality
    assert A.shape == (n_u + n_p, n_u + n_p)
    assert M.shape == (n_u + n_p, n_u + n_p)
    assert A.norm > _TOL
    assert M.norm > _TOL

    # Check Oseen blocks
    assert A_vv.shape == (n_u, n_u)
    assert A_pp.shape == (n_p, n_p)
    assert A_vv.norm > 0
    assert A_pv.norm > 0
    assert A_vp.norm > 0
    assert A_pp.norm < 1e-10

    # Check that A_vp = - A_pv.T
    temp = A_vp.duplicate(copy=True)
    temp.axpy(-1.0, A_pv.T)
    assert temp.norm < _TOL

    # Check mass is SPD and blocks
    assert M.is_numerically_symmetric()
    for _ in range(20):
        x = M.create_vector_right()
        x.set_random()
        assert x.dot(M @ x) > 0
        del x

    assert M_vv.is_numerically_symmetric()
    for _ in range(20):
        x = M_vv.create_vector_right()
        x.set_random()
        assert x.dot(M_vv @ x) > 0
        del x

    assert M_vp.norm < 1e-10
    assert M_pv.norm < 1e-10
    assert M_pp.norm < 1e-10

    log_global(logger, logging.DEBUG, "All tests passed!")
