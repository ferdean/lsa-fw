#!/dolfinx-env/bin/python3

"""Solve the baseflow of the canonical unit cube problem."""

from __future__ import annotations

import logging
import typing
from pathlib import Path


from Meshing import Mesher, Shape, iCellType
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import BoundaryCondition, define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from Solver.baseflow import BaseFlowSolver

from config import load_bc_config, load_facet_config
from lib.loggingutils import setup_logging

logger = logging.getLogger(__name__)

setup_logging(verbose=True)

_CFG_DIR: typing.Final[Path] = Path("config_files") / "3D" / "unit_cube"
_RE: typing.Final[float] = 10.0

# Generate mesh
mesher = Mesher(Shape.UNIT_CUBE, (20, 20, 20), iCellType.TETRAHEDRON)
_ = mesher.generate()
tags = load_facet_config(_CFG_DIR / "mesh_tags.toml")
mesher.mark_boundary_facets(tags)

# Define function spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# Load boundary conditions
bc_cfgs = [
    BoundaryCondition.from_config(cfg) for cfg in load_bc_config(_CFG_DIR / "bcs.toml")
]
bcs = define_bcs(mesher, spaces, bc_cfgs)

# Compute baseflow
baseflow_solver = BaseFlowSolver(spaces, bcs=bcs)
baseflow = baseflow_solver.solve(
    _RE,
    ramp=True,
    steps=1,
    damping_factor=1.0,  # Undamped Newton solve
    show_plot=False,
)

# Get assembler
assembler = LinearizedNavierStokesAssembler(
    base_flow=baseflow, spaces=spaces, re=_RE, bcs=bcs
)
A, M = assembler.assemble_eigensystem()

# Compute transpose (just to add statistical relevance to tests/performance/test_parallel.py)
transpose = A.T
