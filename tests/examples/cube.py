#!/dolfinx-env/bin/python3

"""Solve the baseflow of the canonical unit cube problem (instrumented)."""

from __future__ import annotations

import json
import logging
import time
import typing
from pathlib import Path

from mpi4py import MPI

from config import load_bc_config, load_facet_config
from FEM.bcs import BoundaryCondition, define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import FunctionSpaceType, define_spaces
from lib.loggingutils import setup_logging
from Meshing import Mesher, Shape, iCellType
from Solver.baseflow import BaseFlowSolver

logger = logging.getLogger(__name__)

setup_logging(disabled=True)

_CFG_DIR: typing.Final[Path] = Path("config_files") / "3D" / "unit_cube"
_RE: typing.Final[float] = 10.0
_TIMERS: dict[str, int] = {}

start = time.perf_counter_ns()

# Generate mesh
mesher = Mesher(Shape.UNIT_CUBE, (8, 8, 8), iCellType.TETRAHEDRON)
_ = mesher.generate()
tags = load_facet_config(_CFG_DIR / "mesh_tags.toml")
mesher.mark_boundary_facets(tags)
_TIMERS["mesh_gen_ns"] = time.perf_counter_ns() - start

# Define function spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
_TIMERS["spaces_def_ns"] = time.perf_counter_ns() - _TIMERS["mesh_gen_ns"]

# Load boundary conditions
bc_cfgs = [
    BoundaryCondition.from_config(cfg) for cfg in load_bc_config(_CFG_DIR / "bcs.toml")
]
bcs = define_bcs(mesher, spaces, bc_cfgs)
_TIMERS["bcs_def_ns"] = time.perf_counter_ns() - _TIMERS["spaces_def_ns"]

# Compute baseflow
baseflow_solver = BaseFlowSolver(spaces, bcs=bcs)
baseflow = baseflow_solver.solve(
    _RE,
    ramp=True,
    steps=1,
    damping_factor=1.0,  # Undamped Newton solve
    show_plot=False,
)
_TIMERS["baseflow_compute_ns"] = time.perf_counter_ns() - _TIMERS["spaces_def_ns"]

# Get assembler
assembler = LinearizedNavierStokesAssembler(
    base_flow=baseflow, spaces=spaces, re=_RE, bcs=bcs
)
A, M = assembler.assemble_eigensystem()
_TIMERS["assemble_ns"] = time.perf_counter_ns() - _TIMERS["baseflow_compute_ns"]

# Print on stdout (ref. tests/test_parallel.py)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(json.dumps(_TIMERS))
