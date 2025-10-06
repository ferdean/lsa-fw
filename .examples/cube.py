#!/dolfinx-env/bin/python3

"""Solve the baseflow of the canonical unit cube problem (instrumented for performance determination).

Refer to ./tests/performance for further details.
"""

from __future__ import annotations

import json
import time
import typing
from pathlib import Path

from mpi4py import MPI

from config import load_bc_config, load_facet_config
from FEM.bcs import BoundaryCondition, define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import FunctionSpaceType, define_spaces
from lib.loggingutils import setup_logging
from Meshing.core import Mesher
from Meshing.utils import Shape, iCellType
from Solver.baseflow import BaseFlowSolver

# Suppress logging inside the worker (clean stdout)
setup_logging(disabled=True)

_CFG_DIR: typing.Final[Path] = Path("config_files") / "3D" / "unit_cube"
_RE: typing.Final[float] = 10.0
_TIMERS: dict[str, int] = {}

# Start timer
t0 = time.perf_counter_ns()

# Generate mesh
mesher = Mesher(Shape.UNIT_CUBE, (20, 20, 20), iCellType.TETRAHEDRON)
_ = mesher.generate()
tags = load_facet_config(_CFG_DIR / "mesh_tags.toml")
mesher.mark_boundary_facets(tags)
t1 = time.perf_counter_ns()
_TIMERS["mesh_gen_ns"] = t1 - t0

# Define function spaces
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
t2 = time.perf_counter_ns()
_TIMERS["spaces_def_ns"] = t2 - t1

# Load boundary conditions
bc_cfgs = [
    BoundaryCondition.from_config(cfg) for cfg in load_bc_config(_CFG_DIR / "bcs.toml")
]
bcs = define_bcs(mesher, spaces, bc_cfgs)
t3 = time.perf_counter_ns()
_TIMERS["bcs_def_ns"] = t3 - t2

# Compute baseflow
baseflow_solver = BaseFlowSolver(spaces, bcs=bcs)
baseflow = baseflow_solver.solve(
    _RE,
    ramp=True,
    steps=1,
    damping_factor=1.0,  # Undamped Newton solve
    show_plot=False,
)
t4 = time.perf_counter_ns()
_TIMERS["baseflow_compute_ns"] = t4 - t3

# Assembler eigenproblem
assembler = LinearizedNavierStokesAssembler(
    base_flow=baseflow, spaces=spaces, re=_RE, bcs=bcs
)
A, M = assembler.assemble_eigensystem()
t5 = time.perf_counter_ns()
_TIMERS["assemble_ns"] = t5 - t4

# Print on stdout (ref. tests/test_parallel.py)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(json.dumps(_TIMERS))
