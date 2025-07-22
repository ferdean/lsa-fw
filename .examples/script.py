"""Example of complete pipeline using only scripting tools (i.e., no I/O from file system)."""

from pathlib import Path

import numpy as np

from FEM.bcs import define_bcs, BoundaryConditionType, BoundaryCondition
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import define_spaces, FunctionSpaceType
from lib.loggingutils import setup_logging
from Meshing import Mesher, Geometry
from Meshing.geometries import CylinderFlowGeometryConfig
from Solver.baseflow import BaseFlowSolver


setup_logging(verbose=True)


_RE = 60
_OUTPUT_PATH = Path("out") / "matrices" / f"reynolds_{int(_RE)}"
_OUTPUT_PATH.mkdir(exist_ok=True)


def _marker_fn(x: np.ndarray) -> int:
    if np.isclose(x[0], -20):
        return 1  # Inlet
    if np.isclose(x[0], 60):
        return 2  # Outlet
    if np.isclose(abs(x[1]), 20):
        return 3  # Top and bottom (free-flow)
    return 4  # Cylinder


# Case configuration:
#    - Cylinder flow
#    - Geometrical coordinates centered on cylinder axis
#    - Local refinement around cylinder wake
config = CylinderFlowGeometryConfig(
    dim=2,
    cylinder_radius=0.5,
    cylinder_center=(0.0, 0.0),
    x_range=(-20.0, 60.0),
    y_range=(-20.0, 20.0),
    resolution=1,
    resolution_around_cylinder=0.15,
    influence_radius=40,
    influence_length=40,
)

# Mesher generation (mesh manager)
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, config, cache=None)
mesher.mark_boundary_facets(_marker_fn)

# Function spaces (Taylor-Hood) definition
function_spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

# Boundary conditions definition
bcs_cfg = [
    BoundaryCondition(1, BoundaryConditionType.DIRICHLET_VELOCITY, (1.0, 0.0)),
    BoundaryCondition(2, BoundaryConditionType.NEUMANN_VELOCITY, (0.0, 0.0)),
    BoundaryCondition(3, BoundaryConditionType.NEUMANN_VELOCITY, (0.0, 0.0)),
    BoundaryCondition(4, BoundaryConditionType.DIRICHLET_VELOCITY, (0.0, 0.0)),
]
bcs = define_bcs(mesher, function_spaces, bcs_cfg)

# Baseflow (stationary) computation
bf_solver = BaseFlowSolver(function_spaces, bcs=bcs)
bf = bf_solver.solve(
    _RE,
    ramp=True,
    steps=3,
    damping_factor=1.0,  # Undamped Newton solve
    show_plot=True,
)

# Assemble eigensystem
assembler = LinearizedNavierStokesAssembler(bf, function_spaces, _RE, bcs)
A, M = assembler.assemble_eigensystem()

A.export(_OUTPUT_PATH / "A")
M.export(_OUTPUT_PATH / "M")
