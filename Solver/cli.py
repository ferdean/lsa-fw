"""LSA-FW Solver CLI.

This CLI provides access to the steady base-flow solver implemented in
:mod:`Solver.baseflow`.  It can generate simple benchmark geometries and
solve the stationary Navier--Stokes equations using Newton's method with a
Stokes initial guess.

Subcommands:
- ``baseflow``: build geometry, define spaces and boundary conditions and
  compute the steady flow.
- ``eigen``: load matrices from disk and solve the eigenvalue problem.

Example usage:

    python -m Solver -p baseflow \
        --geometry cylinder_flow \
        --config config_files/2D/cylinder/cylinder_flow.toml \
        --facet-config config_files/2D/cylinder/mesh_tags.toml \
        --bcs config_files/2D/cylinder/bcs.toml \
        --re 80 --steps 3

    # Solve eigenvalue problem from matrices stored on disk
    python -m Solver eigen --A path/to/A.petsc --M path/to/M.petsc \
        --nev 6 --problem-type gnhep

The command can be parallelised with ``mpirun -n <np>`` as all operations are
MPI-aware.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mpi4py import MPI

from config import (
    load_cylinder_flow_config,
    load_step_flow_config,
    load_facet_config,
    load_bc_config,
)
from lib.loggingutils import setup_logging

from Meshing import Mesher, Geometry
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import BoundaryCondition, define_bcs
from FEM.utils import iPETScMatrix

from .baseflow import BaseFlowSolver
from .eigen import EigenSolver, EigensolverConfig
from .utils import iEpsProblemType

logger = logging.getLogger(__name__)


def _run_baseflow(args: argparse.Namespace) -> None:
    if args.geometry == Geometry.CYLINDER_FLOW:
        geo_cfg = load_cylinder_flow_config(args.config)
    elif args.geometry == Geometry.STEP_FLOW:
        geo_cfg = load_step_flow_config(args.config)
    else:
        raise NotImplementedError(f"Unsupported geometry: {args.geometry}")

    mesher = Mesher.from_geometry(args.geometry, geo_cfg, comm=MPI.COMM_WORLD)
    if args.facet_config:
        marker_fn = load_facet_config(args.facet_config)
        mesher.mark_boundary_facets(marker_fn)

    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

    bc_cfgs = [
        BoundaryCondition.from_config(cfg) for cfg in load_bc_config(args.bc_config)
    ]
    bcs = define_bcs(mesher, spaces, bc_cfgs)

    solver = BaseFlowSolver(spaces, bcs=bcs)
    solver.solve(
        args.re,
        ramp=args.steps > 1,
        steps=args.steps,
        damping_factor=args.damping,
        show_plot=args.plot,
    )


def _run_eigensolver(args: argparse.Namespace) -> None:
    A = iPETScMatrix.load(args.A, comm=MPI.COMM_WORLD)
    M = iPETScMatrix.load(args.M, comm=MPI.COMM_WORLD) if args.M else None

    cfg = EigensolverConfig(
        num_eig=args.nev,
        problem_type=args.problem_type,
        atol=args.atol,
        max_it=args.max_it,
    )
    solver = EigenSolver(cfg, A, M)
    if args.target is not None:
        solver.solver.set_target(args.target)
    pairs = solver.solve()
    for idx, (val, _vec) in enumerate(pairs):
        logger.info("Eigenvalue %d: %s", idx, val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSA-FW solver utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the solution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("baseflow", help="Compute steady base flow")
    base.add_argument(
        "--geometry",
        type=Geometry,
        choices=list(Geometry),
        required=True,
        help="Benchmark geometry name",
    )
    base.add_argument("--config", type=Path, required=True, help="Geometry config file")
    base.add_argument(
        "--facet-config",
        dest="facet_config",
        type=Path,
        required=True,
        help="Facet tags config file",
    )
    base.add_argument(
        "--bcs",
        dest="bc_config",
        type=Path,
        required=True,
        help="Boundary condition config file",
    )
    base.add_argument("--re", type=float, default=100.0, help="Target Reynolds number")
    base.add_argument("--steps", type=int, default=3, help="Number of ramp steps")
    base.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Newton damping factor",
    )
    # FIXME: implement a `--output-path` argument and allow to export the baseflow function (or, at least, the vector)
    # to the local disk
    base.set_defaults(func=_run_baseflow)

    eigen = subparsers.add_parser("eigen", help="Solve a matrix eigenproblem")
    eigen.add_argument("--A", type=Path, required=True, help="Path to operator A")
    eigen.add_argument("--M", type=Path, help="Path to operator M")
    eigen.add_argument("--nev", type=int, default=6, help="Number of eigenvalues")
    eigen.add_argument(
        "--problem-type",
        type=iEpsProblemType.from_string,
        choices=list(iEpsProblemType),
        default=iEpsProblemType.GNHEP,
        help="Eigenproblem type",
    )
    eigen.add_argument("--atol", type=float, default=1e-8, help="Solver tolerance")
    eigen.add_argument("--max-it", type=int, default=500, help="Maximum iterations")
    eigen.add_argument(
        "--target",
        type=float,
        default=None,
        help="Spectral target for shift-and-invert",
    )
    eigen.set_defaults(func=_run_eigensolver)

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception as exc:
        logger.exception("Error during CLI execution: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
