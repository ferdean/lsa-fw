"""LSA-FW Solver CLI.

This CLI provides access to the steady base-flow solver implemented in
:mod:`Solver.baseflow`.  It can generate simple benchmark geometries and
solve the stationary Navier--Stokes equations using Newton's method with a
Stokes initial guess.

Subcommands:
- ``baseflow``: build geometry, define spaces and boundary conditions and
  compute the steady flow.

Example usage::

    python -m Solver baseflow \
        --geometry cylinder_flow \
        --config config_files/2D/cylinder/cylinder_flow.toml \
        --facet-config config_files/2D/cylinder/mesh_tags.toml \
        --re 80 --steps 5 --damping 0.8 --plot

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

from .baseflow import BaseFlowSolver

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
        BoundaryCondition.from_config(cfg)
        for cfg in load_bc_config(args.bc_config)
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
        "--bc-config",
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
    base.set_defaults(func=_run_baseflow)

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - CLI
        logger.exception("Error during CLI execution: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
