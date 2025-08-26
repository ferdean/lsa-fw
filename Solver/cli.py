"""LSA-FW Solver CLI.

This module exposes command line utilities for all core routines in the solver package.  It can generate benchmark
geometries, compute steady baseflows, assemble the linearized Navier-Stokes operator and mass matrices, and perform
eigenvalue analysis.

Subcommands:
- ``baseflow``: build geometry, define spaces and boundary conditions, and
  compute the steady flow.
- ``assemble``: solve the baseflow and assemble the linearized operator and
  mass matrices for stability analysis.
- ``eigen``: load matrices from disk and solve for a set of eigenpairs.

Example usage:

    python -m Solver -p assemble \
        --geometry cylinder_flow \
        --config config_files/2D/cylinder/cylinder_flow.toml \
        --facet-config config_files/2D/cylinder/mesh_tags.toml \
        --bcs config_files/2D/cylinder/bcs.toml \
        --re 80 --steps 3 --matrix-dir matrices

The commands can be parallelised with ``mpirun -n <np>`` as all operations
are MPI-aware.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mpi4py import MPI

from config import (
    load_bc_config,
    load_cylinder_flow_config,
    load_facet_config,
    load_step_flow_config,
)
from FEM.bcs import define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.spaces import FunctionSpaceType, define_spaces
from FEM.utils import iPETScMatrix
from lib.cache import CacheStore
from lib.loggingutils import log_global, setup_logging
from Meshing.core import Mesher
from Meshing.geometries import Geometry

from .baseflow import BaseFlowSolver, export_baseflow
from .eigen import EigensolverConfig, EigenSolver
from .utils import iEpsProblemType, iEpsWhich, iSTType, PreconditionerType

logger = logging.getLogger(__name__)


def _run_baseflow(args: argparse.Namespace) -> None:
    cache = CacheStore(args.output_path) if args.output_path else None

    if args.mesh is not None:
        mesher = Mesher.from_file(args.mesh)
    else:
        if args.geometry == Geometry.CYLINDER_FLOW:
            geo_cfg = load_cylinder_flow_config(args.config)
        elif args.geometry == Geometry.STEP_FLOW:
            geo_cfg = load_step_flow_config(args.config)
        else:
            raise NotImplementedError(f"Unsupported geometry: {args.geometry}")
        mesher = Mesher.from_geometry(
            args.geometry, geo_cfg, comm=MPI.COMM_WORLD, cache=cache, key="mesh"
        )
    if args.facet_config:
        marker_fn = load_facet_config(args.facet_config)
        mesher.mark_boundary_facets(marker_fn)

    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

    bc_cfgs = load_bc_config(args.bc_config)
    bcs = define_bcs(mesher, spaces, bc_cfgs)

    solver = BaseFlowSolver(spaces, bcs=bcs)
    solver.solve(
        args.re,
        ramp=args.steps > 1,
        steps=args.steps,
        max_it=args.max_it,
        tol=args.tol,
        damping_factor=args.damping,
        show_plot=args.plot,
        plot_scale=args.plot_scale,
        cache=cache,
        key=f"baseflow_{args.re}",
    )


def _run_assemble(args: argparse.Namespace) -> None:
    cache = CacheStore(args.output_path) if args.output_path else None

    if args.mesh is not None:
        mesher = Mesher.from_file(args.mesh)
    else:
        if args.geometry == Geometry.CYLINDER_FLOW:
            geo_cfg = load_cylinder_flow_config(args.config)
        elif args.geometry == Geometry.STEP_FLOW:
            geo_cfg = load_step_flow_config(args.config)
        else:
            raise NotImplementedError(f"Unsupported geometry: {args.geometry}")
        mesher = Mesher.from_geometry(
            args.geometry, geo_cfg, comm=MPI.COMM_WORLD, cache=cache, key="mesh"
        )
    if args.facet_config:
        marker_fn = load_facet_config(args.facet_config)
        mesher.mark_boundary_facets(marker_fn)

    spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

    bc_cfgs = load_bc_config(args.bc_config)
    bcs = define_bcs(mesher, spaces, bc_cfgs)

    if args.bc_perturbation:
        bc_p_cfgs = load_bc_config(args.bc_perturbation)
        bcs_perturb = define_bcs(mesher, spaces, bc_p_cfgs)
    else:
        bcs_perturb = bcs

    solver = BaseFlowSolver(spaces, bcs=bcs, tags=mesher.facet_tags)
    baseflow = solver.solve(
        args.re,
        ramp=args.steps > 1,
        steps=args.steps,
        max_it=args.max_it,
        tol=args.tol,
        damping_factor=args.damping,
        show_plot=args.plot,
        plot_scale=args.plot_scale,
        cache=cache,
        key=f"baseflow_{args.re}",
    )

    if args.export_baseflow:
        export_baseflow(baseflow, args.export_baseflow)

    assembler = LinearizedNavierStokesAssembler(
        baseflow,
        spaces,
        args.re,
        bcs=bcs_perturb,
        tags=mesher.facet_tags,
        use_sponge=args.use_sponge,
    )
    A, M = assembler.assemble_eigensystem(cache=cache)

    if args.matrix_dir:
        args.matrix_dir.mkdir(parents=True, exist_ok=True)
        A.export(args.matrix_dir / "A.mtx")
        M.export(args.matrix_dir / "M.mtx")


def _run_eigen(args: argparse.Namespace) -> None:
    A = iPETScMatrix.from_path(args.operator)
    M = iPETScMatrix.from_path(args.mass) if args.mass else None

    cfg = EigensolverConfig(
        num_eig=args.num_eig,
        problem_type=args.problem_type,
        atol=args.atol,
        max_it=args.max_it,
    )
    es = EigenSolver(cfg, A, M, check_hermitian=not args.no_check)

    if args.which:
        es.solver.set_which_eigenpairs(args.which)
    if args.st_type:
        es.solver.set_st_type(args.st_type)
    if args.st_pc_type:
        es.solver.set_st_pc_type(args.st_pc_type)
    if args.st_target is not None:
        es.solver.set_target(args.st_target)

    eigenpairs = es.solve()
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
    for idx, (eigval, eigvec) in enumerate(eigenpairs, start=1):
        log_global(logger, logging.INFO, "Eigenvalue %d = %s", idx, eigval)
        if args.export:
            eigvec.real.export(args.export / f"ev_{idx}.dat")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSA-FW solver utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the solution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    base = subparsers.add_parser("baseflow", help="Compute steady baseflow")
    base.add_argument("--mesh", type=Path, help="Path to the mesh file")
    base.add_argument(
        "--geometry",
        type=Geometry,
        choices=list(Geometry),
        help="Benchmark geometry name",
    )
    base.add_argument("--config", type=Path, help="Geometry config file")
    base.add_argument(
        "--facet-config",
        dest="facet_config",
        type=Path,
        help="Facet tags config file",
    )
    base.add_argument(
        "--bcs", dest="bc_config", type=Path, help="Boundary condition config file"
    )
    base.add_argument("--re", type=float, default=100.0, help="Target Reynolds number")
    base.add_argument("--steps", type=int, default=3, help="Number of ramp steps")
    base.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Newton damping factor",
    )
    base.add_argument(
        "--max-it", type=int, default=1000, help="Maximum Newton iterations"
    )
    base.add_argument(
        "--tol", type=float, default=1e-6, help="Newton convergence tolerance"
    )
    base.add_argument(
        "--plot-scale", type=float, default=0.01, help="Velocity plot scale factor"
    )
    base.add_argument(
        "--output-path",
        type=Path,
        help="Directory for caching meshes and baseflow solutions",
    )
    base.set_defaults(func=_run_baseflow)

    asm = subparsers.add_parser(
        "assemble", help="Assemble linearized Navier-Stokes eigensystem"
    )
    asm.add_argument("--mesh", type=Path, help="Path to the mesh file")
    asm.add_argument(
        "--geometry",
        type=Geometry,
        choices=list(Geometry),
        help="Benchmark geometry name",
    )
    asm.add_argument("--config", type=Path, help="Geometry config file")
    asm.add_argument(
        "--facet-config",
        dest="facet_config",
        type=Path,
        help="Facet tags config file",
    )
    asm.add_argument(
        "--bcs", dest="bc_config", type=Path, help="Boundary condition config file"
    )
    asm.add_argument(
        "--bcs-perturbation",
        dest="bc_perturbation",
        type=Path,
        help="Boundary condition config for perturbation",
    )
    asm.add_argument("--re", type=float, default=100.0, help="Target Reynolds number")
    asm.add_argument("--steps", type=int, default=3, help="Number of ramp steps")
    asm.add_argument(
        "--damping",
        type=float,
        default=1.0,
        help="Newton damping factor",
    )
    asm.add_argument(
        "--max-it", type=int, default=1000, help="Maximum Newton iterations"
    )
    asm.add_argument(
        "--tol", type=float, default=1e-6, help="Newton convergence tolerance"
    )
    asm.add_argument(
        "--plot-scale", type=float, default=0.01, help="Velocity plot scale factor"
    )
    asm.add_argument("--use-sponge", action="store_true", help="Apply sponge damping")
    asm.add_argument(
        "--matrix-dir",
        type=Path,
        help="Directory to export assembled matrices",
    )
    asm.add_argument(
        "--export-baseflow", type=Path, help="Directory to export baseflow"
    )
    asm.add_argument(
        "--output-path",
        type=Path,
        help="Directory for caching meshes and baseflow solutions",
    )
    asm.set_defaults(func=_run_assemble)

    eig = subparsers.add_parser(
        "eigen", help="Solve eigenvalue problem from operator matrices"
    )
    eig.add_argument(
        "--operator",
        type=Path,
        required=True,
        help="MatrixMarket file for the system operator",
    )
    eig.add_argument("--mass", type=Path, help="MatrixMarket file for mass matrix")
    eig.add_argument("--num-eig", type=int, default=25, help="Number of eigenpairs")
    eig.add_argument(
        "--problem-type",
        type=lambda s: iEpsProblemType[s.upper()],
        choices=list(iEpsProblemType),
        default=iEpsProblemType.GNHEP,
        help="Eigenproblem type",
    )
    eig.add_argument(
        "--which",
        type=lambda s: iEpsWhich[s.upper()],
        choices=list(iEpsWhich),
        help="Which eigenpairs to compute",
    )
    eig.add_argument(
        "--st-type",
        type=lambda s: iSTType[s.upper()],
        choices=list(iSTType),
        help="Spectral transform type",
    )
    eig.add_argument("--st-target", type=float, help="Spectral transform target value")
    eig.add_argument(
        "--st-pc-type",
        type=lambda s: PreconditionerType[s.upper()],
        choices=list(PreconditionerType),
        help="Preconditioner for spectral transform",
    )
    eig.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    eig.add_argument(
        "--max-it", type=int, default=500, help="Maximum solver iterations"
    )
    eig.add_argument("--export", type=Path, help="Directory to export eigenvectors")
    eig.add_argument("--no-check", action="store_true", help="Disable Hermitian checks")
    eig.set_defaults(func=_run_eigen)

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception as exc:
        log_global(logger, logging.ERROR, "Error during CLI execution: %s", exc)
        raise SystemExit(1)
