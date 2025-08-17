"""LSA-FW FEM CLI.

This command-line interface assembles finite element matrices (A, M)
for linear stability analysis of incompressible Navier-Stokes flows.

Subcommands:
- assemble: Load a mesh, define FE spaces and BCs, compute base flow, and
            assemble the FEM system for eigenvalue analysis.

Example usage:
    # Assemble the FEM system
    python -m FEM assemble --mesh path/to/mesh --re 100 --bcs path/to/bcs

    # Assemble using Taylor-Hood spaces and plot sparsity
    python -m FEM -p assemble --mesh path/to/mesh \
                           --space taylor_hood \
                           --bcs path/to/bcs \
                           --re 500 \
                           --output_path path/for/output

Note that the above commands can be parallelized using 'mpirun -n <number_of_processors> <command>',
as all FEM processes within this module are MPI-aware.
"""

import argparse
import logging
from pathlib import Path

import dolfinx.fem as dfem
from rich.console import Console

from config import load_bc_config
from FEM.bcs import BoundaryConditions, define_bcs
from FEM.operators import LinearizedNavierStokesAssembler
from FEM.plot import plot_mixed_function, spy
from FEM.spaces import define_spaces, FunctionSpaceType, FunctionSpaces
from FEM.utils import iPETScMatrix
from lib.loggingutils import log_global, setup_logging
from Meshing.core import Mesher
from Meshing.plot import PlotMode
from Solver.baseflow import BaseFlowSolver, load_baseflow

console: Console = Console()
logger: logging.Logger = logging.getLogger(__name__)


def _import_bcs(
    file_path: Path, mesher: Mesher, spaces: FunctionSpaces
) -> BoundaryConditions:
    """Load and define BCs from a TOML config."""
    config = load_bc_config(file_path)
    return define_bcs(mesher, spaces, config)


def _get_baseflow(
    spaces: FunctionSpaces,
    base_flow_path: Path | None,
    re: float,
    bcs: BoundaryConditions,
    *,
    show_plot: bool = False,
) -> dfem.Function:
    """Load baseflow from disk or compute it if not provided."""
    if base_flow_path:
        log_global(logger, logging.INFO, "Loading base flow from '%s'", base_flow_path)
        u_base = load_baseflow(base_flow_path, spaces)
    else:
        log_global(
            logger, logging.INFO, "Computing base flow from scratch (Re=%g).", re
        )
        u_base = BaseFlowSolver(spaces, bcs=bcs).solve(re=re)

    if show_plot:
        plot_mixed_function(u_base, PlotMode.INTERACTIVE, scale=0.0, title="Base flow")

    return u_base


def _export_matrices(A: iPETScMatrix, M: iPETScMatrix, output_path: Path) -> None:
    """Export assembled matrices to PETSc binary format."""
    log_global(logger, logging.INFO, f"Saving matrices to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    A.export(output_path / "A.petsc")
    M.export(output_path / "M.petsc")


def assemble_fem(args: argparse.Namespace) -> None:
    """Perform the assembly of the FEM model."""
    mesh_path = args.mesh.resolve()
    output_path = args.output_path.resolve()

    # Load mesh and FE spaces
    mesher = Mesher.from_file(mesh_path)
    log_global(
        logger,
        logging.INFO,
        "Mesh imported with %d cells",
        mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local,
    )
    spaces = define_spaces(mesher.mesh, args.space_type)

    # Boundary conditions
    if args.bcs:
        bcs = _import_bcs(args.bcs.resolve(), mesher, spaces)
    else:
        raise ValueError("Boundary condition config (--bcs) must be provided.")

    # Base flow
    base_flow = _get_baseflow(spaces, args.base_flow, args.re, bcs, show_plot=args.plot)

    # Assembly
    assembler = LinearizedNavierStokesAssembler(base_flow, spaces, args.re, bcs=bcs)
    A, M = assembler.assemble_eigensystem()
    _export_matrices(A, M, output_path / "matrices")

    # Optional plots
    if args.plot:
        plot_dir = output_path / "plots"
        log_global(logger, logging.INFO, f"Saving plots to: {plot_dir}")
        spy(A, plot_dir / "A.png", spaces=spaces)
        spy(M, plot_dir / "M.png", spaces=spaces)


def main() -> None:
    """LSA-FW FEM CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LSA-FW FEM assembly tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot results after assembly"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # assemble
    assemble = subparsers.add_parser("assemble", help="Assemble the FEM model")
    assemble.add_argument(
        "--mesh", type=Path, required=True, help="Path to the mesh file"
    )
    assemble.add_argument(
        "--space",
        dest="space_type",
        type=FunctionSpaceType.from_string,
        choices=list(FunctionSpaceType),
        default=FunctionSpaceType.TAYLOR_HOOD,
        help="Finite element space type",
    )
    assemble.add_argument(
        "--bcs", type=Path, required=True, help="Boundary condition config TOML"
    )
    assemble.add_argument("--base_flow", type=Path, help="Path to base flow folder")
    assemble.add_argument("--re", type=float, default=80.0, help="Reynolds number")
    assemble.add_argument(
        "--output_path",
        type=Path,
        default=Path("out/matrices"),
        help="Output directory",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        if args.command == "assemble":
            assemble_fem(args)
        else:
            parser.error(f"Unknown command '{args.command}'")
    except Exception as e:
        log_global(logger, logging.ERROR, "CLI execution error: %s", e)
        raise SystemExit(1)
