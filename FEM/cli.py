"""LSA-FW FEM CLI.

This command-line interface assembles finite element matrices (A, M)
for linear stability analysis of incompressible Navier-Stokes flows.

Subcommands:
- assemble: Load a mesh, define FE spaces and BCs, compute base flow, and
            assemble the FEM system for eigenvalue analysis.

Example usage:
    # Assemble the FEM system with default BCs and base flow
    python -m FEM assemble --mesh out/mesh/mesh.xdmf --re 100

    # Assemble using Taylor-Hood spaces and plot sparsity
    python -m FEM -p assemble --mesh out/mesh/mesh.xdmf \
                           --space taylor_hood \
                           --re 500 \
                           --output_path out/matrices/
"""

import argparse
import logging
from pathlib import Path
from rich.console import Console

import dolfinx.fem as dfem

from config import load_bc_config

from Meshing.plot import setup_logging  # hack: move to utils
from Meshing import Mesher, Shape

from .spaces import FunctionSpaceType, define_spaces, FunctionSpaces
from .bcs import define_bcs, BoundaryConditions
from .operators import LinearizedNavierStokesAssembler
from .plot import spy
from .utils import iPETScMatrix

console: Console = Console()
logger: logging.Logger = logging.getLogger(__name__)


def _import_mesh(file_path: Path) -> Mesher:
    match file_path.suffix:
        case ".msh":
            shape = Shape.CUSTOM_MSH
        case ".xdmf":
            shape = Shape.CUSTOM_XDMF
        case _:
            raise ValueError(f"Meshes of type '{file_path.suffix}' not supported.")

    mesher = Mesher.from_file(file_path, shape)
    logger.info(
        "Mesh imported with %d cells",
        mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local,
    )
    return mesher


def _import_bcs(
    file_path: Path | None, mesher: Mesher, spaces: FunctionSpaces
) -> BoundaryConditions | None:
    if file_path is None:
        return None
    config = load_bc_config(file_path)
    return define_bcs(mesher.mesh, spaces, mesher.facet_tags, config)


def _import_base_flow(
    spaces: FunctionSpaces, base_flow_path: Path | None
) -> dfem.Function:
    u_base = dfem.Function(spaces.velocity)
    if base_flow_path is None:
        logger.info("No base flow file provided; using zero velocity.")
        u_base.x.array[:] = 0.0
    else:
        logger.info(f"Loading base flow from {base_flow_path}")
        with dfem.Function.load(spaces.velocity, base_flow_path) as f:
            u_base.x.array[:] = f.x.array[:]
    return u_base


def _export_matrices(A: iPETScMatrix, M: iPETScMatrix, output_path: Path) -> None:
    logger.info(f"Output will be saved to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    A.export(output_path / "A.petsc")
    M.export(output_path / "M.petsc")


def assemble_fem(args: argparse.Namespace) -> None:
    """Perform the assembly of the FEM model."""
    mesher = _import_mesh(args.mesh)
    spaces = define_spaces(mesher.mesh, args.space_type)
    bcs = _import_bcs(args.bcs, mesher, spaces)
    base_flow = _import_base_flow(spaces, args.base_flow)
    assembler = LinearizedNavierStokesAssembler(base_flow, spaces, args.re, bcs)

    A, M = assembler.assemble_eigensystem()

    _export_matrices(A, M, args.output_path / "matrices")

    if args.plot:
        logger.info(f"Plots will be saved to: {args.output_path}/plots")
        spy(A.to_aij(), args.output_path / "plots" / "A.svg")
        spy(M.to_aij(), args.output_path / "plots" / "M.svg")


def main():
    """LSA-FW Meshing CLI entry point."""
    parser = argparse.ArgumentParser(description="LSA-FW Meshing tool: assemble")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Flag to plot the results after assembly",
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
        help="Finite element space type (default: Taylor-Hood)",
    )
    assemble.add_argument(
        "--bcs",
        type=Path,
        default=Path("config_files/bcs.toml"),
        help="Path to the boundary condition config file (default: config_files/bcs.toml)",
    )
    assemble.add_argument(
        "--base_flow",
        type=Path,
        default=None,
        help="Optional path to base flow file (PETSc format)",
    )
    assemble.add_argument(
        "--re",
        type=float,
        default=100.0,
        help="Reynolds number (default: 100)",
    )
    assemble.add_argument(
        "--output_path",
        type=Path,
        default=Path("out/matrices"),
        help="Path to the output directory (default: out/matrices)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "assemble":
        assemble_fem(args)
    else:
        console.print("[red]Unknown command. Use --help for usage info.[/red]")
        parser.print_help()
