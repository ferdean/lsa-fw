"""LSA-FW mesh CLI.

This command-line interface allows generating, importing, and exporting meshes.
Supports built-in unit shapes or importing custom mesh files.

Subcommands:
- generate: Create a mesh from a predefined shape (e.g., unit square, box)
- import: Load a mesh from a file (XDMF or MSH) and optionally export it

Example usage:
    # Generate a 2D unit square mesh and export to XDMF (with plot)
    python -m Meshing -p generate --shape unit_square --cell-type triangle \
                               --resolution 32 32 --export mesh.xdmf --format xdmf

    # Import a mesh from a GMSH .msh file and convert to VTK
    python -m Meshing import --from custom_msh --path mesh.msh \
                             --export mesh.vtk --format vtk
"""

import argparse
import pathlib
import logging

from rich.console import Console
from rich.logging import RichHandler

from .core import Mesher
from .types import Shape, Format, iCellType
from .plot import plot_mesh

console: Console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="LSA-FW mesh tool: generate, import, and export meshes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot the mesh after generation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: generate
    generate = subparsers.add_parser(
        "generate", help="Generate a mesh from a predefined shape"
    )
    generate.add_argument(
        "--shape", type=Shape, choices=list(Shape), required=True, help="Mesh shape"
    )
    generate.add_argument(
        "--cell-type",
        type=iCellType.from_string,
        required=True,
        help="Cell type (e.g., triangle, tetrahedron)",
    )
    generate.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        required=True,
        help="Elements per dimension",
    )
    generate.add_argument(
        "--domain", type=float, nargs="+", help="Bounding box: x0 y0 [z0] x1 y1 [z1]"
    )
    generate.add_argument(
        "--export", type=pathlib.Path, help="File path to export the mesh"
    )
    generate.add_argument(
        "--format", type=Format, choices=list(Format), help="Export format"
    )

    # Subcommand: import
    importer = subparsers.add_parser("import", help="Import a mesh from a file")
    importer.add_argument(
        "--from",
        dest="import_type",
        type=Shape,
        choices=[Shape.CUSTOM_XDMF, Shape.CUSTOM_MSH],
        required=True,
    )
    importer.add_argument(
        "--path", type=pathlib.Path, required=True, help="Path to the mesh file"
    )
    importer.add_argument(
        "--gdim", type=int, default=3, help="Geometric dimension of the mesh"
    )
    importer.add_argument(
        "--export", type=pathlib.Path, help="File path to export the mesh"
    )
    importer.add_argument(
        "--format", type=Format, choices=list(Format), help="Export format"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    logger = logging.getLogger(__name__)

    try:
        match args.command:
            case "generate":
                resolution = tuple(args.resolution)
                domain = None
                if args.domain:
                    if len(args.domain) != 2 * len(resolution):
                        raise ValueError(
                            f"--domain must have {2 * len(resolution)} values."
                        )
                    domain = (
                        tuple(args.domain[: len(resolution)]),
                        tuple(args.domain[len(resolution) :]),
                    )

                logger.info(
                    f"[bold cyan]Generating mesh:[/bold cyan] {args.shape}, resolution={resolution}"
                )
                mesher = Mesher(
                    shape=args.shape,
                    n=resolution,
                    cell_type=args.cell_type,
                    domain=domain,
                )
                mesher.generate()

                logger.info(
                    "[green]Mesh generated with"
                    f" {mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local} cells[/green]"
                )

                if args.export and args.format:
                    logger.info(
                        f"[cyan]Exporting mesh to:[/cyan] {args.export} as {args.format}"
                    )
                    mesher.export(args.export, args.format)
                    logger.info("[green]Export complete.[/green]")

                if args.plot:
                    logger.info("[yellow]Plotting mesh...[/yellow]")
                    plot_mesh(mesher.mesh)

            case "import":
                logger.info(
                    f"[bold cyan]Importing mesh:[/bold cyan] {args.path} ({args.import_type})"
                )
                mesher = Mesher.from_file(
                    path=args.path, shape=args.import_type, gdim=args.gdim
                )

                logger.info(
                    "[green]Mesh imported with"
                    f" {mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local} cells[/green]"
                )

                if args.export and args.format:
                    logger.info(
                        f"[cyan]Exporting mesh to:[/cyan] {args.export} as {args.format}"
                    )
                    mesher.export(args.export, args.format)
                    logger.info("[green]Export complete.[/green]")

                if args.plot:
                    logger.info("[yellow]Plotting mesh...[/yellow]")
                    plot_mesh(mesher.mesh)

            case _:
                console.print("[red]Unknown command. Use --help for usage info.[/red]")
                parser.print_help()

    except Exception as e:
        logger.exception(
            f"[red bold]Error:[/red bold] An exception occurred during CLI execution ({e})."
        )
        raise SystemExit(1)
