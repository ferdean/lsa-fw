"""LSA-FW Meshing CLI.

This command-line interface allows generating, importing, exporting,
and benchmarking meshes using pre-defined CFD geometries.

Subcommands:
- generate: Create a mesh from a predefined shape (e.g., unit square, box)
- import: Load a mesh from a file (XDMF or MSH) and optionally export it
- benchmark: Generate a standard CFD benchmark geometry (e.g., cylinder flow)

Example usage:
    # Generate a 2D unit square mesh and export to XDMF (with plot)
    python -m Meshing -p generate --shape unit_square --cell-type triangle \
                                  --facet_config config_files/mesh_tags.toml \
                                  --resolution 32 32 --export mesh.xdmf --format xdmf

    # Import a mesh from a XDMF .xdmf file and convert to VTK
    python -m Meshing import --from custom_xdmf --path mesh.xdmf \
                              --export mesh.vtk --format vtk

    # Generate a benchmark CFD geometry from parameters in TOML (with plot)
    python -m Meshing benchmark --geometry cylinder_flow --params config_files/cylinder_flow.toml \
                                 --export cyl.xdmf --format xdmf
"""

import argparse
import logging
from pathlib import Path

from mpi4py import MPI
from rich.console import Console

from config import load_cylinder_flow_config, load_step_flow_config, load_facet_config
from lib.loggingutils import setup_logging

from .core import Mesher
from .utils import Shape, Format, iCellType, Geometry
from .plot import plot_mesh
from .geometries import get_geometry

console: Console = Console()
logger: logging.Logger = logging.getLogger(__name__)


def _generate(args: argparse.Namespace) -> None:
    resolution = tuple(args.resolution)
    domain = None
    if args.domain:
        if len(args.domain) != 2 * len(resolution):
            raise ValueError(f"--domain must have {2 * len(resolution)} values.")
        domain = (
            tuple(args.domain[: len(resolution)]),
            tuple(args.domain[len(resolution) :]),
        )

    logger.info("Generating mesh: %s, resolution=%s", args.shape, resolution)
    mesher = Mesher(
        shape=args.shape,
        n=resolution,
        cell_type=args.cell_type,
        domain=domain,
    )
    mesher.generate()

    logger.info(
        "Mesh generated with %d cells",
        mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local,
    )
    if args.facet_config:
        marker_fn = load_facet_config(args.facet_config)
        mesher.mark_boundary_facets(marker_fn)

    if args.export:
        logger.info("Exporting mesh to: %s as %s", args.export, args.format)
        mesher.export(args.export, args.format)
        logger.info("Export complete.")

    if args.plot:
        logger.info("Plotting mesh...")
        plot_mesh(mesher.mesh)


def _import(args: argparse.Namespace) -> None:
    logger.info("Importing mesh: %s (%s)", args.path, args.import_type)
    mesher = Mesher.from_file(path=args.path, shape=args.import_type, gdim=args.gdim)

    logger.info(
        "Mesh imported with %d cells",
        mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local,
    )

    if args.export:
        logger.info("Exporting mesh to: %s as %s", args.export, args.format)
        mesher.export(args.export, args.format)
        logger.info("Export complete.")

    if args.plot:
        logger.info("Plotting mesh...")
        plot_mesh(mesher.mesh)


def _benchmark(args: argparse.Namespace) -> None:
    logger.info("Generating benchmark geometry: %s", args.geometry)

    match args.geometry:
        case Geometry.CYLINDER_FLOW:
            config_obj = load_cylinder_flow_config(args.params)
        case Geometry.STEP_FLOW:
            config_obj = load_step_flow_config(args.params)
        case _:
            raise NotImplementedError(f"Unsupported geometry: {args.geometry}")

    mesh = get_geometry(args.geometry, config=config_obj)

    logger.info(
        "Mesh generated with %d cells",
        mesh.topology.index_map(mesh.topology.dim).size_local,
    )

    if args.export:
        from dolfinx.io import XDMFFile, VTKFile

        logger.info("Exporting mesh to: %s", args.export)

        if args.format == Format.XDMF:
            with XDMFFile(MPI.COMM_WORLD, str(args.export), "w") as f:
                f.write_mesh(mesh)
        elif args.format == Format.VTK:
            with VTKFile(MPI.COMM_WORLD, str(args.export), "w") as f:
                f.write_mesh(mesh)
        else:
            raise NotImplementedError("Only XDMF and VTK export supported.")
        logger.info("Export complete.")

    if args.plot:
        logger.info("Plotting mesh...")
        plot_mesh(mesh)


def main():
    """LSA-FW Meshing CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LSA-FW Meshing tool: generate, import, benchmark",
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
    generate.add_argument("--shape", type=Shape, choices=list(Shape), required=True)
    generate.add_argument("--cell-type", type=iCellType.from_string, required=True)
    generate.add_argument("--resolution", nargs="+", type=int, required=True)
    generate.add_argument(
        "--domain", type=float, nargs="+", help="Bounding box: x0 y0 [z0] x1 y1 [z1]"
    )
    generate.add_argument(
        "--facet_config", type=Path, help="Path to config file to define facet tags."
    )
    generate.add_argument("--export", type=Path)
    generate.add_argument("--format", type=Format, choices=list(Format))
    generate.set_defaults(func=_generate)

    # Subcommand: import
    importer = subparsers.add_parser("import", help="Import a mesh from a file")
    importer.add_argument(
        "--from",
        dest="import_type",
        type=Shape,
        choices=[Shape.CUSTOM_XDMF, Shape.CUSTOM_MSH],
        required=True,
    )
    importer.add_argument("--path", type=Path, required=True)
    importer.add_argument("--gdim", type=int, default=3)
    importer.add_argument("--export", type=Path)
    importer.add_argument("--format", type=Format, choices=list(Format))
    importer.set_defaults(func=_import)

    # Subcommand: benchmark
    benchmark = subparsers.add_parser(
        "benchmark", help="Generate a benchmark CFD geometry"
    )
    benchmark.add_argument(
        "--geometry",
        type=Geometry,
        choices=list(Geometry),
        default=Geometry.CYLINDER_FLOW,
        help="Name of the benchmark geometry to generate",
    )
    benchmark.add_argument(
        "--params",
        type=Path,
        default=Path("config_files/cylinder_flow.toml"),
        help="Path to a TOML file with geometry arguments",
    )
    benchmark.add_argument("--export", type=Path, default=Path("out/mesh/mesh.xdmf"))
    benchmark.add_argument(
        "--format", type=Format, choices=list(Format), default=Format.XDMF
    )
    benchmark.set_defaults(func=_benchmark)

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception as e:
        logger.exception("Error: An exception occurred during CLI execution (%s)", e)
        raise SystemExit(1)
