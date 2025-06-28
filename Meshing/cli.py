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
                                  --facet-config path/to/mesh_tags/config/file \
                                  --resolution 32 32 --export path/for/export --format xdmf

    # Import a mesh from a XDMF .xdmf file and convert to VTK
    python -m Meshing import --from custom_xdmf --path path/to/mesh \
                              --export path/for/export --format vtk

    # Generate a benchmark CFD geometry from parameters in TOML (with plot)
    python -m Meshing -p benchmark --geometry cylinder_flow --config path/to/config/path \
                                 --facet-config path/to/mesh_tags/config/file \
                                 --export cyl.xdmf --format xdmf

Note that the above commands can be parallelized using 'mpirun -n <number_of_processors> <command>',
as all meshing processes within this module are MPI-aware.
"""

import argparse
import logging
from pathlib import Path
from rich.console import Console

from mpi4py import MPI

from lib.loggingutils import setup_logging, log_global
from config import (
    load_cylinder_flow_config,
    load_step_flow_config,
    load_facet_config,
)

from .core import Mesher
from .utils import Shape, Format, iCellType, Geometry
from .plot import plot_mesh


console: Console = Console()
logger: logging.Logger = logging.getLogger(__name__)


def _generate(args: argparse.Namespace) -> None:
    """Generate a structured mesh (interval, square, cube, box)."""
    resolution = tuple(args.resolution)
    domain = None
    if args.domain:
        if len(args.domain) != 2 * len(resolution):
            raise ValueError(f"--domain must have {2 * len(resolution)} values.")
        domain = (
            tuple(args.domain[: len(resolution)]),
            tuple(args.domain[len(resolution) :]),
        )

    log_global(logger, logging.INFO, "Generating mesh: %s, resolution=%s", args.shape, resolution)
    mesher = Mesher(
        shape=args.shape,
        n=resolution,
        cell_type=args.cell_type,
        domain=domain,
    )
    mesher.generate()

    log_global(
        logger,
        logging.INFO,
        "Mesh generated with %d cells",
        mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local,
    )

    if args.facet_config:
        marker_fn = load_facet_config(args.facet_config)
        mesher.mark_boundary_facets(marker_fn)

    if args.export:
        fmt = args.format or Format.XDMF
        log_global(logger, logging.INFO, "Exporting mesh to: %s as %s", args.export, fmt)
        mesher.export(args.export, fmt)
        log_global(logger, logging.INFO, "Export complete.")

    if args.plot:
        plot_mesh(mesher.mesh, tags=mesher.facet_tags, show_edges=True)


def _import(args: argparse.Namespace) -> None:
    """Import a mesh from XDMF or MSH using Mesher.from_file."""
    log_global(logger, logging.INFO, "Importing mesh: %s (%s)", args.path, args.import_type)
    mesher = Mesher.from_file(path=args.path, shape=args.import_type, gdim=args.gdim)

    log_global(
        logger,
        logging.INFO,
        "Mesh imported with %d cells",
        mesher.mesh.topology.index_map(mesher.mesh.topology.dim).size_local,
    )

    if args.export:
        fmt = args.format or Format.XDMF
        log_global(logger, logging.INFO, "Exporting mesh to: %s as %s", args.export, fmt)
        mesher.export(args.export, fmt)
        log_global(logger, logging.INFO, "Export complete.")

    if args.plot:
        plot_mesh(mesher.mesh)


def _benchmark(args: argparse.Namespace) -> None:
    """Generate a predefined CFD benchmark geometry via Mesher.from_geometry."""
    log_global(logger, logging.INFO, "Generating benchmark geometry: %s", args.geometry)

    match args.geometry:
        case Geometry.CYLINDER_FLOW:
            cfg = load_cylinder_flow_config(args.config)
        case Geometry.STEP_FLOW:
            cfg = load_step_flow_config(args.config)
        case _:
            raise NotImplementedError(f"Unsupported geometry: {args.geometry}")

    mesher = Mesher.from_geometry(args.geometry, cfg, comm=MPI.COMM_WORLD)
    mesh = mesher.mesh
    log_global(
        logger,
        logging.INFO,
        "Mesh generated with %d cells",
        mesh.topology.index_map(mesh.topology.dim).size_local,
    )

    if args.facet_config:
        marker_fn = load_facet_config(args.facet_config)
        mesher.mark_boundary_facets(marker_fn)

    if args.export:
        fmt = args.format or Format.XDMF
        log_global(logger, logging.INFO, "Exporting mesh to: %s as %s", args.export, fmt)
        mesher.export(args.export, fmt)
        log_global(logger, logging.INFO, "Export complete.")

    if args.plot:
        plot_mesh(mesher.mesh, tags=mesher.facet_tags, show_edges=True)


def main():
    """Meshing CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LSA-FW Meshing tool: generate, import, benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the mesh")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = subparsers.add_parser("generate", help="Generate a structured mesh")
    gen.add_argument("--shape", type=Shape, choices=list(Shape), required=True)
    gen.add_argument("--cell-type", type=iCellType.from_string, required=True)
    gen.add_argument("--resolution", nargs="+", type=int, required=True)
    gen.add_argument(
        "--domain", type=float, nargs="+", help="Bounding box: x0 y0 [z0] x1 y1 [z1]"
    )
    gen.add_argument(
        "--facet-config",
        dest="facet_config",
        type=Path,
        help="TOML file defining facet markers",
    )
    gen.add_argument("--export", type=Path, help="Path to export mesh file")
    gen.add_argument(
        "--format", type=Format, choices=list(Format), help="Export format"
    )
    gen.set_defaults(func=_generate)

    # import
    imp = subparsers.add_parser("import", help="Import mesh from file")
    imp.add_argument(
        "--from",
        dest="import_type",
        type=Shape,
        choices=[Shape.CUSTOM_XDMF, Shape.CUSTOM_MSH],
        required=True,
        help="Type of the input mesh",
    )
    imp.add_argument("--path", type=Path, required=True)
    imp.add_argument("--gdim", type=int, default=3)
    imp.add_argument("--export", type=Path, help="Path to export converted mesh")
    imp.add_argument(
        "--format", type=Format, choices=list(Format), help="Export format"
    )
    imp.set_defaults(func=_import)

    # benchmark
    bench = subparsers.add_parser("benchmark", help="Generate CFD benchmark geometry")
    bench.add_argument(
        "--geometry",
        type=Geometry,
        choices=list(Geometry),
        required=True,
        help="Predefined geometry name",
    )
    bench.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML file with geometry parameters",
    )
    bench.add_argument(
        "--facet-config",
        dest="facet_config",
        type=Path,
        help="TOML file defining facet markers",
    )
    bench.add_argument("--export", type=Path, help="Path to export mesh file")
    bench.add_argument(
        "--format", type=Format, choices=list(Format), help="Export format"
    )
    bench.set_defaults(func=_benchmark)

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        args.func(args)
    except Exception as e:
        log_global(logger, logging.ERROR, "Error during CLI execution: %s", e)
        raise SystemExit(1)
