"""LSA-FW mesh CLI

This command-line interface allows generating, importing, and exporting meshes compatible with FEniCSx.

Supported subcommands:
- generate: Create a mesh from predefined shapes (unit square, cube, etc.)
- import: Load a mesh from a custom XDMF or MSH file and optionally export it
- export: Export a mesh into another format (handled inside generate/import)

Example usage:
    # Generate a 2D unit square mesh and export to XDMF
    lsa-mesher generate --shape unit_square --cell-type triangle \
                        --resolution 32 32 --export mesh.xdmf --format xdmf

    # Import a mesh from a GMSH .msh file and convert to VTK
    lsa-mesher import --from custom_msh --path mesh.msh \
                      --export mesh.vtk --format vtk
"""

import argparse
import pathlib
import logging

from .core import Shape, Format, Mesher
from .utils import CELL_TYPE_MAP

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="LSA-FW mesh tool: generate, import, and export meshes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
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
        type=str,
        choices=list(CELL_TYPE_MAP),
        required=True,
        help="Cell type",
    )
    generate.add_argument(
        "--resolution",
        nargs="+",
        type=int,
        required=True,
        help="Elements per dimension",
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
        "--export", type=pathlib.Path, help="File path to export the mesh"
    )
    importer.add_argument(
        "--format", type=Format, choices=list(Format), help="Export format"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logger.DEBUG if args.verbose else logger.INFO)

    match args.command:
        case "generate":
            resolution = tuple(args.resolution)
            cell_type = CELL_TYPE_MAP[args.cell_type]

            mesher = Mesher(shape=args.shape, n=resolution, cell_type=cell_type)
            mesher.generate()
            mesh = mesher.mesh
            logger.info(
                f"Generated mesh '{args.shape}' with {mesh.topology.index_map(mesh.topology.dim).size_local} cells"
            )

            if args.export and args.format:
                mesher.export(args.export, args.format)
                logger.info(f"Exported mesh to '{args.export}' as '{args.format}'")

        case "import":
            mesher = Mesher(shape=args.import_type, custom_file=args.path)
            mesher.generate()
            mesh = mesher.mesh
            logger.info(f"Imported mesh from '{args.path}' as '{args.import_type}'")

            if args.export and args.format:
                mesher.export(args.export, args.format)
                logger.info(f"Exported mesh to '{args.export}' as '{args.format}'")

        case _:
            parser.print_help()
