# LSA-FW mesh

> Note: Both the code and the docu for this module are in draft status.

This module provides a unified interface for mesh creation, import, export, and visualization in the LSA-FW framework.
It supports both built-in domains and external mesh files compatible with FEniCSx.

## Overview

The codebase is located in `Meshing/` and consists of:
- `core.py`: defines the `Mesher` class for procedural and file-based mesh generation.
- `cli.py`: command-line interface for user interaction with the meshing system.
- `plot.py`: visualization utilities using PyVista.
- `types.py`: internal tzpes for mesh configuration parameters.

## Features

* Supports unit domains (interval, square, cube) and configurable bounding boxes.
* Supports importing custom `.xdmf` or `.msh` files.
* Provides export in `xdmf`, `vtk`, or `gmsh` formats.
* Enables optional mesh visualization via PyVista.
* CLI supports structured argument parsing with logging and verbosity control.

## CLI (external) Usage

Invoke the CLI with `python -m Meshing <command> [options]`.

### Subcommand: `generate`

Creates a structured mesh from a built-in domain.

```bash
python -m Meshing generate \
    --shape unit_square \
    --cell-type triangle \
    --resolution 32 32 \
    --export mesh.xdmf \
    --format xdmf \
    --plot
```

### Subcommand: `import`

Imports an external mesh file and optionally converts its format or visualizes it.

```bash
python -m Meshing import \
    --from custom_msh \
    --path mesh.msh \
    --export mesh.vtk \
    --format vtk \
    --plot
```

## Visualization

The `plotter.py` module uses `dolfinx.plot.vtk_mesh()` to extract VTK-compatible geometry.
Meshes are displayed via PyVista.

The `--plot` option in the CLI invokes the visualizer after mesh creation or import.

> Note: Quadrilateral and hexahedral elements are internally triangulated for compatibility.


