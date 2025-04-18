# Linear Stability Analysis Framework (LSA-FW)

> **Status**: Research prototype in active development  
> **Goal**: Reproduce and extend global linear stability and adjoint-based sensitivity analysis for canonical flows (e.g., cylinder wake).

## Project Objective

The **LSA-FW** project provides a flexible and modular Python framework for performing global linear stability analysis of incompressible 2D and 3D flows.
It focuses on matrix/eigenvalue-based methods, with an emphasis on adjoint-based sensitivity and passive flow control.

The end-goal of the framework is to be generalize to complex geometries and flow setups.

## Theoretical Background

LSA-FW is inspired by several foundational works in hydrodynamic stability.
See `doc/ref/` for a collection of referenced papers and notes.

## Current Features

LSA-FW is modular by design.
Current features include:

- Core Modules
  - Sparse eigenvalue computations for stability and adjoint problems
  - Base flow construction and diagnostics
  - Automatic differentiation support (planned)

- Infrastructure
  - Modular mesh generation & import/export (via [Meshing Module](models/meshing.md))
  - CLI-based experiment runners
  - Support for parameter files (e.g. `.json`)

## Running the Project

### Run in Docker (Recommended)

The repo is fully configured for **VS Code DevContainers**.

- Open the folder in VS Code
- If prompted, choose **“Reopen in Container”**
- All dependencies (FEniCSx, PyVista, GMSH) are pre-installed

### Run Modules

In order to verify the docker environment installation, it is recommended to run the test projects.

From VS Code:

- `F1` → “Run Task” → Select:
  - `Run module` or
  - `Run module with MPI (4 cores)`
- Enter module name (default is `TestProjects`)


### Example: Generate a 2D Cylinder Mesh

```bash
python -m Meshing -p benchmark \
  --geometry cylinder_flow \
  --params params/cylinder2d.json \
  --export cyl.xdmf \
  --format xdmf \
```

See [Meshing module](./doc/models/meshing.md) for full usage.

## Documentation

Documentation is organized into the following folders:

- [arch](./doc/arc/_index.md): System setup, Docker, architecture diagrams.
- [models](./doc/models/_index.md): Module-level documentation (e.g., meshing).
- [ref](./doc/ref/): Theory notes and cited papers.
