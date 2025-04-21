# Linear Stability Analysis Framework (LSA-FW)

> **Status**: Research prototype in active development  
> **Goal**: Reproduce and extend global linear stability and adjoint-based sensitivity analysis for canonical flows (e.g., cylinder wake).

## Project Objective

The **LSA-FW** project provides a flexible and modular Python framework for performing global linear stability analysis of incompressible 2D and 3D flows.
It focuses on matrix/eigenvalue-based methods, with an emphasis on adjoint-based sensitivity and passive flow control.

## Theoretical Background

LSA-FW is inspired by several foundational works in hydrodynamic stability.
See [doc/ref](doc/ref/) for a collection of referenced papers and notes.

## Current Features

LSA-FW is modular by design.
Current features include:

- Core Modules
  - Sparse eigenvalue computations for stability and adjoint problems *(planned)*
  - Base flow construction and diagnostics *(planned)*
  - Automatic differentiation support *(planned)*

- Infrastructure
  - Modular mesh generation and import/export (via [Meshing Module](doc/models/meshing.md))
  - CLI-based experiment runners
  - Support for parameter files (e.g., `.json`)

## Module Overview

### Meshing

Provides all tools for mesh generation, manipulation, and export.
Supports canonical CFD geometries with built-in refinement logic.

- Implements mesh generation for test cases like 2D cylinder flow and step flow.
- Supports multiple output formats (e.g., XDMF, MSH).
- Uses GMSH via command-line and built-in distance/threshold fields for local refinement.
- [Documentation](doc/models/meshing.md)

### FEM

Finite Element Method backend that assembles the linearized Navier–Stokes operator for stability analysis.

- Modular structure for building velocity-pressure function spaces.
- Handles boundary condition enforcement for Dirichlet, Neumann, and Robin types.
- Exposes PETSc matrices for solver consumption.

### Solver *(Planned)*

Linear solver module that will support:

- Eigenvalue solvers for modal stability analysis (via SLEPc)
- Time-stepping solvers for linearized flow evolution (via PETSc TS)
- Preconditioners for large sparse matrices
- Fully decoupled from FEM to allow extensibility

### Visualizer *(Planned)*

Visualization utilities for solution fields, mesh structures, and modal outputs.

- Integration with PyVista and ParaView-compatible formats
- Real-time or post-processing visualization of eigenmodes and flow fields
- Support for animation of time evolution results

## Running the Project

### Run in Docker (Recommended)

The repo is fully configured for **VS Code DevContainers**.

- Open the folder in VS Code
- If prompted, choose **"Reopen in Container"**
- All dependencies (FEniCSx, PyVista, GMSH) are pre-installed

### Run Modules

In order to verify the Docker environment installation, it is recommended to run the test projects.

From VS Code:

- `F1` → "Run Task" → Select:
  - `Run module` or
  - `Run module with MPI (4 cores)`
- Enter module name (default is `TestProjects`)

### Example: Generate a 2D Cylinder Mesh

```bash
python -m Meshing -p benchmark \
  --geometry cylinder_flow \
  --params params/cylinder2d.json \
  --export cyl.xdmf \
  --format xdmf
```

## Documentation

Documentation is organized into the following folders:

- [arch](./doc/arch/_index.md): System setup, Docker, architecture diagrams.
- [models](./doc/models/_index.md): Module-level documentation (e.g., meshing).
- [ref](./doc/ref/): Theory notes and cited papers.
