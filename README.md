# Linear Stability Analysis Framework (LSA-FW)

> **Status:** Research prototype in active development.
> **Documentation is evolving; module details and developer guides are being updated regularly.**

## Project Objective

**LSA-FW** is a flexible, modular Python framework for global linear stability analysis of incompressible 2D and 3D flows.
It integrates matrix/eigenvalue-based methods, with a strong focus on adjoint-based sensitivity analysis and modern reproducibility practices.

**Core Aims:**

* Perform global stability analyses (compute eigenmodes/eigenvalues of flow instabilities).
* Enable adjoint-based sensitivity analyses for flow control and optimization without redundant recomputation of controlled flows.

## Theoretical Background

LSA-FW builds on foundational and modern work in hydrodynamic stability, adjoint methods, and control-oriented sensitivity analysis.
See [doc/ref](doc/ref/) for a collection of reference papers, technical notes, and background material.

## Features and Roadmap

**Mesh Management**

* Built-in generators for canonical domains (unit interval, unit square, unit cube, box)
* Automated mesh creation for CFD benchmarks (e.g., cylinder flow, backward-facing step)
* Import support for externally-generated meshes (XDMF, VTK, GMSH)
* Mesh adaptation routines based on computed baseflow velocity
* Configuration-driven boundary tagging (TOML) for reproducible BC assignment
* *(Planned)* Mesh adaptation via custom fields (vorticity, wavemaker, etc.)

**Finite Element Assembly**

* Configuration-based BC creation: Dirichlet, Neumann, Robin, and periodic (via TOML)
* Function spaces: Taylor-Hood, MINI, SIMPLE; *(Planned)* Discontinuous Galerkin (DG)
* Modular FE assemblers for Stokes, stationary NS, and linearized NS systems
* Fully type-hinted, tested routines compatible with C++ and native Python backends

**Solvers**

* Linear solvers (LU, GMRES, CG, and others) via PETSc
* Nonlinear solvers (Newton; *Planned*: Picard)
* Eigensolver API manager using SLEPc, fully MPI-parallelized
* Direct support for both complex and real PETSc/SLEPc builds
* Validated baseflow computation

**Adjoint-Based Sensitivity Analysis**

* First-order (linear) adjoint-based sensitivity maps for eigenvalue response to steady controls
* *(Planned)* Second-order (quadratic) adjoint-based sensitivity operators, capturing nonlinear effects of finite-amplitude control
* Structural sensitivity computation via direct/adjoint mode overlap
* *(Planned)* General parameter sensitivity (Reynolds number, geometry, etc.) and adjoint-based optimization for open-loop/closed-loop flow control
* Reynolds sensitivity
* Modular adjoint routines, integrated with PETSc/SLEPc eigensolvers

**Visualization & Analysis**

* Native routines for visualization of baseflow fields and eigenmodes
* Export to XDMF/HDF5 for external post-processing
* Logging and reporting system for traceability

**Framework Quality and Extensibility**

* XDMF/HDF5-backed caching to avoid redundant computations
* Modular, type-checked API for extension and research experimentation
* Documentation: Developer, user, and module docs organized for maintainability

## Reproducibility and Dockerized Workflow

LSA-FW is fully containerized, with all dependencies (FEniCSx, PETSc/SLEPc, GMSH, etc.) pre-configured.
VS Code DevContainer support allows for seamless setup: simply open the folder in VS Code and select 'Reopen in Container' for a ready-to-use environment.

For system and Docker architecture details, see [arch/docker.md](./doc/arch/docker.md).

### Build Verification

To verify your Docker/container installation and environment setup, run:

```bash
python /workspaces/lsa-fw/diagnose_build.py
```

**Expected output (example):**

```plaintext
 [>] petsc4py: ... | ScalarType: float64 | PETSc version: (3, 22, 0)
 [>] slepc4py: ... | SLEPc version: (3, 22, 0)
 [>] dolfinx: ... | version: 0.9.0
 [>] mpi4py: ... | version: 4.0.0
 [>] ufl: ... | version: 2024.2.0

 [>] ENV vars:
     [>] PETSc prefix: /usr/local/petsc
     [>] PETSc arch: linux-gnu-real64-32
     [>] PETSc ScalarType: float64 (real build)
```

> **Note:**
> The PETSc/SLEPc `ScalarType` and `arch` lines indicate whether the environment is a *real* or *complex* build.
> Many linear stability and all adjoint/eigenvalue computations involving non-symmetric operators require a complex build.
> **You can switch between REAL and COMPLEX environments using the VS Code task:**
>
> * Press `F1`, select `Tasks: Run Task`, then choose `Switch to real/complex environment`.
>   The current build type will be shown in the output of the verification script.

## Documentation Structure

* [arch](./doc/arch/_index.md): System setup, Docker, architecture diagrams, and devops notes
* [models](./doc/models/_index.md): Module-level documentation and examples
* [ref](./doc/ref/): Theory notes, referenced publications, and technical background

For further usage, module APIs, and developer guides, see module documentation under `doc/models/`.

## *(Planned)* Examples Library

A comprehensive library of practical examples is planned for LSA-FW.
This examples suite will demonstrate typical and advanced use cases, including:

- Mesh generation for canonical and benchmark geometries
- Baseflow and eigenvalue computations for classic CFD test cases
- Adjoint-based sensitivity analysis for various flow control scenarios
- Best practices for visualization, post-processing, and result interpretation
