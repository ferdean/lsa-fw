# Docker Setup for LSA-FW

This document explains how to configure and run the LSA-FW project inside a pre-configured Docker container using Visual Studio Code.


## Overview

The recommended way to run the LSA-FW project is inside a VS Code DevContainer.  
This ensures that all required dependencies (e.g., FEniCSx, GMSH, PyVista, MPI) are available and correctly configured.


## Prerequisites

Ensure the following tools are installed on your machine:

- **Docker**  
  [Download Docker](https://www.docker.com/products/docker-desktop)

- **Visual Studio Code**  
  [Download VS Code](https://code.visualstudio.com/)

- **VS Code Extensions**:
  - [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) (`ms-vscode-remote.remote-containers`)
  - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) (`ms-python.python`)


## Getting Started

### Clone the Repository

Run the following commands in your terminal:

```bash
git clone https://github.com/ferdean/lsa-fw.git
cd lsa-fw
```

### Reopen in DevContainer

- Open the project in VS Code.
- When prompted, click:  
  **“Reopen in Container”**

If no prompt appears, press `F1` → **“Reopen in Container”**

VS Code will now:

- Build a Docker image based on `.devcontainer/devcontainer.json`
- Mount your project folder into the container
- Open a terminal inside the container, with all tools preinstalled

## Environment Details

The DevContainer provides:

- `FEniCSx` (latest main branch)
- `GMSH` CLI (for mesh generation)
- `PyVista` (for visualization)
- `mpi4py`, `numpy`, `scipy`, etc.
- Preconfigured Python interpreter in `.vscode/settings.json`

You can customize this in `.devcontainer/devcontainer.json`.


## Verifying Installation

Currently, only `Meshing` module is fully functional.
Thus, the recommended way to verify the installation is by running, for example:

```bash
python -m Meshing -p generate \
    --shape box \
    --cell-type hexahedron \
    --resolution 40 20 20 \
    --domain 0 0 0 2 1 1 \
```

If PyVista opens the visualization window, everything is working correctly.

## Tips and Troubleshooting

- As the project becomes larger, consider increasing Docker memory in Docker Desktop settings.
- To install extra Python packages, edit `devcontainer/Dockerfile`.

## Manual Setup (Optional)

If you prefer to run outside Docker, you must install:

- `dolfinx` via `conda` or `pip`
- `gmsh` (with `gmsh-sdk`)
- `pyvista`, `vtk`, `mpi4py`, `scipy`

See: https://fenicsproject.org/ for detailed installation instructions.
