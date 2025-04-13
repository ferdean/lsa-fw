# Linear Stability Analysis Framework (Work in Progress)

The goal of this project is to build a general framework for performing linear stability and adjoint-based analysis of 2D/3D flows (like the cylinder wake), based on matrix/eigenvalue methods.

Right now, only a few example problems are available (Poisson and Navier-Stokes).
You can run them via VS Code using:

- F1 → “Run Task” → select "Run module" or "Run module with MPI (4 cores)"
- You’ll be prompted to enter the module name (default is `TestProjects`)

More features and generalization will come as the framework evolves.

## Running in Docker

This project is configured to run inside a VS Code DevContainer.
When you open the project in VS Code, you should see a prompt to reopen it in the container.
This environment comes pre-installed with FEniCSx and all necessary tools.

If you're not using the container, make sure you have a working `dolfinx` environment and update the tasks accordingly.

Further documentation regarding how to set up the environment is a WIP.