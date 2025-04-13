# Linear Stability Analysis Framework (Work in Progress)

This project aims to provide a general framework for performing linear stability and adjoint-based analysis of 2D and 3D flows (e.g., cylinder wake), utilizing matrix and eigenvalue methods.

Currently, only a limited set of example problems is available (Poisson and Navier-Stokes).
These serve as smoke tests to verify the environment setup.

Execution within VS Code is supported via:

- F1 → “Run Task” → Select "Run module" or "Run module with MPI (4 cores)"
- A prompt will request the module name (default is `TestProjects`)

Additional features and broader generalization are planned as the framework develops.

## Running in Docker

The project is configured for execution inside a VS Code DevContainer.
Upon opening the project in VS Code, a prompt should appear offering to reopen it within the container.
This environment includes FEniCSx and all required tools pre-installed.

If the container is not used, ensure a valid `dolfinx` environment is present and update the task definitions accordingly.

Further documentation regarding how to set up the environment is a WIP.