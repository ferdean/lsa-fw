"""Main entry point for LSA-FW FEM."""

from petsc4py import PETSc

from .cli import main

if __name__ == "__main__":
    PETSc.Options().setValue("viewer_binary_skip_info", "")
    main()
