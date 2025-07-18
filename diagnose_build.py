#!/usr/bin/env python3
"""Diagnose environment PETSc/SLEPc/DOLFINx builds"""

import petsc4py
from petsc4py import PETSc, lib as _petsc4py_lib
import slepc4py
from slepc4py import SLEPc
import dolfinx
import mpi4py

__cfg = petsc4py.get_config()
__petsc_arch = _petsc4py_lib.getPathArchPETSc()[1]

print(
    " [>] petsc4py:",
    petsc4py.__file__,
    "| ScalarType:",
    PETSc.ScalarType.__name__,
    "| PETSc version:",
    PETSc.Sys.getVersion(),
)
print(" [>] slepc4py:", slepc4py.__file__, "| SLEPc version:", SLEPc.Sys.getVersion())
print(
    " [>] dolfinx:",
    dolfinx.__file__,
    "| version:",
    getattr(dolfinx, "__version__", "n/a"),
)
print(" [>] mpi4py:", mpi4py.__file__, "| version:", mpi4py.__version__)
try:
    import ufl

    print(" [>] ufl:", ufl.__file__, "| version:", ufl.__version__)
except ImportError:
    print(" [!] ufl: not installed")

print("\n [>] ENV vars:")
print("     [>] PETSc prefix:", __cfg.get("PETSC_DIR", "(not set)"))
print("     [>] PETSc arch:", __petsc_arch, "\n")
