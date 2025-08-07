#!/usr/bin/env python3
"""Diagnose environment PETSc/SLEPc/DOLFINx builds"""

import os

import matplotlib.pyplot as plt
import petsc4py
from petsc4py import PETSc, lib as _petsc4py_lib
import slepc4py
from slepc4py import SLEPc
import mpi4py
import dolfinx

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

try:
    test_path = "/tmp/test_latex.pdf"
    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$\int_\Omega \nabla u \cdot \nabla v \,dx$")
    fig.savefig(test_path)
    print(" [>] LaTeX support in matplotlib is working.")
except Exception:
    print(" [!] LaTeX test plot failed. No LaTeX support in matplotlib is available.")
finally:
    os.remove(test_path)
