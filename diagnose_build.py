#!/usr/bin/env python3
"""Diagnose the local PETSc/SLEPc/DOLFINx build environment."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import petsc4py
from petsc4py import PETSc, lib as _petsc4py_lib
import slepc4py
from slepc4py import SLEPc
import mpi4py
import dolfinx

PETSC_CONFIG = petsc4py.get_config()
PETSC_ARCH = _petsc4py_lib.getPathArchPETSc()[1]

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
print("     [>] PETSc prefix:", PETSC_CONFIG.get("PETSC_DIR", "(not set)"))
print("     [>] PETSc arch:", PETSC_ARCH, "\n")

test_path = Path("/tmp/test_latex.pdf")
fig: plt.Figure | None = None
try:
    plt.rcParams.update({"text.usetex": True})
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$\int_\Omega \nabla u \cdot \nabla v \,dx$")
    fig.savefig(test_path)
    print(" [>] LaTeX support in matplotlib is working.")
except Exception:
    print(" [!] LaTeX test plot failed. No LaTeX support in matplotlib is available.")
finally:
    if fig is not None:
        plt.close(fig)
    if test_path.exists():
        os.remove(test_path)
