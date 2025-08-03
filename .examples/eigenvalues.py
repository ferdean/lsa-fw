"""Quick script to test eigenvalues."""

import logging
from typing import Final
from pathlib import Path

import dolfinx.fem as dfem
import numpy as np
from petsc4py import PETSc

from FEM.plot import plot_mixed_function
from FEM.spaces import define_spaces
from FEM.utils import iPETScMatrix, iPETScVector
from lib.loggingutils import setup_logging
from Meshing.core import Mesher
from Meshing.utils import Shape
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import iEpsProblemType, iSTType, PreconditionerType, iEpsWhich

setup_logging(verbose=True)

_CASE_DIR: Final[Path] = Path("cases") / "cylinder"

mesher = Mesher.from_file(
    _CASE_DIR / "mesh" / "mesh.xdmf", shape=Shape.CUSTOM_XDMF, gdim=2
)
spaces = define_spaces(mesher.mesh)

try:
    for idx in range(50):
        eigenvector = iPETScVector.from_file(
            _CASE_DIR / "eig" / "re_8" / f"ev_{idx+1}.dat"
        )
        eigenfunction = dfem.Function(spaces.mixed)

        ev = np.real(eigenvector.as_array())

        eigenfunction.x.array[:] = ev
        eigenfunction.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
        plot_mixed_function(eigenfunction, title="Eigenmode (real)")

        eigenfunction.x.array[:] = np.imag(eigenvector.as_array())
        eigenfunction.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
        plot_mixed_function(eigenfunction, title="Eigenmode (imag)")

except Exception:
    A = iPETScMatrix.from_path(_CASE_DIR / "matrices" / "re_8" / "A.mtx")
    M = iPETScMatrix.from_path(_CASE_DIR / "matrices" / "re_8" / "M.mtx")
    A.assemble()
    M.assemble()

    # Define eigensolver
    es_cfg = EigensolverConfig(
        num_eig=50,
        problem_type=iEpsProblemType.GNHEP,
        atol=1e-5,
        max_it=500,
    )
    es = EigenSolver(es_cfg, A, M, check_hermitian=False)
    es.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)
    es.solver.set_st_pc_type(PreconditionerType.LU)
    es.solver.set_st_type(iSTType.SINVERT)
    es.solver.set_target(-1.0)  # To do: add ref. for the value

    eigenpairs = es.solve()

    for idx, (eigval, eigvec) in enumerate(eigenpairs, start=1):
        logging.info("Eigenvalue %d = %s", idx, eigval)
        out_file = _CASE_DIR / "eig" / "re_8" / f"ev_{idx}.dat"
        eigvec.real.export(out_file)
