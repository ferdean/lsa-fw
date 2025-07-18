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
from Meshing import Mesher, Geometry
from Meshing.geometries import CylinderFlowGeometryConfig
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import iEpsProblemType, iSTType, PreconditionerType, iEpsWhich


setup_logging(verbose=True)

_INPUT_PATH: Final[Path] = Path("out") / "matrices" / "reynolds_60"
_solve = False


if _solve:
    A = iPETScMatrix.from_path(_INPUT_PATH / "A.mtx")
    M = iPETScMatrix.from_path(_INPUT_PATH / "M.mtx")
    A.assemble()
    M.assemble()

    # Define eigensolver
    es_cfg = EigensolverConfig(
        num_eig=5,
        problem_type=iEpsProblemType.GNHEP,
        atol=1e-5,
        max_it=100,
    )
    es = EigenSolver(es_cfg, A, M, check_hermitian=False)
    es.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)
    es.solver.set_st_pc_type(PreconditionerType.LU)
    es.solver.set_st_type(iSTType.SINVERT)
    es.solver.set_target(0.5 + 0.5j)  # To do: add ref. for the value

    # Solve
    eigenpairs = es.solve()

    for idx, (eigval, eigvec) in enumerate(eigenpairs, start=1):
        logging.info("Eigenval %d = %s", idx, eigval)
        out_file = _INPUT_PATH / f"ev_2_{idx:02d}.dat"
        eigvec.real.export(out_file)

else:
    mesher = Mesher.from_geometry(
        Geometry.CYLINDER_FLOW,
        CylinderFlowGeometryConfig(
            dim=2,
            cylinder_radius=0.5,
            cylinder_center=(0.0, 0.0),
            x_range=(-20.0, 60.0),
            y_range=(-20.0, 20.0),
            resolution=1,
            resolution_around_cylinder=0.15,
            influence_radius=40,
            influence_length=40,
        ),
        cache=None,
    )
    spaces = define_spaces(mesher.mesh)

    for idx in range(4):
        eigenvector = iPETScVector.from_file(_INPUT_PATH / f"ev_{idx+1:02d}.dat")
        eigenfunction = dfem.Function(spaces.mixed)

        eigenfunction.x.array[:] = np.real(eigenvector.as_array())
        eigenfunction.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
        plot_mixed_function(eigenfunction, title="Eigenmode (real)")

        eigenfunction.x.array[:] = np.imag(eigenvector.as_array())
        eigenfunction.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT)
        plot_mixed_function(eigenfunction, title="Eigenmode (imag)")
