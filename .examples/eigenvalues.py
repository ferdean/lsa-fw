"""Cylinder flow: solve EVP on pre-assembled (A, M) matrices for several domain lengths.

This script loads the EVP matrices for multiple cylinder-flow cases (see CASE_NAMES below), solves for the rightmost
eigenpair near a given target, and writes the dominant eigenvalue (sigma) to file. Plots of the corresponding
eigenvector are also exported.

Note:
- The test data (meshes/matrices for the listed cases) is not included in the repository to avoid bloating it with
large assets. Point SAVE_DIR to your local data folder. Refer to ./.examples/params.py for an example of a parameter
analysis script.
"""

from pathlib import Path
from typing import Final

import dolfinx.fem as dfem

from FEM.plot import plot_mixed_function
from FEM.spaces import define_spaces
from Meshing.core import Mesher
from Meshing.plot import PlotMode
from config import load_facet_config
from FEM.utils import iPETScMatrix
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import PreconditionerType, iSTType
from lib.loggingutils import setup_logging

__example_name__ = "cylinder_flow"

SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
target = -0.05

CASE_NAMES: Final[tuple[str, ...]] = (
    "cylinder_flow_L40",
    "cylinder_flow_L60",
    "cylinder_flow_L80",
    "cylinder_flow_L100",
    "cylinder_flow_L120",
    "cylinder_flow_L140",
    "cylinder_flow_L160",
    "cylinder_flow_L170",
    "cylinder_flow_L180",
    "cylinder_flow_L190",
    "cylinder_flow_L200",
    "cylinder_flow_L210",
)

# Output file (aggregated results)
SIGMA_OUT: Final[Path] = SAVE_DIR / "sigma_results.csv"

setup_logging(verbose=True)

# Initialize aggregated results file
SIGMA_OUT.parent.mkdir(parents=True, exist_ok=True)
SIGMA_OUT.write_text("case_name,sigma_real,sigma_imag\n", encoding="utf-8")

for case_name in CASE_NAMES:
    case_dir = SAVE_DIR / case_name
    mesh_file = case_dir / "mesh" / "mesh.xdmf"
    mat_dir = case_dir / "matrices" / "wo_pressure"

    mesher = Mesher.from_file(mesh_file)
    facet_cfg = load_facet_config(CFG_DIR / "facets.toml")
    mesher.mark_boundary_facets(facet_cfg)

    spaces = define_spaces(mesher.mesh)

    A = iPETScMatrix.from_path(mat_dir / "A.mtx")
    M = iPETScMatrix.from_path(mat_dir / "M.mtx")
    A.assemble()
    M.assemble()

    es_cfg = EigensolverConfig(num_eig=5, atol=1e-3)
    es = EigenSolver(A, M, cfg=es_cfg, check_hermitian=False)

    es.solver.set_st_type(iSTType.SINVERT)
    es.solver.set_target(target)
    es.solver.set_st_pc_type(PreconditionerType.LU)

    es.solver.solve()
    sigma = es.solver.get_eigenvalue(0)

    with SIGMA_OUT.open("a", encoding="utf-8") as f:
        f.write(f"{case_name},{sigma.real},{sigma.imag}\n")

    # Also write a per-case text file next to the matrices
    (mat_dir / "sigma_eig0.txt").write_text(
        f"{sigma.real} {sigma.imag}\n", encoding="utf-8"
    )

    # Plot the real part of the eigenvector
    ev = es.solver.get_eigenvector(0)
    ef = dfem.Function(spaces.mixed)
    ef.x.array[:] = ev.real.as_array()

    plot_mixed_function(
        ef,
        PlotMode.STATIC,
        output_path=mat_dir / "ev.png",
    )
    plot_mixed_function(
        ef,
        PlotMode.STATIC,
        output_path=mat_dir / "ev.pdf",
    )
