from pathlib import Path
from typing import Final
from FEM.bcs import define_bcs
from FEM.plot import plot_mixed_function
from FEM.spaces import define_spaces
from Meshing.core import Mesher
from Meshing.plot import PlotMode
from Meshing.utils import Geometry
from Sensitivity import EigenSensitivitySolver
from Solver.baseflow import load_function
from config import load_bc_config, load_cylinder_flow_config, load_facet_config
from lib.loggingutils import setup_logging

setup_logging(verbose=True)

# Setup ---

SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"

cylinder_cfg = load_cylinder_flow_config(CFG_DIR / "geometry_short.toml")
bcs_cfg = load_bc_config(CFG_DIR / "bcs.toml")
facet_cfg = load_facet_config(CFG_DIR / "facets.toml")

mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
mesher.mark_boundary_facets(facet_cfg)

spaces = define_spaces(mesher.mesh)
bcs = define_bcs(mesher, spaces, bcs_cfg)

baseflow = load_function(SAVE_DIR / "baseflow", spaces=spaces)

direct_mode = load_function(SAVE_DIR / "sens_eigs", spaces, name="direct_mode")
adjoint_mode = load_function(SAVE_DIR / "sens_eigs", spaces, name="adjoint_mode")

# Compute structural sensitivity ---

sense_solver = EigenSensitivitySolver(
    spaces, bcs, baseflow, 60, tags=mesher.facet_tags, target=4.9 - 0.24j
)

structural_sens = sense_solver.compute_wavemaker(
    v_func=direct_mode, a_func=adjoint_mode
)
plot_mixed_function(structural_sens, PlotMode.INTERACTIVE, scale=0.0)

# Compute eigenvalue sensitivity to Re ---

baseflow_sense = sense_solver.compute_baseflow_sensitivity()
sense = sense_solver.evaluate_sensitivity(
    60.0, direct_mode, adjoint_mode, baseflow_sense
)
print(sense)
