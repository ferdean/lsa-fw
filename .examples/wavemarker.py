from pathlib import Path
from typing import Final

from FEM.plot import plot_mixed_function
from Meshing.plot import PlotMode

from FEM.spaces import FunctionSpaceType, define_spaces
from Meshing.core import Mesher
from Meshing.utils import Geometry
from Solver.baseflow import export_function, load_function
from config import load_cylinder_flow_config
from Sensitivity.sensitivity import compute_receptivity_from_modes
from lib.loggingutils import setup_logging

setup_logging(verbose=True)

_CFG_DIR: Final[Path] = Path("config_files") / "2D" / "cylinder"
_SAVE_DIR: Final[Path] = Path("cases") / "cylinder"
_REYNOLDS: Final[float] = 60.0


case_dir = _SAVE_DIR / f"reynolds_{_REYNOLDS:.1f}_plots"

# Build mesh/spaces
cylinder_cfg = load_cylinder_flow_config(_CFG_DIR / "geometry_fine.toml")

mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cylinder_cfg)
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)

v_func = load_function(case_dir / "direct_mode", spaces, name="direct_mode")
a_func = load_function(case_dir / "adjoint_mode", spaces, name="adjoint_mode")
receptivity = compute_receptivity_from_modes(v_func, a_func)

export_function(receptivity, case_dir / "receptivity", name="receptivity")

plot_mixed_function(v_func, PlotMode.INTERACTIVE, scale=0.0)
plot_mixed_function(a_func, PlotMode.INTERACTIVE, scale=0.0)
plot_mixed_function(receptivity, PlotMode.INTERACTIVE, scale=0.0)
