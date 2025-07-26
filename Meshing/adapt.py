"""LSA-FW Meshing adapt."""

import logging
import os
import tempfile
import warnings
from pathlib import Path

from basix import CellType as BasixCellType
from basix.ufl import element
import dolfinx.fem as dfem
from dolfinx.fem.petsc import LinearProblem
import dolfinx.io as dio
import dolfinx.mesh as dmesh
import gmsh
import meshio
from mpi4py import MPI
import numpy as np
from ufl import dx, TestFunction, TrialFunction

from FEM.utils import iElementFamily
from lib.loggingutils import log_global, capture_and_log

from .core import Mesher
from .utils import iCellType

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore")


def _extract_mesh_data(
    mesher: Mesher,
) -> tuple[
    dmesh.Mesh, BasixCellType, int, np.ndarray, np.ndarray, tuple[str, np.ndarray]
]:
    """Extract geometric and topological data from the input mesh."""
    try:
        mesh = mesher.mesh
        cell_type = iCellType.from_dolfinx(mesh.topology.cell_type).to_basix()
        gdim = mesh.geometry.dim
        coords = mesh.geometry.x[:, :gdim]
        num_cells = mesh.topology.index_map(gdim).size_local
        conn = mesh.topology.connectivity(gdim, 0).array
        nvpc = conn.size // num_cells
        cell_vertices = conn.reshape((num_cells, nvpc))
        cell_block = (cell_type.name, cell_vertices)
        return mesh, cell_type, gdim, coords, cell_vertices, cell_block
    except RuntimeError as e:
        raise ValueError(
            "Mesh adaptation requires a fully-generated input mesh."
        ) from e


def _project_velocity_magnitude(
    mesh: dmesh.Mesh, baseflow: dfem.Function, cell_type: BasixCellType
) -> dfem.Function:
    """Project the magnitude of the baseflow velocity onto a linear function space.

    Computes the velocity norm at mesh vertices by solving a lumped-mass linear problem,
    yielding a continuous approximation of the velocity magnitude on the mesh.
    """
    velocity = baseflow.sub(0).collapse()
    V = dfem.functionspace(
        mesh,
        element(
            family=iElementFamily.LAGRANGE.to_dolfinx(),
            cell=cell_type,
            degree=1,
        ),
    )
    velocity_norm = dfem.Function(V)
    u, v = TrialFunction(V), TestFunction(V)

    norm_form = dfem.form(
        (
            velocity[0] ** 2
            + velocity[1] ** 2
            + (velocity[2] ** 2 if mesh.geometry.dim == 3 else 0.0)
        )
        ** 0.5
        * v
        * dx
    )
    lump = dfem.form(u * v * dx)
    problem = LinearProblem(lump, norm_form, u=velocity_norm)
    velocity_norm.x.array[:] = problem.solve().x.array[:]
    return velocity_norm


def _scale_and_clamp_velocity(
    velocity_norm: dfem.Function, min_size: int, max_size: int
) -> None:
    """Normalize the size field to lie within [min_size, max_size]."""
    vals = velocity_norm.x.array
    umax = vals.max() if vals.size else 1.0
    scaled = min_size + (vals / umax) * (max_size - min_size)
    velocity_norm.x.array[:] = np.clip(scaled, min_size, max_size)
    velocity_norm.x.scatter_forward()


def _write_background_field(
    pos_path: Path,
    cell_vertices: np.ndarray,
    coords: np.ndarray,
    velocity_norm: dfem.Function,
    gdim: int,
) -> None:
    """Write a Gmsh POS background mesh size file."""
    with pos_path.open("w") as f:
        f.write('View "background" {\n')
        for verts in cell_vertices:
            if gdim == 2:
                i, j, k = verts
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                x3, y3 = coords[k]
                s1, s2, s3 = velocity_norm.x.array[[i, j, k]]
                f.write(
                    f"ST({x1},{y1},0, {x2},{y2},0, {x3},{y3},0)"
                    f"{{{s1}, {s2}, {s3}}};\n"
                )
            else:
                p, s = coords, velocity_norm.x.array
                faces = [
                    (verts[0], verts[1], verts[2]),
                    (verts[0], verts[1], verts[3]),
                    (verts[0], verts[2], verts[3]),
                    (verts[1], verts[2], verts[3]),
                ]
                for a, b, c in faces:
                    f.write(
                        f"ST({p[a][0]},{p[a][1]},{p[a][2]}, "
                        f"{p[b][0]},{p[b][1]},{p[b][2]}, "
                        f"{p[c][0]},{p[c][1]},{p[c][2]})"
                        f"{{{s[a]}, {s[b]}, {s[c]}}};\n"
                    )
        f.write("};\n")


def _export_to_msh(coords: np.ndarray, cell_block: tuple[str, np.ndarray]) -> str:
    """Export the current mesh to a temporary Gmsh .msh file."""
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        msh_file = tmp.name

    meshio.write(
        msh_file,
        meshio.Mesh(points=coords, cells={cell_block[0]: cell_block[1]}),
        file_format="gmsh22",
    )

    return msh_file


def _run_gmsh_remesh(
    msh_file: str,
    pos_file: Path,
    gdim: int,
    comm: MPI.Intracomm,
) -> dmesh.Mesh:
    """Invoke Gmsh to mesh the geometry with the specified background field."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("adaptation")
        gmsh.merge(msh_file)

        if gdim == 2:
            gmsh.model.mesh.classifySurfaces(1e-6, True, True, 0)
            gmsh.model.mesh.createGeometry()

        gmsh.merge(str(pos_file))
        fld = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(fld, "ViewIndex", 0)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # Smooth mesh to avoid low-quality cells near boundaries
        gmsh.option.setNumber("Mesh.Smoothing", 10)

        gmsh.model.mesh.field.setAsBackgroundMesh(fld)
        gmsh.model.mesh.generate(gdim)
        log_global(logger, logging.DEBUG, "Gmsh mesh generation completed.")

        entities = gmsh.model.getEntities(gdim)
        tags = [tag for (_, tag) in entities]
        gmsh.model.addPhysicalGroup(gdim, tags)

        new_mesh, _, _ = dio.gmshio.model_to_mesh(gmsh.model, comm, 0)

    except Exception as e:
        raise RuntimeError(f"Gmsh remeshing failed: {e}")
    finally:
        gmsh.finalize()
        os.remove(msh_file)
        pos_file.unlink()

    return new_mesh


def _flatten_mesh(mesh: dmesh.Mesh) -> dmesh.Mesh:
    """Ensure a 2D topology is represented with 2D geometry."""
    if mesh.topology.dim == 2 and mesh.geometry.dim == 3:
        coords2d = mesh.geometry.x[:, :2].copy()
        conn = mesh.topology.connectivity(mesh.topology.dim, 0).array
        ncells = mesh.topology.index_map(mesh.topology.dim).size_local
        nvpc = conn.size // ncells
        cells2d = conn.reshape((ncells, nvpc))
        return dmesh.create_mesh(mesh.comm, cells2d, coords2d, mesh.ufl_domain())
    return mesh


def adapt_mesh(
    mesher: Mesher,
    baseflow: dfem.Function,
    *,
    min_size: float,
    max_size: float,
) -> Mesher:
    """Adapt the mesh using a background sizing field from the baseflow velocity magnitude.

    Computes the element-wise velocity magnitude of a given baseflow field,  projects it onto a linear finite element
    space, and constructs a background mesh size field scaled between min_size and max_size.  The background field is
    exported to Gmsh as a POS file, merged with the original mesh, and re-meshing is performed with smoothing to improve
    cell quality. The adapted mesh is then converted back into a Mesher object.
    """
    log_global(logger, logging.INFO, "Starting mesh adaptation.")
    mesh, cell_type, gdim, coords, cell_vertices, cell_block = _extract_mesh_data(
        mesher
    )
    log_global(
        logger,
        logging.INFO,
        f"Extracted initial mesh data (gdim={gdim}, cells={len(cell_vertices)}).",
    )

    velocity_norm = _project_velocity_magnitude(mesh, baseflow, cell_type)
    log_global(logger, logging.DEBUG, "Projected velocity magnitude onto P1 space.")

    _scale_and_clamp_velocity(velocity_norm, min_size, max_size)
    log_global(
        logger,
        logging.INFO,
        f"Scaled and clamped sizes to [{min_size}, {max_size}].",
    )

    pos_file = Path("background_size.pos")
    _write_background_field(pos_file, cell_vertices, coords, velocity_norm, gdim)

    with capture_and_log(logger, logging.WARNING):
        # Catch any export warnings and re-direct them to the main logger
        msh_file = _export_to_msh(coords, cell_block)

    log_global(
        logger, logging.DEBUG, f"Exported mesh to temporary *.gmsh file {msh_file}."
    )

    log_global(logger, logging.INFO, "Running re-meshing.")
    new_mesh = _run_gmsh_remesh(msh_file, pos_file, gdim, mesh.comm)
    new_mesh_flattened = _flatten_mesh(new_mesh)
    log_global(logger, logging.INFO, "Mesh adaptation complete.")

    return Mesher.from_mesh(new_mesh_flattened)
