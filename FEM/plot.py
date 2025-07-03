"""LSA-FW FEM plotter.

Provides helper functions for visualizing FEM data, such as matrix sparsity patterns and mixed function fields.
"""

import logging
from pathlib import Path

import dolfinx.fem as dfem
import matplotlib.pyplot as plt
import numpy as np
import pyvista
from dolfinx.plot import vtk_mesh
from scipy.sparse import csr_matrix  # type:ignore[import-untyped]

from FEM.spaces import FunctionSpaces
from lib.loggingutils import log_global

from .utils import iPETScBlockMatrix, iPETScMatrix

logger = logging.getLogger(__name__)


def spy(
    matrix: iPETScMatrix | iPETScBlockMatrix,
    out_path: Path,
    dpi: int = 300,
    *,
    spaces: FunctionSpaces | None = None,
) -> None:
    """Plot the sparsity pattern of an iPETScMatrix as a static PNG.

    If function spaces are given as an argument, the output plot reorders the matrix so that the DOFs are ordered by
    type: [u_1 u_2 ... u_n p_1 p_2 ... p_m].
    """
    if isinstance(matrix, iPETScBlockMatrix):
        matrix = matrix.to_aij()

    comm = spaces.mixed.mesh.comm

    if comm.size > 1:
        log_global(
            logger,
            logging.WARNING,
            "DOF block-wise reordering for sparsity plots is only available in serial runs. "
            "This plot will show the distributed PETSc ordering instead. "
            "For block-ordered sparsity patterns, rerun with a single MPI rank.",
        )

    perm: np.ndarray | None = None
    if spaces is not None:
        _, dofs_u_loc = spaces.mixed.sub(0).collapse()
        _, dofs_p_loc = spaces.mixed.sub(1).collapse()
        dofs_u = comm.gather(dofs_u_loc, root=0)
        dofs_p = comm.gather(dofs_p_loc, root=0)
        if comm.rank == 0:
            global_dofs_u = np.sort(np.concatenate(dofs_u))
            global_dofs_p = np.sort(np.concatenate(dofs_p))
            perm = np.concatenate([global_dofs_u, global_dofs_p])

    csr = _to_csr_global(matrix)
    if csr is None:
        # Non-root ranks drop out here
        return

    if perm is not None:
        csr = csr[perm, :][:, perm]

    _save_plot(csr, out_path.with_suffix(".png"), dpi)


def plot_mixed_function(
    mixed_function: dfem.Function, *, scale: float = 1.0, title: str | None = None
) -> None:
    """Visualize mixed FEM function (velocity P2, pressure P1) using PyVista."""
    comm = mixed_function.function_space.mesh.comm

    # Collapse sub-functions
    u_c = mixed_function.sub(0).collapse()
    p_c = mixed_function.sub(1).collapse()

    # Velocity mesh (P2)
    cells_u, cell_types_u, points_u = vtk_mesh(u_c.function_space)
    gdim = u_c.function_space.mesh.geometry.dim
    u_values = np.zeros((len(u_c.x.array) // gdim, 3), dtype=np.float64)
    u_values[:, :gdim] = u_c.x.array.real.reshape((-1, gdim))

    # Pressure mesh (P1): use original mesh geometry
    cells_p, cell_types_p, points_p = vtk_mesh(p_c.function_space)
    p_values = p_c.x.array.real

    # Gather data to root
    cells_u_list = comm.gather(cells_u, root=0)
    cell_types_u_list = comm.gather(cell_types_u, root=0)
    points_u_list = comm.gather(points_u, root=0)
    u_list = comm.gather(u_values, root=0)

    cells_p_list = comm.gather(cells_p, root=0)
    cell_types_p_list = comm.gather(cell_types_p, root=0)
    points_p_list = comm.gather(points_p, root=0)
    p_list = comm.gather(p_values, root=0)

    if comm.rank == 0:
        # Merge velocity data (P2)
        pts_u = np.vstack(points_u_list)
        offsets_u = np.cumsum([0] + [len(blk) for blk in points_u_list[:-1]])
        all_cells_u = []
        for off, blk in zip(offsets_u, cells_u_list):
            arr = blk.copy()
            i = 0
            while i < len(arr):
                n = arr[i]
                all_cells_u.append(n)
                verts = arr[i + 1 : i + 1 + n] + off
                all_cells_u.extend(verts)
                i += n + 1

        cells_global_u = np.array(all_cells_u, dtype=int)
        cell_types_global_u = np.hstack(cell_types_u_list)
        u_global = np.vstack(u_list)

        # Create grid for velocity
        u_grid = pyvista.UnstructuredGrid(cells_global_u, cell_types_global_u, pts_u)
        u_grid["velocity"] = u_global

        # Glyphs for velocity
        glyphs = u_grid.glyph(orient="velocity", factor=scale)
        plotter_vel = pyvista.Plotter()
        if title:
            plotter_vel.add_text(
                f"{title} — Velocity", position="upper_edge", font_size=12
            )
        plotter_vel.add_mesh(u_grid, show_edges=False, show_scalar_bar=False)
        plotter_vel.add_mesh(glyphs)
        plotter_vel.view_xy()
        plotter_vel.show()

        # Merge pressure data (P1)
        pts_p = np.vstack(points_p_list)
        offsets_p = np.cumsum([0] + [len(blk) for blk in points_p_list[:-1]])
        all_cells_p = []
        for off, blk in zip(offsets_p, cells_p_list):
            arr = blk.copy()
            i = 0
            while i < len(arr):
                n = arr[i]
                all_cells_p.append(n)
                verts = arr[i + 1 : i + 1 + n] + off
                all_cells_p.extend(verts)
                i += n + 1

        cells_global_p = np.array(all_cells_p, dtype=int)
        cell_types_global_p = np.hstack(cell_types_p_list)
        p_global = np.hstack(p_list)

        # Create grid for pressure (P1)
        p_grid = pyvista.UnstructuredGrid(cells_global_p, cell_types_global_p, pts_p)
        p_grid["pressure"] = p_global

        # Plot pressure field
        plotter_p = pyvista.Plotter()
        if title:
            plotter_p.add_text(
                f"{title} — Pressure", position="upper_edge", font_size=12
            )
        plotter_p.add_mesh(p_grid, scalars="pressure", show_edges=False, cmap="viridis")
        plotter_p.view_xy()
        plotter_p.show()


def _to_csr_global(matrix: iPETScMatrix) -> csr_matrix | None:
    mat = matrix.raw
    if "aij" not in matrix.type.lower():
        raise NotImplementedError("Only AIJ-type supported.")

    m, n = matrix.shape
    comm = mat.comm.tompi4py()
    i0, i1 = mat.getOwnershipRange()

    indptr_loc, indices_loc, data_loc = mat.getValuesCSR()

    chunks = comm.gather((i0, i1, indptr_loc, indices_loc, data_loc), root=0)
    if comm.rank != 0:
        return None

    global_indptr = np.empty(m + 1, dtype=indptr_loc.dtype)
    all_idx = []
    all_dat = []
    nnz_off = 0

    for r0, r1, ip, idx, dat in chunks:
        global_indptr[r0 : r1 + 1] = ip + nnz_off
        nnz_off += ip[-1]
        all_idx.append(idx)
        all_dat.append(dat)

    global_indptr[m] = nnz_off
    global_idx = np.concatenate(all_idx)
    global_dat = np.concatenate(all_dat)

    return csr_matrix((global_dat, global_idx, global_indptr), shape=(m, n))


def _save_plot(matrix: csr_matrix, path: Path, dpi: int) -> None:
    """Render and save a CSR matrix sparsity plot."""
    path.parent.mkdir(parents=True, exist_ok=True)

    golden = (1 + 5**0.5) / 2
    fig, ax = plt.subplots(figsize=(6, 6 / golden))

    ax.spy(matrix, markersize=1)
    ax.set(title="Matrix sparsity pattern", xlabel="cols (-)", ylabel="rows (-)")
    ax.tick_params(direction="in", length=4, width=1, labelsize=8)

    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
