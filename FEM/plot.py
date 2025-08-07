"""LSA-FW FEM plotter.

Provides helper functions for visualizing FEM data, such as matrix sparsity patterns and mixed function fields.
"""

import logging
from pathlib import Path
from typing import assert_never, Final

import matplotlib.pyplot as plt
import numpy as np
import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx import fem as dfem
from matplotlib.tri import Triangulation
from mpi4py import MPI
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix  # type:ignore[import-untyped]

from FEM.spaces import FunctionSpaces, define_spaces, FunctionSpaceType
from Meshing.plot import PlotMode
from lib.loggingutils import log_global

from .utils import iPETScBlockMatrix, iPETScMatrix


for log in ("matplotlib", "PIL.PngImagePlugin"):
    logging.getLogger(log).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["LM Sans 10"],
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)

_DEFAULT_FIGSIZE: Final[tuple[float, float]] = (6, 3)


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
    mixed_function: dfem.Function,
    mode: PlotMode,
    *,
    output_path: Path | None = None,
    title: str | None = None,
    domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
    fig_size: tuple[float, float] = _DEFAULT_FIGSIZE,
    scale: float = 1.0,
    streamline_density: float = 1.0,
    cmap: str = "coolwarm",
) -> None:
    """Plot a function defined in a mixed velocity-pressure space."""
    match mode:
        case PlotMode.INTERACTIVE:
            _plot_mixed_function_pyvista(
                mixed_function, title=title, scale=scale, domain=domain, cmap=cmap
            )
        case PlotMode.STATIC:
            if output_path is None:
                raise ValueError("Output path must be provided for static mode.")
            _plot_mixed_function_matplotlib(
                mixed_function,
                output_path,
                domain=domain,
                fig_size=fig_size,
                streamline_density=streamline_density,
            )
        case _:
            assert_never(mode)


def _plot_mixed_function_pyvista(
    mixed_function: dfem.Function,
    title: str | None = None,
    scale: float = 1.0,
    domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
    cmap: str = "RdBu_r",
) -> None:
    comm = mixed_function.function_space.mesh.comm

    u = mixed_function.sub(0).collapse()
    p = mixed_function.sub(1).collapse()

    cu, ctu, xu = vtk_mesh(u.function_space)
    cp, ctp, xp = vtk_mesh(p.function_space)
    gdim = u.function_space.mesh.geometry.dim

    u_vec = np.zeros((len(u.x.array) // gdim, 3))
    u_vec[:, :gdim] = u.x.array.real.reshape((-1, gdim))
    p_vec = p.x.array.real

    cu_list = comm.gather(cu, root=0)
    ctu_list = comm.gather(ctu, root=0)
    xu_list = comm.gather(xu, root=0)
    u_list = comm.gather(u_vec, root=0)

    cp_list = comm.gather(cp, root=0)
    ctp_list = comm.gather(ctp, root=0)
    xp_list = comm.gather(xp, root=0)
    p_list = comm.gather(p_vec, root=0)

    if comm.rank != 0:
        return

    c_u, ct_u, x_u = _flatten_mesh(cu_list, ctu_list, xu_list)
    c_p, ct_p, x_p = _flatten_mesh(cp_list, ctp_list, xp_list)
    u_all = np.vstack(u_list)
    p_all = np.hstack(p_list)

    grid_u = pyvista.UnstructuredGrid(c_u, ct_u, x_u)
    grid_u["velocity"] = u_all
    glyphs = grid_u.glyph(orient="velocity", factor=scale)

    if domain is not None:
        mask = (
            (grid_u.points[:, 0] >= domain[0][0])
            & (grid_u.points[:, 0] <= domain[1][0])
            & (grid_u.points[:, 1] >= domain[0][1])
            & (grid_u.points[:, 1] <= domain[1][1])
        )
        grid_u = grid_u.extract_points(mask, adjacent_cells=True)

    plotter_u = pyvista.Plotter()
    if title:
        plotter_u.add_text(f"{title} — Velocity", position="upper_edge", font_size=12)
    plotter_u.add_mesh(grid_u, show_edges=False, show_scalar_bar=False, cmap=cmap)
    plotter_u.add_mesh(glyphs, cmap=cmap)
    plotter_u.view_xy()
    plotter_u.show()

    grid_p = pyvista.UnstructuredGrid(c_p, ct_p, x_p)
    grid_p["pressure"] = p_all

    if domain is not None:
        mask = (
            (grid_p.points[:, 0] >= domain[0][0])
            & (grid_p.points[:, 0] <= domain[1][0])
            & (grid_p.points[:, 1] >= domain[0][1])
            & (grid_p.points[:, 1] <= domain[1][1])
        )
        grid_p = grid_p.extract_points(mask, adjacent_cells=True)

    plotter_p = pyvista.Plotter()
    if title:
        plotter_p.add_text(f"{title} — Pressure", position="upper_edge", font_size=12)
    plotter_p.add_mesh(
        grid_p,
        scalars="pressure",
        show_edges=False,
        cmap=cmap,
        clim=[-np.abs(p_all).max(), np.abs(p_all).max()],
    )
    plotter_p.view_xy()
    plotter_p.show()


def _plot_mixed_function_matplotlib(
    mixed_function: dfem.Function,
    output_path: Path,
    domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
    fig_size: tuple[float, float] = _DEFAULT_FIGSIZE,
    streamline_density: float = 1.0,
) -> None:
    mesh = mixed_function.function_space.mesh
    if mesh.geometry.dim != 2:
        raise ValueError("Only 2D domains are supported for static plots.")
    if MPI.COMM_WORLD.size > 1:
        log_global(
            logger, logging.WARNING, "Static plotting requires serial execution."
        )

    u_h, p_h = mixed_function.split()
    if u_h.function_space.ufl_element().degree > 1:
        linear = define_spaces(mesh, type=FunctionSpaceType.SIMPLE)
        u = dfem.Function(linear.velocity)
        p = dfem.Function(linear.pressure)
        u.interpolate(u_h)
        p.interpolate(p_h)
    else:
        u, p = u_h, p_h

    cells, cell_types, coords = vtk_mesh(u.function_space)
    tri_indices = np.where(cell_types == 5)[0]
    triangle_cells = []

    i = 0
    tri_id = 0
    while i < len(cells):
        n = cells[i]
        if tri_id in tri_indices:
            triangle_cells.append(cells[i + 1 : i + 1 + n])
        i += n + 1
        tri_id += 1

    if not triangle_cells:
        raise RuntimeError("No linear triangle cells found.")

    triangles = np.array(triangle_cells)
    xy = coords[:, :2]
    tri = Triangulation(xy[:, 0], xy[:, 1], triangles)
    u_vals = u.x.array.reshape((-1, 2))
    u_mag = np.linalg.norm(u_vals, axis=1)
    p_vals = p.x.array

    if domain:
        (xmin, ymin), (xmax, ymax) = domain
    else:
        xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()

    fig, axs = plt.subplots(
        2, 1, figsize=fig_size, constrained_layout=True, sharex=True
    )

    ax0, ax1 = axs
    tpc_u = ax0.tripcolor(tri, u_mag, shading="gouraud", cmap="viridis")
    ax0.set_ylabel(r"$y$ (-)")
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)

    try:
        nx, ny = 300, 300
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
        ug = griddata(xy, u_vals[:, 0], (gx, gy), method="linear")
        vg = griddata(xy, u_vals[:, 1], (gx, gy), method="linear")

        # Mask invalid or out-of-domain areas
        mask = np.isnan(ug) | np.isnan(vg) | (np.hypot(ug, vg) < 1e-10)
        ug = np.ma.array(ug, mask=mask)
        vg = np.ma.array(vg, mask=mask)

        ax0.streamplot(
            gx, gy, ug, vg, color="lightgray", density=streamline_density, linewidth=0.5
        )

    except Exception:
        log_global(
            logger, logging.WARNING, "Skipping streamlines (scipy.interpolate failed)."
        )

    tpc_p = ax1.tripcolor(tri, p_vals, shading="gouraud", cmap="coolwarm")
    ax1.set_xlabel(r"$x$ (-)")
    ax1.set_ylabel(r"$y$ (-)")
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)

    for ax, tpc, label in zip(
        axs,
        [tpc_u, tpc_p],
        [r"$\|\mathbf{\tilde{\bar{u}}}\|_{L_2}$", r"$\tilde{p}$ (-)"],
    ):
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(direction="in")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.2)
        cbar = fig.colorbar(tpc, cax=cax)
        cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8, direction="in")

    ext = output_path.suffix.lower().lstrip(".")
    if ext not in {"pdf", "svg", "png"}:
        raise ValueError("Output format must be one of: pdf, svg, png")
    dpi = 330 if ext == "png" else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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


def _flatten_mesh(cells, cell_types, points):
    offsets = np.cumsum([0] + [len(p) for p in points[:-1]])
    cell_data = []
    for off, blk in zip(offsets, cells):
        i = 0
        while i < len(blk):
            n = blk[i]
            cell_data.append(n)
            cell_data.extend(blk[i + 1 : i + 1 + n] + off)
            i += n + 1
    return np.array(cell_data), np.hstack(cell_types), np.vstack(points)


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
