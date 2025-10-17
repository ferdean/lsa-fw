"""LSA-FW Elasticity plotting utilities."""

import logging
from pathlib import Path
from typing import Final

from matplotlib import pyplot as plt

# from matplotlib.tri import Triangulation
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

from dolfinx.plot import vtk_mesh
from dolfinx import fem as dfem

import numpy as np
import pyvista
from dataclasses import dataclass


for log in ("matplotlib", "PIL.PngImagePlugin"):
    logging.getLogger(log).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


@dataclass
class DisplacementPlotConfig:
    """Configuration of the displacement plots."""

    scale: float = 0.05
    title: str | None = None
    show_edges: bool = True


def _flatten(cells_list, ctypes_list, points_list):
    offsets = np.cumsum([0] + [len(p) for p in points_list[:-1]])
    cell_data = []
    for off, blk in zip(offsets, cells_list):
        i = 0
        while i < len(blk):
            n = blk[i]
            cell_data.append(n)
            cell_data.extend(blk[i + 1 : i + 1 + n] + off)
            i += n + 1
    return np.array(cell_data), np.hstack(ctypes_list), np.vstack(points_list)


def plot_displacement(
    u: dfem.Function, *, cfg: DisplacementPlotConfig | None = None
) -> None:
    """Interactive PyVista plot of a displacement mode."""
    cfg = cfg or DisplacementPlotConfig()
    space = u.function_space
    mesh = space.mesh
    dofs_per_node = mesh.geometry.dim
    comm = mesh.comm

    cells, cell_types, x = vtk_mesh(space)

    # Reshape DOF vector into (n, 3) with zero-padding if needed
    u_vals = u.x.array
    if u_vals.size % dofs_per_node != 0:
        raise RuntimeError("Unexpected DOF layout for vector function.")
    u_vec = np.zeros((u_vals.size // dofs_per_node, 3))
    u_vec[:, :dofs_per_node] = u_vals.reshape((-1, dofs_per_node))

    # Gather to rank 0 (MPI-safe)
    cells_rank = comm.gather(cells, root=0)
    ctype_rank = comm.gather(cell_types, root=0)
    coords_rank = comm.gather(x, root=0)
    u_vec_rank = comm.gather(u_vec, root=0)
    if comm.rank != 0:
        return

    c_all, ct_all, x_all = _flatten(cells_rank, ctype_rank, coords_rank)
    u_all = np.vstack(u_vec_rank)

    # Build grid and attach data
    grid = pyvista.UnstructuredGrid(c_all, ct_all, x_all)
    grid["u"] = u_all
    grid["u_mag"] = np.linalg.norm(u_all[:, :dofs_per_node], axis=1)

    # Choose scalars
    warped = grid.warp_by_vector("u", factor=cfg.scale)

    # Plot
    pl = pyvista.Plotter()
    if cfg.title:
        pl.add_text(cfg.title, position="upper_edge", font_size=12)

    pl.add_mesh(
        warped, scalars="u_mag", show_scalar_bar=True, show_edges=cfg.show_edges
    )
    pl.add_mesh(grid.outline(), line_width=1)  # Add an outline of the original mesh

    if dofs_per_node == 2:
        pl.view_xy()
    pl.show()


# Quiet some noisy libs
for log in ("matplotlib", "PIL.PngImagePlugin"):
    logging.getLogger(log).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Minimal, consistent rc
_RC: Final[dict] = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["LM Sans 10"],
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.bbox": "tight",
}


def _ensure_linear_vector_space(u: dfem.Function) -> dfem.Function:
    V = u.function_space
    el = V.ufl_element()
    deg = getattr(el, "degree", 1)
    if deg <= 1:
        return u
    mesh = V.mesh
    gdim = mesh.geometry.dim
    V1 = dfem.FunctionSpace(mesh, ("Lagrange", 1, (gdim,)))
    out = dfem.Function(V1)
    out.interpolate(u)
    return out


def _gather_vtk(comm, cells, ctypes, coords, node_vals):
    """Gather VTK arrays and nodal values to rank 0 and flatten indexing."""
    Lc = comm.gather(cells, root=0)
    Lt = comm.gather(ctypes, root=0)
    Lx = comm.gather(coords, root=0)
    Lv = comm.gather(node_vals, root=0)
    if comm.rank != 0:
        return None

    offsets = np.cumsum([0] + [len(a) for a in Lx[:-1]])
    flat_cells = []
    for off, blk in zip(offsets, Lc):
        i = 0
        while i < len(blk):
            n = int(blk[i])
            flat_cells.append(np.r_[n, np.asarray(blk[i + 1 : i + 1 + n]) + off])
            i += n + 1
    cells_all = np.concatenate(flat_cells).astype(int)
    ctypes_all = np.hstack(Lt).astype(int)
    X = np.vstack(Lx)
    U = np.vstack(Lv)
    return cells_all, ctypes_all, X, U


def _tri_from_vtk(cells_flat: np.ndarray, ctypes: np.ndarray) -> np.ndarray:
    """Triangles from VTK cell stream (VTK_TRIANGLE=5)."""
    VTK_TRIANGLE = 5
    tri_ids = np.where(ctypes == VTK_TRIANGLE)[0]
    tris = []
    i = 0
    k = 0
    while i < len(cells_flat):
        n = int(cells_flat[i])
        if k in tri_ids:
            c = cells_flat[i + 1 : i + 1 + n]
            if len(c) == 3:
                tris.append(c)
        i += n + 1
        k += 1
    if not tris:
        raise RuntimeError("No linear triangles (VTK_TRIANGLE=5) found.")
    return np.asarray(tris, dtype=int)


def _vtk_cells_iter(cells_flat: np.ndarray) -> list[np.ndarray]:
    out = []
    i = 0
    while i < len(cells_flat):
        n = int(cells_flat[i])
        out.append(np.asarray(cells_flat[i + 1 : i + 1 + n], dtype=int))
        i += n + 1
    return out


def _boundary_triangles_3d(cells_flat: np.ndarray, ctypes: np.ndarray) -> np.ndarray:
    """Boundary faces as triangles for tets/hexes (quadratic downcast by corners)."""
    VTK_TETRA = 10
    VTK_HEXAHEDRON = 12
    VTK_QUADRATIC_TETRA = 24
    VTK_QUADRATIC_HEXAHEDRON = 25
    VTK_TRIQUADRATIC_HEXAHEDRON = 29

    face_count: dict[tuple[int, int, int], int] = {}
    face_keep: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    cells = _vtk_cells_iter(cells_flat)
    if len(cells) != len(ctypes):
        raise RuntimeError("cells/ctypes length mismatch.")

    for nodes, ct in zip(cells, ctypes):
        if ct in (VTK_TETRA, VTK_QUADRATIC_TETRA):
            v = nodes[:4]
            faces = [
                (v[0], v[1], v[2]),
                (v[0], v[1], v[3]),
                (v[0], v[2], v[3]),
                (v[1], v[2], v[3]),
            ]
        elif ct in (
            VTK_HEXAHEDRON,
            VTK_QUADRATIC_HEXAHEDRON,
            VTK_TRIQUADRATIC_HEXAHEDRON,
        ):
            v = nodes[:8]
            quads = [
                (v[0], v[1], v[2], v[3]),
                (v[4], v[5], v[6], v[7]),
                (v[0], v[1], v[5], v[4]),
                (v[1], v[2], v[6], v[5]),
                (v[2], v[3], v[7], v[6]),
                (v[3], v[0], v[4], v[7]),
            ]
            faces = []
            for a, b, c, d in quads:
                faces.append((a, b, c))
                faces.append((a, c, d))
        else:
            raise RuntimeError(f"Unsupported 3D VTK cell type {ct}.")

        for f in faces:
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
            face_keep.setdefault(key, f)

    tris = [face_keep[k] for k, cnt in face_count.items() if cnt == 1]
    if not tris:
        raise RuntimeError("No boundary faces found.")
    return np.asarray(tris, dtype=int)


def _axes3d_style(ax):
    # transparent panes + faint edges + light grid
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
        pane.set_edgecolor((0, 0, 0, 0.25))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(
            {"linewidth": 0.4, "linestyle": "-", "color": (0, 0, 0, 0.12)}
        )
        axis._axinfo["tick"].update({"inward_factor": 0.015, "outward_factor": 0.0})
    ax.grid(True)


def _part(a: np.ndarray, mode: str) -> np.ndarray:
    if mode == "real":
        return a.real if np.iscomplexobj(a) else a
    if mode == "imag":
        return a.imag if np.iscomplexobj(a) else np.zeros_like(a)
    if mode == "abs":
        return np.abs(a) if np.iscomplexobj(a) else a
    raise ValueError("mode must be 'abs', 'real', or 'imag'.")


# ---------- public API ----------


def save_displacement(
    u: dfem.Function,
    out_path: Path,
    *,
    scale: float = 0.05,
    cmap: str = "viridis",
    part: str = "abs",
    show_mesh: bool = True,
    title: str | None = None,
    fig_size: tuple[float, float] | None = None,  # None -> auto per dim
    dpi_png: int = 330,
    elev: float = 18.0,
    azim: float = -50.0,
) -> None:
    """
    Save a static plot of a displacement field.
    - 2D: filled trisurf (Triangulation) + optional wireframe
    - 3D: boundary shaded Poly3DCollection + optional undeformed wireframe

    Vector outputs (.pdf/.svg) recommended; .png supported via dpi_png.
    """
    # Linearize if needed (node-wise mapping)
    u_lin = _ensure_linear_vector_space(u)
    V = u_lin.function_space
    mesh = V.mesh
    gdim = mesh.geometry.dim
    comm = mesh.comm

    # VTK topology + coordinates
    cells, ctypes, coords = vtk_mesh(V)

    # nodal vector values
    vals = u_lin.x.array
    if vals.size % gdim != 0:
        raise RuntimeError("Unexpected DOF layout for vector function.")
    U = vals.reshape((-1, gdim))
    Up = _part(U, part)

    # Gather to rank 0
    gathered = _gather_vtk(comm, cells, ctypes, coords, Up)
    if gathered is None:
        return
    cells_all, ctypes_all, X, U = gathered

    # Deformation & magnitudes
    if gdim == 2:
        Xdef = np.c_[
            X[:, 0] + scale * U[:, 0], X[:, 1] + scale * U[:, 1], np.zeros(len(X))
        ]
        Umag = np.linalg.norm(U, axis=1)
        tris = _tri_from_vtk(cells_all, ctypes_all)
        tri2d = Triangulation(Xdef[:, 0], Xdef[:, 1], triangles=tris)
        fig_w, fig_h = fig_size if fig_size else (4.8, 3.6)
        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            col = ax.tripcolor(
                tri2d, facecolors=Umag[tris].mean(axis=1), shading="flat", cmap=cmap
            )
            if show_mesh:
                ax.triplot(tri2d, lw=0.25, color=(0, 0, 0, 0.25))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            if title:
                ax.set_title(title, pad=6)
            cbar = fig.colorbar(col, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"$\|\tilde{\mathbf{u}}\|$")
            _tight_save(fig, out_path, dpi_png)
            plt.close(fig)
        return

    # --- 3D branch ---
    Xdef = X.copy()
    Xdef[:, :3] = Xdef[:, :3] + scale * U
    Umag = np.linalg.norm(U, axis=1)
    tri_conn = _boundary_triangles_3d(cells_all, ctypes_all)
    tris_xyz_def = [Xdef[idx][:, :3] for idx in tri_conn]
    face_vals = Umag[tri_conn].mean(axis=1)

    fig_w, fig_h = fig_size if fig_size else (5.6, 4.2)
    with plt.rc_context({**_RC, "savefig.bbox": "standard"}):
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.subplots_adjust(right=0.86)  # space for colorbar
        ax = fig.add_subplot(111, projection="3d")

        norm = Normalize(vmin=float(face_vals.min()), vmax=float(face_vals.max()))
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = mappable.to_rgba(face_vals)

        surf = Poly3DCollection(
            tris_xyz_def, facecolors=colors, edgecolor="none", linewidths=0.0
        )
        surf.set_antialiased(True)
        ax.add_collection3d(surf)

        if show_mesh:
            tris_xyz_undeformed = [X[idx][:, :3] for idx in tri_conn]
            wire = Line3DCollection(
                [np.vstack([t, t[0]]) for t in tris_xyz_undeformed],
                linewidths=0.25,
                colors=(0, 0, 0, 0.18),
            )
            ax.add_collection3d(wire)

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        if title:
            ax.set_title(title, pad=6)
        _axes3d_style(ax)

        # equal aspect + small padding
        mins = Xdef[:, :3].min(axis=0)
        maxs = Xdef[:, :3].max(axis=0)
        span = float(np.max(maxs - mins))
        pad = 0.03 * span
        mid = 0.5 * (mins + maxs)
        ax.set_xlim(mid[0] - span / 2 - pad, mid[0] + span / 2 + pad)
        ax.set_ylim(mid[1] - span / 2 - pad, mid[1] + span / 2 + pad)
        ax.set_zlim(mid[2] - span / 2 - pad, mid[2] + span / 2 + pad)

        ax.view_init(elev=elev, azim=azim)

        # colorbar on the right
        cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cb = plt.colorbar(mappable, cax=cax)
        cb.set_label(r"$\|\tilde{\mathbf{u}}\|$")

        _tight_save(fig, out_path, dpi_png)
        plt.close(fig)


def _tight_save(fig: plt.Figure, out_path: Path, dpi_png: int):
    ext = out_path.suffix.lower()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".png":
        fig.savefig(out_path, dpi=dpi_png)
    elif ext in {".pdf", ".svg"}:
        fig.savefig(
            out_path,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=300,
            metadata={"Creator": "LSA-FW displacement saver"},
        )
    else:
        raise ValueError("Use one of: .pdf, .svg, .png")
