"""LSA-FW Elasticity plotting utilities."""

import logging
from pathlib import Path
from typing import Final

from matplotlib import pyplot as plt

# from matplotlib.tri import Triangulation
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


# Matplotlib/LaTeX defaults (vector-friendly)
_PLOT_RC: Final[dict] = {
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

_DEF_3D_RC = {
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["LM Sans 10"],
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.bbox": "tight",
}


def _beautify_3d_axes(ax) -> None:
    # Light fine gridlines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(
            {
                "linewidth": 0.4,
                "linestyle": "-",
                "color": (0, 0, 0, 0.12),  # light gray w/ alpha
            }
        )
        axis._axinfo["tick"].update({"inward_factor": 0.015, "outward_factor": 0.0})
    ax.grid(True)

    # Transparent panes, faint edges
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0.0))
        pane.set_edgecolor((0, 0, 0, 0.25))


def _set_equal_3d(ax, Xdef: np.ndarray) -> None:
    xmin, ymin, zmin = Xdef.min(axis=0)[:3]
    xmax, ymax, zmax = Xdef.max(axis=0)[:3]
    ranges = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    span = float(ranges.max())
    mid = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2], float)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)


@dataclass(frozen=True)
class SaveDisplacementConfig:
    """Static (publication-quality) displacement plot."""

    scale: float = 0.05
    title: str | None = None
    fig_size: tuple[float, float] = (4.5, 3.0)
    cmap: str = "viridis"
    show_mesh: bool = True
    mesh_edge_width: float = 0.3
    mesh_edge_color: str = "0.7"
    show_colorbar: bool = True
    colorbar_label: str = r"$\|\tilde{\mathbf{u}}\|$"
    part: str = "abs"
    dpi_png: int = 330  # used only if saving PNG
    require_2d: bool = False


def _as_part(a: np.ndarray, part: str) -> np.ndarray:
    if part == "real":
        return a.real if np.iscomplexobj(a) else np.asarray(a, dtype=np.float64)
    if part == "imag":
        if not np.iscomplexobj(a):
            return np.zeros_like(a, dtype=np.float64)
        return a.imag
    # "abs"
    return np.abs(a) if np.iscomplexobj(a) else np.asarray(a, dtype=np.float64)


def _tri_from_vtk(cells: np.ndarray, cell_types: np.ndarray) -> np.ndarray:
    """Extract linear triangle connectivity (VTK cell type 5) from VTK arrays."""
    TRI = 5  # VTK_TRIANGLE
    tri_indices = np.where(cell_types == TRI)[0]
    triangles = []
    i = 0
    cell_id = 0
    while i < len(cells):
        n = int(cells[i])
        if cell_id in tri_indices:
            c = cells[i + 1 : i + 1 + n]
            if len(c) == 3:
                triangles.append(c)
        i += n + 1
        cell_id += 1
    if not triangles:
        raise RuntimeError(
            "No linear triangle cells found. Provide/convert to P1 mesh."
        )
    return np.array(triangles, dtype=int)


def _ensure_linear_vector_space(u: dfem.Function) -> dfem.Function:
    """If u is higher-order, interpolate to linear vector Lagrange on the same mesh."""
    V = u.function_space
    el = V.ufl_element()
    # Heuristic: use interpolation for degree > 1; otherwise return as-is
    if getattr(el, "degree", 1) > 1:
        mesh = V.mesh
        # build linear vector Lagrange space
        linear = dfem.FunctionSpace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        u_lin = dfem.Function(linear)
        u_lin.interpolate(u)
        return u_lin
    return u


def _vtk_cells_iter(cells_flat: np.ndarray) -> list[np.ndarray]:
    """Split VTK flat connectivity into a list of per-cell index arrays."""
    out: list[np.ndarray] = []
    i = 0
    while i < len(cells_flat):
        n = int(cells_flat[i])
        out.append(np.asarray(cells_flat[i + 1 : i + 1 + n], dtype=int))
        i += n + 1
    return out


def _boundary_triangles_3d(cells_flat: np.ndarray, ctypes: np.ndarray) -> np.ndarray:
    """
    Extract boundary triangles for 3D meshes with tetrahedra or hexahedra
    (corner nodes only; quadratic variants downcast to corners).
    Returns (NT, 3) with vertex indices.
    """
    VTK_TETRA = 10
    VTK_HEXAHEDRON = 12
    VTK_QUADRATIC_TETRA = 24
    VTK_QUADRATIC_HEXAHEDRON = 25
    VTK_TRIQUADRATIC_HEXAHEDRON = 29  # also handled by corners

    face_count: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    # value = (count, oriented_face)

    cells = _vtk_cells_iter(cells_flat)
    if len(cells) != len(ctypes):
        raise RuntimeError("cells/ctypes length mismatch.")

    for nodes, ctype in zip(cells, ctypes.astype(int)):
        if ctype in (VTK_TETRA, VTK_QUADRATIC_TETRA):
            if len(nodes) < 4:
                raise RuntimeError("Tetra cell with < 4 nodes.")
            v = nodes[:4]
            faces = [
                (v[0], v[1], v[2]),
                (v[0], v[1], v[3]),
                (v[0], v[2], v[3]),
                (v[1], v[2], v[3]),
            ]
        elif ctype in (
            VTK_HEXAHEDRON,
            VTK_QUADRATIC_HEXAHEDRON,
            VTK_TRIQUADRATIC_HEXAHEDRON,
        ):
            if len(nodes) < 8:
                raise RuntimeError("Hexahedron cell with < 8 nodes.")
            v = nodes[:8]
            # 6 quad faces (VTK hex corner ordering)
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
                # split into two tris keeping local orientation
                faces.append((a, b, c))
                faces.append((a, c, d))
        else:
            raise RuntimeError(
                f"Unsupported 3D VTK cell type {ctype}. "
                "Supported: tetra/hex (including quadratic variants)."
            )

        for f in faces:
            key = tuple(sorted(f))
            cnt, _ = face_count.get(key, (0, f))
            face_count[key] = (cnt + 1, f)

    # Boundary faces are those seen exactly once
    tris = [oriented for (cnt, oriented) in face_count.values() if cnt == 1]
    if not tris:
        raise RuntimeError("No boundary faces found.")
    return np.asarray(tris, dtype=int)


def save_displacement(
    u: dfem.Function,
    out_path: Path,
    *,
    cfg: SaveDisplacementConfig | None = None,
) -> None:
    """Save a publication-quality static plot of a displacement field."""
    cfg = cfg or SaveDisplacementConfig()

    V = u.function_space
    mesh = V.mesh
    gdim = mesh.geometry.dim
    comm = mesh.comm

    if cfg.require_2d and gdim != 2:
        raise ValueError(
            "Only 2D domains are supported by save_displacement (set require_2d=False to override)."
        )

    # Interpolate to linear if needed (ensures node-based mapping for plotting)
    u_lin = _ensure_linear_vector_space(u)

    # VTK mesh and coordinates (node ordering matches geometry dofs for P1)
    cells, cell_types, coords = vtk_mesh(u_lin.function_space)

    # Extract vector dof values at nodes
    u_vals = u_lin.x.array
    if u_vals.size % gdim != 0:
        raise RuntimeError("Unexpected DOF layout for vector function.")
    u_vec = u_vals.reshape((-1, gdim))
    # pick requested component part
    if cfg.part in ("real", "imag"):
        u_vec = _as_part(u_vec, cfg.part)
    elif cfg.part == "abs":
        # keep vector direction for deformation, magnitude for coloring
        pass
    else:
        raise ValueError("cfg.part must be one of: 'real', 'imag', 'abs'")

    # MPI-safe gather to rank 0
    cells_L = comm.gather(cells, root=0)
    ctype_L = comm.gather(cell_types, root=0)
    coords_L = comm.gather(coords, root=0)
    uvec_L = comm.gather(u_vec, root=0)

    if comm.rank != 0:
        return

    # Flatten across ranks
    offsets = np.cumsum([0] + [len(p) for p in coords_L[:-1]])
    all_cells = []
    for off, blk in zip(offsets, cells_L):
        i = 0
        while i < len(blk):
            n = int(blk[i])
            all_cells.append(np.array(blk[i + 1 : i + 1 + n]) + off)
            i += n + 1
    cells_all = np.concatenate(
        [
            np.array([len(c)], dtype=int).repeat(1).tolist() + c.tolist()
            for c in all_cells
        ]
    )
    ctypes_all = np.hstack(ctype_L)
    X = np.vstack(coords_L)
    U = np.vstack(uvec_L)

    if gdim == 3:
        # Deformed coordinates and scalar field
        X_def = X.copy()
        X_def[:, :3] = X_def[:, :3] + cfg.scale * U
        u_mag = np.linalg.norm(U, axis=1)

        # Boundary triangulation
        tri_conn = _boundary_triangles_3d(cells_all, ctypes_all)

        # Build triangle vertex lists for Poly3DCollection
        tris_xyz = [X_def[idx][:, :3] for idx in tri_conn]
        # Face scalar = mean of vertex magnitudes
        face_vals = np.mean(u_mag[tri_conn], axis=1)

    with plt.rc_context(
        {**_PLOT_RC, "savefig.bbox": "standard"}
    ):  # disable tight for this fig
        fig = plt.figure(figsize=(max(cfg.fig_size[0], 5.2), max(cfg.fig_size[1], 4.2)))
        # reserve space on the right for colorbar
        fig.subplots_adjust(right=0.86)  # 0.86 ≈ leave ~14% for cbar
        ax = fig.add_subplot(111, projection="3d")

        # Colormap and normalization
        norm = Normalize(vmin=float(face_vals.min()), vmax=float(face_vals.max()))
        cmap = cm.get_cmap(cfg.cmap)
        colors = cmap(norm(face_vals))

        surf = Poly3DCollection(
            tris_xyz, facecolors=colors, edgecolor="none", linewidths=0.0
        )
        surf.set_antialiased(True)
        ax.add_collection3d(surf)

        if cfg.show_mesh:
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
        if cfg.title:
            ax.set_title(cfg.title, pad=6)

        _beautify_3d_axes(ax)

        # ---- equal aspect with a small padding to avoid left cut ----
        xmin, ymin, zmin = X_def.min(axis=0)[:3]
        xmax, ymax, zmax = X_def.max(axis=0)[:3]
        span = float(np.max([xmax - xmin, ymax - ymin, zmax - zmin]))
        pad = 0.03 * span  # 3% pad on each side prevents cropping
        cx, cy, cz = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
        ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
        ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
        ax.set_zlim(cz - span / 2 - pad, cz + span / 2 + pad)

        # ---- place colorbar in its own axes outside the 3D plot ----
        norm = Normalize(vmin=float(face_vals.min()), vmax=float(face_vals.max()))
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_vals)

        # cleaner view
        ax.view_init(elev=18, azim=-50)

        # save
        ext = out_path.suffix.lower().lstrip(".")
        dpi = cfg.dpi_png if ext == "png" else None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
