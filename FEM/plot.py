"""LSA-FW FEM plotter.

Provides static visualization of PETSc matrix sparsity.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix  # type:ignore[import-untyped]

from .utils import iPETScMatrix


def spy(matrix: iPETScMatrix, out_path: Path, dpi: int = 300) -> None:
    """Plot the sparsity pattern of an iPETScMatrix as a static PNG."""
    if not _is_root(matrix):
        return

    csr = _to_csr(matrix)
    _save_plot(csr, out_path.with_suffix(".png"), dpi)


def _is_root(matrix: iPETScMatrix) -> bool:
    return matrix.comm.tompi4py().rank == 0


def _to_csr(matrix: iPETScMatrix) -> csr_matrix:
    """Convert an AIJ-type iPETScMatrix to SciPy CSR (rank 0 only)."""
    if "aij" not in matrix.type.lower():
        raise NotImplementedError("Only AIJ-type matrices can be converted.")

    mat = matrix.raw
    m, n = matrix.shape
    i0, i1 = mat.getOwnershipRange()

    # Local CSR pieces
    indptr = [0]
    indices = []
    data = []

    for row in range(i0, i1):
        cols, vals = mat.getRow(row)
        indices.extend(cols)
        data.extend(vals)
        indptr.append(len(indices))

    # Gather to root
    comm = mat.comm.tompi4py()
    gathered = comm.gather((data, indices, indptr), root=0)
    if comm.rank != 0:
        return None

    # Stitch full CSR
    full_data, full_indices, full_indptr = [], [], [0]
    nnz = 0
    for d, i, p in gathered:
        full_data.extend(d)
        full_indices.extend(i)
        full_indptr.extend(nnz + np.array(p[1:]))
        nnz = full_indptr[-1]

    return csr_matrix((full_data, full_indices, full_indptr), shape=(m, n))


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
