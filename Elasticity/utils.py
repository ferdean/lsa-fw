from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np
from FEM.utils import iComplexPETScVector, iPETScMatrix, iPETScVector
import dolfinx.fem as dfem


def _mat_vec_dot(
    mat: iPETScMatrix, v: iPETScVector, w: iPETScVector | None = None
) -> complex:
    rhs = w or v
    mat_w = mat @ rhs  # iPETScMatrix @ iPETScVector -> iPETScVector
    return v.dot(mat_w)


def _vec_from_array_like(mat: iPETScMatrix, arr: np.ndarray) -> iPETScVector:
    v = mat.create_vector_right()
    v.set_array(np.array(arr, copy=True))
    v.assemble()
    return v


def _clean_eigenvalue(eigenvalue: complex) -> complex:
    er, ei = eigenvalue.real, eigenvalue.imag
    if er < 0 and abs(er) < 1e-6:
        er = 0.0
    if abs(ei) < 1e-3:
        ei = 0.0
    return complex(er, ei)


def _get_freq_from_eigenvalue(eigenvalue: complex) -> tuple[float, float, float]:
    eval = _clean_eigenvalue(eigenvalue)
    wn = math.sqrt(max(eval.real, 0.0))
    fn = wn / (2.0 * math.pi)
    eta_r = (eval.imag / (wn**2)) if wn > 0 else 0.0
    return wn, fn, eta_r


@dataclass(frozen=True)
class Eigenmode:
    """Eigenmode data wrapper."""

    value: complex
    """Raw eigenvalue."""
    function: dfem.Function
    """Raw eigenvector, as a dolfinx function"""
    wn: float
    """Natural frequency, in rad/s."""
    fn: float
    """Natural frequency, in Hz."""
    eta_r: float
    """Modal loss factor. Note that eigenvalue := ωn^2 + i * eta_r * ωn^2."""
    rq_omega2: float
    """Rayleigh quotient (wn^2)."""
    mass_chk: bool
    """Flag indicating validity of the eigenvalue after mass-normalization (v^H M v = 1)."""


def process_modes(
    eigenpairs: Iterable[tuple[float | complex, iComplexPETScVector]],
    stiffness: iPETScMatrix,
    mass: iPETScMatrix,
    function_space: dfem.FunctionSpace,
    *,
    skip_below_hz: float = 0.1,
):
    """Post-process eigen-modes.

    The post-process step includes:
      - Mass-normalization of the eigenvectors
      - Computation of frequencies
      - Sorting by (ascending) frequency
      - Eliminate spurious modes
    """
    out: list[Eigenmode] = []
    for eigval, eigvec in eigenpairs:
        eigvec_real = eigvec.real.as_array()
        v = _vec_from_array_like(stiffness, eigvec_real)

        # Mass-normalize (alpha = 1/sqrt(v^H M v))
        v_mass_v = _mat_vec_dot(mass, v, v)
        alpha = (
            1.0 / math.sqrt(float(np.real(v_mass_v)))
            if np.real(v_mass_v) > 0.0
            else 1.0
        )
        v.scale(alpha)

        # Checks/diagnostics with the normalized vector
        v_mass_v_norm = _mat_vec_dot(mass, v, v)  # ~ 1
        v_stiffness_v = _mat_vec_dot(stiffness, v, v)  # ~ ω_n^2

        u = dfem.Function(function_space)
        u.x.array[:] = v.as_array().real

        wn, fn, eta_r = _get_freq_from_eigenvalue(eigval)
        if fn < skip_below_hz:
            continue

        out.append(
            Eigenmode(
                value=eigval,
                function=u,
                wn=wn,
                fn=fn,
                eta_r=eta_r,
                rq_omega2=float(np.real(v_stiffness_v)),
                mass_chk=bool(np.isclose(v_mass_v_norm, 1)),
            ),
        )

    # Sort by frequency
    out.sort(key=lambda m: m.fn)
    return out
