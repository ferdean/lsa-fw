"""Tests for Solver.eigen.py"""

import pytest
import logging
import numpy as np
from petsc4py import PETSc

from FEM.utils import iPETScMatrix, iPETScVector
from Solver.eigen import EigenSolver, EigensolverConfig, iEpsProblemType
from Solver.utils import iEpsSolver, iEpsWhich, iSTType, PreconditionerType


@pytest.fixture(autouse=True)
def capinfo(caplog) -> pytest.LogCaptureFixture:
    """Fixture to capture and assert on log warnings."""
    caplog.set_level(logging.WARNING)
    return caplog


@pytest.fixture
def diagonal_matrix() -> iPETScMatrix:
    """Define a diagonal 3×3 matrix."""
    A = iPETScMatrix.zeros((3, 3))
    A[0, 0] = 1.0
    A[1, 1] = 1.5
    A[2, 2] = -42.0
    A.assemble()
    return A


@pytest.fixture
def identity_matrix() -> iPETScMatrix:
    """Define a 3x3 identity matrix."""
    eye = iPETScMatrix.zeros((3, 3))
    for i in range(3):
        eye[i, i] = 1.0
    eye.assemble()
    return eye


@pytest.fixture
def config_hermitian() -> EigensolverConfig:
    """Configuration for a generalized Hermitian eigenproblem (GHEP)."""
    return EigensolverConfig(
        num_eig=3, problem_type=iEpsProblemType.GHEP, atol=1e-3, max_it=100
    )


@pytest.fixture
def config_nonhermitian() -> EigensolverConfig:
    """Configuration for a generalized non-Hermitian eigenproblem (GNHEP)."""
    return EigensolverConfig(
        num_eig=2, problem_type=iEpsProblemType.GNHEP, atol=1e-6, max_it=100
    )


@pytest.fixture
def config_spd() -> EigensolverConfig:
    """Configuration for a standard Hermitian eigenproblem (HEP) on SPD matrix."""
    return EigensolverConfig(
        num_eig=5, problem_type=iEpsProblemType.HEP, atol=1e-8, max_it=200
    )


@pytest.fixture
def random_spd_matrix() -> iPETScMatrix:
    """Define a random 5x5 symmetric positive-definite matrix."""
    rs = np.random.RandomState(42)
    X = rs.randn(5, 5)
    A_np = X.T @ X + np.eye(5) * 1e-3
    # build PETSc matrix from NumPy array
    A = iPETScMatrix.zeros(A_np.shape)
    for i in range(5):
        for j in range(5):
            A[i, j] = float(A_np[i, j])
    A.assemble()
    return A


def test_invalid_constructor(diagonal_matrix: iPETScMatrix) -> None:
    """Test that initializing iEpsSolver with only M raises a ValueError."""
    with pytest.raises(ValueError):
        iEpsSolver(M=diagonal_matrix)


def test_config_and_solver_properties(
    diagonal_matrix: iPETScMatrix, config_hermitian: EigensolverConfig
) -> None:
    """Test that EigenSolver propagates config correctly into the raw EPS solver."""
    es = EigenSolver(config_hermitian, A=diagonal_matrix)
    assert es.config is config_hermitian

    solver = es.solver
    assert isinstance(solver, iEpsSolver)

    tol, max_it = solver.raw.getTolerances()
    assert tol == config_hermitian.atol
    assert max_it == config_hermitian.max_it

    nev, _, _ = solver.raw.getDimensions()
    assert nev == config_hermitian.num_eig

    assert solver.raw.getProblemType() == config_hermitian.problem_type.to_slepc()


def test_dummy_solver(
    config_hermitian: EigensolverConfig, diagonal_matrix: iPETScMatrix
) -> None:
    """Test that the solver finds eigenvalues of a diagonal matrix (Ax=λx)."""
    es = EigenSolver(config_hermitian, A=diagonal_matrix)

    pairs = es.solve()
    found = sorted(val for val, _ in pairs)

    expected = sorted([1.0, 1.5, -42.0])
    assert found == pytest.approx(expected, abs=config_hermitian.atol)


def test_dummy_generalized_solver(
    config_hermitian: EigensolverConfig,
    diagonal_matrix: iPETScMatrix,
    identity_matrix: iPETScMatrix,
) -> None:
    """Test generalized solve Ax=λIx gives same result as standard solve."""
    es = EigenSolver(config_hermitian, A=diagonal_matrix, M=identity_matrix)
    found = sorted(val for val, _ in es.solve())
    expected = sorted([1.0, 1.5, -42.0])
    assert found == pytest.approx(expected, abs=config_hermitian.atol)


def test_nonhermitian_generalized(config_nonhermitian: EigensolverConfig) -> None:
    """Test that solver handles a non-Hermitian 2x2 Jordan block correctly."""
    A = iPETScMatrix.zeros((2, 2))
    A[0, 0], A[0, 1], A[1, 0], A[1, 1] = 1.0, 1.0, 0.0, 1.0
    A.assemble()

    es = EigenSolver(config_nonhermitian, A=A)
    vals = sorted(val.real for val, _ in es.solve())

    assert vals == pytest.approx([1.0, 1.0], abs=config_nonhermitian.atol)


def test_smallest_magnitude_selection(
    diagonal_matrix: iPETScMatrix, config_hermitian: EigensolverConfig
) -> None:
    """Test that selecting smallest-magnitude eigenpairs works correctly."""
    es = EigenSolver(config_hermitian, A=diagonal_matrix)

    es.solver.set_which_eigenpairs(iEpsWhich.SMALLEST_MAGNITUDE)
    es.solve()
    found = sorted(val for val, _ in es.solver.get_all_eigenpairs_up_to(2))

    assert found == pytest.approx([1.0, 1.5], abs=config_hermitian.atol)


def test_warning_on_non_hermitian(
    diagonal_matrix: iPETScMatrix,
    config_hermitian: EigensolverConfig,
    capinfo: pytest.LogCaptureFixture,
) -> None:
    """Test that non-Hermitian A with Hermitian config logs a warning."""
    diagonal_matrix[0, 1] = 0.1
    diagonal_matrix.assemble()

    _ = EigenSolver(config_hermitian, A=diagonal_matrix)

    assert any("assumes Hermitian A" in rec.getMessage() for rec in capinfo.records)


def test_eigenvector_size(
    diagonal_matrix: iPETScMatrix, config_nonhermitian: EigensolverConfig
) -> None:
    """Test that each returned vector has correct dimension."""
    pairs = EigenSolver(config_nonhermitian, A=diagonal_matrix).solve()
    for _, vec in pairs:
        assert isinstance(vec, iPETScVector)
        assert vec.size == diagonal_matrix.shape[0]


def test_random_spd_matches_numpy(
    random_spd_matrix: iPETScMatrix, config_spd: EigensolverConfig
) -> None:
    """Test that solver on SPD matrix matches numpy.linalg.eigvals."""
    vals = sorted(
        float(v.real) for v, _ in EigenSolver(config_spd, A=random_spd_matrix).solve()
    )
    rs = np.random.RandomState(42)
    X = rs.randn(5, 5)
    ref = sorted(np.linalg.eigvalsh(X.T @ X + np.eye(5) * 1e-3))
    assert vals == pytest.approx(ref, rel=1e-6)


def test_shift_invert_with_epsilon():
    """Test that small epsilon avoids zero pivot in shift-invert factorization."""
    A = iPETScMatrix.zeros((3, 3))
    base = [1.0, 1.0 + 1e-8, 1.0 + 2e-8]
    eps = 1e-9
    for i, val in enumerate(base):
        A[i, i] = val + eps
    A.assemble()

    cfg = EigensolverConfig(
        num_eig=3, problem_type=iEpsProblemType.HEP, atol=1e-12, max_it=500
    )
    es = EigenSolver(cfg, A=A)
    es.solver.set_st_type(iSTType.SINVERT)
    es.solver.set_target(1.0)
    found = sorted(float(v.real) for v, _ in es.solve())

    assert found == pytest.approx(base, rel=1e-6)


def test_singular_M_errors(diagonal_matrix: iPETScMatrix) -> None:
    """Test that generalized solve with singular M raises a PETSc.Error."""
    M = iPETScMatrix.zeros((3, 3))
    M[0, 0] = 1.0
    M.assemble()
    cfg = EigensolverConfig(
        num_eig=2, problem_type=iEpsProblemType.GHEP, atol=1e-6, max_it=200
    )
    with pytest.raises(PETSc.Error):
        EigenSolver(cfg, A=diagonal_matrix, M=M).solve()


def test_repeated_eigenvalues(diagonal_matrix: iPETScMatrix) -> None:
    """Test that algebraic multiplicity and vector independence hold for repeated eigenvalues."""
    D = diagonal_matrix
    D.zero_all_entries()
    for idx, val in enumerate([2.0, 2.0, 3.0]):
        D[idx, idx] = val
    D.assemble()
    cfg = EigensolverConfig(
        num_eig=3, problem_type=iEpsProblemType.HEP, atol=1e-8, max_it=200
    )

    pairs = EigenSolver(cfg, A=D).solve()
    vals = sorted(float(v.real) for v, _ in pairs)

    near_two = [v for v in vals if abs(v - 2.0) <= cfg.atol]
    assert len(near_two) == 2

    vecs = np.vstack([vec.as_array() for _, vec in pairs]).T
    rank = np.linalg.matrix_rank(vecs)
    assert rank == 3


@pytest.mark.parametrize("pc_type", list(PreconditionerType))
def test_set_st_pc_type_sets_petsc_pc(
    pc_type: PreconditionerType, diagonal_matrix: iPETScMatrix
) -> None:
    """Test that set_st_pc_type correctly sets PC type on KSP."""
    solver = iEpsSolver(A=diagonal_matrix)
    solver.set_problem_type(iEpsProblemType.HEP)
    solver.set_dimensions(number_eigenpairs=3)
    solver.set_tolerances(atol=1e-8, max_it=50)
    solver.set_st_type(iSTType.SINVERT)
    solver.set_target(2.0)

    solver.set_st_pc_type(pc_type)
    pc = solver.raw.getST().getKSP().getPC()
    assert isinstance(pc, PETSc.PC)
    assert pc.getType() == pc_type.name.lower()
