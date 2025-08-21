# LSA-FW Eigenvalue Solver

> [Back to Solver Index](solver.md)

---

## Introduction

The eigensolver provides scalable routines for large sparse eigenvalue problems arising in linear stability analysis.
It targets both standard and generalized problems and is built on top of SLEPc (via `slepc4py`) and PETSc.

It integrates directly with matrices assembled in the FEM module (e.g. linearized Navier–Stokes operators), and exposes a light, composable API for:

* Spectral targeting (focus on eigenvalues near a region of interest)
* Spectral transforms (shift, shift-and-invert, Cayley, polynomial filtering)
* Preconditioning for accelerated convergence

This design allows the extraction of unstable eigenmodes from large-scale systems (up to millions of degrees of freedom) with efficient Krylov subspace methods.

## Problem Statement

We consider the generalized eigenvalue problem (EVP):

$$
\mathbf{A} \mathbf{x} = \lambda \mathbf{B} \mathbf{x},
$$

where $\mathbf{A}, \mathbf{B} \in \mathbb{C}^{n \times n}$ are sparse operators,
$\lambda \in \mathbb{C}$ is the eigenvalue, and $\mathbf{x} \in \mathbb{C}^n$ the corresponding eigenvector.

### Common problem types

SLEPc classifies problems into several types, depending on the structure of $\mathbf{A}$ and $\mathbf{B}$:

* HEP (Hermitian Eigenvalue Problem): $\mathbf{A} = \mathbf{A}^*$, $\mathbf{B} = I$.
* NHEP (Non-Hermitian Eigenvalue Problem): $\mathbf{A} \neq \mathbf{A}^*$, $\mathbf{B} = I$.
* GHEP (Generalized Hermitian Eigenvalue Problem): $\mathbf{A} = \mathbf{A}^*$, $\mathbf{B} = \mathbf{B}^* > 0$.
* GNHEP (Generalized Non-Hermitian Eigenvalue Problem): $\mathbf{A} \neq \mathbf{A}^*$, $\mathbf{B} = \mathbf{B}^* > 0$.
* GHIEP (Generalized Hermitian Indefinite Eigenvalue Problem): $\mathbf{A} = \mathbf{A}^*$, $\mathbf{B} = \mathbf{B}^*$, but $\mathbf{B}$ is indefinite.
* PGNHEP (Polynomial Generalized Non-Hermitian Eigenvalue Problem): $\mathbf{A}$ is replaced by a polynomial operator in $\lambda$.

Hermitian problems admit more robust solvers (Lanczos, LOBPCG) with guaranteed orthogonality properties, while non-Hermitian problems (e.g. convection–diffusion, Navier–Stokes linearization) require more general methods (Arnoldi, Krylov–Schur).

## Spectral Transform and Targeting

Large sparse eigenvalue problems rarely allow computation of the full spectrum.
Instead, the goal is to compute only a subset of eigenvalues, typically those with the largest real part (stability), largest magnitude (dominance), or closest to a physical frequency/shift.

### Spectral Targeting

By default, Krylov methods tend to converge to dominant eigenvalues (e.g., largest in magnitude).
To access interior eigenvalues, one introduces spectral transforms:

$$
T(\mathbf{A}) \mathbf{x} = \mu \mathbf{x},
$$

where eigenvalues of $T(\mathbf{A})$ are related to those of $\mathbf{A}$, but reordered or mapped to favor convergence in the desired spectral region.

### Shifting

A spectral shift modifies the operator:

$$
\mathbf{A}_\sigma = \mathbf{A} - \sigma \mathbf{I},
$$

so that eigenvalues are translated by $-\sigma$:

$$
\lambda_\sigma = \lambda - \sigma.
$$

This does not change the eigenvectors, but allows us to 'nudge' the spectrum.
For example, to extract eigenvalues near $\sigma = 0$, one shifts so that these eigenvalues are close to the origin.
Krylov methods converge faster when the target eigenvalues dominate in the shifted spectrum.

### Shift-and-Invert (SINVERT)

The shift-and-invert transform is one of the most powerful tools for interior eigenvalues.
It defines the operator:

$$
T(\mathbf{A}) = (\mathbf{A} - \sigma \mathbf{I})^{-1}.
$$

The mapping of eigenvalues is:

$$
\mu = \frac{1}{\lambda - \sigma}.
$$

Hence, eigenvalues $\lambda$ close to the shift $\sigma$ are mapped to large magnitudes $|\mu|$.
Krylov methods, which converge to dominant eigenvalues, now preferentially extract those near $\sigma$.

This transform requires solving linear systems of the form:

$$
(\mathbf{A} - \sigma \mathbf{I}) \mathbf{y} = \mathbf{v},
$$

which is computationally expensive but manageable with efficient preconditioners (e.g., ILU, multigrid).
The accuracy of the inner linear solver is critical, since it directly influences eigenvalue accuracy.

### Cayley Transform

The Cayley transform provides a rational mapping

$$
T(\mathbf{A}) = (\mathbf{A} - \sigma \mathbf{I})^{-1} (\mathbf{A} + \sigma \mathbf{I}),
$$

with eigenvalue mapping

$$
\mu = \frac{\lambda + \sigma}{\lambda - \sigma}.
$$

This improves spectral separation in some cases (e.g., symmetric indefinite problems), but at the cost of solving two shifted systems per iteration.
It is less common in CFD contexts but useful in indefinite systems or highly clustered spectra.

### Polynomial Filters

Instead of rational transforms, one can apply polynomial filtering:

$$
T(\mathbf{A}) = p(\mathbf{A}),
$$

where $p(\lambda)$ is designed so that $p(\lambda) \approx 1$ near the region of interest and $p(\lambda) \approx 0$ elsewhere.
Chebyshev polynomials are a common choice due to their minimax property and stability.
Polynomial filters avoid inner solves but typically require higher Krylov dimensions.

### Preconditioned Transforms

In practice, spectral transforms are combined with preconditioning, i.e., approximating the inverse action with a solver tailored to the operator. For example:

$$
(\mathbf{A} - \sigma \mathbf{I})^{-1} \approx M^{-1},
$$

where $M^{-1}$ is a preconditioner.
The combination allows one to retain fast Krylov convergence while avoiding exact factorization of large sparse matrices.

### Selection Policies

SLEPc provides multiple policies to decide which eigenvalues are computed after transformation:

* `LARGEST_MAGNITUDE` – dominant eigenvalues in modulus.
* `LARGEST_REAL` / `SMALLEST_REAL` – extremal real parts (stability).
* `TARGET_REAL` – closest to a given real shift.
* `TARGET_IMAGINARY` – closest to a given imaginary shift.
* Interval/rectangle selection – eigenvalues inside a real interval or a complex box.

These are combined with spectral transforms to enable efficient extraction of physical modes in stability analysis.

## Practical considerations

* Hermitian checks: when a Hermitian problem type is requested, the solver can warn if $A$ or $B$ are not numerically Hermitian.
This requires symmetry or conjugate-symmetry probing and can be expensive on large operators, so it is optional (enabled by default and can be disabled with `check_hermitian=False`).
* Real vs complex builds: with a PETSc real build, complex eigenvectors arrive as separate real and imaginary parts; with a complex build they arrive as a single complex vector.
The wrapper `iComplexPETScVector` abstracts this.
* Preconditioning: for shift-invert, configure the spectral transform's inner KSP and PC (for example, LU, ICC, GAMG).
LU is robust but memory intensive; ICC/ILU are lighter but more problem dependent.
* Dimensions and tolerances: control the number of eigenpairs (`nev`), absolute tolerance, and iteration limits.
Subspace size can be tuned via SLEPc options if needed.
* Diagnostics: after `solve()`, the number of converged eigenpairs and, when available, inner iteration counts are reported.

## API reference

### EigensolverConfig

```python
@dataclass(frozen=True)
class EigensolverConfig:
    num_eig: int = 25
    problem_type: iEpsProblemType = iEpsProblemType.GNHEP
    atol: float = 1e-8
    max_it: int = 500
```

Supported problem types: `HEP`, `NHEP`, `GHEP`, `GNHEP`, `PGNHEP`, `GHIEP`.

### EigenSolver

```python
EigenSolver(
    cfg: EigensolverConfig,
    A: iPETScMatrix,
    M: iPETScMatrix | None = None,
    *,
    check_hermitian: bool = True,
)
```

Constructs an eigensolver on matrices $A$ (and optional $M$), validates shapes, and optionally warns on Hermitian mismatches.

## Usage examples

### Basic generalized solve with target on the real axis

```python
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import iEpsProblemType, iEpsWhich
from FEM.utils import iPETScMatrix

A = iPETScMatrix.from_path("cases/cylinder/matrices/re_50/A.mtx")
M = iPETScMatrix.from_path("cases/cylinder/matrices/re_50/M.mtx")
A.assemble(); M.assemble()

cfg = EigensolverConfig(num_eig=6, problem_type=iEpsProblemType.GNHEP, atol=1e-8, max_it=500)
es = EigenSolver(cfg, A, M)
es.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)
es.solver.set_target(0.1)

pairs = es.solve()
for k, (lam, vec) in enumerate(pairs, 1):
    print(f"[{k}] λ = {lam}")
```

### Shift-and-invert with preconditioning

```python
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import iEpsProblemType, iSTType, PreconditionerType, iEpsWhich

cfg = EigensolverConfig(num_eig=6, problem_type=iEpsProblemType.GHEP, atol=1e-8, max_it=500)
es = EigenSolver(cfg, A, M)

# Target interior eigenvalues near sigma = 0.5
es.solver.set_st_type(iSTType.SINVERT)
es.solver.set_target(0.5)

# Preconditioned ST inner KSP
es.solver.set_st_pc_type(PreconditionerType.ICC)
es.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)

pairs = es.solve()
for idx, (val, vec) in enumerate(pairs, start=1):
    print(f"Eigenvalue {idx} = {val}")
```

### Interval and rectangle selection

```python
from Solver.utils import iEpsWhich

es.solver.set_which_eigenpairs(iEpsWhich.ALL)
es.solver.set_interval(a=0.0, b=1.0)  # Real interval [0,1]
# Or, for complex rectangle:
# es.solver.set_interval_complex(a=0.0, b=1.0, c=-0.2, d=0.2)
pairs = es.solve()
```

### Polynomial filtering

```python
from Solver.utils import iSTType, iEpsWhich

# User can configure FILTER via PETSc/SLEPc options (degree, window) and combine with which/target
es.solver.set_st_type(iSTType.FILTER)
es.solver.set_which_eigenpairs(iEpsWhich.TARGET_MAGNITUDE)
pairs = es.solve()
```
