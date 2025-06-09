# LSA-FW Solver Module

> [Back to index](_index.md)

---

## Overview

The `Solver` package provides tools to compute base flows and to solve
eigenvalue problems arising in stability analysis.  It builds on PETSc
and SLEPc through thin wrapper classes that expose a user friendly API
without hiding the underlying functionality.

At the moment the focus is on the generalized eigenvalue problem
$A x = \lambda M x$, which is used to compute the global modes of a
linearized Navier–Stokes operator.  Additional utilities include
plotting helpers for visualising eigenvectors.

## Eigensolver API

The high level interface is the `EigenSolver` class which accepts an
`EigensolverConfig` dataclass.  The configuration collects the number of
eigenpairs, tolerance and problem type.  Internally the solver delegates
to `iEpsSolver`, a lightweight wrapper around `slepc4py.EPS`.

```python
from Solver.eigen import EigenSolver, EigensolverConfig, iEpsProblemType

cfg = EigensolverConfig(
    num_eig=6,
    problem_type=iEpsProblemType.GHEP,
    atol=1e-8,
    max_it=500,
)
es = EigenSolver(cfg, A, M)
values_and_vectors = es.solve()
```

The returned list contains `(eigenvalue, eigenvector)` tuples.  The
`iEpsSolver` object can be accessed through `es.solver` to customise
spectral transforms, preconditioners or target shifts.

## Plotting Utilities

`Solver.plot` provides a `plot_eigenvector` function that renders a
solution field using PyVista.  It takes a `Mesher` instance, the finite
element spaces and the eigenvector to plot.  Results can be displayed
interactively or saved to disk as screenshots.

## Future Work

Planned extensions include iterative Newton solvers for the steady base
flow computation and time-stepping capabilities for transient
simulations.

