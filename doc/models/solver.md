# LSA-FW Solver Module

> [Back to index](_index.md)

---

## Solver Formulation

Many problems in fluid dynamics and stability analysis reduce, after finite element discretization, to the solution of algebraic systems of equations.
These include:

* Linear systems of the form $\mathbf{A} \mathbf{x} = \mathbf{b},$  where $\mathbf{A} \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$) is sparse and $\mathbf{b}$ is a known right-hand side.

* Nonlinear systems of the form $F(x) = 0,$ where $F: \mathbb{R}^n \to \mathbb{R}^n$ is typically derived from discretized PDE residuals.
Such systems are solved using iterative Newton-type methods, which require repeated linear solves.

* Generalized eigenvalue problems of the form $\mathbf{A} \mathbf{x} = \lambda \mathbf{B x},$ with $\mathbf{A, B} \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$), where $\mathbf{B}$ is often symmetric positive-definite.
These arise in linear stability analysis, where $\lambda$ represents growth rates or frequencies of perturbations.

The `Solver` module provides a unified interface to these problem classes, built on **PETSc** (for scalable linear and nonlinear solvers) and **SLEPc** (for large-scale eigenvalue problems).

## Implementation

The `Solver` package acts as the computational backbone of LSA-FW.
It integrates with finite element operators assembled in the FEM module and provides robust, configurable solution strategies.

It leverages

* **PETSc** Krylov solvers and preconditioners for efficient sparse linear algebra,
* **PETSc SNES** (Scalable Nonlinear Equations Solvers) for Newton iterations, and
* **SLEPc EPS** (Eigenvalue Problem Solvers) for large sparse eigenvalue problems.

## Module Structure

| File                             | Purpose                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `baseflow.py`                    | Steady baseflow solver (Newton iterations for incompressible Navier–Stokes) |
| `linear.py`                      | Wrappers for PETSc and SciPy linear solvers                                  |
| `nonlinear.py` / `nonlinear2.py` | Interfaces for Newton-type nonlinear problem solvers                         |
| `eigen.py`                       | Generalized eigensolver based on SLEPc                                       |
| `utils.py`                       | Shared solver utilities, enumerations, and helper classes                    |

## Submodules Index

* [Baseflow solver](solver-baseflow.md)
* [Linear solvers](solver-linear.md)
* [Nonlinear solvers](solver-nonlinear.md)
* [Eigenvalue solvers](solver-eigen.md)

## Public API and Usage Examples

### Python Scripting

The `Solver` package can be used directly in Python to compute steady baseflows, assemble linear stability matrices, and solve eigenvalue problems.

#### Steady Baseflow

```python
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import define_bcs, BoundaryCondition
from Meshing.core import Mesher
from Meshing.geometries import Geometry
from config import load_cylinder_flow_config, load_bc_config
from Solver.baseflow import BaseFlowSolver

# Load geometry and boundary condition configs
geo_cfg = load_cylinder_flow_config("...")
bc_cfgs = [BoundaryCondition.from_config(cfg) 
           for cfg in load_bc_config("...")]

# Generate mesh
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, geo_cfg)

# Define spaces and boundary conditions
spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
bcs = define_bcs(mesher, spaces, bc_cfgs)

# Solve steady baseflow at Re=80
solver = BaseFlowSolver(spaces, bcs=bcs)
solution = solver.solve(re=80.0, steps=3)  # Contains velocity and pressure fields
```

This script computes the steady Navier–Stokes baseflow around a cylinder at Reynolds number 80.

#### Eigenvalue Problem

```python
from Solver.eigen import EigenSolver, EigensolverConfig
from Solver.utils import iEpsProblemType, iEpsWhich, iSTType, PreconditionerType
from FEM.utils import iPETScMatrix

# Load (assembled) matrices
A = iPETScMatrix.from_path("...")
M = iPETScMatrix.from_path("...")
A.assemble()
M.assemble()

# Configure eigensolver
es_cfg = EigensolverConfig(
    num_eig=4,
    problem_type=iEpsProblemType.PGNHEP,
    atol=1e-6,
    max_it=500,
)
es = EigenSolver(es_cfg, A, M, check_hermitian=False)
es.solver.set_which_eigenpairs(iEpsWhich.TARGET_REAL)
es.solver.set_st_pc_type(PreconditionerType.LU)
es.solver.set_st_type(iSTType.SINVERT)
es.solver.set_target(0.0)

# Solve and export eigenpairs
eigenpairs = es.solve()
for idx, (eigval, eigvec) in enumerate(eigenpairs, start=1):
    print(f"Eigenvalue {idx} = {eigval}")
    eigvec.real.export("...")
```

This script loads stability matrices, configures a shift-invert eigensolver, and computes unstable eigenvalues and eigenmodes.

### CLI Usage

All solver functionality is also accessible from the command line.
Running with `--help` shows all options:

```bash
python -m Solver --help
```

Currently, the CLI provides one subcommand: `baseflow`.

#### Steady Baseflow (`baseflow`)

```bash
python -m Solver -p baseflow \
    --geometry cylinder_flow \
    --config config_files/2D/cylinder/cylinder_flow.toml \
    --facet-config config_files/2D/cylinder/mesh_tags.toml \
    --bcs config_files/2D/cylinder/bcs.toml \
    --re 80 --steps 3
```

This command generates the cylinder-flow mesh, applies boundary conditions, and computes the steady Navier–Stokes solution at \$Re=80\$.
The run can be parallelised with MPI:

```bash
mpirun -n 4 python -m Solver -p baseflow --geometry cylinder_flow --config ...
```

Output (mesh, baseflow solution, and logs) can be cached with `--output-path`.
