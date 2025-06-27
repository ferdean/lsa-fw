# LSA-FW FEM Module (Operators)

> [Back to FEM Overview](fem.md)

---

Linear stability analysis of fluid flows requires linearizing the Navier-Stokes equations around a given base flow.
This module provides tools to assemble the discrete matrices for this linearized system.

Base flows can be computed with the `SteadyNavierStokesAssembler` described in
the [Steady Base Flow Solver](fem.md#steady-base-flow-solver) section.

The module supports both time-stepping simulations and eigenanalysis for stabilities, and it provides PETSc features for parallel assembly, block matrix storage and preconditioning.

## Variational Forms for Linearized Navier-Stokes

The base flow, assumed divergence-free, satisfying the steady Navier-Stokes equations at Reynolds $Re$ is considered known, and thus, an input to the module.
Accordingly, the study of the linearized problem consists on the study of small perturbations on the flow, so that 

$$
 \left\lbrace
 \begin{aligned}
    &\mathbf{u}(\mathbf{x}, t) = \overline{\mathbf{u}}(\mathbf{x}) + \mathbf{u}'(\mathbf{x}, t)\\
    &p(\mathbf{x}, t) = \overline{p}(\mathbf{x}) + p'(\mathbf{x}, t)
\end{aligned}
\right.,
$$

where $\overline{\mathbf{u}}, \overline{p}$ represent the base flow, and $\mathbf{u}', p'$ represent the perturbations.

The linearized Navier-Stokes equations for the perturbation are defined as 

$$
\left\lbrace
\begin{aligned}
    s\mathbf{u}' + (\overline{\mathbf{u}}\cdot\nabla)\mathbf{u}' + (\mathbf{u}'\cdot\nabla)\overline{\mathbf{u}} &= -\nabla p' + \frac{1}{Re}\,\Delta \mathbf{u}' \\
    \nabla\cdot \mathbf{u}' &= 0
\end{aligned}
\right.,
$$

where $s$ is the growth rate in an eigenanalysis.
For time evolution, one would have $\partial_t \mathbf{u}'$ instead.

Each term in these equations corresponds to a bilinear form that will be assembled into a matrix:

| Term                        | Expression                                  | Description                                                                                  |
|-----------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------|
| Growth rate term            | $s\mathbf{u}'$                             | Represents temporal evolution; in eigenanalysis, $s$ is the spectral growth rate; in time-dependent simulations, it corresponds to $\partial_t \mathbf{u}'$. |
| Convection term             | $(\overline{\mathbf{u}}\cdot\nabla)\mathbf{u}'$       | Advection of the perturbation by the base flow; transports momentum downstream.              |
| Shear term                  | $(\mathbf{u}'\cdot\nabla)\overline{\mathbf{u}}$       | Interaction of the perturbation with base flow gradients; captures the effect of base flow shear on perturbation dynamics. |
| Pressure gradient term      | $-\nabla p'$                               | Pressure force acting on the perturbation velocity field; contributes to acceleration and enforces incompressibility. |
| Viscous diffusion term      | $\frac{1}{Re}\,\Delta \mathbf{u}'$         | Represents momentum diffusion due to viscosity; forms a symmetric positive semi-definite operator. |
| Divergence (continuity) term| $\nabla\cdot \mathbf{u}'$                  | Enforces incompressibility of the perturbation velocity field; couples velocity and pressure in a saddle-point system. |

In a finite element context, suitable [function spaces](fem-spaces.md) are chosen.
From now on in this document, $V$ is used for the velocity (vector) field, and $Q$ is used for the pressure (scalar) field. 

Let $\mathbf{v}$ be a test function in $V$ and $q$ be a test function in $Q$.
Then, the bilinear forms associated with each term in the linearized Navier–Stokes equations are defined below.
These forms are assembled into discrete operators acting on the velocity and pressure function spaces.
Certainly. Here's your section reordered to match the order in which the terms appear in both the equations and the explanatory table:

### Mass Form

$$
m(\mathbf{u}, \mathbf{v}) = \int_\Omega \mathbf{u} \cdot \mathbf{v}\,dx
$$

This bilinear form defines the velocity mass matrix $M_v$, which is block-diagonal in many finite element bases.
It is used to represent the $s\mathbf{u}$ term in eigenproblems or $\partial_t \mathbf{u}$ in time evolution.
Since pressure does not evolve in time, the full mass matrix is block-diagonal with zero entries in the pressure-pressure block.

### Convection Form

$$
a_{\text{conv}}(\mathbf{u}, \mathbf{v}) = \int_\Omega ((\overline{\mathbf{u}}\cdot\nabla)\mathbf{u}) \cdot \mathbf{v}\,dx
$$

This non-symmetric form contributes to the convection matrix $C_1$.

### Shear Form

$$
a_{\text{shear}}(\mathbf{u}, \mathbf{v}) = \int_\Omega ((\mathbf{u}\cdot\nabla)\overline{\mathbf{u}}) \cdot \mathbf{v}\,dx
$$

This form defines the shear matrix $C_2$.

### Pressure Coupling Forms

From the pressure gradient and continuity terms, integration by parts leads to:

$$
b(\mathbf{v}, p) = -\int_\Omega p\,(\nabla\cdot \mathbf{v})\,dx,
\qquad
b^T(\mathbf{u}, q) = \int_\Omega q\,(\nabla\cdot \mathbf{u})\,dx
$$

These define the gradient operator $G$ and divergence operator $D$, coupling pressure and velocity in the saddle-point structure.
For compatible function spaces, the discrete operators satisfy $D = -G^T$ up to a sign convention.

### Viscous Form

$$
a_{\text{visc}}(\mathbf{u}, \mathbf{v}) = \frac{1}{Re}\int_\Omega \nabla \mathbf{u} : \nabla \mathbf{v}\,dx
$$

Here, the colon '$:$' denotes the Frobenius inner product (or double contraction) between the velocity gradient tensors, defined as

$$
\nabla \mathbf{u} : \nabla \mathbf{v} = \sum_{i,j} \frac{\partial u_i}{\partial x_j} \frac{\partial v_i}{\partial x_j}
$$

This symmetric bilinear form yields the diffusion (stiffness) matrix for the velocity field, modeling viscous dissipation via the Laplacian operator.

### Physical Interpretation

The viscous term $a_{\text{visc}}$ yields a symmetric positive semi-definite stiffness matrix that models momentum diffusion and dissipative effects.
The convection and shear terms, $a_{\text{conv}}$ and $a_{\text{shear}}$, collectively define the linearized advection operator, and are responsible for non-self-adjoint behavior when the base flow is nonzero.

The pressure and continuity terms enforce the incompressibility constraint, coupling velocity and pressure through a saddle-point system with a nullspace in pressure.
The mass form $m$ accounts for temporal evolution or eigenvalue scaling and is essential for both time-marching and spectral stability analyses.

## Discrete System Structure

The variational forms described above are assembled into a block matrix system that captures the dynamics of the linearized Navier–Stokes equations in mixed velocity-pressure form.

The resulting algebraic system reads

$$
\begin{bmatrix}
    s\mathbf{M} + \mathbf{C}_1 + \mathbf{C}_2 + \mathbf{B} & \mathbf{G} \\
    \mathbf{D} & 0
\end{bmatrix}
\begin{bmatrix}
    \mathbf{u} \\
    p
\end{bmatrix}
=
\begin{bmatrix}
    0 \\
    0
\end{bmatrix}.
$$

This system arises in both eigenvalue problems (where $s$ is a complex eigenvalue) and time-stepping schemes (where $s = \frac{1}{\Delta t}$ or similar).
Its saddle-point nature reflects the incompressibility condition and requires appropriate solvers and pre-conditioners for efficient numerical solution.

## Implementation

The `operators` module implement the above defined variational forms through the `VariationalForms` collector and exposes an assembler, `LinearizedNavierStokesAssembler`, to compute their associated [PETSc](https://petsc.org/release/) matrices.

All bilinear forms are implemented as static methods, each returning a variational form suitable for FEniCSx assembly routines.

### Assembler

The `LinearizedNavierStokesAssembler` encapsulates the full operator assembly pipeline for the linearized problem.

The core interface is

```python
assembler = LinearizedNavierStokesAssembler(
    base_flow: dolfinx.fem.Function,
    spaces: FunctionSpaces,
    re: float,
    bcs: BoundaryConditions| None
)
```

where

- `base_flow` is a dolfinx function in the velocity space, representing $\overline{\mathbf{u}}$,
- `spaces` is a `FunctionSpaces` container defining velocity and pressure spaces (ref. [fem-spaces](fem-spaces.md)),
- `re` is the Reynolds number, and
- `bcs` is an optional `BoundaryConditions` object (ref. [fem-bcs](fem-bcs.md)).

The assembler provides the following key methods:

- `assemble_mass_matrix()` — Velocity mass matrix
- `assemble_viscous_matrix()` — Viscous (Laplacian) matrix
- `assemble_convection_matrix()` — Base flow convection operator
- `assemble_shear_matrix()` — Shear (base flow gradient) operator
- `assemble_pressure_gradient_matrix()` — Pressure gradient matrix
- `assemble_divergence_matrix()` — Divergence matrix
- `assemble_robin_matrix()` — Optional contribution from Robin boundaries
- `assemble_neumann_rhs(g, ds)` — RHS vector from Neumann boundary data

For the sake of efficiency, all matrices are cached internally on first use.
`LinearizedNavierStokesAssembler` also accepts a `CacheStore` object to persist
assembled matrices to disk using HDF5/XDMF files. The `clear_cache()` method
invalidates the in-memory cache only.

### Discrete System Assembly

#### Full Linear Operator

The `assemble_linear_operator()` method returns the full system operator $A$ as a PETSc nested matrix,

$$
\mathbf{A} = \begin{bmatrix} \mathbf{A}_{uu} & \mathbf{G} \\ \mathbf{D} & 0 \end{bmatrix},
$$

where

- $\mathbf{A}_{uu} = \mathbf{B} + \mathbf{C}_1 + \mathbf{C}_2 + \mathbf{R}$ with diffusion, convection, shear, and optional Robin term,
- $\mathbf{G}$ comes from the pressure gradient, and
- $\mathbf{D}$ comes from the velocity divergence.

### Eigenvalue System

The `assemble_eigensystem()` returns a pair of matrices `(A, M)` such that

$$
\mathbf{A} \mathbf{x} = s \mathbf{M} \mathbf{x}, \quad \text{with } \mathbf{x} = \begin{bmatrix} \mathbf{u} \\ p \end{bmatrix},
$$

where

- $\mathbf{A}$ is the full linear operator, and
- $\mathbf{M}$ is the block mass matrix, such that
  $$
  \mathbf{M} = \begin{bmatrix} \mathbf{M}_u & 0 \\ 0 & 0 \end{bmatrix}
  $$

The velocity mass matrix $\mathbf{M}_u$ is nonzero, while the pressure block is null.

### Nullspace Handling

The `attach_pressure_nullspace(mat: iPETScMatrix)` method attaches a constant pressure nullspace to a PETSc matrix.

This will be necessary for saddle-point systems where pressure is only defined up to a constant.
This function ensures Krylov solvers (e.g. MINRES) handle the system correctly.

## Design Notes

- All forms are assembled with DOLFINx's `assemble_matrix` and respect supplied Dirichlet BCs.
- Block systems are returned as `iPETScMatrix` wrappers over PETSc `MatNest`.
- Matrices are cached internally; use `clear_cache()` to recompute. A
  persistent cache can be enabled by providing a `CacheStore`.
- Integration with SLEPc is supported via `assemble_eigensystem()`.

## Further Notes

### Parallel Assembly and Matrix Structure

The assembly can be made fully parallel over MPI and leverages PETSc structures to preserve sparsity and block structure.

Benefits include

- memory efficiency, as Zero blocks (e.g., pressure-pressure) are never allocated,
- modularity, since each block is handled separately, and 
- preconditioner support, as the design enables `PCFieldSplit` to recognize velocity and pressure fields.

### Using the Assembled Matrices in Eigenvalue Analysis

The assembled `(A, M)` pair from `assemble_eigensystem()` can be directly passed to SLEPc.
A typical usage pattern looks like:

```python
from slepc4py import SLEPc
E = SLEPc.EPS().create(comm)
E.setOperators(A.raw, M.raw)  # iPETScMatrix wrappers
E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
E.setFromOptions()
E.solve()
```

### Using the Assembled Matrices in Time Integration

Instead of eigenvalue analysis, the matrices can be used for time stepping.
For example, applying implicit Euler:

$$
(\mathbf{M} - \Delta t \mathbf{A}) \mathbf{u}_{n+1} = \mathbf{M} \mathbf{u}_n.
$$
