# LSA-FW FEM Module

> [Back to index](_index.md)

---

## FEM Mathematical Formulation

Let $\Omega \subset \mathbb{R}^d$ be a bounded domain with Lipschitz boundary $\partial\Omega$ in a $d$-dimensional space, where $d \in \mathbb{N}$ is the spatial dimension. 
Consider a partial differential equation expressed in weak (variational) form, such as

$$
\text{Find } u \in V \text{ such that } a(u, v) = f(v) \quad \forall v \in V,
$$

where

- $V$ is a Hilbert space, often a Sobolev space such as $H^1(\Omega)$, $[H^1(\Omega)]^d$, or $H(\text{div}; \Omega)$,
- $a(\cdot, \cdot)$ is a continuous bilinear form representing the weak form of the differential operator,
- $f \in V'$ is a continuous linear functional representing the forcing terms or boundary data.

In this formulation, $V$ serves both as the trial space (the space of unknown solutions $u$) and the test space (the space of admissible variations $v$).
In symmetric formulations, the trial and test spaces coincide; for more general or mixed formulations, one may use distinct spaces $V$ and $W$:

$$
\text{Find } u \in V \text{ such that } a(u, w) = f(w) \quad \forall w \in W.
$$

To approximate this infinite-dimensional problem, the finite element method restricts the solution space to a finite-dimensional subspace $V_h \subset V$, defined over a mesh (or triangulation) $\mathcal{T}_h$ of $\Omega$:

$$
\text{Find } u_h \in V_h \text{ such that } a(u_h, v_h) = f(v_h) \quad \forall v_h \in V_h.
$$

Here, $u_h$ is the finite element approximation of the exact solution $u$, and the problem reduces to solving a finite system of algebraic equations.
The triangulation $\mathcal{T}_h$ partitions $\Omega$ into simple elements (e.g., triangles, quadrilaterals, tetrahedra), over which polynomial basis functions are defined.

Note that the term "triangulation" refers to the process of partitioning the domain $\Omega$ into smaller, simpler subdomains (elements) over which the solution is approximated.
Despite the name, a triangulation does not have to consist of triangles only (refer to [Meshing](meshing.md) for details).

More precisely, the triangulation $\mathcal{T}_h$ is a subdivision of $\Omega$ into elements $K \subset \mathbb{R}^d$, such that:

- the elements are non-overlapping,
- the intersection of any two elements is either empty, a shared vertex, a shared edge, or a shared face, and
- the elements cover the closure of $\Omega$.

> See: Ciarlet (2002), Ern & Guermond (2004), Brenner & Scott (2008) for foundational references.

## Implementation

The `FEM` module provides function space definitions, boundary condition configuration, and variational form support for incompressible Navier–Stokes problems in the LSA-FW framework.

It is built on top of FEniCSx and supports modular, extensible finite element formulations with a focus on clean APIs.

## Module Structure

| File               | Purpose                                                             |
|--------------------|---------------------------------------------------------------------|
| `spaces.py`        | Function space definitions for velocity and pressure                |
| `bcs.py`           | Dirichlet, Neumann, and Robin boundary condition specification      |
| `utils.py`         | Shared helpers and enums                                            |
| `operators.py`     | Linearized and steady Navier--Stokes assemblers |
| `plot.py`          | Utils for matrix (sparsity) visualization                           |

## Submodules Index

- [Function spaces](fem-spaces.md)
- [Boundary conditions](fem-bcs.md)
- [Linearized operators](fem-operators.md)
- [Steady base flow solver](#steady-base-flow-solver)

## Steady Base Flow Solver

`SteadyNavierStokesAssembler` implements a Newton iteration for the steady
incompressible Navier–Stokes equations. It constructs the residual and Jacobian
forms and exposes a `solve()` method that returns the converged velocity and
pressure fields.

## Future Extensions

- Support for Discontinuous Galerkin (DG)
- Time-dependent boundary conditions
- Built-in stabilized equal-order elements
