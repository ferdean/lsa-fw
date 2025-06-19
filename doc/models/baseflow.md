# Base Flow Solver Implementation

## Mathematical Formulation

This utility module solves the steady, incompressible Navier-Stokes equation in non-dimensional form,

$$
\left\lbrace
\begin{align*}
(\mathbf{u} \cdot \nabla ) \mathbf{u} + \nabla p - \frac{1}{\text{Re}} \Delta\mathbf{u} = \mathbf{0} \\
\nabla \cdot \mathbf{u} = 0
\end{align*}\right.,
$$

subject to the specified boundary conditions. 
Here $\text{Re}$ is the Reynolds number, $\mathbf{u}$ is the velocity field, and $p$ is the pressure field.

Dirichlet conditions impose prescribed velocity or pressure values on the boundaries.
Neumann conditions impose surface forces (such as specified normal or tangential stresses) on the boundaries.
Robin conditions combine aspects of velocity and surface force specifications, often used to model outflow or semi-permeable boundaries.

### Variational Form

The above system is solved for $\mathbf{u}$ and $p$ in the given finite element spaces (e.g., Taylor–Hood, $[P_2]^d - P_1$).

Let $\mathbf{v}$ be a test function in the velocity space and $q$ a test function in the pressure space.
The weak form of the steady Navier–Stokes equations is formulated as 

$$
\text{Find } (\mathbf{u}, p) \text{ such that } \quad \forall (\mathbf{v}, q):
$$

$$
\begin{align*}
\int_\Omega (\mathbf{u} \cdot \nabla \mathbf{u}) \cdot \mathbf{v}\,dx 
+ \frac{1}{\text{Re}}\int_\Omega \nabla \mathbf{u} : \nabla \mathbf{v}\,dx 
-& \int_\Omega p\,(\nabla \cdot \mathbf{v})\,dx 
+ \int_\Omega q\,(\nabla \cdot \mathbf{u})\,dx 
+ \int_{\Gamma_R} \alpha\, (\mathbf{u}\cdot \mathbf{v})\,ds 
= \cdots\\
& \cdots= \int_{\Gamma_N} \mathbf{g}\cdot \mathbf{v}\,ds 
+ \int_{\Gamma_R} \alpha\, (\mathbf{g}\cdot \mathbf{v})\,ds,
\end{align*}
$$

Here’s the updated version of your documentation section, revised carefully to match the **current** solver behavior (iterative GMRES + Hypre setup, not direct LU):

## Nonlinear Solver

Two nonlinear solvers are supported, Newton's method and Piccard's iteration.

- **Newton's Method:**
  The full residual is linearized at each iteration.
  This includes the derivative of the convective term with respect to both the transported and transporting velocity fields, resulting in the full Jacobian.
  Newton's method typically converges quadratically near the solution but may require a good initial guess, especially at high Reynolds numbers.

- **Picard Iteration:**
  This is a fixed-point iteration where the convecting velocity in the nonlinear term $(\mathbf{u}\cdot\nabla)\mathbf{u}$ is lagged.
  Specifically, $(\mathbf{u}^{(k)}\cdot\nabla)\mathbf{u}$ is used at iteration $k$, with $\mathbf{u}^{(k)}$ treated as given and $\mathbf{u}$ as unknown. 
  This simplifies the Jacobian, making the method more robust and ensuring monotonic convergence, although it often converges more slowly than Newton.

Solver parameters such as

- Reynolds number,
- nonlinear solver type,
- convergence tolerance,
- maximum number of iterations,
- linear solver type, and
- preconditioner type

are encapsulated in a configuration file. 
This allows full configurability.

At each nonlinear iteration, the resulting linear system is solved using an iterative PETSc Krylov solver combined with a preconditioner.
The system matrix and residual vector are assembled at every step, and Dirichlet boundary conditions are enforced strongly during assembly.

## API

The solver returns two `dolfinx.fem.Function` objects,

- `u`, with the steady velocity field, and
- `p`, with the steady pressure field.

These belong to the user-specified function spaces and are fully compatible with the perturbation and stability analysis modules.
For example, the `LinearizedNavierStokesAssembler` (refer to [FEM-operators documentation](fem-operators.md)) expects the base velocity field `u` to assemble the perturbation system matrices.

In the project workflow, if no precomputed base flow is available, the code can call `solve_base_flow` automatically. 

### Example

```python
from lib import base_flow

from FEM.spaces import FunctionSpaceType, define_spaces
from FEM.bcs import BoundaryCondition, BoundaryConditions, define_bcs

# Mesh and spaces
mesh = ...
tags = ...
spaces = define_spaces(mesh, FunctionSpaceType.TAYLOR_HOOD)
bc_configs = [...] 
bcs = define_bcs(mesh, spaces, tags, bc_configs)

# Solver configuration
config = base_flow_solver.BaseFlowSolverConfig(
    re=100.0,
    solver_type=base_flow_solver.NonlinearSolverType.NEWTON,
    tol=1e-8,
    max_iterations=100,
    linear_solver_type=base_flow_solver.LinearSolverType.GMRES,
    preconditioner_type=base_flow_solver.PreconditionerType.HYPRE
)

# Solve for base flow
u_base, p_base = base_flow_solver.solve_base_flow(mesh, spaces, bcs, config)
```

After solving, `u_base` can be passed to the `LinearizedNavierStokesAssembler` for stability analysis.

## Implementation Details

- A mixed function space $W = V \times Q$ is created, coupling velocity and pressure into a single system.

- Dirichlet boundary conditions are enforced strongly at every iteration using PETSc tools, ensuring physical and mathematical correctness.

- For each nonlinear iteration:
  - The residual vector and Jacobian matrix are assembled.
  - A Krylov iterative solver (e.g., `gmres`) with a preconditioner (e.g., `hypre`) solves the linear system $ J\Delta w = -R$.
  - The solution update $\Delta w$ is applied to the current iterate.
  - Convergence is monitored based on the residual norm.

- Logging reports the start of the solve, each iteration’s residual norm, and final convergence status, providing clear feedback for debugging or performance tracking.


## Notes and Assumptions

- It is assumed that the problem has at least one pressure degree of freedom constrained (e.g., a pressure Dirichlet condition or a constraint on the mean pressure) to avoid singularity.

- The linear solver and preconditioner are configurable and can be tuned for larger or stiffer problems.
The default (`gmres` + `hypre`) is considered robust for general use cases.

- Picard iteration is more robust for higher Reynolds numbers or poor initial guesses, while Newton’s method is faster when convergence is achievable.

- Future extensions could introduce under-relaxation strategies.
