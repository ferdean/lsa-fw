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

The solver now employs Newton's method with a **Stokes flow** initial guess and solves the resulting linear systems using sparse **LU factorization**.
For improved robustness, especially at larger Reynolds numbers, the Reynolds number can be ramped from 1 to the target value over a user-defined number of steps.

## Nonlinear Solver (Newton)

At iteration $k$ the Navier--Stokes residual is linearized around the current approximation $(\mathbf{u}^{(k)},p^{(k)})$.
The Jacobian therefore contains the derivative of the convective term with respect to both the transported and transporting velocity fields.
The Newton update $(\Delta\mathbf{u},\Delta p)$ is obtained from
\[
  J\,\Delta w = -R(\mathbf{u}^{(k)},p^{(k)}),
\]
and the iterate is updated with an optional damping factor $\alpha \in (0,1]$,
\[
  (\mathbf{u}^{(k+1)},p^{(k+1)}) = (\mathbf{u}^{(k)},p^{(k)}) + \alpha\,\Delta w.
\]

### Stokes Initial Guess

Before running Newton iterations the solver assembles and solves the associated Stokes problem,
\[
  -\tfrac{1}{\mathrm{Re}}\Delta\mathbf{u} + \nabla p = \mathbf{0},\qquad
  \nabla \cdot \mathbf{u} = 0,
\]
subject to the same boundary conditions.
The Stokes solution provides a physically meaningful and smooth starting point for the nonlinear iterations.

### Linear Solver

Both the Stokes solve and all Newton steps are performed via direct LU factorization using `scipy.sparse.linalg.splu`.
While this approach is memory intensive, it yields robust convergence for the moderate mesh sizes used in the included examples and avoids the need for preconditioners.
Solver options such as the target Reynolds number, number of ramping steps and damping factor are passed directly to `BaseFlowSolver.solve`.

## API

The solver returns two `dolfinx.fem.Function` objects,

- `u`, with the steady velocity field, and
- `p`, with the steady pressure field.

These belong to the user-specified function spaces and are fully compatible with the perturbation and stability analysis modules.
For example, the `LinearizedNavierStokesAssembler` (refer to [FEM-operators documentation](fem-operators.md)) expects the base velocity field `u` to assemble the perturbation system matrices.

In the project workflow, if no precomputed base flow is available, the code can call `solve_base_flow` automatically. 

### Example

```python
from Meshing import Mesher, Geometry
from FEM.spaces import define_spaces, FunctionSpaceType
from FEM.bcs import BoundaryCondition, define_bcs
from Solver.baseflow import BaseFlowSolver

# Generate mesh and mark boundaries (omitted for brevity)
mesher = Mesher.from_geometry(Geometry.CYLINDER_FLOW, cfg)
mesher.mark_boundary_facets(marker_fn)

spaces = define_spaces(mesher.mesh, FunctionSpaceType.TAYLOR_HOOD)
bcs = define_bcs(mesher, spaces, [BoundaryCondition.from_config(c) for c in cfg_bcs])

solver = BaseFlowSolver(spaces, bcs=bcs)
baseflow = solver.solve(
    re=100.0,
    ramp=True,
    steps=5,
    damping_factor=0.8,
)
```

After solving, `u_base` can be passed to the `LinearizedNavierStokesAssembler` for stability analysis.

## Implementation Details

- A mixed function space $W = V \times Q$ couples velocity and pressure into a single unknown.
- Dirichlet boundary conditions are imposed strongly using PETSc wrappers.
- The Stokes matrix is assembled once and factorized via LU to obtain the initial guess.
- At every Newton step the Jacobian and residual are assembled and the linear correction is computed via the cached LU factorization.
- The residual norm is monitored and logged at each iteration together with the current Reynolds number when ramping is enabled.


## Notes and Assumptions

- It is assumed that the problem has at least one pressure degree of freedom constrained (e.g., a pressure Dirichlet condition or a constraint on the mean pressure) to avoid singularity.
- The LU factorization is cached and reused across Newton iterations, so the method is best suited for small to medium size problems where direct solvers are affordable.
- Picard iterations are no longer supported; Newton with optional damping is the default strategy.

- Future extensions could introduce under-relaxation strategies.

## CLI Integration

The solver is available through the module `Solver.cli`.  A typical command line invocation is

```bash
python -m Solver baseflow \
    --geometry cylinder_flow \
    --config config_files/2D/cylinder/cylinder_flow.toml \
    --facet-config config_files/2D/cylinder/mesh_tags.toml \
    --re 80.0 --steps 5 --damping 0.8 --plot
```

The command generates the geometry, applies the specified boundary conditions and computes the steady base flow.
Results can optionally be plotted or exported for later use in the FEM assembly module.
