# LSA-FW FEM Function Spaces

> [Back to FEM Overview](fem.md)

---

## Function Spaces

As already formulated, the finite element method looks for the solution of a PDE in a finite-dimensional subspace of an appropriate infinite-dimensional function space.  
These function spaces define (and constrain) the regularity and structure of the admissible solutions.

In the case of an incompressible Navier–Stokes flow, the standard mixed formulation involves:

- a velocity field $\mathbf{u} \in [H^1(\Omega)]^d$, which includes vector-valued functions with square-integrable weak derivatives, and
- a pressure field $p \in L^2(\Omega)$, which includes scalar functions with square-integrable values.

Thus, the corresponding function spaces are constructed such that

$$
\begin{align*}
   & u_h \in V_h \subset [H^1(\Omega)]^d,\\
   & p_h \in Q_h \subset L^2(\Omega).
\end{align*}
$$

### Sobolev Spaces

The Sobolev space $H^1(\Omega)$ is defined as

$$
H^1(\Omega) = \left\lbrace u \in L^2(\Omega) \;\big|\; \frac{\partial u}{\partial x_i} \in L^2(\Omega) \;\; \text{for all } i = 1, \ldots, d \right\rbrace
$$

This space consists of functions that are square-integrable over $\Omega$, along with their first-order weak partial derivatives, which must also be square-integrable.

The Sobolev embedding theorem states that if the domain $\Omega$ is sufficiently regular (e.g., bounded with a Lipschitz-continuous boundary), and the spatial dimension satisfies $d \leq 3$, then:

$$
H^1(\Omega) \subset C^0(\overline{\Omega}).
$$

Here, $C^0(\overline{\Omega})$ denotes the space of continuous functions on the closure of $\Omega$, written $\overline{\Omega}$, which includes both the interior of $\Omega$ and its boundary $\partial\Omega$:

$$
\overline{\Omega} = \Omega \cup \partial\Omega.
$$

This embedding implies that functions in $H^1(\Omega)$ are not only integrable but also continuous up to the boundary of the domain.
This property is particularly important in the context of finite element methods, as it ensures that the finite element approximation is a continuous function — essential for physical realism and numerical stability.

The vector-valued Sobolev space $[H^1(\Omega)]^d$ is used to approximate vector fields.
In the context of this framework, this is of special interest for the (multidimensional) velocity field.  
It is defined as

$$
[H^1(\Omega)]^d = \{ \mathbf{u} = (u_1, \dots, u_d) \mid u_i \in H^1(\Omega) \text{ for } i = 1, \cdots, d \}.
$$

This space is usually the go-to for incompressible Navier–Stokes flows, as vector fields require continuity across element boundaries.

For certain problems — especially those involving conservation of fluxes across interfaces (e.g., Darcy flow or mixed formulations of incompressible flow), the divergence-conforming space can be used, defined as

$$
H(\text{div}; \Omega) = \left\lbrace \mathbf{v} \in [L^2(\Omega)]^d \,\middle|\, \nabla \cdot \mathbf{v} \in L^2(\Omega) \right\rbrace.
$$

While conservation of mass is a feature of many PDEs — including the Navier–Stokes equations — not all formulations or discretizations require explicit divergence-conforming spaces.

## Finite Element Spaces

As before, let $\mathcal{T}_h$ denote a conforming triangulation of $\Omega$.
Finite element spaces $V_h \subset V$ are constructed as global piece-wise polynomial spaces over $\mathcal{T}_h$, assembled from each local element $K \in \mathcal{T}_h$.

Each finite element is defined via $\left(K, P_K, \Sigma_K \right)$, where

- $K \subset \mathbb{R}^d$ is a reference element,
- $P_K$ is a polynomial space,
- $\Sigma_K$ is a set of degrees of freedom, represented by linearly independent functionals on $P_K$.

Each $\sigma \in \Sigma_K$ can be thought of as a rule that extracts a certain piece of information from a function — for example, evaluation at a node, an average over a face, or the value of a normal derivative.

A central requirement is that the map

$$
p \mapsto (\sigma(p))_{\sigma \in \Sigma_K}
$$

is a linear isomorphism from $P_K$ to $\mathbb{R}^{|\Sigma_K|}$.
That means that

- every function $p \in P_K$ is uniquely determined by its values under the functionals in $\Sigma_K$, and
- every tuple of real numbers $(c_{\sigma \in \Sigma_K})$ corresponds to a unique function $p \in P_K$ such that $\sigma(p) = c_\sigma$.

This property is called unisolvence.
It ensures that interpolation is well-defined: given a set of target values $(c_\sigma)$, there exists a unique polynomial in $P_K$ that matches these values at the associated DOFs.

### Lagrange Elements

Lagrange elements are the most common finite elements conforming to $H^1(\Omega)$.
They are defined by

- polynomial shape functions of degree $k$ (e.g., $\mathbb{P}_k$),
- degrees of freedom associated with nodal values (at vertices, edges, or element interiors), and
- global $C^0$-continuity across element boundaries.

Examples:

- $\mathbb{P}_1$: piecewise linear functions on triangles or tetrahedra,
- $\mathbb{P}_2$: quadratic functions, with additional DOFs at edge midpoints.

These elements are widely used for scalar fields or vector fields requiring continuity (e.g., velocity in incompressible flow).

### Bubble/MINI Elements

Bubble elements are enriched spaces that include interior functions vanishing on the element boundary.
A typical example is the cubic bubble function on a triangle:

$$
b(x, y) = 27 \lambda_1 \lambda_2 \lambda_3,
$$

where $\lambda_i$ are barycentric coordinates.
These enrichments do not affect inter-element continuity but improve approximation properties locally.

In mixed formulations, bubble functions are used in the MINI element to ensure inf-sup stability.

The inf-sup condition (also known as the LBB condition) is necessary to guarantee well-posedness of saddle-point problems involving velocity and pressure.
The interpretation of this condition is that it ensures that the pressure space is not "too large" compared to the velocity space.

The MINI element then uses

$$
V_h = [\mathbb{P}_1 \oplus B]^d, \quad Q_h = \mathbb{P}_1,
$$

where $B$ is a bubble space.

This pairing is simple, low-order, and inf-sup stable, making it effective for incompressible flows with minimal computational overhead.

### Mixed Elements: Taylor–Hood and Others

Taylor–Hood elements are classical mixed finite elements for incompressible flow.

They combine

- a higher-order continuous velocity space: $[\mathbb{P}_2]^d$, and
- a continuous linear pressure space: $\mathbb{P}_1$.

This pairing is inf-sup stable and more accurate than MINI, especially on smooth solutions.

Other mixed elements include:

- [MINI](#bubblemini-elements): $[\mathbb{P}_1 \oplus B]^d - \mathbb{P}_1$,
- equal-order: $[\mathbb{P}_1]^d - \mathbb{P}_1$ (requires stabilization techniques, as it is naturally non-LBB stable), and
- non-conforming: e.g., Crouzeix–Raviart elements for velocity, with discontinuous pressure.

Each pairing must be carefully chosen based on the governing equations, regularity, and solver requirements.

## Implementation

The `spaces` module provides a unified interface for defining compatible function spaces for incompressible Navier–Stokes problems within LSA-FW.

Each function space pair is selected via the `FunctionSpaceType` enum and constructed over a given mesh using `basix.ufl.element`.
The function spaces are returned as a `FunctionSpaces` dataclass that encapsulates

- a vector-valued velocity space (e.g., $[P_2]^d$, $[P_1 + B]^d$), and  
- a scalar pressure space (e.g., $P_1$).

The core interface is:

```python
def define_spaces(mesh: dolfinx.mesh.Mesh, type: FunctionSpaceType) -> FunctionSpaces
```

### Design Details

Function spaces are constructed explicitly from polynomial elements:

- **Taylor–Hood** uses continuous quadratic elements (`P2`) for velocity and linear (`P1`) for pressure.
- **MINI** enriches the linear velocity space with a cubic bubble element, using `enriched_element([P1, bubble])` to ensure inf-sup stability at low order.
- **Simple (P1–P1)** uses the same linear space for both fields.
This pairing is not inf-sup stable and should be combined with stabilization techniques.
- **Discontinuous Galerkin (DG)** support is reserved for future versions.

Vector-valued elements are constructed by passing `shape=(mesh.geometry.dim,)`, ensuring the correct dimensionality (2D or 3D) of the velocity field.

Polynomial degrees are fixed internally to guarantee known stability and performance properties.
For advanced use cases requiring manual element construction, this interface can be bypassed.

### Supported Function Space Types

| Type            | Velocity Space               | Pressure Space | Properties                                |
|-----------------|------------------------------|----------------|--------------------------------------------|
| `TAYLOR_HOOD`   | $[P_2]^d$                  | $P_1$         | Inf-sup stable, classical mixed pair       |
| `MINI`          | $[P_1 + B]^d$              | $P_1$         | Low-order enriched pair, inf-sup stable    |
| `SIMPLE`        | $[P_1]^d$                  | $P_1$         | Not inf-sup stable, requires stabilization |
| `DG`            | Not implemented              | Not implemented| Reserved for future DG formulations        |

### Example

```python
from FEM.spaces import define_spaces, FunctionSpaceType

spaces = define_spaces(mesh, FunctionSpaceType.MINI)
V, Q = spaces.velocity, spaces.pressure
mixed_space = spaces.mixed  # This is the recommended space for NS assemblers
```

This approach allows for consistent and composable space definitions across the LSA-FW framework, and supports downstream modules such as boundary condition handling, weak form assembly, and solver setup.
