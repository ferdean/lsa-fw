# LSA-FW FEM Module (Boundary Conditions)

> [Back to FEM Overview](fem.md)

---

## Boundary Conditions

Let $\partial\Omega = \Gamma_D \cup \Gamma_N \cup \Gamma_R$ be a partition into Dirichlet, Neumann, and Robin boundaries.

### Dirichlet Conditions

Prescribed values:

$$
u = u_D \quad \text{on } \Gamma_D.
$$

These are imposed strongly, typically by modifying the system matrix and/or the solution vector directly.

### Neumann Conditions

Prescribed normal derivatives:

$$
\frac{\partial u}{\partial n} = g \quad \text{on } \Gamma_N.
$$

These appear naturally in the variational formulation:

$$
\int_{\Gamma_N} g v \, ds.
$$

They are incorporated as boundary integrals in the weak form, without altering the stiffness matrix.

### Robin Conditions

Weighted combination:

$$
\alpha u + \beta \frac{\partial u}{\partial n} = g \quad \text{on } \Gamma_R.
$$

Typical weak form contribution (in penalty form):

$$
\int_{\Gamma_R} \alpha u v \, ds \quad \text{and/or} \quad \int_{\Gamma_R} g v \, ds.
$$

In practice, implementation may vary depending on whether $\alpha$ or $\beta$ dominates.
You may see approximate Robin conditions introduced via penalty/stabilization terms of the form:

$$
\int_{\Gamma_R} \alpha (g - u) v \, ds.
$$

> For more on weak imposition and penalty methods, see Ern & Guermond (2004).

## Implementation

The `bcs` module provides a clean interface for defining boundary conditions within the LSA-FW.

Each boundary condition is described using a `BoundaryCondition` object, which specifies

- the facet marker (an integer label from `MeshTags`),
- the [type](#supported-types) of condition (Dirichlet, Neumann, or Robin),
- the value, and
- optionally the robin penalty parameter.

The boundary conditions are grouped and returned as a `BoundaryConditions` dataclass that holds

- strong Dirichlet boundary conditions (applied directly using `dolfinx.fem.dirichletbc`), and
- weak Neumann/Robin forms (as `ufl.Form` terms to be added to the variational formulation).

The core interface is now simplified to:

```python
def define_bcs(
    mesher: Mesher,
    spaces: FunctionSpaces,
    configs: Sequence[BoundaryCondition],
) -> BoundaryConditions
```

The `Mesher` instance provides the mesh and facet tags, so the caller only
needs to supply the configuration objects.

### Design Details

Dirichlet BCs are imposed by
  - interpolating the value into a `Function` on the corresponding space,
  - locating the degrees of freedom on the boundary facets, and
  - creating a `DirichletBC` object.
  
Constant values are wrapped as callable interpolators via internal helpers to support both constant and spatially-varying expressions.

Neumann and Robin BCs are handled via surface integrals added to the weak form.
This is

- for Neumann terms, $\int_{\Gamma_N} g \cdot v \, ds$, and
- for Robin terms, $\int_{\Gamma_R} \alpha u \cdot v \, ds + \int_{\Gamma_R} \alpha g \cdot v \, ds$.

Measure handling (`ds(marker)`) is abstracted via the `iMeasure` utility to ensure consistent access to tagged boundaries.

### Supported Types

| Type                   | Description |
|------------------------|-----------------------------------------------------|
| `DIRICHLET_VELOCITY`   | Strong velocity condition |
| `DIRICHLET_PRESSURE`   | Strong pressure condition |
| `NEUMANN`              | Natural Neumann condition (weak form) |
| `ROBIN`                | Robin condition via penalty terms in weak formulation |
| `PERIODIC`             | Periodic condition via DOF mapping |


### Example

```python
from FEM.bcs import define_bcs, BoundaryCondition, BoundaryConditionType

bcs = define_bcs(
    mesher,
    spaces,
    configs=[
        BoundaryCondition(marker=1, type=BoundaryConditionType.DIRICHLET_VELOCITY, value=(1.0, 0.0)),
        BoundaryCondition(marker=3, type=BoundaryConditionType.ROBIN, value=(0.0, 0.0), robin_alpha=10.0),
    ],
)
```

The example above manually instantiates BoundaryCondition objects in Python, but the intended design is to read these configurations from external files (e.g., JSON or TOML), enabling declarative problem specification and clean separation between code and input.

This would allow users, for example, to just define a config `bcs` file, such as 

```json
[
  {
    "marker": 1,
    "type": "DIRICHLET_VELOCITY",
    "value": [1.0, 0.0]
  },
  {
    "marker": 3,
    "type": "ROBIN",
    "value": [0.0, 0.0],
    "robin_alpha": 10.0
  }
]
```

Periodic boundary conditions can also be defined by specifying two markers in
`value`:

```toml
[[BC]]
marker = 4
type = "periodic"
value = [1, 2] # facets tagged 1 map onto facets tagged 2
```

