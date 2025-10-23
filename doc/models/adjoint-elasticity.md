# LSA-FW Adjoint-Based Sensitivity - Elasticity

## Motivation

Adjoint-based sensitivity analysis is actively used for eigenvalue problems in structural dynamics (mechanical vibrations).
The approach has long been studied to evaluate how natural frequencies change with design or physical parameters.
Classical work by Fox and Kapoor [[1]](#references) established formulas for the rates of change of eigenvalues and eigenvectors.
Since then, many researchers have developed efficient adjoint or 'design sensitivity' methods for vibration eigenproblems.

Contemporary studies continue to apply adjoint-based sensitivity in vibration analysis.
For instance, Zheng et al. [[2]](#references) derive formulas for acoustic cavity resonance frequencies using an adjoint method with right and left eigenvectors – a concept equally applicable to structural natural frequencies.
The importance of eigenfrequency sensitivities in mechanical systems is well recognized; efforts have been made to compute modal sensitivities of elastic structures in various contexts.

## Theoretical Background

Consider the elasticity generalized eigenvalue problem for free vibrations of an undamped solid,

$$
\mathbf{K}(\mu)\mathbf{v} = \lambda\mathbf{M}(\mu)\mathbf{v},
$$

where $\mathbf{K}$ is the structural stiffness matrix, $\mathbf{M}$ is the mass matrix, and $\lambda = \omega^2$ is the eigenvalue, with $\omega$ being the natural frequency.

The system depends on a parameter $\mu$, which may represent a material property, a shape parameter, or a boundary condition.

The corresponding adjoint (left) eigenproblem is

$$
\mathbf{a}^T \mathbf{K}(\mu) = \lambda\mathbf{a}^T \mathbf{M}(\mu),
$$

with normalization $\mathbf{a}^T \mathbf{M}\mathbf{v} = 1$.

Differentiating the primal eigenvalue equation with respect to $\mu$ yields

$$
(\partial_\mu \mathbf{K})\mathbf{v} + \mathbf{K}(\partial_\mu \mathbf{v})
= (\partial_\mu \lambda)\mathbf{M}\mathbf{v} + \lambda(\partial_\mu \mathbf{M})\mathbf{v} + \lambda\mathbf{M}(\partial_\mu \mathbf{v}),
$$

which rearranges as

$$
(\partial_\mu \mathbf{K})\mathbf{v} - (\partial_\mu \lambda)\mathbf{M}\mathbf{v} - \lambda(\partial_\mu \mathbf{M})\mathbf{v} + [\mathbf{K} - \lambda\mathbf{M}](\partial_\mu \mathbf{v}) = \mathbf{0}.
$$

Projecting onto the adjoint mode by multiplying from the left with $\mathbf{a}^T$ gives

$$
\mathbf{a}^T (\partial_\mu \mathbf{K})\mathbf{v} - (\partial_\mu \lambda)\mathbf{a}^T \mathbf{M}\mathbf{v} - \lambda\mathbf{a}^T (\partial_\mu \mathbf{M})\mathbf{v} + \mathbf{a}^T [\mathbf{K} - \lambda\mathbf{M}](\partial_\mu \mathbf{v}) = \mathbf{0}.
$$

Since $\mathbf{a}^T[\mathbf{K} - \lambda\mathbf{M}] = \mathbf{0}^T$ by definition of the adjoint problem and $\mathbf{a}^T \mathbf{M}\mathbf{v} = 1$ by normalization, the sensitivity of $\lambda$ with respect to $\mu$ follows as

$$
\partial_\mu \lambda
= \mathbf{a}^T \left(\partial_\mu \mathbf{K} - \lambda\partial_\mu \mathbf{M}\right)\mathbf{v}.
$$

Additionally, as the eigenproblem matrices are symmetric, one can set $\mathbf{a} = \mathbf{v}$, so the formula further simplifies:

$$
\partial_\mu \lambda
= \mathbf{v}^T \left(\partial_\mu \mathbf{K} \right) \mathbf{v} - \lambda\mathbf{v}^T\left(\partial_\mu \mathbf{M}\right)\mathbf{v} \quad \square
$$

### Symmetry and Adjoint Simplification

**Lemma.**
Let $\mathbf{K}, \mathbf{M} \in \mathbb{R}^{n\times n}$ with $\mathbf{K} = \mathbf{K}^T$, $\mathbf{M} = \mathbf{M}^T$, and $\mathbf{x}^T \mathbf{M}\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$.  
Then every right eigenpair $(\lambda, \mathbf{v})$ of

$$
\mathbf{K}\mathbf{v} = \lambda\mathbf{M}\mathbf{v}
$$

has a real eigenvalue $\lambda$, and the left eigenvector $\mathbf{a}$ can be chosen equal to $\mathbf{v}$.

**Proof (Reality of the eigenvalues).**

Define the $\mathbf{M}$-inner product $\langle \mathbf{x}, \mathbf{y}\rangle_\mathbf{M} := \mathbf{x}^T \mathbf{M}\mathbf{y}$ and the operator $\mathbf{A} := \mathbf{M}^{-1}\mathbf{K}$.
Then, for any $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,

$$
\langle \mathbf{x}, \mathbf{A}\mathbf{y}\rangle_\mathbf{M}
= \mathbf{x}^T \mathbf{M}(\mathbf{M}^{-1}\mathbf{K})\mathbf{y}
= \mathbf{x}^T \mathbf{K}\mathbf{y}
= (\mathbf{K}\mathbf{x})^T \mathbf{y}
= \langle \mathbf{A}\mathbf{x}, \mathbf{y}\rangle_\mathbf{M}.
$$

Hence $\mathbf{A}$ is self-adjoint in $\langle\cdot,\cdot\rangle_\mathbf{M}$ and, as such, possesses only real eigenvalues $\quad \square$

**Proof (Left–right coincidence).**

Transposing the right eigenvalue equation,

$$
(\mathbf{K}\mathbf{v})^T = (\lambda \mathbf{M}\mathbf{v})^T
\quad\Rightarrow\quad
\mathbf{v}^T \mathbf{K}^T = \lambda\mathbf{v}^T \mathbf{M}^T.
$$

Since $\mathbf{K}^T = \mathbf{K}$ and $\mathbf{M}^T = \mathbf{M}$, it follows directly that $\mathbf{a} = \mathbf{v} \quad \square$

Therefore, for symmetric real matrices $\mathbf{K}$ and $\mathbf{M}$, all eigenvalues are real and the adjoint vector coincides with the direct mode, simplifying the adjoint sensitivity expression.


### Why $\mathbf{K}$ and $\mathbf{M}$ are Symmetric in Linear Isotropic Elasticity

Let $\Omega \subset \mathbb{R}^3$ be a Lipschitz domain, and $u,v \in V := \{w \in [H^1(\Omega)]^3 : w = 0 \text{ on } \Gamma_D\}$.

The small-strain tensor is

$$
\varepsilon(\mathbf{u}) := \tfrac{1}{2}(\nabla u + \nabla u^T),
$$

and for an isotropic, linear solid,

$$
\sigma(\mathbf{u}) = \mathbb{C} : \varepsilon(\mathbf{u})
= 2\mu\varepsilon(\mathbf{u}) + \lambda\text{tr}(\varepsilon(\mathbf{u}))\mathbf{I},
$$

with Lamé parameters $\mu > 0$, $\lambda \ge 0$, and density $\rho(x) > 0$.

Define the bilinear forms

$$
a(\mathbf{u},\mathbf{v}) := \int_\Omega \sigma(\mathbf{u}):\varepsilon(\mathbf{v})\mathrm{d}x, \qquad
m(\mathbf{u},\mathbf{v}) := \int_\Omega \rho \mathbf{u} \cdot \mathbf{v}\mathrm{d}x.
$$

With a conforming FE space $V_h \subset V$ and nodal basis $\{\phi_i\}$,

$$
K_{ij} = a(\phi_j, \phi_i), \qquad M_{ij} = m(\phi_j, \phi_i).
$$

**Lemma (Symmetry of the stiffness bilinear form).**  
For all $\mathbf{u},\mathbf{v} \in V$, $a(\mathbf{u},\mathbf{v}) = a(\mathbf{v},\mathbf{u})$.

**Proof.**

Using the constitutive law,

$$
a(\mathbf{u},\mathbf{v}) = \int_\Omega \left[2\mu\varepsilon(\mathbf{u}):\varepsilon(\mathbf{v}) + \lambda\text{tr}(\varepsilon(\mathbf{u}))\text{tr}(\varepsilon(\mathbf{v}))\right]\mathrm{d} x.
$$

Since $\varepsilon(\mathbf{u})$ and $\varepsilon(\mathbf{v})$ are symmetric tensors,
$\varepsilon(\mathbf{u}):\varepsilon(\mathbf{v}) = \varepsilon(\mathbf{v}):\varepsilon(\mathbf{u})$ and
$\text{tr}(\varepsilon(\mathbf{u}))\text{tr}(\varepsilon(\mathbf{v})) = \text{tr}(\varepsilon(\mathbf{v}))\text{tr}(\varepsilon(\mathbf{u}))$.

Hence $a(\mathbf{u},\mathbf{v}) = a(\mathbf{v},\mathbf{u}) \quad \square$.  

**Lemma (Symmetry of the mass bilinear form).**  
For all $u,v \in V$, $m(\mathbf{u},\mathbf{v}) = m(\mathbf{v},\mathbf{u})$.

**Proof.**

The scalar product is commutative and $\rho$ is scalar:

$$
m(\mathbf{u},\mathbf{v}) = \int_\Omega \rho \mathbf{u}\cdot \mathbf{v}\mathrm{d}x
= \int_\Omega \rho \mathbf{v}\cdot \mathbf{u}\mathrm{d}x
= m(\mathbf{v},\mathbf{u}) \quad \square
$$

**Corollary (Matrix symmetry).**

For any conforming FE space $V_h$ with basis $\{\phi_i\}$,

$$
K_{ij} = K_{ji}, \qquad M_{ij} = M_{ji}.
$$

Thus both $\mathbf{K}$ and $\mathbf{M}$ are real symmetric matrices.


## Special case: Sensitivity of the Eigenvalue to Density

Consider the generalized eigenvalue problem for free vibrations of an elastic solid,

$$
\mathbf{K}\mathbf{v} = \lambda\mathbf{M}\mathbf{v},
$$

where $\mathbf{K}$ and $\mathbf{M}$ are the stiffness and mass matrices, and $\lambda = \omega^2$.

Both matrices depend on the material parameters.  
For the present case, we take the density $\rho$ as the parameter of interest, i.e. $\mathbf{K}$ is independent of $\rho$ while $\mathbf{M} = \mathbf{M}(\rho)$.

### Derivation

With the above derivation, and using the independence of $\mathbf{K}$ on t he parameter,

$$
\frac{\partial \lambda}{\partial \rho} = -\lambda\mathbf{v}^T(\partial_\rho \mathbf{M})\mathbf{v}.
$$


### Specialization to Density

For a consistent-mass formulation,

$$
\mathbf{M}(\rho) = \int_\Omega \rho(\mathbf{x}){\bf N}^T{\bf N}\mathrm{d}\Omega,
$$

hence its derivative is

$$
\partial_\rho \mathbf{M} = \int_\Omega {\bf N}^T{\bf N}\mathrm{d}\Omega.
$$

At the continuous level, this corresponds to

$$
\mathbf{a}^T (\partial_\rho \mathbf{M}) \mathbf{v} = \int_\Omega \mathbf{a}\cdot \mathbf{v}\mathrm{d}\Omega.
$$

Because the eigenproblem is Hermitian ($\mathbf{a} = \mathbf{v}$ for symmetric $\mathbf{K}$ and $\mathbf{M}$), the expression simplifies to

$$
\frac{\partial \lambda}{\partial \rho} = -\lambda\frac{\int_\Omega |\mathbf{v}|^2\mathrm{d}\Omega} {\int_\Omega \rho|\mathbf{v}|^2\mathrm{d}\Omega}. \quad \square
$$



### Interpretation

- The derivative is **negative**: increasing density lowers the eigenfrequency.
- The numerator $\int |\mathbf{v}|^2$ measures the kinetic energy distribution, while the denominator $\int \rho |\mathbf{v}|^2$ corresponds to the modal mass.
- If the mode is **mass-normalized**, i.e.,

$$
\mathbf{v}^T \mathbf{M}\mathbf{v} = \int_\Omega \rho |\mathbf{v}|^2\mathrm{d}\Omega = 1,
$$

  the formula reduces to

$$
\frac{\partial \lambda}{\partial \rho} = -\lambda \int_\Omega |\mathbf{v}|^2\mathrm{d}\Omega.
$$

This analytical result can be used to validate symbolic solvers.

## Implementation Detail

### Scalar Integration

The main implementation design decision has been to keep using the assembler's UFL forms in `ElasticityAssembler`, but assembling scalars instead of full derivative matrices.

As it has already been proven earlier in this document, 

$$
\partial_\mu \lambda
= \mathbf{v}^T \left(\partial_\mu \mathbf{K}  - \lambda\partial_\mu \mathbf{M}\right)\mathbf{v}.
$$

The matrices $\mathbf{K}$ and $\mathbf{M}$ originate from continuous bilinear forms, 

$$
a(\mathbf{u}, \mathbf{v}) = \int_\Omega \mathcal{A}(\mathbf{x}, \mu; \mathbf{u}, \mathbf{v}) \mathrm{d}x,
$$

$$
m(\mathbf{u}, \mathbf{v}) = \int_\Omega \mathcal{M}(\mathbf{x}, \mu; \mathbf{u}, \mathbf{v}) \mathrm{d}x,
$$

where $\mathcal{A}$, $\mathcal{M}$ are the integrands depending on the material model and on $\mu$.

Differentiating with respect to $\mu$ yields 

$$
\partial_\mu a(\mathbf{u}, \mathbf{v}) = \int_\Omega \partial_\mu \mathcal{A}(\mathbf{x}, \mu; \mathbf{u}, \mathbf{v}) \mathrm{d}x,
$$

$$
\partial_\mu m(\mathbf{u}, \mathbf{v}) = \int_\Omega \partial_\mu \mathcal{M}(\mathbf{x}, \mu; \mathbf{u}, \mathbf{v}) \mathrm{d}x.
$$

At the discrete level, and focusing on just one of the matrices, 

$$
\left( \partial_\mu K_{ij}\right) = \partial_\mu a(\phi_j, \phi_i),
$$

so

$$
\mathbf{v}^T (\partial_\mu \mathbf{K}) \mathbf{v} = \sum_{i, j} v_i v_j \partial_\mu a(\phi_j, \phi_i) = \partial_\mu a(\mathbf{v}_h, \mathbf{v}_h),
$$

where $\mathbf{v}_h = \sum_j v_j \phi_j$ is the finite-element representation of the eigenmode.

Thus, the contraction $\mathbf{v}^T (\partial_\mu \mathbf{K}) \mathbf{v}$ is exactly equal to evaluating the derivative form $\partial_\mu a(\mathbf{v}_h, \mathbf{v}_h)$ - a **scalar integral**.

As a consequence, the adjoint formula only involves two scalar contractions, we never need the full matrices $\partial_\mu \mathbf{K}$ and $\partial_\mu \mathbf{M}$.

Because each contraction equals the corresponding bilinear form evaluated at $(\mathbf{v}_h, \mathbf{v}_h)$, we can compute it directly as a scalar finite-element integral, which is exactly what `assemble_scalar(form(...))` does in `FEniCSx`.

### Natural Frequency Sensitivity

Applying similar steps as the ones for the postprocessing of the eigenvalues, one can obtain the sensitivity of the natural frequencies as a function of the sensitivity of the eigenvalues.

Given that the angular natural frequency is defined as $\omega = \sqrt{\lambda}$ and, in Hertz, $f = \omega / (2\pi)$, 

$$
\frac{\partial f}{\partial \mu} = \frac{1}{8 \pi^2 f}\frac{\partial \lambda}{\partial \mu}.
$$

Sometimes, one can also be interested in relative (fractional) sensitivity, 

$$
\frac{1}{f} \frac{\partial f}{\partial \mu} = \frac{1}{2\lambda} \frac{\partial \lambda}{\partial \mu}.
$$

### Detail

The sensitivity evaluation is fully implemented at the variational level, directly within the finite-element framework.
All material parameters (e.g., $E$, $\nu$, $\rho$) are represented as *UFL coefficients* — typically cellwise constant DG-0 functions — so that symbolic differentiation with respect to any such function becomes possible.

#### Generic formulation

The adjoint-based formula is evaluated by assembling two scalar integrals.

Instead of explicitly forming the derivative matrices $\partial_\mu\mathbf{K}$ and $\partial_\mu\mathbf{M}$, the same result is obtained by differentiating the *variational forms* symbolically and testing them with the converged eigenmode $\mathbf{v}_h$:

$$
\frac{\partial\lambda}{\partial\mu} = a'(\mathbf{v}_h\mathbf{v}_h) - \lambda m'(\mathbf{v}_h,\mathbf{v}_h),
$$

where $a'$ and $m'$ denote the derivatives of the bilinear forms $a(u,v)$ and $m(u,v)$ with respect to $\mu$.

In practice, this becomes:

```python
dk = ufl.derivative(k_vv, param, dparam)
dm = ufl.derivative(m_vv, param, dparam)
val_k = assemble_scalar(form(dk))
val_m = assemble_scalar(form(dm))
```

and the sensitivity is returned as $\partial_\mu\lambda = \text{val}_k - \lambda\cdot\text{val}_m$.

This formulation is fully general: any parameter appearing in the material laws, source terms, or boundary data can be chosen as `param`, making the approach directly extensible to anisotropic or heterogeneous problems.

#### Validation

To ensure correctness, the implementation is verified on the density sensitivity of the first elastic mode of a steel plate.
This case provides two independent references:

1. **Analytical specialization** (derived earlier),
   
$$
\frac{\partial\lambda}{\partial\rho} = -\lambda \frac{\int_\Omega |\mathbf{v}|^2\mathrm{d}\Omega} {\int_\Omega \rho|\mathbf{v}|^2\mathrm{d}\Omega},
$$

   which for mass-normalized modes simplifies to $-\lambda\int_\Omega|\mathbf{v}|^2\mathrm{d}\Omega.$

2. **Finite-difference approximation** obtained by re-assembling $\mathbf{M}$ for perturbed densities $\rho\pm h$ and recomputing the first eigenpair:

$$
\frac{\partial\lambda}{\partial\rho} \approx \frac{\lambda(\rho+h) - \lambda(\rho-h)}{2h}.
$$

#### Results

For the first mode of the benchmark plate $E=200\;\mathrm{GPa}$, $\nu=0.3$, $\rho=8000\;\mathrm{kg/m^3}$,
the computed sensitivities perfectly coincide across all methods:

| Method                           | Sensitivity [Hz / (kg m⁻³)]                    | Rel. change |
| :------------------------------- | :--------------------------------------------: | :---------: |
| Adjoint / variational derivative |                    −0.002805                   |   −0.006 %  |
| Analytical formula               |                    −0.002805                   |   −0.006 %  |
| Finite differences               |                    −0.002804                   |   −0.006 %  |

The full agreement demonstrates both the correctness of the symbolic differentiation approach and its numerical stability.
Because the implementation operates at the variational level, it immediately generalizes to sensitivities with respect to *any spatially distributed parameter*, not only uniform density.

## References

[1] Fox, R. L., and Kapoor, M. P. (1968). Rates of Change of Eigenvalues and Eigenvectors. *AIAA Journal*, 6(12).

[2] Zheng, C., Zhao, W., Gao, H., Du, L., Zhang, Y., & Bi, C. (2021). Sensitivity analysis of acoustic eigenfrequencies by using a boundary element method. *The Journal of the Acoustical Society of America*, 149(3).
