# LSA-FW Elasticity Analysis

The LSA-FW project aims to develop a generalizable FEniCSx-based framework.
While its primary focus is the linear stability analysis of incompressible Navier–Stokes flows, a key goal is to make it straightforward to implement other classes of physical problems within the same architecture.

To demonstrate this extensibility, the `Elasticity` module serves as a working example.
It provides an interface to compute eigenfrequencies and eigenmodes of undamped, isotropic materials under small-strain conditions.

## Notation and Concepts

In a deformable solid, the position of any material point at time $t$ is

$$
\mathbf{x}(\mathbf{X}, t) = \mathbf{X} + \mathbf{u}(\mathbf{X}, t),
$$

where $\mathbf{x}$ is the point's position in the deformed configuration, $\mathbf{X}$ is its position in the undeformed configuration, and $\mathbf{u}$ is the displacement vector.

In finite-strain theory, the (Green–Lagrange) strain tensor $E$ is

$$
E := \frac{1}{2}\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^{T} + (\nabla \mathbf{u})^{T}\nabla \mathbf{u}\right).
$$

For sufficiently small displacements, the quadratic term $(\nabla \mathbf{u})^{T}\nabla \mathbf{u}$ is negligible. In the small-strain (infinitesimal) limit, the strain tensor reduces to

$$
\varepsilon := \frac{1}{2}\left(\nabla \mathbf{u} + (\nabla \mathbf{u})^{T}\right).
$$

## Governing Equations

We consider small-strain, undamped, isotropic linear elasticity on a domain $\Omega \subset \mathbb{R}^d$ with boundary $\partial\Omega = \Gamma_D \cup \Gamma_N$ (disjoint, up to measure zero). The balance of linear momentum reads
$$
\rho\ddot{\mathbf{u}} = \nabla\cdot\boldsymbol{\sigma} + \mathbf{f} \quad \text{in }\Omega,
$$
with Dirichlet and Neumann boundary conditions
$$
\mathbf{u} = \bar{\mathbf{u}} \ \text{on }\Gamma_D, 
\qquad 
\boldsymbol{\sigma}\mathbf{n} = \bar{\mathbf{t}} \ \text{on }\Gamma_N.
$$
Here $\rho$ is the mass density, $\mathbf{f}$ is the body force density, and $\boldsymbol{\sigma}$ is the Cauchy stress tensor.

Under Hooke's law for small strains,
$$
\boldsymbol{\sigma} = 2\mu\,\varepsilon(\mathbf{u}) + \lambda\,\mathrm{tr}\big(\varepsilon(\mathbf{u})\big)\,\mathbf{I},
\qquad 
\varepsilon(\mathbf{u}) := \tfrac{1}{2}\big(\nabla \mathbf{u} + \nabla \mathbf{u}^{T}\big),
$$
with Lamé parameters $\lambda,\mu$ (equivalently $E,\nu$). Using $\mathrm{tr}(\varepsilon(\mathbf{u})) = \nabla\cdot\mathbf{u}$, the strong form becomes
$$
\rho\ddot{\mathbf{u}} = \nabla\cdot\Big(2\mu\,\varepsilon(\mathbf{u}) + \lambda\,(\nabla\cdot\mathbf{u})\,\mathbf{I}\Big) + \mathbf{f} \quad \text{in }\Omega.
$$

**Free vibration (eigenproblem).** For unforced, homogeneous boundary conditions and harmonic motion $\mathbf{u}(\mathbf{x},t)=\hat{\mathbf{u}}(\mathbf{x})e^{\mathrm{i}\omega t}$,
$$
-\omega^{2}\rho\,\hat{\mathbf{u}} = \nabla\cdot\Big(2\mu\,\varepsilon(\hat{\mathbf{u}}) + \lambda\,(\nabla\cdot\hat{\mathbf{u}})\,\mathbf{I}\Big) \quad \text{in }\Omega,
$$
which discretizes to the generalized EVP
$$
\mathbf{K}\hat{\mathbf{u}} = \omega^{2}\mathbf{M}\hat{\mathbf{u}}.
$$
Material stability (strong ellipticity) requires $\mu>0$ and $\lambda + \tfrac{2}{d}\mu > 0$ (equivalently bulk modulus $\kappa=\lambda+\tfrac{2}{3}\mu>0$ in 3D).
Proof is left as a *to do*.

## Weak Formulation

Let

$$
\mathcal{V} := \big\{\mathbf{v}\in[H^1(\Omega)]^d : \mathbf{v}=\mathbf{0} \ \text{on }\Gamma_D\big\}.
$$

Multiplying the strong form by a test function $\mathbf{v}\in\mathcal{V}$ and integrate over $\Omega$:

$$
\int_{\Omega} \rho\ddot{\mathbf{u}}\cdot \mathbf{v}\,\mathrm{d}\Omega
=
\int_{\Omega} \nabla\cdot\boldsymbol{\sigma}\cdot \mathbf{v}\,\mathrm{d}\Omega
+
\int_{\Omega} \mathbf{f}\cdot \mathbf{v}\,\mathrm{d}\Omega.
$$

Using integration by parts and the symmetry of $\boldsymbol{\sigma}$ (so $\boldsymbol{\sigma}:\omega(\mathbf{v})=0$ with $\omega(\mathbf{v})$ the skew part),

$$
\int_{\Omega} \nabla\cdot\boldsymbol{\sigma}\cdot \mathbf{v}\,\mathrm{d}\Omega
=
\int_{\Gamma_N} (\boldsymbol{\sigma}\mathbf{n})\cdot \mathbf{v}\,\mathrm{d}\Gamma
-
\int_{\Omega} \boldsymbol{\sigma}:\nabla \mathbf{v}\,\mathrm{d}\Omega
=
\int_{\Gamma_N} \bar{\mathbf{t}}\cdot \mathbf{v}\,\mathrm{d}\Gamma
-
\int_{\Omega} \boldsymbol{\sigma}:\varepsilon(\mathbf{v})\,\mathrm{d}\Omega.
$$

Substituting Hooke's law yields the standard bilinear and linear forms:

$$
\boxed{
\begin{aligned}
\text{Find }\mathbf{u}(t)\in \mathcal{V}\text{ such that }\forall\,\mathbf{v}\in\mathcal{V}:\quad
&\int_{\Omega} \rho\ddot{\mathbf{u}}\cdot \mathbf{v}\,\mathrm{d}\Omega
+
\underbrace{\int_{\Omega} \big(2\mu\,\varepsilon(\mathbf{u}):\varepsilon(\mathbf{v}) + \lambda\,(\nabla\cdot\mathbf{u})(\nabla\cdot\mathbf{v})\big)\,\mathrm{d}\Omega}_{a(\mathbf{u},\mathbf{v})}
\\[4pt]
&=
\underbrace{\int_{\Omega} \mathbf{f}\cdot \mathbf{v}\,\mathrm{d}\Omega + \int_{\Gamma_N} \bar{\mathbf{t}}\cdot \mathbf{v}\,\mathrm{d}\Gamma}_{\ell(\mathbf{v})}.
\end{aligned}
}
$$

For free vibrations with homogeneous data ($\mathbf{f}=\mathbf{0}$, $\bar{\mathbf{t}}=\mathbf{0}$) and the harmonic ansatz, the weak eigenproblem is:

$$
\text{Find }(\omega^2,\hat{\mathbf{u}})\text{ with }\hat{\mathbf{u}}\in\mathcal{V}\setminus\{\mathbf{0}\}\text{ such that }
\quad
a(\hat{\mathbf{u}},\mathbf{v}) = \omega^{2}\,\underbrace{\int_{\Omega}\rho\,\hat{\mathbf{u}}\cdot \mathbf{v}\,\mathrm{d}\Omega}_{m(\hat{\mathbf{u}},\mathbf{v})}
\quad \forall\,\mathbf{v}\in\mathcal{V}.
$$

## Test case: Thin Plate

