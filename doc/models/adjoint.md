# LSA-FW Adjoint-Based Sensitivity

## Notation and scope

We study the generalized eigenvalue problem (EVP)

$$
\mathbf{A}(\mu)\,\mathbf{v}=\sigma\,\mathbf{B}\,\mathbf{v}, \tag{1}
$$

with parameter vector $\mu\in\mathbb{R}^p$.
Throughout, $\mathbf{v}$ denotes a **direct (right) eigenvector**, $\mathbf{a}$ the **adjoint (left) eigenvector**, $\sigma\in\mathbb{C}$ the eigenvalue, $\mathbf{A}$ the linearized stability operator (non-Hermitian in general), and $\mathbf{B}$ the mass matrix (possibly singular for incompressible formulations).
Conjugate transpose is $(\cdot)^H$.


## Motivation

Adjoint modes reduce eigenvalue sensitivity to just one direct solve, one adjoint solve, and cheap projections.
After computing a direct eigenpair $(\sigma,\mathbf{v})$ of $(1)$, one can obtain the adjoint mode $\mathbf{a}$ from the (discrete) adjoint problem

$$
\mathbf{A}(\mu)^H\,\mathbf{a}=\sigma^*\,\mathbf{B}^H\,\mathbf{a}.
$$

Then, for each parameter component $\mu_i$,

$$
\boxed{\displaystyle
\frac{\partial \sigma}{\partial \mu_i}
= \frac{\mathbf{a}^{H}\,(\partial_{\mu_i}\mathbf{A})\,\mathbf{v}}
       {\mathbf{a}^{H}\mathbf{B}\,\mathbf{v}} }.
$$

This is a key result that will be derived later in this document.

Once $(\mathbf{v},\mathbf{a})$ is known, each sensitivity component can be obtained from a single single inner product; no additional eigen-solves are required.

### Cost model and scaling

#### Finite-difference baseline (no adjoint)

To obtain $\partial \sigma/\partial \mu_i$ by finite differences one must

1. reassemble $\mathbf{A}(\mu\pm \delta \mathbf{e}_i)$ (full FE integration + BC application),
2. re-solve the eigenproblem (often with shift–invert; typically the dominant cost), and
3. repeat for $i=1,\dots,p$.

Consequently, cost scales like

$$
\underbrace{(2p)}_{\text{central differences}}
\times \big(\underbrace{C_{\mathrm{asm}}}_{\text{full assembly}}
+\underbrace{C_{\mathrm{eig}}}_{\text{eigen-solve}}\big),
$$

which is prohibitive for moderate/large $p$ or large 3D problems where $C_{\mathrm{eig}}$ (factorizations) dominates.

#### Adjoint route

By using the adjoint theory, once can instead compute:

1. One direct eigenpair $(\sigma,\mathbf{v})$: cost $\approx C_{\mathrm{eig}}$.
2. One adjoint eigenvector $\mathbf{a}$ by solving the transposed EVP (or by requesting left eigenvectors in the same SLEPc run) - cost $\approx C_{\mathrm{eig}}$.
    * No FE reassembly; at most a sparse transpose $\mathbf{A}\mapsto \mathbf{A}^{H}$.

3. For each $\mu_i$, evaluate the numerator $\mathbf{a}^{H}(\partial_{\mu_i}\mathbf{A})\mathbf{v}$:
   * If $\partial_{\mu_i}\mathbf{A}$ has an **explicit** term (e.g. $-\tfrac{1}{\mathrm{Re}^{2}}\Delta$), apply the preassembled operator (or a single FE form) to $\mathbf{v}$ and dot with $\mathbf{a}$.
   * If it has an **implicit** baseflow term, compute the baseflow sensitivity once via a *linear* steady-Jacobian solve (reuse Newton Jacobian/factorization), then evaluate the corresponding convection/shear forms on $\mathbf{v}$ and $\mathbf{a}$.

Thus, the total cost in this case is

$$
2\,C_{\mathrm{eig}}
\;+\;\underbrace{C_{\mathrm{bf\text{-}sens}}}_{\text{optional, once per physical parameter}}
\;+\;\underbrace{p\,C_{\mathrm{ip}}}_{\text{$p$ inner products / form applications}},
$$

with $C_{\mathrm{ip}}\ll C_{\mathrm{eig}}$ and $C_{\mathrm{bf\text{-}sens}}\ll C_{\mathrm{eig}}$ (one linear solve on the steady Jacobian, typically reusable).

### Practical consequences

* **One adjoint gives all sensitivities.** After computing $\mathbf{a}$ once, $\partial \sigma/\partial \mu_i$ for any number of parameters reduces to $\mathbf{a}^{H}(\partial_{\mu_i}\mathbf{A})\mathbf{v}$.
* **Matrix transposes, not reassembly.** The adjoint EVP uses $\mathbf{A}^{H}$; PETSc/SLEPc avoid FE re-integration and preserve sparsity/partitioning.
* **Reuse heavy factorizations.** With shift–invert, the expensive factorization for the direct problem has an adjoint analogue performed once.
Finite differences would repeat factorizations per parameter perturbation.
* **Favorable parallel behavior.** No parameter-dependent reassembly implies less communication, and better cache locality.
Operator applications and inner products are bandwidth-bound and scale well.


## Theoretical Background

### Generalized Eigenproblem in Global Stability Theory

We consider a steady incompressible flow $\mathbf{u}(\mathbf{x}, \mu)$ (with pressure $p(\mathbf{x}, \mu)$) that depends on a parameter $\mu$ (e.g., the Reynolds number).
Small perturbations $\mathbf{u}'(\mathbf{x}, t)$ about this base state evolve according to the linearized Navier-Stokes operator, as described in [LSA-FW FEM operators](./fem-operators.md).

Seeking normal-mode solutions $\mathbf{u}'(\mathbf{x}, t) = \mathbf{v}(\mathbf{x}) e^{\sigma t}$ leads to a generalized eigenvalue problem for the perturbation fields.
In discretized form (e.g., after FEM discretization), this can be written as $(1)$, where $\mathbf{A}$ is the system matrix (in our case, the linear stability operator) depending on the baseflow and parameters, $\mathbf{B}$ is the mass matrix, $\sigma$ is the complex eigenvalue ($\sigma = \alpha + \mathrm{i}\omega$, for growth rate $\alpha$ and frequency $\omega$), and $\mathbf{v}$ is the direct eigenvector.

For the incompressible Navier–Stokes equations, $\mathbf{A}$ is generally non-Hermitian and $\mathbf{B}$ is singular (containing zero rows/columns associated with pressure degrees of freedom).

The **adjoint** eigenproblem is defined by the bi-orthogonal eigenvectors $\mathbf{a}$ satisfying

$$
\mathbf{a}^H \mathbf{A}(\mu) = \sigma^* \mathbf{a}^H \mathbf{B} \tag{2},
$$

where $\mathbf{a}$ is the adjoint eigenvector associated with $\sigma^*$ (the complex conjugate of $\sigma$).

### Total Derivative of an Eigenvalue via Adjoint Theory

The main goal of this study is to analyze the sensitivity of the eigenvalue $\sigma$ to a small change in the parameter $\mu$.
Note that $\mu$ can affect the eigenvalue both explicitly, through the coefficients in $\mathbf{A}$ that depend on $\mu$, and implicitly, via the baseflow.

The total derivative $d\sigma/d\mu$ can be obtained using an adjoint-based approach.

Differentiating (1) with respect to $\mu$, noting that $\sigma = \sigma(\mu)$ and $\mathbf{v} = \mathbf{v}(\mu)$, we obtain

$$
(\partial_\mu \mathbf{A})\mathbf{v} + \mathbf{A}(\partial_\mu \mathbf{v}) - (\partial_\mu \sigma)\mathbf{B}\mathbf{v} - \sigma \mathbf{B} (\partial_\mu \mathbf{v}) = 0, \tag{3}
$$

where $\partial_\mu \equiv \frac{\partial}{\partial \mu}$.
Using (1), equation (3) simplifies to

$$
(\partial_\mu \mathbf{A})\mathbf{v} - (\partial_\mu \sigma) \mathbf{B} \mathbf{v} + (\mathbf{A} - \sigma \mathbf{B})(\partial_\mu \mathbf{v})=0.
$$

The term $(\mathbf{A}-\sigma \mathbf{B})(\partial_\mu \mathbf{v})$ vanishes when projected onto the adjoint mode $\mathbf{a}$, since $\mathbf{a}^H(\mathbf{A}-\sigma \mathbf{B})=0$ by (2).
Taking the inner product with $\mathbf{a}^H$ and using this adjoint orthogonality, we isolate the eigenvalue derivative

$$
\frac{d\sigma}{d\mu} = \frac{\mathbf{a}^H(\partial_\mu \mathbf{A})\mathbf{v}}{\mathbf{a}^H\mathbf{B}\mathbf{v}}, \tag{4}
$$

which is the *total* derivative of the eigenvalue with respect to the parameter.

Equation (4) is the fundamental sensitivity result: it shows that the eigenvalue drift $d\sigma$ is proportional to the structural perturbation $d\mathbf{A}$ applied to the system, weighted by the direct and adjoint mode shapes.

This equation is the basis for eigenvalue sensitivity analysis in global stability as introduced by Bottaro *et al.* [[1]](#references).
In particular, (4) reproduces the expression used to define baseflow sensitivity and optional perturbations, and it underlies the 'wavemaker' analysis of Giannetti and Luchini [[2]](#references) for global modes.

> Remark: In practice, one often scales $\mathbf{a}$ and $\mathbf{v}$ such that $\mathbf{a}^H \mathbf{B} \mathbf{v} = 1$, in which case $d\sigma/d\mu = \mathbf{a}^H (\partial_\mu \mathbf{A})\mathbf{v}$.


### Baseflow Sensitivity Equations

The parametric derivative $\partial_\mu \mathbf{A}$ in $(4)$ generally consists of two parts,

* an explicit $\mu$-dependence (e.g., coefficients like $1/\text{Re}$ in the operator), and
* and implicit dependence through the baseflow $\mathbf{\overline{u}}(\mu)$.

The later arises because the baseflow itself changes with $\mu$, altering the linearized operator.

To compute this *implicit* component, one must obtain the sensitivity of the baseflow tot he parameter, $\mathbf{\overline{u}}_\mu(x) \equiv \partial \mathbf{\overline{u}}/\partial \mu$ (and similarly $\overline{p}_\mu$ for pressure).
This is accomplished by differentiating the steady-state governing equations with respect to $\mu$.

As detailed in the [LSA-FW baseflow solver](solver-baseflow.md) documentation, the baseflow $(\overline{\mathbf{u}}, \overline{p})$ satisfies the incompressible Navier-Stokes equations, 

$$
\left\lbrace
\begin{align*}
(\mathbf{\overline{u}} \cdot \nabla ) \mathbf{\overline{u}} + \nabla \overline{p} - \frac{1}{\text{Re}} \Delta\mathbf{\overline{u}} = \mathbf{f} \\
\nabla \cdot \mathbf{\overline{u}} = 0
\end{align*}\right., \tag{5}
$$

where the parameter $\mu$ can, for example, be $\mu = \text{Re}$, or enter the equations via fluid properties, boundary conditions or forcing.

Differentiating $(5)$ with respect to $\mu$ yields a linear system for the baseflow sensitivity.
For the case of $\mu = \text{Re}$ with no forcing, this differentiation gives 

$$
\left\lbrace
\begin{align*}
(\mathbf{\overline{u}} \cdot \nabla )\partial_\text{Re} \mathbf{\overline{u}} + (\mathbf{\partial_\text{Re} \overline{u}} \cdot \nabla )\mathbf{\overline{u}} + \nabla \partial_\text{Re} \overline{p} - \frac{1}{\text{Re}} \Delta\partial_\text{Re} \mathbf{\overline{u}} =-\frac{1}{\text{Re}^2} \Delta \mathbf{\overline{u}} \\
\nabla \cdot \partial_\text{Re} \mathbf{\overline{u}} = 0
\end{align*}\right.. \tag{6}
$$

Here the right-hand side comes from the explicit $\text{Re}$-dependence of the diffusion term.

Equation $(6)$ is a linear inhomogeneous Navier–Stokes system that can be solved subject to the same boundary conditions as the baseflow.

In more general terms, for an arbitrary parameter $\mu$, we would write the steady-state equations as $F(\mathbf{\overline{u}},\mu)=0$ and obtain the linearized system

$$
\mathcal{L}_B(\mathbf{\overline{u}}) = -F_\mu(\mathbf{\overline{u}},\mu) \tag{7}
$$

where $\mathcal{L}_B(\mathbf{\overline{u}})$ is the Jacobian of the steady Navier–Stokes operator with respect to the state, and $-F_\mu$ is the partial derivative of the steady equations with respect to the parameter (the source term).
Equation $(6)$ is a specific instance of $(7)$. 

Solving this linear system yields $\mathbf{\overline{u}}_\mu(\mathbf{\overline{x}})$, the rate of change of the baseflow with the parameter. 

In summary, $\mathbf{\overline{u}}_\mu$ quantifies the indirect effect of the parameter on the flow (through altering the base state).


## Perturbation Operator Variation due to Baseflow Change

We can now determine how the above-developed variations in the baseflow modify the linear stability operator $\mathbf{A}$.
The operator $\mathbf{A}(\mu)$ can be viewed as a functional of the baseflow $\mathbf{\overline{u}}$.
For a small baseflow perturbation $\delta \mathbf{\overline{u}}(x)$, the corresponding first-order change in the linearized operator can be expressed as a linear mapping $\delta \mathbf{A} = (\partial \mathbf{A}/\partial \mathbf{\overline{u}})\delta \mathbf{\overline{u}}$.

In the incompressible Navier–Stokes linearization, the baseflow enters the perturbation operator through the convection (advection) terms.
Using an index-free notation, the Frechét derivative of $\mathbf{A}$ with respect to $\mathbf{\overline{u}}$ applied to an arbitrary vector field $w(\mathbf{x})$ and acting on a perturbation $q=(\mathbf{u}',p')$ is

$$
(\partial_\mathbf{\overline{u}}\mathbf{A})[w]q = (w\cdot \nabla) \mathbf{u}' + (\mathbf{u}' \cdot \nabla) w, \tag{8}
$$

which represents the two ways a change in baseflow can affect the linearized momentum equations.

The first term on the right hand side is the variation of the convective transport of perturbations (the baseflow $\mathbf{\overline{u}}$ is perturbed by $w$).

The second term is the variation of the 'shear' or baseflow gradient term.

Consequently, the total parameter dependence of $\mathbf{A}$ can now be decomposed as 

$$
\partial_\mu \mathbf{A} = \underbrace{(\partial_\mu \mathbf{A})_\text{explicit}}_{\text{coeff. change}} + \underbrace{\partial_\mathbf{\overline{u}}\mathbf{A}[\mathbf{\overline{u}_\mu}]}_{\text{baseflow change}}, \tag{9}
$$

where the first term is non-zero only if $\mu$ appears explicitly in the equations (for example, in the $\mu = \text{Re}$ case), and the second term accounts for the operator change due to baseflow adjustments.

To see the effect ot the baseflow variation on the eigenvalue, we insert the above decomposition $(8)$ into the sensitivity formula $(4)$.
The contribution from the baseflow change is then

$$
\mathbf{a}^H [(\partial_\mathbf{\overline{u}} \mathbf{A})[\mathbf{\overline{u}}_\mu]] \mathbf{v}.
$$.

Using the definition $(8)$ and converting to a continuous inner-product notation, this term can be written as an integral over the flow domain,

$$
\mathbf{a}^{H}\!\left[(\partial_{\mathbf{\overline{u}}}\mathbf{A})[\mathbf{\overline{u}}_{\mu}]\right]\mathbf{v}
=\int_{\Omega} \mathbf{a}^{*}(x)\cdot\Big[(\mathbf{\overline{u}}_{\mu}\cdot\nabla)\mathbf{v}+(\mathbf{v}\cdot\nabla)\mathbf{\overline{u}}_{\mu}\Big]\; d\Omega ,
$$

where $\mathbf{a}^{*}$ denotes the complex-conjugate of $\mathbf{a}$ (since $\mathbf{a}^{H}$ is the conjugate transpose).
By applying integration by parts to shift derivatives off of $\mathbf{v}$ and onto $\mathbf{a}^{*}$ (and assuming the boundary terms vanish under the chosen adjoint boundary conditions or homogeneous far-field conditions), this expression can be symmetrized.

In particular, for divergence-free fields one finds:

$$
\int_{\Omega} \mathbf{a}^{*}\cdot\big[(\mathbf{\overline{u}}_{\mu}\cdot\nabla)\mathbf{v}\big]\; d\Omega
= -\int_{\Omega} \big(\mathbf{\overline{u}}_{\mu}\cdot\nabla \mathbf{a}^{*}\big)\cdot \mathbf{v} \; d\Omega ,
$$

and

$$
\int_{\Omega} \mathbf{a}^{*}\cdot\big[(\mathbf{v}\cdot\nabla)\mathbf{\overline{u}}_{\mu}\big]\; d\Omega
= \int_{\Omega} \big(\mathbf{\overline{u}}_{\mu}\cdot\nabla \mathbf{a}^{*}\big)\cdot \mathbf{v} \; d\Omega ,
$$

where we have used $\nabla\!\cdot \mathbf{\overline{u}}_{\mu}=\nabla\!\cdot \mathbf{v}=0$ and assumed sufficient decay or cancellation on the boundaries.

Notably, both terms yield the same integrand after integration by parts.
Adding them together, we obtain:

$$
\mathbf{a}^{H}\!\left[(\partial_{\mathbf{\overline{u}}}\mathbf{A})[\mathbf{\overline{u}}_{\mu}]\right]\mathbf{v}
= -\,2\int_{\Omega} \big(\mathbf{\overline{u}}_{\mu}\!\cdot\nabla \mathbf{a}^{*}\big)\cdot \mathbf{v} \, d\Omega ,
\tag{10}
$$

which is a manifestly symmetric overlap of $\mathbf{\overline{u}}_{\mu}$, the adjoint mode gradient, and the direct mode.
This symmetric form highlights that regions where the baseflow sensitivity $\mathbf{\overline{u}}_{\mu}$ is large **and** the direct and adjoint mode have large gradients (indicating a strong convective 'self-interaction') will contribute most to shifting the eigenvalue.

In other words, $(10)$ identifies the sensitive regions of the flow (often called the 'wavemaker' when localized). 

The minus sign in $(10)$ indicates that if the baseflow change $\mathbf{\overline{u}}_\mu$ reinforces the adjoint mode's feedback on the direct mode, it will increase the growth rate if $d\sigma/d\mu$ is positive.

## Case: Eigenvalue Sensitivity to Reynolds

To make the foregoing formulation concrete, we specialize to the sensitivity of an eigenvalue with respect to the Reynolds number $\mu = \text{Re}$ in an incompressible flow.

In this case, the explicit part of $\partial_{\text{Re}}\mathbf{A}$ comes from the viscous term, as the linear stability operator contains Laplacian diffusion terms with coefficient $1/\text{Re}$, so 

$$
\partial_{\text{Re}}(1/\text{Re}) = -1/\text{Re}^{2}
$$.

Thus,  the explicit operator derivative is

$$
\big(\partial_{\text{Re}}\mathbf{A}\big)_{\text{explicit}}
= -\,\frac{1}{\text{Re}^{2}}\,\Delta , \tag{11}
$$

i.e., a scaling of the Laplacian (acting on the velocity perturbation).

Meanwhile, the implicit part requires solving the baseflow sensitivity equations $(6)$ for $\mathbf{\overline{u}}_{\text{Re}}(x)$.
Once it is obtained, one can evaluate $(\partial_{\mathbf{\overline{u}}}\mathbf{A})[\mathbf{\overline{u}}_{\text{Re}}]$ as described by $(8)$.

Substituting these into the general sensitivity formula $(4)$, we obtain a complete expression for the eigenvalue sensitivity to $\text{Re}$.

Using $\mathbf{a}$ for the adjoint mode and $\mathbf{v}$ for the direct mode (with $\mathbf{a}^{H}\mathbf{B}\,\mathbf{v}=1$), the result can be written as

$$
\frac{d\sigma}{d\text{Re}}
= \mathbf{a}^{H}\!\left[\big(\partial_{\text{Re}}\mathbf{A}\big)_{\text{explicit}}
+ \big(\partial_{\overline{\mathbf{u}}}\mathbf{A}\big)\!\left[\overline{\mathbf{u}}_{\text{Re}}\right]\right]\mathbf{v}\,,
$$

or, in expanded form, as the sum of an explicit and a baseflow contribution,

$$
\frac{d\sigma}{d\text{Re}}
= -\,\frac{1}{\text{Re}^{2}}\int_{\Omega} \nabla \mathbf{a}^{*} : \nabla \mathbf{v}\, d\Omega
\;-\; \int_{\Omega}\!\left[\,
\big(\overline{\mathbf{u}}_{\text{Re}}\!\cdot\!\nabla \mathbf{a}^{*}\big)\!\cdot\!\mathbf{v}
+ \big(\overline{\mathbf{u}}_{\text{Re}}\!\cdot\!\nabla \mathbf{v}\big)\!\cdot\!\mathbf{a}^{*}
\,\right]\! d\Omega ,
\tag{12}
$$

where the first term comes from $(11)$ and the second term comes from the definition $(8)$ (prior to symmetrization).

All quantities in $(12)$ are understood to be functions on the flow domain $\Omega$: $\mathbf{a}^{*}$ and $\mathbf{v}$ are the adjoint and direct eigenmode velocity fields (the pressure components do not appear in these integrals, consistent with using the $\mathbf{B}$–inner product on velocities), and $\overline{\mathbf{u}}_{\text{Re}}$ is the solution of $(6)$ for the baseflow variation due to $\text{Re}$.

The colon in $\nabla \mathbf{a}^{*} : \nabla \mathbf{v}$ denotes the Frobenius inner product of the velocity-gradient tensors.

Equation $(12)$ is a **complete sensitivity formula** for how a global mode's eigenvalue $\sigma$ shifts with $\text{Re}$ in an incompressible flow:

- The first term represents the direct effect of changing $\text{Re}$ (e.g., increasing $\text{Re}$ reduces viscous damping, typically destabilizing the mode if $\nabla \mathbf{a}^{*}\!:\!\nabla \mathbf{v}$ is positive on average).

- The second term encapsulates the indirect effect.
As $\text{Re}$ changes, the baseflow $\overline{\mathbf{u}}$ is altered, which in turn can stabilize or destabilize the mode depending on the structure of $\overline{\mathbf{u}}_{\text{Re}}$ relative to the mode shapes.

Note that both integrals in $(12)$ are to be evaluated with the initial baseflow (at the $\text{Re}$ about which sensitivity is computed), and $\overline{\mathbf{u}}_{\text{Re}}$ is obtained from the linear system $(6)$ at that same baseflow.

The symmetric form derived in $(10)$ can also be applied to simplify $(12)$.
Noting that

$$
\int_{\Omega}\big(\overline{\mathbf{u}}_{\text{Re}}\!\cdot\!\nabla \mathbf{v}\big)\!\cdot\!\mathbf{a}^{*}\, d\Omega
= \int_{\Omega}\big(\overline{\mathbf{u}}_{\text{Re}}\!\cdot\!\nabla \mathbf{a}^{*}\big)\!\cdot\!\mathbf{v}\, d\Omega
$$

under the divergence–free assumption, we can combine the two baseflow terms.
Then $(12)$ becomes

$$
\frac{d\sigma}{d\text{Re}}
= -\,\frac{1}{\text{Re}^{2}}\int_{\Omega} \nabla \mathbf{a}^{*} : \nabla \mathbf{v}\, d\Omega
\;-\; 2 \int_{\Omega} \big(\overline{\mathbf{u}}_{\text{Re}}\!\cdot\!\nabla \mathbf{a}^{*}\big)\!\cdot\!\mathbf{v}\, d\Omega ,
\tag{13}
$$

which is equivalent to $(12)$ but highlights the symmetric adjoint–direct interaction driving the baseflow sensitivity.

Equation $(13)$ clearly delineates the two physical mechanisms by which $\text{Re}$ influences the eigenvalue: through viscous effects (explicit $-1/\text{Re}^{2}$ term) and through modification of the base-state advection (implicit term with $\overline{\mathbf{u}}_{\text{Re}}$).

This formulation is consistent with earlier sensitivity studies of global modes and provides a rigorous foundation for computing eigenvalue gradients.

## Assumptions and Further Considerations

### Adjoint Boundary Conditions

In the above derivation, it is assumed that the direct and adjoint eigenmodes satisfy complementary homogeneous boundary conditions.

For example, if the direct problem imposes homogeneous Dirichlet (no-slip) velocity at a wall, the adjoint problem imposes homogeneous Dirichlet on the adjoint velocity at that wall as well (so that the integration by parts on viscous terms yields no surface contributions).

At outflow boundaries, if the direct problem uses do-nothing or convective outflow (zero pressure), the adjoint problem typically requires an inflow condition (adjoint velocities vanishing or set to enforce zero incoming characteristics) to eliminate boundary terms. 

We assume here that such conditions are in place so that the integrations by parts [above](#perturbation-operator-variation-due-to-baseflow-change) are valid.

### Divergence-Free Spaces

It has been assumed that the velocity perturbations and baseflow sensitivity fields are divergence-free (incompressible).

In a discretized setting, one typically works with mixed function spaces(velocity-pressure), as described in [LSA-FW FEM Spaces](./fem-spaces.md).
The inner product $\langle \mathbf{a}, \mathbf{B}, \mathbf{v} \rangle$ in $(4)$ is effectively an $L^2$ inner product on velocities because $\mathbf{B}$ has the effect of weighting the velocity components (and annihilating pressure components).

The pressure does not explicitly appear in the sensitivity formula except through its influence on $\mathbf{\overline{u}}_\mu$ and the enforcement of $\nabla\cdot \mathbf{\overline{u}}_\mu = 0$.
In a finite-element weak formulation, the pressure acts as a Lagrange multiplier enforcing incompressibility; differentiating this constraint yields $\nabla\cdot \mathbf{\overline{u}}_\mu = 0$ in $(6)$.
We have also used $\nabla\cdot \mathbf{v}=0$ and $\nabla\cdot \mathbf{\overline{u}}_\mu=0$ to simplify terms in the integration by parts.

### Linearity

The sensitivity analysis here is first-order (linear perturbations in both the baseflow and operator).
It assumes the baseflow variation $\mathbf{\overline{u}}_\mu\;d\mu$ is small enough that higher-order terms are negligible.

This is generally valid in the asymptotic sense for infinitesimal $d\mu$.
If one requires large parameter changes, higher-order derivatives (as recently studied by Müller et al. [[3]](#references)) or a re-linearization may be needed for accuracy, but those are beyond the scope of this first-order analysis.

## Implementation within LSA-FW *(Planned)*

> Note that this section is an implementation plan (mostly for dev purposes), as no adjoint sensitivity is supported by LSA-FW at the time of writing.
> It could be that the actual implementation gets reintegrated before this section gets updated, causing an inconsistency between code and documentation.
> We will try to make this inconsistency, if any, as short as possible.

This section outlines the planned implementation path in LSA-FW.
The intent is to make the workflow reproducible, modular, and aligned with existing assemblers and solver wrappers. 

### Overview and Interfaces

The implementation will be structured as a small set of composable routines. 
A possible API could include:

* `compute_baseflow_sensitivity(mu: Parameter) -> (u_mu, p_mu)`
* `solve_direct_mode(target: ShiftSpec) -> (sigma, v)`
* `solve_adjoint_mode(pair_with: (sigma, v)) -> a`
* `evaluate_sensitivity(mu, v, a, u_mu) -> d_sigma_d_mu`
* `validate(mu, step: float, spec) -> error_report`

All routines will reuse the existing FEM operators, spaces, boundary-condition machinery, and PETSc/SLEPc wrappers.

### Baseflow sensitivity $\left(\overline{\mathbf{u}}_ {\mu},\;\overline{p}_{\mu}\right)$



1. Let $F(\overline{\mathbf{u}}, \overline{p}; \mu)=0$ be the steady NS residual assembled by the (existing) steady solver.
At the converged baseflow, the Jacobian

$$
\mathbf{J} = \partial_{(\overline{\mathbf{u}},\,\overline{p})} F(\overline{\mathbf{u}}, \overline{p}; \mu)
$$

can be assembled (same forms, spaces, and BCs as the Newton step).

2. The baseflow sensitivity can be computed from the linearized system

$$
\mathbf{J}
\begin{bmatrix}
   \overline{\mathbf{u}}_{\mu} \\[2pt] p_{\mu}
\end{bmatrix} = -\,F_{\mu}(\overline{\mathbf{u}}, \overline{p}; \mu).
$$

3. For $\mu = \text{Re}$ (with $\nu = 1/\text{Re}$), $F_{\mu}$ is analytic:

   $$
   F_{\mu}^{(u)} = +\frac{1}{\text{Re}^{2}}\,\Delta\,\overline{\mathbf{u}}.
   $$

   The right-hand side can therefore reuse the viscous form to evaluate $\Delta\overline{\mathbf{u}}$ and be scaled by $1/\text{Re}^{2}$.

4. The linear solve can reuse the steady solver'’'s PETSc configuration (e.g., LU or GMRES+PC).
The solution $(\overline{\mathbf{u}}_{\mu}, p_{\mu})$ can be cached for subsequent sensitivity evaluations.

> **Notes.** Adjoint-compatible BCs should be enforced so that subsequent integrations by parts are valid.
> The nullspace treatment used by the steady solver must be retained.

### Direct eigenmode $\left(\mathbf{v}\right)$

Direct eigenpairs $(\sigma, \mathbf{v})$ can already be computed within LSA-FW. 
Refer to the [eigensolver documentation](./solver-eigen.md).

### Adjoint eigenmode $\left(\mathbf{a}\right)$

Once the eigenmode under study is obtained, we can obtain the adjoint $\mathbf{a}$ associated with $\sigma^{*}$ and normalize bi-orthogonally.

Two equivalent options are possible:

* Left eigenvectors via SLEPc. EPS can be configured to return left eigenvectors directly, paired with $(\sigma,\mathbf{v})$.
* Explicit adjoint operator. The transpose/conjugate of the assembled blocks can be used to solve

  $$
  \mathbf{A}^{H}\mathbf{a} = \sigma^{*}\,\mathbf{B}^{H}\mathbf{a}.
  $$


In both cases, $\mathbf{a}$ should be scaled to satisfy the bi-orthogonality

$$
\mathbf{a}^{H}\mathbf{B}\,\mathbf{v} = 1.
$$

> **Notes.** Adjoint BCs should mirror the direct BCs so that boundary terms vanish in the sensitivity integrals.

### Sensitivity $\left(d\sigma/d\mu\right)$

Once the above terms have been computed, we can assemble the scalar derivative using explicit and baseflow-induced terms.

1. Explicit term (example $\mu = \text{Re}$).

   $$
   \mathbf{a}^{H} \big(\partial_{\text{Re}}\mathbf{A}\big)\mathbf{v}
   = -\frac{1}{\text{Re}^{2}} \int_{\Omega} \nabla \mathbf{a}^{*} : \nabla \mathbf{v}\, d\Omega.
   $$

   In UFL this can be written as follows:

   ```python
   assemble(inner(grad(conj(a)), grad(v)) * dx) * (-1.0 / re**2)
   ```

2. Baseflow (implicit) term. Using $\overline{\mathbf{u}}_{\mu}$:

   $$
   \mathbf{a}^{H}\!\left[(\partial_{\overline{\mathbf{u}}}\mathbf{A})[\overline{\mathbf{u}}_{\mu}]\right]\mathbf{v}
   = \int_{\Omega}
   \Big[(\overline{\mathbf{u}}_{\mu}\!\cdot\!\nabla\mathbf{v})\!\cdot\!\mathbf{a}^{*}
       +(\mathbf{v}\!\cdot\!\nabla\overline{\mathbf{u}}_{\mu})\!\cdot\!\mathbf{a}^{*}\Big]\, d\Omega.
   $$

   This can be evaluated by reusing the existing convection and shear forms with $\overline{\mathbf{u}}_{\mu}$ as the base field.

The final result can be reported as

$$
\frac{d\sigma}{d\mu}
= \underbrace{\mathbf{a}^{H}(\partial_{\mu}\mathbf{A})\mathbf{v}}_{\text{explicit}}
+ \underbrace{\mathbf{a}^{H}\!\left[(\partial_{\overline{\mathbf{u}}}\mathbf{A})[\overline{\mathbf{u}}_{\mu}]\right]\mathbf{v}}_{\text{baseflow}},
$$

with $\Re(\cdot)$ as growth-rate sensitivity and $\Im(\cdot)$ as frequency sensitivity.

### Validation strategy

The implementation will include a finite-difference check:

1. Recompute baseflow and eigenpair at $\mu \pm \Delta\mu$.
2. Form the centered difference

   $$
   \frac{\sigma(\mu+\Delta\mu)-\sigma(\mu-\Delta\mu)}{2\,\Delta\mu}
   $$

   and compare against the adjoint prediction.
3. Perform at least two $\Delta\mu$ values to confirm first-order convergence.

Validation cases will include a 2D cylinder and a channel flow where literature values exist.


## References

[1] Luchini, P., Bottaro, A. (2014). Adjoint Equations in Stability Analysis. *Annual Review of Fluid Mechanics*, 46, 493–517. (Supplemental Appendix cited in this repo)

[2] Luchini, P., Giannetti, F. (2007). Structural sensitivity of the first instability of the cylinder wake. *J. Fluid Mech.*, 581, 167–197.

[3] Müller, J. S., Knechtel, S. J., et al. (2024). Combining Bayesian optimization with adjoint-based gradients for efficient control of flow instabilities.