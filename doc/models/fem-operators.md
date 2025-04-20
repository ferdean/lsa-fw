# LSA-FW FEM Module (Operators)

> [Back to FEM Overview](fem.md)

---

Linear stability analysis of fluid flows requires linearizing the Navier-Stokes equations around a given base flow.
This module provides tools to assemble the discrete matrices for this linearized system.

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
    sM + C_1 + C_2 + D & G \\
    D & 0
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

<!-- Test code -->

```python
from typing import Optional, Sequence, Tuple
import logging

import numpy as np
from dolfinx import fem
from dolfinx.fem import Function, FunctionSpace, dirichletbc, form
from petsc4py import PETSc
from ufl import TrialFunction, TestFunction, inner, grad, div, dx, dot

logger = logging.getLogger(__name__)


class VariationalForms:

    @staticmethod
    def mass(u: TrialFunction, v: TestFunction, degree: int):
        return inner(u, v) * dx(metadata={"quadrature_degree": degree})

    @staticmethod
    def viscous(u: TrialFunction, v: TestFunction, Re: float, degree: int):
        return (1.0 / Re) * inner(grad(u), grad(v)) * dx(metadata={"quadrature_degree": degree})

    @staticmethod
    def convection(u: TrialFunction, v: TestFunction, u_base: Function, degree: int):
        return inner(dot(u_base, grad(u)), v) * dx(metadata={"quadrature_degree": degree})

    @staticmethod
    def shear(u: TrialFunction, v: TestFunction, u_base: Function, degree: int):
        return inner(dot(u, grad(u_base)), v) * dx(metadata={"quadrature_degree": degree})

    @staticmethod
    def pressure_gradient(p: TrialFunction, v: TestFunction, degree: int):
        return -p * div(v) * dx(metadata={"quadrature_degree": degree})

    @staticmethod
    def divergence(u: TrialFunction, q: TestFunction, degree: int):
        return div(u) * q * dx(metadata={"quadrature_degree": degree})

    @staticmethod
    def neumann_rhs(v: TestFunction, g: Function, degree: int):
        return inner(v, g) * dx(metadata={"quadrature_degree": degree})


def export_petsc_matrix(mat: PETSc.Mat, path: str):
    """
    Export a PETSc matrix to a binary file.

    Parameters:
    ----------
    mat : PETSc.Mat
        The matrix to export.
    path : str
        The target file path (typically with `.bin` extension).

    Notes:
    ------
    The binary format can be read by PETSc viewers or converted using `petsc/bin/matlab/PetscBinaryRead`.
    """
    viewer = PETSc.Viewer().createBinary(path, mode=PETSc.Viewer.Mode.WRITE, comm=mat.comm)
    mat.view(viewer)


def attach_pressure_nullspace(Q: FunctionSpace, mat: PETSc.Mat) -> PETSc.MatNullSpace:
    """
    Attach a constant-pressure nullspace to a matrix for saddle-point systems.

    Parameters:
    ----------
    Q : FunctionSpace
        The pressure function space (typically scalar-valued).
    mat : PETSc.Mat
        The PETSc matrix corresponding to the block system.

    Returns:
    -------
    PETSc.MatNullSpace
        The nullspace object associated with the constant-pressure mode.

    Notes:
    ------
    This is required for correct Krylov solver behavior when solving
    incompressible Navier–Stokes problems with pressure in a nullspace.
    """
    dim = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    vec = PETSc.Vec().createMPI(dim, comm=mat.comm)
    vec.set(1.0)
    vec.assemble()
    ns = PETSc.MatNullSpace().create(constant=True, vectors=[vec])
    mat.setNullSpace(ns)
    return ns


class LinearizedNSOperator:
    """
    Finite element operator assembler for linearized incompressible Navier-Stokes systems.
    Supports eigenvalue and time-dependent analyses on mixed velocity-pressure formulations.
    """

    def __init__(
        self,
        V: FunctionSpace,
        Q: FunctionSpace,
        base_flow: Function,
        Re: float,
        u_bcs: Optional[Sequence[fem.DirichletBC]] = None,
        p_bcs: Optional[Sequence[fem.DirichletBC]] = None,
    ):
        if V.mesh != Q.mesh:
            raise ValueError("Velocity and pressure spaces must be defined on the same mesh.")
        if not isinstance(base_flow, Function) or base_flow.function_space != V:
            raise ValueError("base_flow must be a dolfinx.Function defined in the velocity space V.")

        self.V = V
        self.Q = Q
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.p = TrialFunction(Q)
        self.q = TestFunction(Q)
        self.base_flow = base_flow
        self.Re = Re
        self.u_bcs = u_bcs or []
        self.p_bcs = p_bcs or []

        self.quad_degree = max(V.ufl_element().degree(), Q.ufl_element().degree()) + 1

    def _assemble(self, a: fem.Form, bcs: Sequence[fem.DirichletBC]) -> PETSc.Mat:
        mat = fem.petsc.assemble_matrix(a, bcs=bcs)
        mat.assemble()
        return mat

    def assemble_mass_matrix(self) -> PETSc.Mat:
        logger.info("Assembling velocity mass matrix.")
        a_mass = form(VariationalForms.mass(self.u, self.v, self.quad_degree))
        return self._assemble(a_mass, self.u_bcs)

    def assemble_diffusion_matrix(self) -> PETSc.Mat:
        a_diff = form(VariationalForms.viscous(self.u, self.v, self.Re, self.quad_degree))
        return self._assemble(a_diff, self.u_bcs)

    def assemble_convection_matrix(self) -> PETSc.Mat:
        a_conv = form(VariationalForms.convection(self.u, self.v, self.base_flow, self.quad_degree))
        return self._assemble(a_conv, self.u_bcs)

    def assemble_shear_matrix(self) -> PETSc.Mat:
        a_shear = form(VariationalForms.shear(self.u, self.v, self.base_flow, self.quad_degree))
        return self._assemble(a_shear, self.u_bcs)

    def assemble_pressure_gradient_matrix(self) -> PETSc.Mat:
        a_pg = form(VariationalForms.pressure_gradient(self.p, self.v, self.quad_degree))
        return self._assemble(a_pg, self.p_bcs)

    def assemble_divergence_matrix(self) -> PETSc.Mat:
        a_div = form(VariationalForms.divergence(self.u, self.q, self.quad_degree))
        return self._assemble(a_div, self.u_bcs)

    def assemble_linearized_operator(self) -> PETSc.Mat:
        logger.info("Assembling full linearized Navier-Stokes operator.")
        A = self.assemble_diffusion_matrix()
        for add_term in [
            self.assemble_convection_matrix(),
            self.assemble_shear_matrix(),
        ]:
            A.axpy(1.0, add_term, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

        G = self.assemble_pressure_gradient_matrix()
        D = self.assemble_divergence_matrix()

        mat = PETSc.Mat().createNest([[A, G], [D, None]])
        mat.assemble()
        attach_pressure_nullspace(self.Q, mat)
        return mat

    def assemble_generalized_system(self) -> Tuple[PETSc.Mat, PETSc.Mat]:
        logger.info("Assembling generalized eigenvalue system (A, M).")
        A = self.assemble_linearized_operator()
        M_v = self.assemble_mass_matrix()

        n_p = self.Q.dofmap.index_map.size_local * self.Q.dofmap.index_map_bs
        M_p = PETSc.Mat().createAIJ([n_p, n_p], comm=M_v.comm)
        M_p.setUp()

        M = PETSc.Mat().createNest([[M_v, None], [None, M_p]])
        M.assemble()
        return A, M

    def assemble_neumann_rhs(self, g: Function) -> PETSc.Vec:
        logger.info("Assembling Neumann right-hand side vector.")
        rhs_form = form(VariationalForms.neumann_rhs(self.v, g, self.quad_degree))
        b = fem.petsc.assemble_vector(rhs_form)
        fem.petsc.apply_lifting(b, [rhs_form], bcs=[self.u_bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.u_bcs)
        return b

    def pin_pressure_dof(self, A: PETSc.Mat, dof: int = 0):
        logger.info(f"Pinning pressure DoF at index {dof}.")
        A.zeroRowsColumns([dof], diag=1.0)

    def export_matrix(self, A: PETSc.Mat, path: str):
        logger.info(f"Exporting matrix to {path}.")
        export_petsc_matrix(A, path)
```