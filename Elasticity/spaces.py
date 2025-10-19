"""LSA-FW Elasticity function spaces handler."""

import dolfinx.mesh as dmesh
from basix.ufl import element
from dolfinx.fem import FunctionSpace, functionspace

from Meshing.utils import iCellType
from FEM.utils import iElementFamily


def define_space(
    mesh: dmesh.Mesh, *, degree: int = 1, gdim: int | None = None
) -> FunctionSpace:
    """Define the displacement function space (vector-valued Sobolev space of order 1)."""
    cell = iCellType.from_dolfinx(mesh.topology.cell_type).to_basix()
    element_h1 = element(
        family=iElementFamily.LAGRANGE.to_dolfinx(),
        cell=cell,
        degree=degree,
        shape=(gdim or mesh.geometry.dim,),
    )
    return functionspace(mesh, element_h1)
