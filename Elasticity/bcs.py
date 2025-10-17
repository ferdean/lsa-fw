"""LSA-FW Elasticity boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from Meshing.core import Mesher
import dolfinx.mesh as dmesh
import dolfinx.fem as dfem

import numpy as np


@dataclass(frozen=True)
class BoundaryConditions:
    """Generic boundary conditions collector."""

    dirichlet: list[dfem.DirichletBC] = field(default_factory=list)
    neumann: list[tuple[int, dfem.Function]] = field(default_factory=list)
    robin: list[tuple[int, dfem.Constant, dfem.Function]] = field(default_factory=list)


@dataclass(frozen=True)
class ComponentDirichlet:
    """Dirichlet constraint on selected displacement components."""

    tags: Iterable[int]
    components: Sequence[int]
    value: float | Sequence[float] | Callable[[np.ndarray], np.ndarray] = 0.0


@dataclass(frozen=True)
class AxisNormalBc:
    """Axis-aligned surrogate for a normal-displacement constraint."""

    tags: Iterable[int]
    axis: int
    value: float | Callable[[np.ndarray], np.ndarray] = 0.0


def _as_value_fn(
    val: float | Sequence[float] | Callable[[np.ndarray], np.ndarray], size: int
) -> Callable[[np.ndarray], np.ndarray]:
    if callable(val):
        return val
    vec = np.atleast_1d(val).astype(float)
    if vec.size == 1 and size > 1:
        vec = np.repeat(vec, size)
    assert vec.size == size, f"value size {vec.size} != {size}"

    def _fn(x: np.ndarray) -> np.ndarray:
        return np.tile(vec[:, None], (1, x.shape[1]))

    return _fn


def _build_component_dirichlet_bcs(
    mesh: dmesh.Mesh,
    V: dfem.FunctionSpace,  # vector space
    facet_tags: dmesh.MeshTags,
    specs: Iterable[ComponentDirichlet],
) -> list[tuple[int, dfem.DirichletBC]]:
    tdim = mesh.topology.dim
    vsize = int(V.dofmap.bs)
    # Cache subspaces and zero funcs
    subs, Vi_list = [], []
    for i in range(vsize):
        Vi, _ = V.sub(i).collapse()
        subs.append(V.sub(i))
        Vi_list.append(Vi)

    out: list[tuple[int, dfem.DirichletBC]] = []
    for spec in specs:
        val_fn = _as_value_fn(spec.value, vsize)
        comp_fn = {}
        for c in spec.components:
            f = dfem.Function(Vi_list[c])
            f.interpolate(lambda x, c=c: val_fn(x)[c, :])
            comp_fn[c] = f
        for tag in spec.tags:
            facets = facet_tags.find(tag)
            for c in spec.components:
                dofs = dfem.locate_dofs_topological(
                    (subs[c], Vi_list[c]), tdim - 1, facets
                )
                bc = dfem.dirichletbc(comp_fn[c], dofs, subs[c])
                out.append((tag, bc))
    return out


def _build_axis_normal_bcs(
    mesh: dmesh.Mesh,
    V: dfem.FunctionSpace,
    facet_tags: dmesh.MeshTags,
    specs: Iterable[AxisNormalBc],
) -> list[tuple[int, dfem.DirichletBC]]:
    # Axis-normal is just a ComponentDirichlet on a single component
    comp_specs = [
        ComponentDirichlet(tags=s.tags, components=(s.axis,), value=s.value)
        for s in specs
    ]
    return _build_component_dirichlet_bcs(mesh, V, facet_tags, comp_specs)


def define_bcs(
    mesher: Mesher,
    V: dfem.FunctionSpace,
    *,
    component: Iterable[ComponentDirichlet] = (),
    axis_normal: Iterable[AxisNormalBc] = (),
):
    if mesher.facet_tags is None:
        raise ValueError("Mesh boundaries are not properly tagged.")
    mesh, tags = mesher.mesh, mesher.facet_tags

    bcs = []
    bcs += _build_component_dirichlet_bcs(mesh, V, tags, component)
    bcs += _build_axis_normal_bcs(mesh, V, tags, axis_normal)

    return BoundaryConditions(dirichlet=bcs)
