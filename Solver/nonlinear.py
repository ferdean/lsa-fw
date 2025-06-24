"""LSA-FW Nonlinear Solver."""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

import dolfinx.fem as dfem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc

from FEM.operators import BaseAssembler


logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
    }
)
logging.getLogger("matplotlib.ticker").disabled = True
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True


class NewtonSolver:
    """Newton solver implementation, using PETSc for linear solves."""

    def __init__(self, assembler: BaseAssembler) -> None:
        """Initialize."""
        self._assembler = assembler
        self._residual_history: list[float] = []

        self._dwh = dfem.Function(assembler.spaces.mixed)
        self._A, self._b = self._assembler.get_matrix_forms()

        self._solver = PETSc.KSP().create(assembler.spaces.mixed.mesh.comm)
        self._solver.setOperators(self._A.raw)

    def _reassemble(self) -> None:
        with self._b.raw.localForm() as loc_b:
            loc_b.set(0)
        self._A.zero_all_entries()
        assemble_matrix(
            self._A.raw, self._assembler.jacobian, bcs=list(self._assembler.bcs)
        )
        self._A.assemble()

        assemble_vector(self._b.raw, self._assembler.residual)
        self._b.ghost_update(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self._b.scale(-1)

        apply_lifting(
            self._b.raw,
            [self._assembler.jacobian],
            [list(self._assembler.bcs)],
            x0=[self._assembler.sol.x.petsc_vec],
            alpha=1.0,
        )
        set_bc(
            self._b.raw,
            [*self._assembler.bcs],
            self._assembler.sol.x.petsc_vec,
            1.0,
        )
        self._b.ghost_update(
            addv=PETSc.InsertMode.INSERT_VALUES,
            mode=PETSc.ScatterMode.FORWARD,
        )

    def _solve_linear(self) -> None:
        self._solver.solve(self._b.raw, self._dwh.x.petsc_vec)
        self._dwh.x.scatter_forward()

    def solve(
        self, *, max_it: int = 500, atol: float = 1e-5, damping_factor: float = 1.0
    ) -> dfem.Function:
        """Solve nonlinear problem."""
        if not (0.0 < damping_factor <= 1.0):
            logger.warning(
                "Damping factor should be in the (0, 1] range, but %.2f was given. Using 1.00 (undamped).",
                damping_factor,
            )
            damping_factor = 1.0

        logger.debug("Newton solver started.")

        it = 0
        while it < max_it:
            try:
                self._reassemble()
                self._solve_linear()
                self._assembler.sol.x.array[:] += damping_factor * self._dwh.x.array

                residual = self._dwh.x.petsc_vec.norm(0)

                logger.debug("Iteration %d - residual = %.2e", it, residual)

                if residual < atol:
                    return self._assembler.sol

                if math.isinf(residual):
                    break

                it += 1
                self._residual_history.append(residual)

            except KeyboardInterrupt:
                last_residual = (
                    self._residual_history[-1] if self._residual_history else 0.0
                )
                logger.warning(
                    "Solver was interrupted by the user. Newton did not fully converge (last residual: %.6f)",
                    last_residual,
                )
                break

        logger.warning(
            "Newton did not converge after %d iterations (last residual: %.6f)",
            it,
            self._residual_history[-1],
        )

    def plot_residuals(
        self, output_path: Path = Path(".") / "newton_residuals.png"
    ) -> None:
        """Plot the residuals and save them to disk."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(
            self._residual_history, label="Newton residuals", color="k", linewidth=1.5
        )
        ax.set_xlabel("Iteration (-)")
        ax.set_ylabel("Residual norm (-)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(which="major", linestyle="-", linewidth=0.8)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle=":", color="gray", linewidth=0.5)
        ax.tick_params(which="major", direction="in", length=3, width=0.5)
        ax.tick_params(which="minor", direction="in", length=1.25, width=0.5)
        fig.tight_layout()
        fig.savefig(output_path, dpi=330)
        plt.close(fig)

        logger.info("Residual plot saved to %s", output_path)
