from TestProjects.examples.poisson import solve_poisson
from TestProjects.examples.navier_stokes import run_navier_stokes

from typing import Callable

_EXAMPLES_MAP: dict[str, Callable] = {
    "simple poisson problem": solve_poisson,
    "navier-stokes": run_navier_stokes,
}

if __name__ == "__main__":
    print("running test projects:")
    for name, func in _EXAMPLES_MAP.items():
        print(f"    * solving: {name}")
        func()
        print()
