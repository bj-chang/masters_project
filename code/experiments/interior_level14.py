"""level 14 on its own since it is slow"""
import os
import sys

_PROJECT_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_CODE not in sys.path:
    sys.path.insert(0, _PROJECT_CODE)

import numpy as np
from firedrake import (
    DirichletBC, Function, FunctionSpace, Mesh, SpatialCoordinate,
    TestFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional,
    continue_annotation, pause_annotation, set_working_tape,
)

from mesh_generation.random_refine import interior_random_refined_mesh
from meshdep.optimisers import (
    solve_with_hilbert_lbfgs, solve_with_scipy_external_check,
)

ALPHA = 1.0e-4
EPS = 1.0e-7
MAX_SCIPY = 5000
MAX_HILBERT = 500
P_REFINE = 0.35
SEED = 42
N_INITIAL = 8
REGION = (0.3, 0.7, 0.3, 0.7)
TMP_DIR = "/tmp/interior_level14"
os.makedirs(TMP_DIR, exist_ok=True)
TARGET_LEVEL = 14


def build_interior_mesh(target_level):
    return interior_random_refined_mesh(
        target_level, region=REGION, seed=SEED,
        p_refine=P_REFINE, n_initial=N_INITIAL, tmp_dir=TMP_DIR)


def build_jhat(mesh, riesz_map):
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    bc = DirichletBC(V, 0.0, "on_boundary")
    m = Function(V)
    continue_annotation()
    with set_working_tape() as tape:
        u = Function(V)
        v = TestFunction(V)
        F = inner(grad(u), grad(v)) * dx - m * v * dx
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat


def main():
    print(f"Interior random-refined mesh, level={TARGET_LEVEL}")
    mesh, realised, n_cells = build_interior_mesh(TARGET_LEVEL)
    print(f"cells={n_cells}, realised h_max/h_min={realised:.2f}")
    print()

    print("Running Hilbert L-BFGS (L^2) first (fast)...", flush=True)
    Jhat_H = build_jhat(mesh, riesz_map="L2")
    result_H = solve_with_hilbert_lbfgs(Jhat_H, eps=EPS,
                                        max_iter=MAX_HILBERT, history=5)
    hilbert_iters = result_H["iterations"]
    print(f"Hilbert L-BFGS (L^2): {hilbert_iters}", flush=True)
    print()

    print("Running SciPy L-BFGS-B (l^2) (slow)...", flush=True)
    Jhat_S = build_jhat(mesh, riesz_map="l2")
    result_S = solve_with_scipy_external_check(
        Jhat_S, eps=EPS, test_riesz_map="L2", maxiter=MAX_SCIPY,
    )
    scipy_iters = result_S["iterations"]
    print(f"SciPy L-BFGS-B (l^2): {scipy_iters}", flush=True)
    print()

    print(f"=== RESULT: L=14, ratio={realised:.2f}, cells={n_cells} ===")
    print(f"SciPy L-BFGS-B (l^2):  {scipy_iters}")
    print(f"Hilbert L-BFGS (L^2):  {hilbert_iters}")


if __name__ == "__main__":
    main()
