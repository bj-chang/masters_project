"""mirror experiment: h1 regularised problem on the interior meshes. h1 pc stays flat, l2 pc grows to the cap"""
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
from firedrake.petsc import PETSc
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

from mesh_generation.random_refine import interior_random_refined_mesh
from meshdep.optimisers import (
    solve_with_hilbert_lbfgs, solve_with_scipy_external_check,
)
from meshdep.preconditioners import RieszMapPCContext

ALPHA = 1.0e-4
EPS = 1.0e-7
MAX_SCIPY = 5000
MAX_HILBERT = 500
MAX_NLS = 50
P_REFINE = 0.35
SEED = 42
N_INITIAL = 8
REGION = (0.3, 0.7, 0.3, 0.7)
TMP_DIR = "/tmp/interior_random_h1_sweep"
os.makedirs(TMP_DIR, exist_ok=True)

LEVELS = (4, 8, 12, 14)


def build_interior_mesh(target_level):
    return interior_random_refined_mesh(
        target_level, region=REGION, seed=SEED,
        p_refine=P_REFINE, n_initial=N_INITIAL, tmp_dir=TMP_DIR)


def _build_jhat(mesh, riesz_map):
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
    return Jhat, V


def run_scipy_h1(mesh):
    Jhat, _ = _build_jhat(mesh, riesz_map="l2")
    result = solve_with_scipy_external_check(
        Jhat, eps=EPS, test_riesz_map="H1", maxiter=MAX_SCIPY,
    )
    return result["iterations"]


def run_hilbert_h1(mesh):
    Jhat, _ = _build_jhat(mesh, riesz_map="H1")
    result = solve_with_hilbert_lbfgs(
        Jhat, eps=EPS, max_iter=MAX_HILBERT, history=5,
    )
    return result["iterations"]


def run_nls_h1_pc(mesh):
    Jhat, V = _build_jhat(mesh, riesz_map="H1")
    params = {
        'tao_type': 'nls',
        'tao_gatol': EPS,
        'tao_grtol': 1.0e-7,
        'tao_gttol': 0.0,
        'tao_max_it': MAX_NLS,
        'tao_nls_ksp_rtol': 1.0e-4,
    }
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    ksp = solver.tao.getKSP()
    pc = ksp.getPC()
    pc.setType('python')
    pc.setPythonContext(RieszMapPCContext(V, riesz_map="H1"))
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    outer = solver.tao.getIterationNumber()
    inner = len(ksp.getConvergenceHistory())
    return outer, inner


def main():
    print("Interior random-refined meshes: H^1 sweep")
    print(f"  region={REGION}, p={P_REFINE}, seed={SEED}, n_ini={N_INITIAL}")
    print(f"  alpha={ALPHA}, eps={EPS:.0e}")
    print()

    rows = []
    for level in LEVELS:
        print(f"--- level {level} ---", flush=True)
        mesh, realised, n_cells = build_interior_mesh(level)
        print(f"  cells={n_cells}, realised h_max/h_min={realised:.2f}",
              flush=True)

        print(f"  running Hilbert L-BFGS (H^1)...", flush=True)
        hilbert_iters = run_hilbert_h1(mesh)
        print(f"    -> {hilbert_iters}", flush=True)

        print(f"  running TAO/NLS + Riesz H^1 PC...", flush=True)
        out_pc, inn_pc = run_nls_h1_pc(mesh)
        print(f"    -> outer={out_pc}, inner CG total={inn_pc}", flush=True)

        print(f"  running SciPy L-BFGS-B (l^2, convergence in H^1)...",
              flush=True)
        scipy_iters = run_scipy_h1(mesh)
        print(f"    -> {scipy_iters}", flush=True)

        rows.append({
            "level": level, "realised": realised, "n_cells": n_cells,
            "scipy_h1": scipy_iters, "hilbert_h1": hilbert_iters,
            "nls_out_pc_h1": out_pc, "nls_inn_pc_h1": inn_pc,
        })

    print()
    print("=" * 88)
    print("FINAL TABLE (H^1 on interior meshes)")
    print("=" * 88)
    print(f"{'level':>5}  {'ratio':>6}  {'cells':>6}  "
          f"{'SciPy(l2)->H1':>13}  {'Hilbert(H1)':>11}  "
          f"{'PC out':>7}  {'PC inn':>7}")
    print("-" * 88)
    for r in rows:
        print(f"{r['level']:>5d}  {r['realised']:>6.2f}  {r['n_cells']:>6d}  "
              f"{r['scipy_h1']:>13d}  {r['hilbert_h1']:>11d}  "
              f"{r['nls_out_pc_h1']:>7d}  {r['nls_inn_pc_h1']:>7d}")
    print("=" * 88)


if __name__ == "__main__":
    main()
