"""levels 12 and 14 for the matching table"""
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
from meshdep.preconditioners import RieszMapPCContext

ALPHA = 1.0e-4
EPS = 1.0e-7
MAX_NLS = 50
P_REFINE = 0.35
SEED = 42
N_INITIAL = 8
REGION = (0.3, 0.7, 0.3, 0.7)
TMP_DIR = "/tmp/interior_extra_levels"
os.makedirs(TMP_DIR, exist_ok=True)

LEVELS = (0, 2, 6, 10)


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


def run_nls(mesh, pc_type):
    riesz_map = "L2" if pc_type in ("none", "L2") else "H1"
    Jhat, V = _build_jhat(mesh, riesz_map=riesz_map)
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
    if pc_type != 'none':
        pc = ksp.getPC()
        pc.setType('python')
        pc.setPythonContext(RieszMapPCContext(V, riesz_map=pc_type))
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    outer = solver.tao.getIterationNumber()
    inner = len(ksp.getConvergenceHistory())
    return outer, inner


def main():
    print("Interior random-refined meshes, extra levels for the poster plot")
    print(f"  region={REGION}, p={P_REFINE}, seed={SEED}")
    print()

    rows = []
    for level in LEVELS:
        print(f"--- level {level} ---", flush=True)
        mesh, realised, n_cells = build_interior_mesh(level)
        print(f"  cells={n_cells}, realised h_max/h_min={realised:.2f}",
              flush=True)

        print(f"  baseline (no PC)...", flush=True)
        out_b, inn_b = run_nls(mesh, pc_type='none')
        print(f"    -> outer={out_b}, inner={inn_b}", flush=True)

        print(f"  L^2 Riesz PC...", flush=True)
        out_l2, inn_l2 = run_nls(mesh, pc_type='L2')
        print(f"    -> outer={out_l2}, inner={inn_l2}", flush=True)

        print(f"  H^1 Riesz PC...", flush=True)
        out_h1, inn_h1 = run_nls(mesh, pc_type='H1')
        print(f"    -> outer={out_h1}, inner={inn_h1}", flush=True)

        rows.append({
            "level": level, "realised": realised, "n_cells": n_cells,
            "baseline_outer": out_b, "baseline_inner": inn_b,
            "l2_outer": out_l2, "l2_inner": inn_l2,
            "h1_outer": out_h1, "h1_inner": inn_h1,
        })

    print()
    print("=" * 92)
    print("FINAL TABLE (extra interior levels)")
    print("=" * 92)
    print(f"{'level':>5}  {'ratio':>6}  {'cells':>6}  "
          f"{'base out':>8}  {'base inn':>8}  "
          f"{'L2 out':>7}  {'L2 inn':>7}  "
          f"{'H1 out':>7}  {'H1 inn':>7}")
    print("-" * 92)
    for r in rows:
        print(f"{r['level']:>5d}  {r['realised']:>6.2f}  {r['n_cells']:>6d}  "
              f"{r['baseline_outer']:>8d}  {r['baseline_inner']:>8d}  "
              f"{r['l2_outer']:>7d}  {r['l2_inner']:>7d}  "
              f"{r['h1_outer']:>7d}  {r['h1_inner']:>7d}")
    print("=" * 92)


if __name__ == "__main__":
    main()
