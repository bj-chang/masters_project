"""matching experiment on the interior random meshes: baseline vs L2 pc vs H1 pc on the l2 regularised problem"""
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
GATOL = 1.0e-7
MAX_IT = 50
P_REFINE = 0.35
SEED = 42
N_INITIAL = 8
REGION = (0.3, 0.7, 0.3, 0.7)
TMP_DIR = "/tmp/interior_random_meshes_item3"
os.makedirs(TMP_DIR, exist_ok=True)

LEVELS = (4, 8, 12, 14)


def build_interior_mesh(target_level):
    return interior_random_refined_mesh(
        target_level, region=REGION, seed=SEED,
        p_refine=P_REFINE, n_initial=N_INITIAL, tmp_dir=TMP_DIR)


def build_jhat(mesh):
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = Function(V), TestFunction(V)
    m = Function(V)
    F = inner(grad(u), grad(v)) * dx - m * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    d = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    fwd_params = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    m.zero()
    continue_annotation()
    with set_working_tape() as tape:
        solve(F == 0, u, bcs=bc, solver_parameters=fwd_params)
        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m), tape=tape)
    pause_annotation()
    return Jhat, V


def solve_and_count(Jhat, V, use_pc):
    params = {
        'tao_type': 'nls',
        'tao_gatol': GATOL,
        'tao_grtol': 1.0e-7,
        'tao_gttol': 0.0,
        'tao_max_it': MAX_IT,
        'tao_nls_ksp_rtol': 1.0e-4,
    }
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    ksp = solver.tao.getKSP()
    if use_pc:
        pc = ksp.getPC()
        pc.setType('python')
        pc.setPythonContext(RieszMapPCContext(V, riesz_map="L2"))
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    outer = solver.tao.getIterationNumber()
    inner = len(ksp.getConvergenceHistory())
    return outer, inner


def main():
    print("Item 3 on interior random-refined meshes")
    print(f"  region={REGION}, p={P_REFINE}, seed={SEED}, n_ini={N_INITIAL}")
    print(f"  alpha={ALPHA}, gatol={GATOL:.0e}")
    print()

    rows = []
    for lvl in LEVELS:
        mesh, realised, n_cells = build_interior_mesh(lvl)
        print(f"level={lvl}, cells={n_cells}, realised={realised:.2f}",
              flush=True)

        Jhat1, V1 = build_jhat(mesh)
        out0, inn0 = solve_and_count(Jhat1, V1, use_pc=False)
        print(f"  baseline: outer={out0}, inner KSP total={inn0}", flush=True)

        Jhat2, V2 = build_jhat(mesh)
        out1, inn1 = solve_and_count(Jhat2, V2, use_pc=True)
        print(f"  Riesz PC: outer={out1}, inner KSP total={inn1}", flush=True)

        rows.append((lvl, realised, n_cells, out0, inn0, out1, inn1))

    print()
    print("=" * 78)
    print("FINAL TABLE (interior random-refined meshes)")
    print("=" * 78)
    print(f"{'level':>5}  {'realised':>9}  {'cells':>7}  "
          f"{'base outer':>10}  {'base inner':>10}  "
          f"{'pc outer':>8}  {'pc inner':>8}")
    for lvl, realised, n_cells, out0, inn0, out1, inn1 in rows:
        print(f"{lvl:>5d}  {realised:>9.2f}  {n_cells:>7d}  "
              f"{out0:>10d}  {inn0:>10d}  "
              f"{out1:>8d}  {inn1:>8d}")
    print("=" * 78)


if __name__ == "__main__":
    main()
