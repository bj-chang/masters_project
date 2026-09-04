"""total inner cg iterations for tao nls on the graded meshes, unpreconditioned vs the L2 riesz pc wired on by hand. tables in 12.1 and 12.4"""
import os
import re
import sys
import io

_PROJECT_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_CODE not in sys.path:
    sys.path.insert(0, _PROJECT_CODE)

from firedrake import *
from firedrake.adjoint import *
from firedrake.petsc import PETSc
from pyadjoint.optimization.tao_solver import TAOSolver

from meshdep.meshes import graded_unit_square_from_file
from meshdep.preconditioners import RieszMapPCContext

ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 50
MESH_DIR = "code/mesh_generation"
TARGET_RATIOS = (4, 16, 64, 128)


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


results = []
for r in TARGET_RATIOS:
    print(f"\nLoading Netgen mesh, target r={r}...", flush=True)
    mesh, realised = graded_unit_square_from_file(
        f"{MESH_DIR}/graded_R{r}.msh"
    )
    n_cells = mesh.num_cells()
    print(f"  {n_cells} cells, realised h_max/h_min = {realised:.2f}",
          flush=True)

    Jhat1, V1 = build_jhat(mesh)
    out0, inn0 = solve_and_count(Jhat1, V1, use_pc=False)
    print(f"  baseline (no PC): outer={out0}, inner KSP total={inn0}",
          flush=True)

    Jhat2, V2 = build_jhat(mesh)
    out1, inn1 = solve_and_count(Jhat2, V2, use_pc=True)
    print(f"  with Riesz L2 PC: outer={out1}, inner KSP total={inn1}",
          flush=True)

    results.append((r, realised, n_cells, out0, inn0, out1, inn1))


print()
print("=" * 78)
print("FINAL TABLE")
print("=" * 78)
print(f"{'r':>4}  {'h_ratio':>8}  {'cells':>7}  "
      f"{'baseline outer':>14}  {'baseline inner':>14}  "
      f"{'riesz outer':>11}  {'riesz inner':>11}")
for r, realised, n_cells, out0, inn0, out1, inn1 in results:
    print(f"{r:>4}  {realised:>8.2f}  {n_cells:>7d}  "
          f"{out0:>14d}  {inn0:>14d}  "
          f"{out1:>11d}  {inn1:>11d}")
print("=" * 78)
