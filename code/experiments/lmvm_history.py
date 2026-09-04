"""tao lmvm with riesz_map=L2 on the graded meshes, history 5 then 100. tables in 10.9"""
import os
import sys

_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
    assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    set_working_tape,
)
from firedrake.petsc import PETSc
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

from meshdep.meshes import graded_unit_square_from_file

MESH_DIR = "/home/bjcwsl/masters_project/code/mesh_generation"
RATIOS = [4, 8, 16, 32, 64, 128]
ALPHA = 1.0e-4
GATOL = 1.0e-7
CONFIGS = [("l2", 5), ("L2", 5), ("L2", 100)]
MAX_IT = 2000


def build_jhat(mesh, riesz_map="L2"):
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
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map), tape=tape)
    pause_annotation()
    return Jhat


def run(Jhat, history):

    PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', history)
    params = {"tao_type": "lmvm", "tao_gatol": GATOL,
              "tao_grtol": 0.0, "tao_gttol": 0.0, "tao_max_it": MAX_IT}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    try:
        solver.solve()
    except Exception:
        pass
    tao = solver.tao
    return tao.getIterationNumber(), tao.getConvergedReason() > 0


results = {c: [] for c in CONFIGS}
cells, realised = [], []

for r in RATIOS:
    mesh, real = graded_unit_square_from_file(f"{MESH_DIR}/graded_R{r}.msh")
    cells.append(mesh.num_cells())
    realised.append(real)
    print(f"\n=== target r={r}  realised={real:.2f}  cells={mesh.num_cells()}",
          flush=True)
    for rm, h in CONFIGS:
        its, ok = run(build_jhat(mesh, rm), h)
        results[(rm, h)].append((its, ok))
        print(f"    riesz={rm:>2}, history {h:>3}: {its:>5} iterations, "
              f"converged={ok}", flush=True)

print("\n===== SUMMARY =====")
print("target r :", RATIOS)
print("cells    :", cells)
for c in CONFIGS:
    row = [f"{i}{'' if ok else '*'}" for i, ok in results[c]]
    print(f"riesz={c[0]:>2} history {c[1]:>3}:", row)
print("(* = did not reach the tolerance within "
      f"{MAX_IT} iterations)")
