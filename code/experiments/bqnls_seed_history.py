"""bqnls in all four seed/history combos on the graded meshes. gatol 1e-7 with grtol and gttol off. isolation table in 10.11"""
import os
import sys

_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
    TrialFunction, assemble, dx, grad, inner, pi, sin, solve,
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
RATIOS = [4, 16, 64, 128]
ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 2000
CONFIGS = [("none", 5), ("none", 100), ("Mh", 5), ("Mh", 100)]


def build_jhat(mesh):
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
        Jhat = ReducedFunctional(J, Control(m, riesz_map="L2"), tape=tape)
    pause_annotation()
    return Jhat, V


def run(mesh, seed, history):
    Jhat, V = build_jhat(mesh)
    PETSc.Options().setValue('-tao_bqnls_mat_lmvm_hist_size', history)
    params = {"tao_type": "bqnls", "tao_gatol": GATOL,
              "tao_grtol": 0.0, "tao_gttol": 0.0, "tao_max_it": MAX_IT}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    if seed == "Mh":
        u, v = TrialFunction(V), TestFunction(V)
        Mh = assemble(inner(u, v) * dx).petscmat
        solver.tao.getLMVMMat().setLMVMJ0(Mh)
    try:
        solver.solve()
    except Exception:
        pass
    tao = solver.tao
    return tao.getIterationNumber(), tao.getConvergedReason() > 0


results = {c: [] for c in CONFIGS}
for r in RATIOS:
    mesh, realised = graded_unit_square_from_file(f"{MESH_DIR}/graded_R{r}.msh")
    print(f"\n=== r={r}  realised={realised:.2f}  cells={mesh.num_cells()}",
          flush=True)
    for seed, hist in CONFIGS:
        its, ok = run(mesh, seed, hist)
        results[(seed, hist)].append((its, ok))
        print(f"    seed={seed:>4}, history {hist:>3}: {its:>5} iterations, "
              f"converged={ok}", flush=True)

print("\n===== SUMMARY (absolute gatol=1e-7, grtol=gttol=0) =====")
print("ratios:", RATIOS)
for c in CONFIGS:
    print(f"seed={c[0]:>4} history {c[1]:>3}:",
          [f"{i}{'' if ok else '*'}" for i, ok in results[c]])
print(f"(* = did not converge within {MAX_IT} iterations)")
