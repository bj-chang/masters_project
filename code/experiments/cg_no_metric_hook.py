"""checks cg exposes no seed and no inner ksp, and that switching riesz_map leaves its iterates identical while lmvm changes. table in 12.8"""
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
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

from meshdep.meshes import graded_unit_square_from_file

MESH = "/home/bjcwsl/masters_project/code/mesh_generation/graded_R16.msh"
ALPHA = 1.0e-4
BUDGET = 40

mesh, realised = graded_unit_square_from_file(MESH)
print(f"mesh r=16  realised={realised:.2f}  cells={mesh.num_cells()}\n")


def build(riesz):
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
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz), tape=tape)
    pause_annotation()
    return Jhat


def run(tao_type, riesz):
    params = {"tao_type": tao_type, "tao_gatol": 1.0e-30,
              "tao_grtol": 0.0, "tao_gttol": 0.0,
              "tao_max_it": BUDGET, "tao_max_funcs": 20 * BUDGET}
    solver = TAOSolver(MinimizationProblem(build(riesz)), parameters=params)
    try:
        solver.solve()
    except Exception:
        pass
    return solver


print("--- 1 & 2: what hooks does each solver expose? ---")
for tao_type in ("cg", "lmvm", "nls"):
    solver = run(tao_type, "L2")
    tao = solver.tao
    try:
        tao.getLMVMMat()
        seed = "yes"
    except Exception as e:
        seed = f"no ({type(e).__name__})"
    try:
        ksp = tao.getKSP()
        inner_ksp = "none" if ksp is None or ksp.handle == 0 else "yes"
    except Exception as e:
        inner_ksp = f"no ({type(e).__name__})"
    print(f"  {tao_type:>5}: LMVM seed = {seed:<22} inner KSP = {inner_ksp}")

print("\n--- 3: does the Riesz map change the iterates? ---")
for tao_type in ("cg", "lmvm"):
    vals = {}
    for riesz in ("l2", "L2"):
        solver = run(tao_type, riesz)
        vals[riesz] = float(solver.tao.getObjectiveValue())
    same = vals["l2"] == vals["L2"]
    print(f"  {tao_type:>5}: J(l2) = {vals['l2']:.16e}")
    print(f"         J(L2) = {vals['L2']:.16e}   -> "
          f"{'IDENTICAL' if same else 'different'}")
