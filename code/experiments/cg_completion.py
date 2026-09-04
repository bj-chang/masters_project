"""tao cg run to convergence with the function evaluation cap raised. takes hours at the higher ratios"""
import os
import sys
import time

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
ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 200000
MAX_FUNCS = 400000
RATIOS = [4, 16]


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
    return Jhat


for r in RATIOS:
    mesh, realised = graded_unit_square_from_file(f"{MESH_DIR}/graded_R{r}.msh")
    print(f"\n=== r={r}  realised={realised:.2f}  cells={mesh.num_cells()}",
          flush=True)
    params = {"tao_type": "cg",
              "tao_gatol": GATOL, "tao_grtol": 0.0, "tao_gttol": 0.0,
              "tao_max_it": MAX_IT, "tao_max_funcs": MAX_FUNCS}
    solver = TAOSolver(MinimizationProblem(build_jhat(mesh)), parameters=params)
    t0 = time.time()
    try:
        solver.solve()
    except Exception as exc:
        print(f"   (raised {type(exc).__name__})", flush=True)
    tao = solver.tao
    print(f"   cg: {tao.getIterationNumber()} iterations, "
          f"reason={int(tao.getConvergedReason())}, "
          f"{time.time() - t0:.0f} s", flush=True)
