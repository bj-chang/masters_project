"""tao nls at gatol 1e-30. newton lands on the optimum in one step and the tape reports ~1e-18, so the floor is machine precision and not what stalled lmvm"""
import os
import sys

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

PETSc.Options().setValue('-tao_lmvm_mat_lmvm_hist_size', 100)

ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 2000
MESH_DIR = "code/mesh_generation"


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
    return Jhat


def run(Jhat, tao_type):
    params = {'tao_type': tao_type,
              'tao_gatol': GATOL,
              'tao_grtol': 0.0, 'tao_gttol': 0.0,
              'tao_max_it': MAX_IT}
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    try:
        solver.solve()
        its = solver.tao.getIterationNumber()
        reason = solver.tao.getConvergedReason()
        return f"{its} (r={reason})"
    except Exception as e:
        return f"FAIL: {e.__class__.__name__}"


TARGET_RATIOS = (4, 16, 64, 128)


print(f"2D Poisson control on Netgen meshes, alpha={ALPHA}, "
      f"gatol={GATOL:.0e}")
print(f"{'target r':>9}  {'realised':>9}  {'LMVM (hist=100)':>18}  "
      f"{'NLS':>14}")
for r in TARGET_RATIOS:
    mesh, realised = graded_unit_square_from_file(
        f"{MESH_DIR}/graded_R{r}.msh"
    )
    print(f"{r:>9d}  {realised:>9.2f}  running LMVM and NLS...",
          flush=True)
    lmvm_iters = run(build_jhat(mesh), 'lmvm')
    nls_iters = run(build_jhat(mesh), 'nls')
    print(f"{r:>9d}  {realised:>9.2f}  "
          f"{lmvm_iters:>18}  {nls_iters:>14}", flush=True)
