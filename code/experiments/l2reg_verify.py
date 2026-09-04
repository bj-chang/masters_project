"""same-minimiser check on the l2 regularised problem. l2 and h1 pc agree in J* to 15 digits so the h1 growth is conditioning, not a wrong answer"""
import sys
sys.path.insert(0, "/home/bjcwsl/masters_project/code")

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
    assemble, dx, grad, inner, pi, sin, solve, sqrt,
)
from firedrake.adjoint import (
    Control, ReducedFunctional,
    continue_annotation, pause_annotation, set_working_tape,
)
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver
from mesh_generation.random_refine import interior_random_refined_mesh
from meshdep.preconditioners import RieszMapPCContext

ALPHA = 1.0e-4
REGION = (0.3, 0.7, 0.3, 0.7)
TMP_DIR = "/tmp/interior_random_h1_sweep"
LU = {"ksp_type": "preonly", "pc_type": "lu"}


def build(mesh, riesz_map):
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
        solve(F == 0, u, bcs=bc, solver_parameters=LU)

        J = assemble(0.5 * (u - d) ** 2 * dx + 0.5 * ALPHA * m ** 2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map), tape=tape)
    pause_annotation()
    return Jhat, V


def run(mesh, pc_kind, monitor=False):
    Jhat, V = build(mesh, pc_kind)
    params = {
        "tao_type": "nls", "tao_gatol": 1e-10, "tao_grtol": 1e-7,
        "tao_gttol": 0.0, "tao_max_it": 50, "tao_nls_ksp_rtol": 1e-4,
    }
    if monitor:
        params["tao_monitor"] = None
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    ksp = solver.tao.getKSP()
    pc = ksp.getPC()
    pc.setType("python")
    pc.setPythonContext(RieszMapPCContext(V, riesz_map=pc_kind))
    ksp.setConvergenceHistory(reset=False)
    if monitor:
        print(f"  --- tao_monitor for H1-PC (outer residual per Newton step) ---")
    m_star = solver.solve()
    outer = solver.tao.getIterationNumber()
    inner = len(ksp.getConvergenceHistory())
    J_star = Jhat(m_star)
    m_norm = sqrt(assemble(m_star * m_star * dx))
    return m_star, J_star, m_norm, outer, inner


for lvl in (8, 10):
    mesh, realised, ncells = interior_random_refined_mesh(
        lvl, region=REGION, seed=42, p_refine=0.35, n_initial=8,
        tmp_dir=TMP_DIR)
    print(f"\n===== L={lvl}, cells={ncells}, ORIGINAL L2-regularised problem =====")
    m_l2, J_l2, n_l2, o_l2, i_l2 = run(mesh, "L2")
    m_h1, J_h1, n_h1, o_h1, i_h1 = run(mesh, "H1", monitor=True)
    diff = sqrt(assemble((m_l2 - m_h1) ** 2 * dx))
    print(f"  L2-PC : J*={J_l2:.10e}  outer={o_l2} inner={i_l2}")
    print(f"  H1-PC : J*={J_h1:.10e}  outer={o_h1} inner={i_h1}")
    print(f"  |J*_L2 - J*_H1| = {abs(J_l2 - J_h1):.3e}")
    print(f"  ||m*_L2 - m*_H1||/||m*|| = {diff / n_l2:.3e}  "
          f"(tiny => same optimum => H1's high/growing count is genuine)")
