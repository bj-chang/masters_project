"""checks pyadjoints RieszMapPC reproduces the hand wired context to the iteration"""
import os
import sys

_PROJECT_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _PROJECT_CODE not in sys.path:
    sys.path.insert(0, _PROJECT_CODE)

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
    assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional,
    continue_annotation, pause_annotation, set_working_tape,
)
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

from meshdep.meshes import graded_unit_square_from_file
from meshdep.preconditioners import RieszMapPCContext

ALPHA = 1.0e-4
GATOL = 1.0e-7
MAX_IT = 50
MESH_DIR = "code/mesh_generation"
L2_RATIOS = (4, 16, 64, 128)


NEW_PC = "pyadjoint.optimization.tao_solver.RieszMapPC"


def build_jhat(mesh, riesz_map):
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
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat, V


def _base_params():
    return {
        'tao_type': 'nls',
        'tao_gatol': GATOL,
        'tao_grtol': 1.0e-7,
        'tao_gttol': 0.0,
        'tao_max_it': MAX_IT,
        'tao_nls_ksp_rtol': 1.0e-4,

        'tao_view': ':tao_view_output.log'
    }


def run_baseline(Jhat):
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=_base_params())
    ksp = solver.tao.getKSP()
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    return (solver.tao.getIterationNumber(),
            len(ksp.getConvergenceHistory()),
            ksp.getPC().getType())


def run_new_pc_via_options(Jhat):
    params = _base_params()
    params['tao_nls_pc_type'] = 'python'
    params['tao_nls_pc_python_type'] = NEW_PC
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    ksp = solver.tao.getKSP()
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    return (solver.tao.getIterationNumber(),
            len(ksp.getConvergenceHistory()),
            ksp.getPC().getType())


def run_reference_pc(Jhat, V, riesz_map):
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=_base_params())
    ksp = solver.tao.getKSP()
    pc = ksp.getPC()
    pc.setType('python')
    pc.setPythonContext(RieszMapPCContext(V, riesz_map=riesz_map))
    ksp.setConvergenceHistory(reset=False)
    solver.solve()
    return (solver.tao.getIterationNumber(),
            len(ksp.getConvergenceHistory()),
            ksp.getPC().getType())


def main():
    all_pass = True

    print("=" * 70)
    print("L2 sweep -- new PC vs baseline vs reference PCContext")
    print("=" * 70)
    for r in L2_RATIOS:
        mesh, realised = graded_unit_square_from_file(
            f"{MESH_DIR}/graded_R{r}.msh")
        print(f"\n--- r={r}, realised h_max/h_min={realised:.2f} ---",
              flush=True)

        Jhat, V = build_jhat(mesh, "L2")
        o0, i0, t0 = run_baseline(Jhat)
        print(f"  baseline (no PC):       outer={o0}, inner={i0:>4}, "
              f"pc={t0!r}", flush=True)

        Jhat, V = build_jhat(mesh, "L2")
        o1, i1, t1 = run_new_pc_via_options(Jhat)
        print(f"  new RieszMapPC (opts):  outer={o1}, inner={i1:>4}, "
              f"pc={t1!r}", flush=True)

        Jhat, V = build_jhat(mesh, "L2")
        o2, i2, t2 = run_reference_pc(Jhat, V, "L2")
        print(f"  reference PCContext:    outer={o2}, inner={i2:>4}, "
              f"pc={t2!r}", flush=True)

        installed = (t1 == 'python')
        matches = (i1 == i2)
        flat = (i1 < i0)
        ok = installed and matches and flat
        all_pass = all_pass and ok
        print(f"  PC installed: {installed} | matches reference: {matches} "
              f"({i1} vs {i2}) | below baseline: {flat} ({i1} < {i0})",
              flush=True)
        print(f"  --> {'PASS' if ok else 'FAIL'}", flush=True)

    print()
    print("=" * 70)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
