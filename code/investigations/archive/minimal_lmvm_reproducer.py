"""Minimal reproducer for D. Ham's question.

Does vanilla pyadjoint TAOSolver / LMVM with Control(m, riesz_map="L2")
give mesh-dependent iteration counts on Schwedes' Poisson
distributed-control problem?

No custom infrastructure: only Firedrake + pyadjoint. We do NOT call
tao.setGradientNorm() ourselves, add no convergence callback, and patch
nothing. The Riesz map is set only at the Control level. Uniform meshes.
"""

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4
RESOLUTIONS = [16, 32, 64]


def solve_one(N):
    mesh = UnitSquareMesh(N, N)
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
        # Exact (LU) state solve so the discrete gradient can reach
        # gatol=1e-7; an iterative solver's tolerance would floor it.
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * ALPHA * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map="L2"), tape=tape)
    pause_annotation()

    params = {
        "tao_type": "lmvm",
        "tao_monitor": None,
        "tao_gatol": 1.0e-7,
        "tao_grtol": 0.0,
        "tao_gttol": 0.0,
        "tao_lmvm_num_vecs": 5,     # history 5 (verified via tao_view)
        "tao_max_it": 10000,
        "tao_max_funcs": 20000,
    }
    print(f"\n===== N = {N} =====", flush=True)
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    m_opt = solver.solve()

    its = solver.tao.getIterationNumber()
    _, _, gnorm, _, _, reason = solver.tao.getSolutionStatus()
    final_J = float(Jhat(m_opt))
    return N, its, final_J, gnorm, int(reason)


def main():
    rows = [solve_one(N) for N in RESOLUTIONS]
    print("\n\n=== Vanilla TAO/LMVM, Control(m, riesz_map='L2'), "
          "uniform meshes ===")
    print(f"{'N':>4}  {'iters':>6}  {'final J':>14}  "
          f"{'final |g|_L2':>14}  {'reason':>6}")
    print("-" * 56)
    for N, its, fJ, g, reason in rows:
        print(f"{N:>4}  {its:>6}  {fJ:>14.6e}  {g:>14.6e}  {reason:>6}")


if __name__ == "__main__":
    main()
