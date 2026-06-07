from firedrake import *
from firedrake.adjoint import *
from pyadjoint import Control as PAControl, MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver, RieszMapMat

ALPHA = 1.0e-4
MESH_DIR = "code/mesh_generation"
RATIOS = [4, 16]


def build_problem(mesh):
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
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * ALPHA * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map="L2"), tape=tape)
    pause_annotation()
    return Jhat


def run(Jhat, override):
    params = {
        "tao_type": "lmvm",
        "tao_monitor": None,
        "tao_gatol": 1.0e-7,
        "tao_grtol": 0.0,
        "tao_gttol": 0.0,
        "tao_lmvm_num_vecs": 5,
        "tao_max_it": 2000,
        "tao_max_funcs": 4000,
    }
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    if override:
        c = PAControl(Jhat.controls[0].control, riesz_map="L2")
        minv = RieszMapMat([c], comm=solver.tao.getComm())
        solver.tao.setGradientNorm(minv)
        solver._keep = (minv, c)
    m_opt = solver.solve()
    its = solver.tao.getIterationNumber()
    _, _, gnorm, _, _, reason = solver.tao.getSolutionStatus()
    return its, float(Jhat(m_opt)), gnorm, int(reason)


def main():
    summary = []
    for ratio in RATIOS:
        mesh = Mesh(f"{MESH_DIR}/graded_R{ratio}.msh")
        for override in (False, True):
            tag = "override" if override else "vanilla"
            print(f"\n===== ratio {ratio}, {tag} =====", flush=True)
            its, fJ, g, reason = run(build_problem(mesh), override)
            summary.append((ratio, tag, its, fJ, g, reason))

    print("\n\n=== graded-mesh TAO/LMVM (riesz_map='L2') ===")
    print(f"{'ratio':>5}  {'variant':>8}  {'iters':>6}  {'final J':>13}  "
          f"{'final |g|':>12}  {'reason':>6}")
    for ratio, tag, its, fJ, g, reason in summary:
        print(f"{ratio:>5}  {tag:>8}  {its:>6}  {fJ:>13.6e}  "
              f"{g:>12.4e}  {reason:>6}")


if __name__ == "__main__":
    main()
