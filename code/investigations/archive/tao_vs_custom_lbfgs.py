"""Before / after for the mesh-dependence investigation.

Two ways of solving the SAME Poisson distributed-control problem with
the L2 inner product on the control, compared on uniform and graded
meshes:

  solve_tao_lmvm      BEFORE. Plain pyadjoint TAOSolver / LMVM with
                      Control(m, riesz_map="L2"). This is the "proper"
                      route -- a maintained library optimiser -- but it
                      is mesh-dependent: ~6 iterations on a uniform mesh,
                      several hundred on a graded mesh.

  solve_custom_lbfgs  FIX (less proper). A hand-rolled L-BFGS in the L2
                      inner product. Mesh-independent (single-digit
                      iterations on every mesh), but a bespoke
                      implementation rather than a library solver.

The two differ in one detail that turns out to matter: the custom
L-BFGS rescales its initial Hessian each step by
gamma_k = <s,y>_L2 / <y,y>_L2 (standard L-BFGS), whereas TAO, given a
fixed custom initial Hessian B0 = M^{-1} via setLMVMH0, does not. On a
uniform mesh the fixed scale is about right so TAO is fine; on a graded
mesh it is not. See graded_lmvm_diagnose.py for the experiment pinning
this down. Run from ~/masters_project/.
"""

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver

ALPHA = 1.0e-4
MESH_DIR = "code/mesh_generation"
LU = {"ksp_type": "preonly", "pc_type": "lu"}


def build_problem(mesh):
    """Poisson distributed control, L2 Riesz map on the control."""
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
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * ALPHA * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map="L2"), tape=tape)
    pause_annotation()
    return Jhat, V


def solve_tao_lmvm(Jhat):
    """BEFORE: library TAO/LMVM. Proper, but mesh-dependent on graded meshes."""
    params = {
        "tao_type": "lmvm",
        "tao_gatol": 1.0e-7, "tao_grtol": 0.0, "tao_gttol": 0.0,
        "tao_lmvm_num_vecs": 5, "tao_max_it": 3000, "tao_max_funcs": 10000,
    }
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    solver.solve()
    its = solver.tao.getIterationNumber()
    converged = solver.tao.getSolutionStatus()[5] > 0
    return its, converged


def solve_custom_lbfgs(Jhat, V, eps=1.0e-7, max_iter=3000, history=5):
    """FIX (less proper): hand-rolled L-BFGS in the L2 inner product.

    The one line that TAO lacks is the gamma_k rescaling of the initial
    Hessian (marked below).
    """
    def L2(a, b):
        return float(assemble(inner(a, b) * dx))

    def primal_grad():
        return Jhat.derivative().riesz_representation(
            riesz_map="L2", solver_options=LU)

    m = Function(V).assign(Jhat.controls[0].tape_value())
    Jc = float(Jhat(m))
    g = primal_grad()
    gN = L2(g, g) ** 0.5
    S, Y, R = [], [], []

    for k in range(max_iter):
        if gN <= eps:
            return k, True
        q = Function(V).assign(g)
        al = []
        for s_i, y_i, r_i in zip(reversed(S), reversed(Y), reversed(R)):
            a = r_i * L2(s_i, q)
            al.append(a)
            q.assign(q - a * y_i)
        gam = 1.0
        if S:                       # <-- adaptive B0 rescaling TAO does not do
            yy = L2(Y[-1], Y[-1])
            gam = (1.0 / R[-1]) / yy if yy > 0 else 1.0
        z = Function(V).assign(gam * q)
        for s_i, y_i, r_i, a in zip(S, Y, R, reversed(al)):
            b = r_i * L2(y_i, z)
            z.assign(z + (a - b) * s_i)
        p = Function(V).assign(-1.0 * z)
        d0 = L2(g, p)
        if d0 >= 0.0:
            p.assign(-1.0 * g)
            d0 = -L2(g, g)
        alpha = 1.0
        ok = False
        mt = Function(V)
        for _ in range(60):
            mt.assign(m + alpha * p)
            Jt = float(Jhat(mt))
            if Jt <= Jc + 1.0e-4 * alpha * d0:
                ok = True
                break
            alpha *= 0.5
        if not ok:
            return k + 1, False
        gn = primal_grad()
        s = Function(V).assign(mt - m)
        yv = Function(V).assign(gn - g)
        sy = L2(s, yv)
        if sy > 1.0e-20:
            if len(S) >= history:
                S.pop(0); Y.pop(0); R.pop(0)
            S.append(s); Y.append(yv); R.append(1.0 / sy)
        m.assign(mt)
        Jc = Jt
        g.assign(gn)
        gN = L2(g, g) ** 0.5
    return max_iter, False


def main():
    meshes = [
        ("uniform N=32", UnitSquareMesh(32, 32)),
        ("graded R4", Mesh(f"{MESH_DIR}/graded_R4.msh")),
        ("graded R16", Mesh(f"{MESH_DIR}/graded_R16.msh")),
    ]
    rows = []
    for label, mesh in meshes:
        Jhat, V = build_problem(mesh)
        tao_its, tao_conv = solve_tao_lmvm(Jhat)
        Jhat, V = build_problem(mesh)
        cus_its, cus_conv = solve_custom_lbfgs(Jhat, V)
        rows.append((label, tao_its, tao_conv, cus_its, cus_conv))

    print("\n=== same problem, L2 inner product on the control ===")
    print(f"{'mesh':>14}  {'TAO/LMVM (before)':>18}  "
          f"{'custom L-BFGS (fix)':>20}")
    for label, ti, tc, ci, cc in rows:
        tcell = f"{ti}{'' if tc else ' (capped)'}"
        ccell = f"{ci}{'' if cc else ' (capped)'}"
        print(f"{label:>14}  {tcell:>18}  {ccell:>20}")


if __name__ == "__main__":
    main()
