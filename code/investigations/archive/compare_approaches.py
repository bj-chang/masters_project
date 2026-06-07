"""The mesh-dependence investigation in one comparison.

Same Poisson distributed-control problem, L2 inner product on the
control, on a uniform mesh and a graded (Netgen) mesh.

Stage 1  ORIGINAL: library TAO/LMVM with Control(m, riesz_map="L2").
         Fine on the uniform mesh, mesh-dependent on the graded one.

Stage 2  WORKAROUND: a hand-rolled L-BFGS in the L2 inner product.
         Mesh-independent on both.

Stage 3  WHY. Two things we initially blamed, ruled out by the numbers,
         and the real cause:
           (3a) the "control update" override (build a second
                Control and call setGradientNorm again): identical to
                Stage 1, so it was NOT the cause;
           (3b) the custom L-BFGS with its initial-Hessian rescaling
                turned OFF: matches Stage 1, so the rescaling IS the
                cause. TAO, given a fixed custom B0 = M^{-1} via
                setLMVMH0, does not apply the standard
                gamma_k = <s,y>/<y,y> rescaling.

Run from ~/masters_project/.
"""

from firedrake import *
from firedrake.adjoint import *
from pyadjoint import Control as PAControl, MinimizationProblem
from pyadjoint.optimization.tao_solver import TAOSolver, RieszMapMat

ALPHA = 1.0e-4
MESH_DIR = "code/mesh_generation"
LU = {"ksp_type": "preonly", "pc_type": "lu"}


def build(mesh):
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


def tao_lmvm(Jhat, control_update_override=False):
    """Library TAO/LMVM. `control_update_override` replays the redundant
    setGradientNorm override (Stage 3a)."""
    params = {
        "tao_type": "lmvm",
        "tao_gatol": 1.0e-7, "tao_grtol": 0.0, "tao_gttol": 0.0,
        "tao_lmvm_num_vecs": 5, "tao_max_it": 3000, "tao_max_funcs": 10000,
    }
    solver = TAOSolver(MinimizationProblem(Jhat), parameters=params)
    if control_update_override:
        c = PAControl(Jhat.controls[0].control, riesz_map="L2")
        minv = RieszMapMat([c], comm=solver.tao.getComm())
        solver.tao.setGradientNorm(minv)
        solver._keep = (minv, c)
    solver.solve()
    return solver.tao.getIterationNumber()


def custom_lbfgs(Jhat, V, rescale=True, eps=1.0e-7, max_iter=3000, history=5):
    """Hand-rolled L-BFGS in the L2 inner product. `rescale` toggles the
    gamma_k initial-Hessian rescaling (off => Stage 3b)."""
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
            return k
        q = Function(V).assign(g)
        al = []
        for s_i, y_i, r_i in zip(reversed(S), reversed(Y), reversed(R)):
            a = r_i * L2(s_i, q)
            al.append(a)
            q.assign(q - a * y_i)
        gam = 1.0
        if S and rescale:
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
            return max_iter
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
    return max_iter


def main():
    meshes = [
        ("uniform N=32", UnitSquareMesh(32, 32)),
        ("graded R16", Mesh(f"{MESH_DIR}/graded_R16.msh")),
    ]
    res = {}
    for label, mesh in meshes:
        res[("s1", label)] = tao_lmvm(*([build(mesh)[0]]))
        Jhat, V = build(mesh)
        res[("s2", label)] = custom_lbfgs(Jhat, V, rescale=True)
        res[("s3a", label)] = tao_lmvm(build(mesh)[0],
                                       control_update_override=True)
        Jhat, V = build(mesh)
        res[("s3b", label)] = custom_lbfgs(Jhat, V, rescale=False)

    labels = [m[0] for m in meshes]
    print("\n=== iteration counts (same problem, L2 control metric) ===")
    head = f"{'':40}" + "".join(f"{l:>14}" for l in labels)
    print(head)
    rows = [
        ("s1", "Stage 1  TAO/LMVM (riesz_map=L2)"),
        ("s2", "Stage 2  custom L-BFGS"),
        ("s3a", "Stage 3a TAO + control-update override"),
        ("s3b", "Stage 3b custom L-BFGS, rescaling OFF"),
    ]
    for key, name in rows:
        print(f"{name:40}"
              + "".join(f"{res[(key, l)]:>14}" for l in labels))
    print("\n3a == Stage 1  -> the control update was a no-op, not the cause.")
    print("3b == Stage 1  -> the missing gamma_k rescaling is the cause.")


if __name__ == "__main__":
    main()
