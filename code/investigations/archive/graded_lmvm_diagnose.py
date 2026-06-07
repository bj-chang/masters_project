"""Why is TAO/LMVM with Control(m, riesz_map="L2") slow on graded meshes?

Vanilla pyadjoint TAOSolver/LMVM with Control(m, riesz_map="L2") takes
~6 iterations on uniform meshes but 522-670 on the Netgen graded meshes
(522 with the Armijo line search, 670 with more-thuente), for the same
Poisson distributed-control problem. This script identifies the cause.

Ruled out by separate experiments:
  - the project's setGradientNorm override (vanilla == override);
  - inaccurate Riesz solves (default L2 Riesz agrees with LU to 1e-16
    on both uniform and graded meshes; printed below);
  - the inner product in the BFGS recursion (with a correct line search,
    swapping the recursion metric L2 -> l2 only changes 7 -> 19 iters);
  - the TAO line search and LMVM scale_type (522-670 across variants).

The cause is the *adaptive initial-Hessian rescaling*. Standard L-BFGS
rescales B0 each step by gamma_k = <s,y>/<y,y>. When pyadjoint installs
a custom J0 = M^{-1} via setLMVMH0, TAO uses it as a FIXED initial
Hessian and does not apply this rescaling (which is why scale_type has
no effect). Below, ONE L-BFGS is run with the rescaling on and off:
with it, 7 iterations; without it, 522 -- matching vanilla TAO/Armijo
exactly. On uniform meshes the fixed scale is already about right, so
the rescaling barely matters; that is why TAO looks fine there.

Run from ~/masters_project/.
"""

import numpy as np

from firedrake import *
from firedrake.adjoint import *

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


def lbfgs(Jhat, V, rescale, eps=1.0e-7, max_iter=3000, history=5):
    """L-BFGS in the L2 inner product. `rescale` toggles the standard
    B0 rescaling gamma_k = <s,y>_L2 / <y,y>_L2."""

    def L2(a, b):
        return float(assemble(inner(a, b) * dx))

    def primal_grad():
        return Jhat.derivative().riesz_representation(
            riesz_map="L2", solver_options=LU)

    control = Jhat.controls[0]
    m = Function(V).assign(control.tape_value())
    Jc = float(Jhat(m))
    g = primal_grad()
    gN = L2(g, g) ** 0.5
    S, Y, R = [], [], []

    for k in range(max_iter):
        if gN <= eps:
            return k, True, gN
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
            return k + 1, False, gN
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
    return max_iter, False, gN


def riesz_accuracy(Jhat, V):
    Jhat(Function(V))
    dual = Jhat.derivative()
    gd = dual.riesz_representation(riesz_map="L2")
    gl = dual.riesz_representation(riesz_map="L2", solver_options=LU)
    diff = Function(V).assign(gd - gl)
    return (float(assemble(inner(diff, diff) * dx)) ** 0.5
            / float(assemble(inner(gl, gl) * dx)) ** 0.5)


def main():
    meshes = [
        ("uniform N=32", UnitSquareMesh(32, 32)),
        ("graded R16", Mesh(f"{MESH_DIR}/graded_R16.msh")),
    ]

    print("=== default vs LU L2 Riesz solve (rules out solve-accuracy) ===")
    for label, mesh in meshes:
        Jhat, V = build(mesh)
        print(f"  {label:14s}  rel diff = {riesz_accuracy(Jhat, V):.2e}")

    print("\n=== L2 L-BFGS, adaptive B0 rescaling on vs off ===")
    print(f"{'mesh':>14}  {'gamma rescale':>13}  {'iters':>6}  {'conv':>5}")
    for label, mesh in meshes:
        for rescale in (True, False):
            Jhat, V = build(mesh)
            its, conv, _ = lbfgs(Jhat, V, rescale)
            print(f"{label:>14}  {str(rescale):>13}  {its:>6}  "
                  f"{str(conv):>5}", flush=True)
    print("\nWithout rescaling, graded R16 matches vanilla TAO/LMVM"
          " (522, Armijo); with it, 7. That rescaling is the cause.")


if __name__ == "__main__":
    main()
