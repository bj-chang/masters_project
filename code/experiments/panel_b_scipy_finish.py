"""finishes the panel (b) scipy row at r64 and r128, capped so it terminates"""
from pyadjoint import minimize
from meshdep.meshes import graded_unit_square_from_file

MAXITER = 2000
MAXFUN = 3000
EPS = 1.0e-7


class _Converged(Exception):
    pass


def build_h1reg_l2metric(mesh, alpha):
    from firedrake import (
        DirichletBC, Function, FunctionSpace, SpatialCoordinate,
        TestFunction, assemble, dx, grad, inner, pi, sin, solve,
    )
    from firedrake.adjoint import (
        Control, ReducedFunctional, continue_annotation,
        pause_annotation, set_working_tape,
    )
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
        J = assemble(0.5 * (u - d)**2 * dx
                     + 0.5 * alpha * (m**2 + inner(grad(m), grad(m))) * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map="l2"), tape=tape)
    pause_annotation()
    return Jhat, V, m


def run(mesh, realised, r, alpha=1.0e-4):
    from firedrake import Function, assemble, dx, grad, inner

    Jhat, V, m_func = build_h1reg_l2metric(mesh, alpha)
    control = Jhat.controls[0]

    def h1_norm_of_grad(xk):
        m_xk = Function(V)
        m_xk.dat.data[:] = xk
        Jhat(m_xk)
        Jhat.derivative()
        primal = m_func._ad_convert_riesz(
            control.block_variable.adj_value, riesz_map="H1")
        return float(assemble(
            (inner(primal, primal) + inner(grad(primal), grad(primal)))
            * dx)) ** 0.5

    it = [0]
    last = [None]

    def cb(xk):
        it[0] += 1
        g = h1_norm_of_grad(xk)
        last[0] = g
        if it[0] % 25 == 0:
            print(f"    [R{r}] iter {it[0]:4d}  |g|_H1={g:.3e}", flush=True)
        if g <= EPS:
            raise _Converged()

    options = {"gtol": 1e-30, "ftol": 1e-30,
               "maxiter": MAXITER, "maxfun": MAXFUN}
    converged = False
    try:
        minimize(Jhat, method="L-BFGS-B", options=options, callback=cb)
        converged = last[0] is not None and last[0] <= EPS
    except _Converged:
        converged = True

    status = (f"CONVERGED at {it[0]}" if converged
              else f"NOT converged after {it[0]} iters (|g|_H1={last[0]:.3e}, "
                   f"hit maxiter={MAXITER}/maxfun={MAXFUN})")
    print(f"R{r} (realised {realised:.2f}): {status}", flush=True)
    return it[0], converged, last[0]


def main():
    for r in (64, 128):
        mesh, realised = graded_unit_square_from_file(
            f"mesh_generation/graded_R{r}.msh")
        print(f"=== R{r}  realised={realised:.2f}  cells={mesh.num_cells()} ===",
              flush=True)
        run(mesh, realised, r)


if __name__ == "__main__":
    main()
