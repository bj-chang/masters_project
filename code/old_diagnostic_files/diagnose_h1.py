"""Why is the TAO LMVM H1 row not converging?

Three quick comparisons at ratio 4, n=32:

  (a) TAO LMVM with riesz_map='H1' and NO override.
      Expect: TAO's own gnorm is ||g||_H1, convergence should be fast.

  (b) Same with override convergence_riesz_map='H1'.
      Should be identical to (a) since the override Mat equals the
      original one.

  (c) Sanity: TAO LMVM with riesz_map='L2' (known to converge in ~6
      iters). For comparison.

Print TAO's verbose monitor for each and the converged reason.
"""

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    set_working_tape,
)

from pdeopt.meshes import graded_unit_square_tensor
from pdeopt.optimisers import solve_with_tao


def build_problem(mesh, alpha=1.0e-4, riesz_map="H1"):
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
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat


def main():
    mesh, ratio = graded_unit_square_tensor(h_ratio=4.0, n=32)
    print(f"ratio-4 mesh, n=32, realised={ratio:.2f}\n")

    for label, riesz, override in [
        ("H1, no override", "H1", None),
        ("H1 + override=H1", "H1", "H1"),
        ("L2, no override (sanity)", "L2", None),
    ]:
        print(f"\n=== {label} ===")
        Jhat = build_problem(mesh, riesz_map=riesz)
        result = solve_with_tao(
            Jhat, tao_gatol=1.0e-7,
            tao_max_it=50, tao_max_funcs=200,
            history=5, verbose=True, convergence_riesz_map=override,
        )
        print(f"  iters={result['iterations']}, "
              f"converged={result['converged']}, J={result['final_J']:.6e}")


if __name__ == "__main__":
    main()
