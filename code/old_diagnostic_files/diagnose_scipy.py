"""Quick test of solve_with_scipy_external_check.

On a ratio-4, n=32 graded mesh, run SciPy L-BFGS-B with the l^2
internal metric but the L^2 external convergence check. Compare
against TAO L^2 row (which converges) and Schwedes' Table 2.2
(panel a, ratio 4: SciPy 25 iters, TAO LMVM-L2 22 iters).
"""

import numpy as np

from firedrake import (
    DirichletBC, Function, FunctionSpace, SpatialCoordinate,
    TestFunction, assemble, dx, grad, inner, pi, sin, solve,
)
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, pause_annotation,
    set_working_tape,
)

from pdeopt.meshes import graded_unit_square_tensor
from pdeopt.optimisers import (
    solve_with_tao,
    solve_with_scipy_external_check,
)


def build_problem(mesh, alpha=1.0e-4, riesz_map="l2"):
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

    print("--- SciPy L-BFGS-B, l^2 algo, L^2 external check ---")
    Jhat = build_problem(mesh, riesz_map="l2")
    result = solve_with_scipy_external_check(
        Jhat, eps=1e-7, test_riesz_map="L2", maxiter=1000,
    )
    print(f"  iterations          = {result['iterations']}")
    print(f"  converged           = {result['converged']}")
    print(f"  final J             = {result['final_J']:.6e}")
    print(f"  final ||g||_L2      = {result['final_grad_norm_H']}")

    print("\n--- TAO LMVM, L^2 algo, L^2 external check (sanity) ---")
    Jhat2 = build_problem(mesh, riesz_map="L2")
    result2 = solve_with_tao(
        Jhat2, tao_gatol=1e-7, tao_max_it=500, tao_max_funcs=1000,
        history=5, convergence_riesz_map="L2",
    )
    print(f"  iterations = {result2['iterations']}")
    print(f"  final J    = {result2['final_J']:.6e}")


if __name__ == "__main__":
    main()
