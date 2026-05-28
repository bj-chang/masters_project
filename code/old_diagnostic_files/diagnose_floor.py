"""Fast diagnostic: where does the L^2 gradient norm get stuck?

Runs TAO/LMVM (riesz_map='l2') with the L^2 convergence-metric override
and verbose monitor enabled, capped at 200 iterations. Prints the TAO
residual at each iteration so we can see whether the L^2 gradient norm
is genuinely decreasing toward 1e-7 (in which case we just need more
iterations) or stuck at some floor (in which case we have a deeper bug).
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
from pdeopt.optimisers import solve_with_tao


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
        # Match poisson_control: LU forward solve to keep noise floor low.
        solve(F == 0, u, bcs=bc,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
        J = assemble(0.5 * (u - d)**2 * dx + 0.5 * alpha * m**2 * dx)
        Jhat = ReducedFunctional(J, Control(m, riesz_map=riesz_map),
                                 tape=tape)
    pause_annotation()
    return Jhat, V


def main():
    n = 32
    for ratio_target in [4.0]:
        mesh, ratio = graded_unit_square_tensor(h_ratio=ratio_target, n=n)
        print(f"\n##### ratio={ratio_target}, n={n}, realised={ratio:.2f} #####")

        for algo_riesz in ["l2", "L2"]:
            print(f"\n--- TAO/LMVM, riesz_map={algo_riesz!r}, "
                  f"convergence='L2', gatol=1e-7, cap=500 ---")
            Jhat, V = build_problem(mesh, riesz_map=algo_riesz)
            result = solve_with_tao(
                Jhat, tao_gatol=1.0e-7,
                tao_max_it=500, tao_max_funcs=1000,
                history=5, verbose=False, convergence_riesz_map="L2",
            )
            print(f"  iterations = {result['iterations']}")
            print(f"  converged  = {result['converged']}")
            print(f"  final J    = {result['final_J']:.6e}")


if __name__ == "__main__":
    main()
